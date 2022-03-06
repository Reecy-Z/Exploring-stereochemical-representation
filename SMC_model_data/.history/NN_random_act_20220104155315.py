import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

DEVICE = torch.device("cuda:0")
file = 'suzuki_MALD.csv'
def get_data(file,seed):
    data = pd.read_csv(file,header=None)
    data = data.sample(frac=1, random_state=seed)

    features_total = data.iloc[:,:-1]
    yields_total = data.iloc[:,-1]

    features = np.array(features_total)
    yields = np.array(yields_total)
    yields = yields.flatten()

    X_train = features[:4032,:]
    y_train = yields[:4032]
    X_test = features[4032:,:]
    y_test = yields[4032:]

    featuresTrain = torch.from_numpy(X_train)
    targetsTrain = torch.from_numpy(y_train)
    featuresTest = torch.from_numpy(X_test)
    targetsTest = torch.from_numpy(y_test)

    train_loader = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
    valid_loader = torch.utils.data.TensorDataset(featuresTest,targetsTest)

    batch_size = len(X_train)
    batch_size_test = len(X_test)

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size = batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_loader, batch_size = batch_size_test, shuffle=True)
    return train_loader, valid_loader

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    """
    return ((y_true - y_pred) ** 2).mean() / 2

class Act_fun(nn.Module):
    def __init__(self):
        super(Act_fun, self).__init__()
        
    def forward(self, x):
        x = torch.sigmoid(x)
        x = x * 100
        return x

def define_model(file,n_layers,out_features_total,p_total):
    data = pd.read_csv(file,header=None)
    in_features = len(list(data)) - 1
    layers = []
    for index in range(n_layers):
        layers.append(nn.Linear(in_features, out_features_total[index]))
        layers.append(nn.BatchNorm1d(out_features_total[index]))
        layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(p_total[index]))

        in_features = out_features_total[index]
    layers.append(nn.Linear(in_features,1))
    layers.append(nn.BatchNorm1d(1))
    layers.append(Act_fun())

    return nn.Sequential(*layers)

n_layers = 3
out_features_total = [1087,1654,751]
p_total = [0.3218304497425615,0.4234575918676726,0.3423559591863924]
optimizer_name = 'Adam'
lr = 0.0008430633857811011
EPOCHS = 10000

r2_10 = []
for item, seed in enumerate(range(10)):
    model = define_model(file,n_layers,out_features_total,p_total)
    model.to(DEVICE)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    r2_train_total = []
    r2_test_total = []
    print('随即划分第{}次'.format(item+1))
    train_loader, valid_loader = get_data(file,seed)
    
    # Training of the model.
    for epoch in range(EPOCHS):
        r2_train = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * len(data) >= len(data):
                break

            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data.to(torch.float))
            output = output.squeeze(-1)
            loss = squared_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output
            r2_train.append(r2_score((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy())) 

        r2_train = np.array(r2_train).mean()
        r2_train_total.append(r2_train)
        
        if (epoch+1) % 10 == 0:
            print('------------------------------------------')
            print('开始训练第{}轮'.format(epoch+1))
            print('r2_train:{}'.format(r2_train))
        
        if (epoch+1) == EPOCHS:
            target = np.array((target.to(torch.float)).data.cpu().numpy()).reshape(-1,1)
            pred = np.array(pred.data.cpu().numpy()).reshape(-1,1)
            target_pred_train = np.concatenate((target, pred),axis = 1)
            np.savetxt('MALD_act_random_train_target_pred_'+ str(seed+1) + '.csv',target_pred_train,delimiter=',')
            np.savetxt('MALD_act_random_r2_train_'+ str(seed+1) + '.csv',r2_train_total,delimiter=',')

        # Validation of the model.
        model.eval()
        
        r2_test = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                if batch_idx * len(data) >= len(data):
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data.to(torch.float))
                pred = output
                r2_test.append(r2_score((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy())) 

        r2_test = np.array(r2_test).mean()
        r2_test_total.append(r2_test)

        if (epoch+1) % 10 == 0:
            print('r2_test:{}'.format(r2_test))
            print('------------------------------------------')
        
        if (epoch+1) == EPOCHS:
            target = np.array((target.to(torch.float)).data.cpu().numpy()).reshape(-1,1)
            pred = np.array(pred.data.cpu().numpy()).reshape(-1,1)
            target_pred_test = np.concatenate((target, pred),axis = 1)
            np.savetxt('MALD_act_random_test_target_pred_'+ str(seed+1) + '.csv',target_pred_test,delimiter=',')
            np.savetxt('MALD_act_random_r2_test_'+ str(seed+1) + '.csv',r2_test_total,delimiter=',')
    r2_10.append(r2_test)
r2_10 = np.array(r2_10).mean()
print(r2_10)


