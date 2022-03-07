import pandas as pd
import numpy as np
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

file = 'FMSSD_384.csv'
Train_384 = 'FMSSD_384.csv'
Test_171 = 'FMSSD_171.csv'
Test_216 = 'FMSSD_216.csv'
Test_304 = 'FMSSD_304.csv'

Train_G_384 = 'G_384.csv'
Test_G_171 = 'G_171.csv'
Test_G_216 = 'G_216.csv'
Test_G_304 = 'G_304.csv'

def get_Train_Test(X=None,Y=None):
    for i,x in enumerate(X):
        x_y = pd.read_csv(x,header=None)
        y  = pd.read_csv(Y[i],header=None)
        x_y['G'] = y
        x = np.array(x_y.iloc[:,0:-1])
        y = np.array(x_y.iloc[:,-1]).flatten()
        if i == 0:
            featuresTrain = torch.from_numpy(x)
            targetsTrain = torch.from_numpy(y)
            train_loader = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
            batch_size = len(x)
            train_loader = torch.utils.data.DataLoader(train_loader, batch_size = batch_size, shuffle=True)
        else:
            featuresTest = torch.from_numpy(x)
            targetsTest = torch.from_numpy(y)
            batch_size_test = len(y)
            valid_loader = torch.utils.data.TensorDataset(featuresTest,targetsTest)
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
        x = (x-0.5) * 2 * 3.13482
        return x

def define_model(file,n_layers,out_features_total,p_total):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    data = pd.read_csv(file,header=None)
    in_features = len(list(data))
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

DEVICE = torch.device("cuda:0")
EPOCHS = 20000
n_layers = 3
out_features_total = [1087,1654,751]
p_total = [0.3218304497425615,0.4234575918676726,0.3423559591863924]
optimizer_name = 'Adam'
lr = 0.002

model = define_model(file,n_layers,out_features_total,p_total)
model.to(DEVICE)
optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
train_loader,valid_loader = get_Train_Test(X=[Train_384,Test_216],Y=[Train_G_384,Test_G_216])

mae_train_total = []
mae_test_total = []
for epoch in range(EPOCHS):
    mae_train = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
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
        mae_train.append(mean_absolute_error((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))

    mae_train = np.array(mae_train).mean()
    mae_train_total.append(mae_train)
    if (epoch+1) % 10 == 0:
        print('------------------------------------------')
        print('start training {} epoch'.format(epoch+1))
        print('mae_train:{}'.format(mae_train))

    if (epoch+1) == EPOCHS:
        target = np.array((target.to(torch.float)).data.cpu().numpy()).reshape(-1,1)
        pred = np.array(pred.data.cpu().numpy()).reshape(-1,1)
        # target = np.array(target).reshape(-1,1)
        # pred = np.array(pred).reshape(-1,1)
        target_pred_train = np.concatenate((target, pred),axis = 1)
        np.savetxt('FMSSD_G_subset_train_target_pred_'+ 'test_sub'+ '.csv',target_pred_train,delimiter=',')
        np.savetxt('FMSSD_G_subset_r2_train_'+ 'test_sub' + '.csv',mae_train_total,delimiter=',')

    # Validation of the model.
    model.eval()
    
    mae_test = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            if batch_idx * len(data) >= len(data):
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data.to(torch.float))
            pred = output
            mae_test.append(mean_absolute_error((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))

    mae_test = np.array(mae_test).mean()
    mae_test_total.append(mae_test)
    if (epoch+1) % 10 == 0:
        print('mae_test:{}'.format(mae_test))
        print('------------------------------------------')
    
    if (epoch+1) == EPOCHS:
        target = np.array((target.to(torch.float)).data.cpu().numpy()).reshape(-1,1)
        pred = np.array(pred.data.cpu().numpy()).reshape(-1,1)
        # target = np.array(target).reshape(-1,1)
        # pred = np.array(pred).reshape(-1,1)
        target_pred_test = np.concatenate((target, pred),axis = 1)
        np.savetxt('FMSSD_G_subset_test_target_pred_'+ 'test_sub' + '.csv',target_pred_test,delimiter=',')
        np.savetxt('FMSSD_G_subset_r2_test_'+ 'test_sub' + '.csv',mae_test_total,delimiter=',')


