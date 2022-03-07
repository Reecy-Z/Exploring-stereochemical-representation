import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.utils import shuffle
from draw_ANSA_80 import plot_ANN_80_20_MAE_update

DEVICE = torch.device("cuda:0")
file = 'MAD_total.csv'

def get_data(file):
    data = pd.read_csv(file,header=None)
    Train = np.array(data[(data.iloc[:,-2]) < 80])
    Test = np.array(data[(data.iloc[:,-2]) >= 80])
    Train = shuffle(Train,random_state = 42)

    X_train = np.array(Train[:,0:-2])
    y_train = np.array(Train[:,-1]).flatten()
    X_test = np.array(Test[:,0:-2])
    y_test = np.array(Test[:,-1]).flatten()

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
    in_features = len(list(data)) - 2
    layers = []
    for index in range(n_layers):
        layers.append(nn.Linear(in_features, out_features_total[index]))
        layers.append(nn.BatchNorm1d(out_features_total[index]))
        layers.append(nn.Dropout(p_total[index]))
        layers.append(nn.Sigmoid())
        
        in_features = out_features_total[index]
    layers.append(nn.Linear(in_features,1))
    layers.append(nn.BatchNorm1d(1))
    layers.append(Act_fun())

    return nn.Sequential(*layers)

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    """
    return ((y_true - y_pred) ** 2).mean() / 2

n_layers = 3
out_features_total = [1087,1654,751]
p_total = [0.3218304497425615,0.4234575918676726,0.3423559591863924]
optimizer_name = 'Adam'
lr = 0.002

EPOCHS = 100

model = define_model(file,n_layers,out_features_total,p_total)
model.to(DEVICE)
optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
train_loader, valid_loader = get_data(file)

mae_train_total = []
mae_test_total = []
# Training of the model.
r2_best = 0
for epoch in range(EPOCHS):
    mae_train = []
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
        mae_train.append(mean_absolute_error((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))

        train_target = target.to(torch.float).data.cpu().numpy()
        train_pred = pred.data.cpu().numpy()

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
        np.savetxt('MAD_G_80_train_target_pred.csv',target_pred_train,delimiter=',')
        np.savetxt('MAD_G_80_mae_train.csv',mae_train_total,delimiter=',')
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

            test_target = target.to(torch.float).data.cpu().numpy()
            test_pred = pred.data.cpu().numpy()

    mae_test = np.array(mae_test).mean()
    mae_test_total.append(mae_test)

    r2 = r2_score(test_target,test_pred)
    rmse_test = mean_squared_error(test_target,test_pred)**0.5
    if r2 > r2_best:
        r2_best = r2
        plot_ANN_80_20_MAE_update(test_pred,
                                  train_pred,
                                  r2,
                                  mae_test,
                                  rmse_test,
                                  train_target,
                                  test_target,
                                  epoch)

    if (epoch+1) % 10 == 0:
        print('mae_test:{}'.format(mae_test))
        print('r2_test:{}'.format(r2))
        print('------------------------------------------')
    
    if (epoch+1) == EPOCHS:
        target = np.array((target.to(torch.float)).data.cpu().numpy()).reshape(-1,1)
        pred = np.array(pred.data.cpu().numpy()).reshape(-1,1)
        # target = np.array(target).reshape(-1,1)
        # pred = np.array(pred).reshape(-1,1)
        target_pred_test = np.concatenate((target, pred),axis = 1)
        np.savetxt('MAD_G_80_test_target_pred.csv',target_pred_test,delimiter=',')
        np.savetxt('MAD_G_80_mae_test.csv',mae_test_total,delimiter=',')

