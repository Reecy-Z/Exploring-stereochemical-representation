import pandas as pd
import numpy as np
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import r2_score


DEVICE = torch.device("cuda:0")
EPOCHS = 10000
OUT_DIR = './'
file = 'features_3960_sun.csv'
test_1 = 'test1.csv'
test_2 = 'test2.csv'
test_3 = 'test3.csv'
test_4 = 'test4.csv'

def get_Train_Test(objs_train,Test):
    Train = pd.concat(objs_train, ignore_index=False)
    X_train = np.array(Train.iloc[:,0:-1])
    y_train = np.array(Train.iloc[:,-1]).flatten()
    X_test = np.array(Test.iloc[:,0:-1])
    y_test = np.array(Test.iloc[:,-1]).flatten()

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

def get_data(test_1,test_2,test_3,test_4):
    test_1 = pd.read_csv(test_1,header=None)
    test_2 = pd.read_csv(test_2,header=None)
    test_3  = pd.read_csv(test_3,header=None)
    test_4 = pd.read_csv(test_4,header=None)

    objs_train_234 = [test_2,test_3,test_4]
    objs_train_134 = [test_1,test_3,test_4]
    objs_train_124 = [test_1,test_2,test_4]
    objs_train_123 = [test_1,test_2,test_3]

    Train_234, Test_1 = get_Train_Test(objs_train_234,test_1)
    Train_134, Test_2 = get_Train_Test(objs_train_134,test_2)
    Train_124, Test_3 = get_Train_Test(objs_train_124,test_3)
    Train_123, Test_4 = get_Train_Test(objs_train_123,test_4)

    return [Train_234, Train_134, Train_124, Train_123], [Test_1, Test_2, Test_3, Test_4]

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

model = define_model(file,n_layers,out_features_total,p_total)
model.to(DEVICE)
optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

Train_loader,Valid_loader = get_data(test_1,test_2,test_3,test_4)
r2_train_total = []
r2_test_total = []
for epoch in range(EPOCHS):
    r2_train = []
    model.train()
    for batch_idx, (data, target) in enumerate(Train_loader[3]):
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
        np.savetxt('MALD_act_subset_train_target_pred_'+ 'test4' + '.csv',target_pred_train,delimiter=',')
        np.savetxt('MALD_act_subset_r2_train_'+ 'test4' + '.csv',r2_train_total,delimiter=',')

    # Validation of the model.
    model.eval()
    
    r2_test = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(Valid_loader[3]):
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
        np.savetxt('MALD_act_subset_test_target_pred_'+ 'test4' + '.csv',target_pred_test,delimiter=',')
        np.savetxt('MALD_act_subset_r2_test_'+ 'test4' + '.csv',r2_test_total,delimiter=',')



