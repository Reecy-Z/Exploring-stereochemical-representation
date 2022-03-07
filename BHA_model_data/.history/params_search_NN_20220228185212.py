import os
import optuna
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import r2_score

DEVICE = torch.device("cuda:0")
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 20000
OUT_DIR = './'

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    """
    return ((y_true - y_pred) ** 2).mean() / 2

def define_model(trial,file):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 6)
    layers = []

    data = pd.read_csv(file,header=None)
    in_features = len(list(data)) - 1
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 2000)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
        layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features,1))
    # layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def get_data(file,seed):
    data = pd.read_csv(file,header=None)
    data = data.sample(frac=1, random_state=seed)

    features_total = data.iloc[:,:-1]
    yields_total = data.iloc[:,-1]

    features = np.array(features_total)
    yields = np.array(yields_total)
    yields = yields.flatten()

    X_train = features[:2772,:]
    y_train = yields[:2772]
    X_test = features[2772:,:]
    y_test = yields[2772:]

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


def objective(trial):

    # Generate the model.
    model = define_model(trial,'MAD_total.csv').to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 0.0001, 0.2, log=False)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader = get_data('MAD_total.csv',1)
    best_r2 = 0
    # Training of the model.
    for epoch in range(EPOCHS):
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

        # Validation of the model.
        model.eval()
        # correct = 0
        
        r2_total = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * len(data) >= len(data):
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                # data = Variable(data.to(torch.float))  
                output = model(data.to(torch.float))
                # Get the index of the max log-probability.
                # pred = output.argmax(dim=1, keepdim=True)
                pred = output
                # correct += pred.eq(target.view_as(pred)).sum().item()
                pred = torch.where(torch.isnan(pred), torch.full_like(pred, 0), pred)
                pred = torch.where(torch.isinf(pred), torch.full_like(pred, 0), pred)
                r2_total.append(r2_score((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy())) 
                # rmse += mean_squared_error((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()) ** 0.5

        r2_total = np.array(r2_total).mean()
        # rmse_total = rmse / min(len(valid_loader.dataset), N_VALID_EXAMPLES)
        if (epoch+1) % 10 == 0:
            print('------------------------------------------')
            print('start training {} epoch'.format(epoch+1))
            print('r2_test:{}'.format(r2_total))
            print('------------------------------------------')
        trial.report(r2_total, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()



    if r2_total > best_r2:
        best_r2 = r2_total
        for f in os.listdir(OUT_DIR):
            if(os.path.isfile(OUT_DIR+f)) and 'best' in f:#必須是目录，不要是文件
                os.remove(OUT_DIR+f)
        torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            },'{}_best_state_dict.pkl'.format(trial.number))
        
        torch.save(model,'{}_best_model.pkl'.format(trial.number))
        
    return r2_total

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("foo.log", mode="w"))
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    study = optuna.create_study(direction="maximize")
    study.optimize(objective)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    with open('foo.log') as f:
        assert f.readline() == "Start optimization.\n"
        assert f.readline().startswith("Finished trial#0 with value:")
    
