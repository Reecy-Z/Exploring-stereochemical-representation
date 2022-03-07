import logging
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# 划分3个小数据集
Train_384 = 'feature_384.csv'
Test_171 = 'feature_171.csv'
Test_216 = 'feature_216.csv'
Test_304 = 'feature_304.csv'

Train_G_384 = 'G_384.csv'
Test_G_171 = 'G_171.csv'
Test_G_216 = 'G_216.csv'
Test_G_304 = 'G_304.csv'

Train_384 = pd.read_csv(Train_384,header=None)
Test_171 = pd.read_csv(Test_171,header=None)
Test_216 = pd.read_csv(Test_216,header=None)
Test_304 = pd.read_csv(Test_304,header=None)

Train_G_384  = pd.read_csv(Train_G_384,header=None)
Test_G_171 = pd.read_csv(Test_G_171,header=None)
Test_G_216 = pd.read_csv(Test_G_216,header=None)
Test_G_304= pd.read_csv(Test_G_304,header=None)

Train_384['G'] = Train_G_384
Test_171['G'] = Test_G_171
Test_216['G'] = Test_G_216
Test_304['G'] = Test_G_304

X_train_384 = np.array(Train_384.iloc[:,0:-1])
y_train_384 = np.array(Train_384.iloc[:,-1]).flatten()

X_test_171 = np.array(Test_171.iloc[:,0:-1])
y_test_171 = np.array(Test_171.iloc[:,-1]).flatten()

X_test_216 = np.array(Test_216.iloc[:,0:-1])
y_test_216 = np.array(Test_216.iloc[:,-1]).flatten()

X_test_304 = np.array(Test_304.iloc[:,0:-1])
y_test_304 = np.array(Test_304.iloc[:,-1]).flatten()

def get_data(data,seed):
    data = data.sample(frac=1, random_state=seed)
    features_total = data.iloc[:,:-2]
    yields_total = data.iloc[:,-1]

    features = np.array(features_total)
    yields = np.array(yields_total)
    yields = yields.flatten()

    X_train = features[:600,:]
    y_train = yields[:600]
    X_test = features[600:,:]
    y_test = yields[600:]
    return X_train,y_train,X_test,y_test

def get_model(trial):
    params = {'max_depth': trial.suggest_int("max_depth", 1, 20),
              'learning_rate': trial.suggest_float("learning_rate", 0.01, 1),
              'n_estimators': trial.suggest_int("n_estimators", 1, 300), 
              'silent':1, 
              'objective':'reg:linear', 
              'nthread':-1,
              'gamma': trial.suggest_float("gamma", 0, 10), 
              'min_child_weight': trial.suggest_int("min_child_weight", 0, 10), 
              'max_delta_step': trial.suggest_int("max_delta_step", 0, 10), 
              'subsample': trial.suggest_float("subsample", 0.1, 1), 
              'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1), 
              'colsample_bylevel':1, 
              'reg_alpha':0, 
              'reg_lambda':1, 
              'scale_pos_weight':1, 
              'seed':1440}
    model = xgb.XGBRegressor(**params)
    return model

def objective(trial):
    mae_mean_total = []

    model = get_model(trial)
    model.fit(X_train_384, y_train_384, eval_metric='mae', verbose = True)

    pred_171 = model.predict(X_test_171)
    mae_171 = mean_absolute_error(pred_171,y_test_171)
    print('171数据集mae为:{}'.format(mae_171))

    pred_216 = model.predict(X_test_216)
    mae_216 = mean_absolute_error(pred_216,y_test_216)
    print('216数据集mae为:{}'.format(mae_216))

    pred_304 = model.predict(X_test_304)
    mae_304 = mean_absolute_error(pred_304,y_test_304)
    print('304数据集mae为:{}'.format(mae_304))

    mae_mean_total.append(mae_171)
    mae_mean_total.append(mae_216)
    mae_mean_total.append(mae_304)
    mae_mean_total = np.array(mae_mean_total).mean()
    print(mae_mean_total)
    trial.report(mae_mean_total,trial.number)

        # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return mae_mean_total


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("foo_xgboost_3_split.log", mode="w"))
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    study = optuna.create_study(direction="minimize")
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
    
    with open('foo_xgboost_3_split.log') as f:
        assert f.readline() == "Start optimization.\n"
        assert f.readline().startswith("Finished trial#0 with value:")