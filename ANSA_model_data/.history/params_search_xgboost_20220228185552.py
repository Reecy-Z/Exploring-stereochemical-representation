import logging
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

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

file = file = 'FMSSD_total.csv'
data = pd.read_csv(file,header=None)

def objective(trial):
    mae_mean = []
    for seed in range(10):
        print('random split {}'.format(seed+1))
        X_train,y_train,X_test,y_test = get_data(data,seed)
        model = get_model(trial)
        model.fit(X_train, y_train, eval_metric='mae', verbose = True)
        pred = model.predict(X_test)
        mae = mean_absolute_error(pred,y_test)
        print(mae)
        mae_mean.append(mae)
    mae_mean = np.array(mae_mean).mean()
    print('mean of 10 times:{}'.format(mae_mean))
    trial.report(mae_mean,trial.number)

        # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return mae_mean

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("foo_xgboost.log", mode="w"))
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
    
    with open('foo_xgboost.log') as f:
        assert f.readline() == "Start optimization.\n"
        assert f.readline().startswith("Finished trial#0 with value:")

