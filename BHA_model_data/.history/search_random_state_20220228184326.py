import random
import optuna
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

file = 'features_3960_sun.csv'
data = pd.read_csv(file,header=None)
# r2_mean = []
# for seed in range(10):
    # print("第{}次随机划分".format(seed+1))
data = data.sample(frac=1, random_state=9)

features_total = data.iloc[:,:-1]
yields_total = data.iloc[:,-1]

features = np.array(features_total)
yields = np.array(yields_total)
yields = yields.flatten()

X_train = features[:2772,:]
y_train = yields[:2772]
X_test = features[2772:,:]
y_test = yields[2772:]

def define_model(trial):

    random_state = trial.suggest_int('random_state', 0, 100)
    model = RandomForestRegressor(n_estimators = 200, min_samples_split = 2, min_impurity_decrease = 0.05, max_features = 'sqrt', 
                                max_depth = 35, criterion = 'mse', bootstrap = False,random_state = random_state)
    return model


def objective(trial):
    model = define_model(trial)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(pred,y_test)
    trial.report(r2,trial.number)

        # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return r2


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("foo_random_state.log", mode="w"))
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
    
    with open('foo_random_state.log') as f:
        assert f.readline() == "Start optimization.\n"
        assert f.readline().startswith("Finished trial#0 with value:")