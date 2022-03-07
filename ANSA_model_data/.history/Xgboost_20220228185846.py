import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

file = 'FMSSD_total.csv'

def get_data(file,seed):
    data = pd.read_csv(file,header=None)
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

def get_model():
    params = {'max_depth': 23, 'learning_rate': 0.16318915920493737, 'n_estimators': 62, 'gamma': 0.17735702393456645, 
              'min_child_weight': 5, 'max_delta_step': 100, 'subsample': 0.9332230398089334, 'colsample_bytree': 0.807431352173656,
              'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'seed':1440, 'silent':1, 'objective':'reg:linear', 'nthread':-1, 'colsample_bylevel':1}
    model = xgb.XGBRegressor(**params)
    return model

mae_total = []
for seed in range(10):
    print('random split {}'.format(seed+1))
    model = get_model()
    X_train,y_train,X_test,y_test = get_data(file,seed)
    model.fit(X_train, y_train, eval_metric='mae', verbose = True)
    pred = model.predict(X_test)
    mae = mean_absolute_error(pred,y_test)
    print(mae)
    mae_total.append(mae)
mae_mean = np.array(mae_total).mean()
print(mae_mean)