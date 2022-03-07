import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

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
    # 目前最好参数 mae_mean = 0.14355861009355367
    # params = {'max_depth': 14, 'learning_rate': 0.02765181316110176, 'n_estimators': 175, 'gamma': 0.003202373087580569, 
    #           'min_child_weight': 3, 'max_delta_step': 3, 'subsample': 0.6386357867883933, 'colsample_bytree': 0.5504889379121635,
    #           'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'seed':1440, 'silent':1, 'objective':'reg:linear', 'nthread':-1, 'colsample_bylevel':1}
    params = {'max_depth': 18, 'learning_rate': 0.1255430518663882, 'n_estimators': 150, 'gamma': 0.11753830592381259, 
              'min_child_weight': 1, 'max_delta_step': 47, 'subsample': 0.9693940497564076, 'colsample_bytree': 0.9275176209308276,
              'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'seed':1440, 'silent':1, 'objective':'reg:linear', 'nthread':-1, 'colsample_bylevel':1}
    model = xgb.XGBRegressor(**params)
    return model

mae_total = []
for seed in range(10):
    print('第{}次随机划分'.format(seed+1))
    model = get_model()
    X_train,y_train,X_test,y_test = get_data(file,seed)
    # , eval_set = [(X_test, y_test)],early_stopping_rounds=100
    model.fit(X_train, y_train, eval_metric='mae', verbose = True)
    pred = model.predict(X_test)
    mae = mean_absolute_error(pred,y_test)
    print(mae)
    mae_total.append(mae)
mae_mean = np.array(mae_total).mean()
print(mae_mean)