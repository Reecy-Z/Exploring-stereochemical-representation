import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

Train_384 = 'FMSSD_384.csv'
Test_171 = 'FMSSD_171.csv'
Test_216 = 'FMSSD_216.csv'
Test_304 = 'FMSSD_304.csv'

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

def get_model():
    params = {'max_depth': 23, 'learning_rate': 0.16318915920493737, 'n_estimators': 62, 'gamma': 0.17735702393456645, 
              'min_child_weight': 5, 'max_delta_step': 100, 'subsample': 0.9332230398089334, 'colsample_bytree': 0.807431352173656,
              'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'seed':1440, 'silent':1, 'objective':'reg:linear', 'nthread':-1, 'colsample_bylevel':1}
    model = xgb.XGBRegressor(**params)
    return model

mae_mean_total = []
model = get_model()
model.fit(X_train_384, y_train_384, eval_metric='mae', verbose = True)

pred_171 = model.predict(X_test_171)
mae_171 = mean_absolute_error(pred_171,y_test_171)
print('mae of 171:{}'.format(mae_171))

pred_216 = model.predict(X_test_216)
mae_216 = mean_absolute_error(pred_216,y_test_216)
print('mae of 216:{}'.format(mae_216))

pred_304 = model.predict(X_test_304)
mae_304 = mean_absolute_error(pred_304,y_test_304)
print('mae of 304:{}'.format(mae_304))

mae_mean_total.append(mae_171)
mae_mean_total.append(mae_216)
mae_mean_total.append(mae_304)
mae_mean_total = np.array(mae_mean_total).mean()
print(mae_mean_total)
