import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv('sorted_output_with_img_name.csv')
df.drop('Image Name',axis=1,inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(int) - 1 


X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)
dtrain = xgb.DMatrix(X_train,y_train,enable_categorical=True)
dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {
    'max_depth':2,
    'eta':0.3,
    'objective':'multi:softprob',
    'num_class':7,
    'eval_metric': 'mlogloss'
}

epoch=500
eval_res={}
evals = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(params,dtrain,epoch,evals,evals_result=eval_res)

model.save_model('xgboost_new.model')