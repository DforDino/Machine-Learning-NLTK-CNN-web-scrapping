#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

train_org_df=pd.read_csv('train.csv',parse_dates=['timestamp'])
macro_org_df=pd.read_csv('macro.csv',parse_dates=['timestamp'])
train_df=pd.merge(train_org_df,macro_org_df,how='left',on='timestamp')

test_df=pd.read_csv('test.csv',parse_dates=['timestamp'])
testm_df=pd.merge(test_df,macro_org_df,how='left',on='timestamp')

rng = np.random.RandomState(0)
tmp_df=train_df
y_train=train_df['price_doc'].as_matrix()
dropped_df=tmp_df.drop(['modern_education_share','price_doc','timestamp','id'],axis=1)
dropped_test_df=testm_df.drop(['modern_education_share','timestamp','id'],axis=1)
Data_FULL_arr=[dropped_df, dropped_test_df]
DATA_Full=pd.concat(Data_FULL_arr)


DATA_Full_dummy=pd.get_dummies(DATA_Full)
X_full = DATA_Full_dummy.as_matrix()
X_full=np.nan_to_num(X_full)

X_train=X_full[0:len(y_train), :]
X_test=X_full[len(y_train): , :]

#y_full=dataset[:,[291]]
#relevent_columns=[i for i in range(2,291)]
#relevent_columns1=[i for i in range(292,391)]
#rel_col=np.concatenate([relevent_columns,relevent_columns1],axis=0)
#X_full=dataset[:,rel_col]

n_samples = X_train.shape[0]
n_features = X_train.shape[1]




# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=500)
estimator.fit(X_train, y_train)
predicted_y= estimator.predict(X_test)
#score = cross_val_score(estimator, X_full, y_full).mean()
#print("Score with the entire dataset = %.2f" % score)


AA=[list(testm_df['id']), predicted_y]
f=open('submit.csv', 'w')
f.write('id,price_doc\n')
for i,a in enumerate(AA[0]):
    f.write('{0},{1}\n'.format(AA[0][i], AA[1][i]))
f.close()


