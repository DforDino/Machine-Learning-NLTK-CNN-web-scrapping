# coding: utf-8
#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import gc
from sklearn.cluster import KMeans
#from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import Imputer
#from sklearn.model_selection import cross_val_score

train_org_df=pd.read_csv('train.csv', parse_dates=['timestamp'])
macro_org_df=pd.read_csv('macro.csv',parse_dates=['timestamp'])
train_df=pd.merge(train_org_df,macro_org_df,how='left',on='timestamp')

test_df=pd.read_csv('test.csv',parse_dates=['timestamp'])
testm_df=pd.merge(test_df,macro_org_df,how='left',on='timestamp')

#KK=train_df['build_year']
#
#def ff(KK):
#   for i in range(0,len(KK)):
#        PP1[i]=KK[i]%10
#        PP2[i]=(KK[i]/10)%10
#        PP3[i]=(KK[i]/100)%10
#        PP4[i]=(KK[i]/1000)%10
#        if KK[i] > 2017:  
#            if PP4[i] > 2:
#                QQ4[i] = 2
#                if PP3[i] > 0:
#                    QQ3[i] = 0
#                    if PP2[i] > 1:
#                        QQ2[i] = 1
#                        if PP1[i] > 7:
#                            QQ1[i] = 7
#                            KK[i] = QQ4[i]*1000 + QQ3[i]*100 + QQ2[i]*10 + QQ1[i]
#                            return KK
#                            
#        else:
#            if PP3[i] > 0:
#                    QQ3[i] = 0
#                    if PP2[i] > 1:
#                        QQ2[i] = 1
#                        if PP1[i] > 7:
#                            QQ1[i] = 7
#                            KK[i] = QQ4[i]*1000 + QQ3[i]*100 + QQ2[i]*10 + QQ1[i]
#                            return KK
#             else:
#               if PP2[i] > 1:
#                        QQ2[i] = 1
#                        if PP1[i] > 7:
#                            QQ1[i] = 7
#                            KK[i] = QQ4[i]*1000 + QQ3[i]*100 + QQ2[i]*10 + QQ1[i]
#                            return KK
#                else:
#                   if PP1[i] > 7:
#                            QQ1[i] = 7
#                            KK[i] = QQ4[i]*1000 + QQ3[i]*100 + QQ2[i]*10 + QQ1[i]
#                            return KK
#                   else:
#                       return KK

business_rent=np.zeros(len(train_df))
economy_rent=np.zeros(len(train_df))


for index, row in train_df.iterrows():
#    print(index)
    if row['num_room']==1 :
        business_rent[index]=row['rent_price_1room_bus']
        economy_rent[index]=row['rent_price_1room_eco']
    
    if row['num_room']==2 :
        business_rent[index]=row['rent_price_2room_bus']
        economy_rent[index]=row['rent_price_2room_eco']
    if row['num_room']==3 :
        business_rent[index]=row['rent_price_3room_bus']
        economy_rent[index]=row['rent_price_3room_eco']
    if row['num_room']>3 :
        business_rent[index]=row['rent_price_4+room_bus']
        economy_rent[index]=row['rent_price_4+room_bus']
    if np.isnan(row['num_room']) :
        business_rent[index]=np.mean([row['rent_price_1room_bus'], row['rent_price_2room_bus'], row['rent_price_3room_bus'], row['rent_price_4+room_bus']  ])
        economy_rent[index]=np.mean([row['rent_price_1room_eco'], row['rent_price_2room_eco'], row['rent_price_3room_eco']  ])


train_df['eco_rent_combined']=economy_rent
train_df['bus_rent_combined']=business_rent


del business_rent
del economy_rent

gc.collect()

business_rent=np.zeros(len(testm_df))
economy_rent=np.zeros(len(testm_df))


for index, row in testm_df.iterrows():
#    print(index)

    if row['num_room']==1 :
        business_rent[index]=row['rent_price_1room_bus']
        economy_rent[index]=row['rent_price_1room_eco']
    
    if row['num_room']==2 :
        business_rent[index]=row['rent_price_2room_bus']
        economy_rent[index]=row['rent_price_2room_eco']
    if row['num_room']==3 :
        business_rent[index]=row['rent_price_3room_bus']
        economy_rent[index]=row['rent_price_3room_eco']
    if row['num_room']>3 :
        business_rent[index]=row['rent_price_4+room_bus']
        economy_rent[index]=row['rent_price_4+room_bus']
    if np.isnan(row['num_room']) :
        business_rent[index]=np.mean([row['rent_price_1room_bus'], row['rent_price_2room_bus'], row['rent_price_3room_bus'], row['rent_price_4+room_bus']  ])
        economy_rent[index]=np.mean([row['rent_price_1room_eco'], row['rent_price_2room_eco'], row['rent_price_3room_eco']  ])


testm_df['eco_rent_combined']=economy_rent
testm_df['bus_rent_combined']=business_rent


#def room_rent_drop(df):
#    room_rent_df = df
#    room_rent_df.ix[room_rent_df['num_room']== 1,['rent_price_4+room_bus', 'rent_price_3room_bus', 'rent_price_2room_bus',  'rent_price_3room_eco', 'rent_price_2room_eco', ]]=0
#    room_rent_df.ix[room_rent_df['num_room']== 2,['rent_price_4+room_bus', 'rent_price_3room_bus',  'rent_price_1room_bus', 'rent_price_3room_eco',  'rent_price_1room_eco']]=0
#    room_rent_df.ix[room_rent_df['num_room']== 3,['rent_price_4+room_bus',  'rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_2room_eco', 'rent_price_1room_eco']]=0
#    room_rent_df.ix[room_rent_df['num_room'] >= 4,[ 'rent_price_3room_bus', 'rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_3room_eco', 'rent_price_2room_eco', 'rent_price_1room_eco']]=0
#    
#    return room_rent_df
#train_df=room_rent_drop(train_df)
#pd.set_option('display.max_columns',500)
#pd.set_option('display.max_rows',100)
#train_df['rent_eco_all']=train_df[['rent_price_2room_eco', 'rent_price_1room_eco', 'rent_price_3room_eco']].sum(axis=1)
#train_df.rent_eco_all= train_df.loc[train_df.num_room.isnull(),['rent_price_2room_eco', 'rent_price_1room_eco', 'rent_price_3room_eco']].mean(axis=1)
#### below this is your script

##########################
#######################
rng = np.random.RandomState(0)
tmp_df=train_df
y_train=train_df['price_doc'].as_matrix()

#new
features=train_df.columns
use_fea=list(features[2:15])
# expand
use_fea.append('railroad_km')
use_fea.append('cafe_count_5000')
use_fea.append('cafe_count_2000')
use_fea.append('metro_km_avto')
use_fea.append('metro_min_walk')
use_fea.append('bus_terminal_avto_km')
use_fea.append('big_market_km')
use_fea.append('oil_urals')
use_fea.append('mortgage_rate')
use_fea.append('unemployment')
use_fea.append('eco_rent_combined')
use_fea.append('bus_rent_combined')

train_fea=train_df.columns
test_fea=testm_df.columns

drop_train_fea=[i for i in train_fea if i not in use_fea]
dropped_df=tmp_df.drop(drop_train_fea,axis=1)

drop_test_fea=[i for i in test_fea if i not in use_fea]
dropped_test_df=testm_df.drop(drop_test_fea, axis=1)

################################

Q = train_df['life_sq'].isnull()
newtrain_df1 = train_df[~Q]
newtrain_df2 = train_df[Q]
y_train_1 = newtrain_df1['life_sq'].as_matrix()
X_train_1 = newtrain_df1['full_sq'].as_matrix()
X_test_1 = newtrain_df2['full_sq'].as_matrix()
#y_train_1_working = y_train_1.reshape(len(y_train_1),1)
X_train_1_working = X_train_1.reshape(len(X_train_1),1)
X_test_1_working = X_test_1.reshape(len(X_test_1),1) 
#estimator_1 = SVR(kernel='linear', C=1e3)
estimator_1 = LinearRegression()
estimator_1.fit(X_train_1_working, y_train_1)
predicted_y_test_1= estimator_1.predict(X_test_1_working)

YY=train_df['life_sq'].as_matrix()
i=0
for index, row in newtrain_df2.iterrows():
      YY[index]=predicted_y_test_1[i]
      i-i+1

train_df['life_sq']= YY  
        
drop_train_fea=[i for i in train_fea if i not in use_fea]
dropped_df=train_df.drop(drop_train_fea,axis=1)

drop_test_fea=[i for i in test_fea if i not in use_fea]
dropped_test_df=testm_df.drop(drop_test_fea, axis=1)        

################################

##############################
#drop_train_fea=[i for i in train_fea if i not in use_fea]
#droppedm_df=trainm_df.drop(drop_train_fea,axis=1)
#M = droppedm_df['num_room'].isnull()
#droppedm_df1 = droppedm_df[~M]
#droppedm_df2 = droppedm_df[M]

###########################
#droppedm_test_df=dropped_test_df
#N = droppedm_test_df['num_room'].isnull()
#droppedm_test_df1 = droppedm_test_df[~N]
#droppedm_test_df2 = droppedm_test_df[N]
#S = testm_df['num_room'].isnull()
#testm_df1=testm_df[~S
#dropped_df=tmp_df.drop(['modern_education_share','price_doc','timestamp','id'],axis=1)
#dropped_test_df=testm_df.drop(['modern_education_share','timestamp','id'],axis=1)
Data_FULL_arr_A=[dropped_df, dropped_test_df]
DATA_Full_A=pd.concat(Data_FULL_arr_A)


DATA_Full_dummy=pd.get_dummies(DATA_Full_A)
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

#D = train_df['num_room'].isnull()
#df_A=train_df[~D]
#y_train_A = df_A['price_doc'].as_matrix()

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
estimator.fit(X_train, y_train)
predicted_y= estimator.predict(X_test)
#score = cross_val_score(estimator, X_full, y_full).mean()
#print("Score with the entire dataset = %.2f" % score)

#dropped_df=tmp_df.drop(['modern_education_share','price_doc','timestamp','id'],axis=1)
#dropped_test_df=testm_df.drop(['modern_education_share','timestamp','id'],axis=1)
#df_B=dropped_df.drop('num_room',axis=1)
#df_B_test=dropped_test_df.drop('num_room',axis=1)
#Data_FULL_arr_B=[df_B, df_B_test]
#DATA_Full=pd.concat(Data_FULL_arr_B)


#ATA_Full_dummy=pd.get_dummies(DATA_Full)
#X_full = DATA_Full_dummy.as_matrix()
#X_full=np.nan_to_num(X_full)


#X_train=X_full[0:len(y_train), :]
#X_test=X_full[len(y_train): , :]

#y_full=dataset[:,[291]]
#relevent_columns=[i for i in range(2,291)]
#relevent_columns1=[i for i in range(292,391)]
#rel_col=np.concatenate([relevent_columns,relevent_columns1],axis=0)
#X_full=dataset[:,rel_col]

#n_samples = X_train.shape[0]
#n_features = X_train.shape[1]




# Estimate the score on the entire dataset, with no missing values
#estimator2 = RandomForestRegressor(random_state=0, n_estimators=100, verbose=1)
#estimator2.fit(X_train, y_train)
#predicted_y2= estimator2.predict(X_test)

#kmeans = KMeans(n_clusters=5, random_state=0).fit(X_test)
#test_kmeans_labels=kmeans.labels_
#num_sam=[sum(test_kmeans_labels==i) for i in range(5)]




AA=[list(testm_df['id']), predicted_y]
f=open('submit.csv', 'w')
f.write('id,price_doc\n')
for i,a in enumerate(AA[0]):
    #if (AB[0][i] == AA[0][i]) :
    #       f.write('{0},{1}\n'.format(AB[0][i], AA[1][i]))
   # else 
           #result=(AB[1][i]+AA[1][i])*0.5
           f.write('{0},{1}\n'.format(AA[0][i], AA[1][i]))
           
f.close()

