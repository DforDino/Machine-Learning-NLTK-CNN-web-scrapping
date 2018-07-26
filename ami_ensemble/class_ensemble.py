# coding: utf-8
# %load num-rooms.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_org_df=pd.read_csv('train.csv',parse_dates=['timestamp'])
update_train=pd.read_csv('BAD_ADDRESS_FIX.csv')
train_org_df.update(update_train,overwrite=True)
train_org_df.price_doc = train_org_df.price_doc/1000000.
macro_org_df=pd.read_csv('macro.csv',parse_dates=['timestamp'])
train_df=pd.merge(train_org_df,macro_org_df,how='left',on='timestamp')
test_org_df=pd.read_csv('test.csv',parse_dates=['timestamp'])
test_df=pd.merge(test_org_df,macro_org_df,on='timestamp',how='left')

# move price_doc to last
price_df=train_df.price_doc
train_df=train_df.drop('price_doc', axis=1)
train_df['price_doc']=price_df

# wrong build year update from 1691 to 1991
train_df.set_value(26332,'build_year',1991);
train_df.set_value(30275,'build_year',1971);
train_df.set_value(30150,'build_year',2015);
train_df.set_value(10089,'build_year',2007);
train_df.set_value(10089,'state',3);
train_df.set_value(15220,'build_year',1965); # was 4965
train_df.set_value(13992,'build_year',2017) ; # was 20
test_df.set_value(2995,'build_year',2015);
#train_df.full_sq.set_value(4678,'full_sq',65); ### doubt anni daa
#train_df.set_value(27460,'price_doc', 7.1249624) 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
    
class Poselenie_Vnukovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows):
        print (train_rows.shape)
        self.train_target=train_rows.price_doc
        train_rows=train_rows.drop(['price_doc','sub_area','product_type','build_year'],axis=1)
        train_rows1=train_rows.iloc[:,2:20]
        print (train_rows1.shape)
        for fea in train_rows1.columns:
            print (fea)
            train_rows1[fea].fillna(train_rows1[fea].median(), inplace=True)
        print (train_rows1.shape)
        
        self.train=train_rows1
        estimator1 = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        #print (type(estimator))
        estimator1.fit(self.train, self.train_target)
        self.estimator=estimator1
        print ('jaya jagannath')
    @staticmethod
    def drop_test(test_df):
      
        return test_df
    def predict_price(self,test_row):
        test_df=test_row.drop(['sub_area','product_type','build_year'],axis=1)
        test_df=test_df.iloc[:,2:20]
        for fea in test_df.columns:
            test_df[fea].fillna(test_df[fea].median(),inplace=True)
        for fea in test_df.columns:
            test_df[fea].fillna(0,inplace=True)
       
        predicted_price= self.estimator.predict(test_df)
        #if test_row[2]< 35:
        #    predicted_price=predicted_price-1
        return predicted_price
            

class Staroe_Krjukovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows):
        
        target=train_rows.price_doc
        train_rows=train_rows.drop(['price_doc','sub_area','product_type','build_year'],axis=1)
        train_rows1=train_rows.iloc[:,2:20]
        print (train_rows1.shape)
        for fea in train_rows1.columns:
           
            train_rows1[fea].fillna(train_rows1[fea].median(), inplace=True)
        
        x=train_rows1.full_sq
        x=x.reshape((len(x),1))        
        y=target
        y=y.reshape((len(y),1))
        estimator1 = LinearRegression()    
        estimator1.fit(x, y)
        self.estimator=estimator1
        print ('jaya jagannath')
    @staticmethod
    def drop_test(test_df):

        return test_df
    def predict_price(self,test_row):
        test_df=test_row.drop(['sub_area','product_type','build_year'],axis=1)
        test_df=test_df.iloc[:,2:20]
        for fea in test_df.columns:
            test_df[fea].fillna(test_df[fea].median(),inplace=True)
        #for fea in test_df.columns:
            #test_df[fea].fillna(0,inplace=True)
        x1=test_df.full_sq
        x1=x1.reshape(len(x1),1)
        predicted_price= self.estimator.predict(x1)
        #if test_row[2]< 35:
        #    predicted_price=predicted_price-1
        return predicted_price


s1 = test_df[test_df.sub_area=='Staroe Krjukovo']
print(s1)

a1=Staroe_Krjukovo(train_df[train_df.sub_area=='Staroe Krjukovo'])
s1=Staroe_Krjukovo.drop_test(s1)
print (s1)
#Poselenie_Vnukovskoe.predict_price(s1)
predicted=[]
for pos, row in s1.iterrows():
    y=a1.predict_price(s1.loc[pos:pos+1])
    predicted.append(y)
print (predicted)
    
#plt.scatter(s1.full_sq,predicted, color='red')
plt.scatter(train_df[train_df.sub_area=='Staroe Krjukovo'].full_sq, train_df[train_df.sub_area=='Staroe Krjukovo'].price_doc)
plt.show()
plt.scatter(s1.full_sq,predicted, color='red')
#plt.show()
for fea in ['Staroe Krjukovo']:#train_df.okrugs.unique():
    print(fea)
    plot1=train_df[(train_df.sub_area==fea)][['price_doc','num_room','kitch_sq','build_year','full_sq','sub_area','railroad_station_walk_min','state']]
    #plot1=plot1[(plot1.price_doc < 80) & (plot1.full_sq <200)]
    plot1.sortlevel()
    plt.scatter(list(plot1['full_sq']), list(plot1['price_doc']))
    plt.title(fea)
    plt.xlabel('full_sq')
    plt.ylabel('price_doc in millions')
    #plt.savefig(filename)
    plt.show()
#plot1[plot1.price_doc>5]
