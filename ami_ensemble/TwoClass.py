#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:12:05 2017

@author: dino
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn import svm

class Predict_Class():
    #_data = pd.DataFrame
    def __init__(self, train_rows):
        self.highcutoff = 100
        
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        
        s3=sa[sa.price_doc>3]
        s4=sa[sa.price_doc <= 2.1]

        t3=s3.price_doc
        t4=s4.price_doc
        
        s3=s3.drop(['price_doc','sub_area', 'full_sq' ,'product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area', 'full_sq','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
                
        d3=np.ones(len(t3))
        d4=np.ones(len(t4))*-1
        print ("s3 shape",s3.shape)
        X = pd.concat([s4,s3,s4])
        y = np.concatenate([d4,d3,d4],axis=0)
        self.y=y
        self.X=X
        #print (X)
        #self.decision_clf = svm.SVC()
        self.decision_clf = RandomForestClassifier()
        self.decision_clf.fit(X, y)
        
        
    def predict_class(self,test_row):
                pclass=self.decision_clf.predict(test_row)[0]
                return pclass


selectedRows=(train_df.sub_area=='Chertanovo Severnoe') | \
             (train_df.sub_area=='Krjukovo') | \
             (train_df.sub_area=='Rostokino') | \
             (train_df.sub_area=='Basmannoe') |\
             (train_df.sub_area=='Poselenie Vnukovskoe') |\
             (train_df.sub_area=='Poselenie Desjonovskoe') |\
             (train_df.sub_area=='Poselenie Sosenskoe') |\
             (train_df.sub_area=='Poselenie Filimonkovskoe') |\
             (train_df.sub_area=='Poselenie Voskresenskoe') |\
             (train_df.sub_area=='Novo-Peredelkino') |\
             (train_df.sub_area=='Akademicheskoe') |\
             (train_df.sub_area=='Sokol') |\
             (train_df.sub_area=='Poselenie Krasnopahorskoe') |\
             (train_df.sub_area=='Zjuzino') |\
             (train_df.sub_area=='Matushkino') |\
             (train_df.sub_area=='Otradnoe') |\
             (train_df.sub_area=='Mitino') |\
             (train_df.sub_area=='Vojkovskoe') |\
             (train_df.sub_area=='Rjazanskij') |\
             (train_df.sub_area=='Severnoe Butovo') |\
             (train_df.sub_area=='Staroe Krjukovo') |\
             (train_df.sub_area=='Golovinskoe') |\
             (train_df.sub_area=='Kosino-Uhtomskoe') |\
             (train_df.sub_area=='Veshnjaki') |\
             (train_df.sub_area=='Horoshevo-Mnevniki') |\
             (train_df.sub_area=='Pechatniki') |\
             (train_df.sub_area=='Tekstil\'shhiki') |\
             (train_df.sub_area=='Solncevo') |\
             (train_df.sub_area=='Sviblovo') |\
             (train_df.sub_area=='Silino') |\
             (train_df.sub_area=='Butyrskoe') |\
             (train_df.sub_area=='Birjulevo Vostochnoe') |\
             (train_df.sub_area=='Caricyno') |\
             (train_df.sub_area=='Taganskoe') |\
             (train_df.sub_area=='Kapotnja') |\
             (train_df.sub_area=='Orehovo-Borisovo Juzhnoe') |\
             (train_df.sub_area=='Dmitrovskoe') |\
             (train_df.sub_area=='Juzhnoe Medvedkovo') |\
             (train_df.sub_area=='Sokolinaja Gora') |\
             (train_df.sub_area=='Lianozovo') |\
             (train_df.sub_area=='Zapadnoe Degunino') |\
             (train_df.sub_area=='Novogireevo') |\
             (train_df.sub_area=='Gol\'janovo') |\
             (train_df.sub_area=='Bogorodskoe') |\
             (train_df.sub_area=='Presnenskoe') |\
             (train_df.sub_area=='Timirjazevskoe') |\
             (train_df.sub_area=='Jasenevo') |\
             (train_df.sub_area=='Altuf\'evskoe') |\
             (train_df.sub_area=='Severnoe Medvedkovo') |\
             (train_df.sub_area=='Vyhino-Zhulebino') |\
             (train_df.sub_area=='Filevskij Park') |\
             (train_df.sub_area=='Kotlovka') |\
             (train_df.sub_area=='Jaroslavskoe') |\
             (train_df.sub_area=='Severnoe Izmajlovo') |\
             (train_df.sub_area=='Perovo') |\
             (train_df.sub_area=='Nizhegorodskoe') |\
             (train_df.sub_area=='Jakimanka') |\
             (train_df.sub_area=='Ivanovskoe') |\
             (train_df.sub_area=='Severnoe Tushino') |\
             (train_df.sub_area=='Nagatino-Sadovniki') |\
             (train_df.sub_area=='Ramenki') |\
             (train_df.sub_area=='Bibirevo') |\
             (train_df.sub_area=='Zjablikovo') |\
             (train_df.sub_area=='Meshhanskoe') |\
             (train_df.sub_area=='Chertanovo Juzhnoe') |\
             (train_df.sub_area=='Danilovskoe') |\
             (train_df.sub_area=='Hovrino') |\
             (train_df.sub_area=='Vostochnoe Degunino') |\
             (train_df.sub_area=='Birjulevo Zapadnoe') |\
             (train_df.sub_area=='Donskoe') |\
             (train_df.sub_area=='Chertanovo Central\'noe') |\
             (train_df.sub_area=='Losinoostrovskoe') |\
             (train_df.sub_area=='Vostochnoe Izmajlovo') |\
             (train_df.sub_area=='Orehovo-Borisovo Severnoe') |\
             (train_df.sub_area=='Preobrazhenskoe') |\
             (train_df.sub_area=='Fili Davydkovo') |\
             (train_df.sub_area=='Moskvorech\'e-Saburovo') |\
             (train_df.sub_area=='Teplyj Stan') |\
             (train_df.sub_area=='Lomonosovskoe') |\
             (train_df.sub_area=='Ljublino') |\
             (train_df.sub_area=='Strogino') |\
             (train_df.sub_area=='Koptevo') |\
             (train_df.sub_area=='Babushkinskoe') |\
             (train_df.sub_area=='Troparevo-Nikulino') |\
             (train_df.sub_area=='Cheremushki') |\
             (train_df.sub_area=='Levoberezhnoe') |\
             (train_df.sub_area=='Prospekt Vernadskogo') |\
             (train_df.sub_area=='Nagatinskij Zaton') |\
             (train_df.sub_area=='Savelki') |\
             (train_df.sub_area=='Poselenie Kokoshkino')
         
             
             
             
             
             
             
             
             


TwoClassdata = train_df[selectedRows]

TwoClassModel=Predict_Class(TwoClassdata)






