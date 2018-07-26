import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

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
        
        
    def predict(self,test_row):
                pclass=self.decision_clf.predict(test_row)
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


####################################Poselenie Sosenskoe#######################
class MasterModel_ALL():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
        
        random_estimator4 = RandomForestRegressor(random_state=0, n_estimators=500)
        random_estimator4.fit(s4,t4)
        self.random_estimator4=random_estimator4
        
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
        flag=0
        if ((test_row.state.isnull()) & (test_row.life_sq.isnull())).iloc[0]:
            flag=1
        test_row=test_row.drop(['sub_area','product_type'],axis=1)
        test_row=test_row.iloc[:,2:20]
        for fea in test_row.columns:
            test_row[fea].fillna(test_row[fea].median(),inplace=True)
        for fea in test_row.columns:
            test_row[fea].fillna(0,inplace=True)
        dum=test_row.iloc[0]['full_sq']
        #print (dum)
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       predicted_price1= self.random_estimator4.predict(test_row)[0]
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator4.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price

gc.collect()
###############################################################






##




############################################################

class MasterModel():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
        
        random_estimator4 = RandomForestRegressor(random_state=0, n_estimators=500)
        random_estimator4.fit(s4,t4)
        self.random_estimator4=random_estimator4
        
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
        flag=0
        if ((test_row.state.isnull()) & (test_row.life_sq.isnull())).iloc[0]:
            flag=1
        test_row=test_row.drop(['sub_area','product_type'],axis=1)
        test_row=test_row.iloc[:,2:20]
        for fea in test_row.columns:
            test_row[fea].fillna(test_row[fea].median(),inplace=True)
        for fea in test_row.columns:
            test_row[fea].fillna(0,inplace=True)
        dum=test_row.iloc[0]['full_sq']
        #print (dum)
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       predicted_price1= self.random_estimator4.predict(test_row)[0]
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator4.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price

gc.collect()


########################################################################3



sub_area_name='Poselenie Novofedorovskoe'

a11=MasterModel(train_df[train_df.sub_area==sub_area_name], TwoClassModel)

#a1=MasterModel_ALL(train_df, TwoClassModel)


b1=test_df[test_df.sub_area==sub_area_name]

#b1=Poselenie_Sosenskoe.drop_test(b1)
#print (b1)
#Poselenie_Vnukovskoe.predict_price(s1)
predicted=[]
for pos, row in b1.iterrows():
    y=a11.predict_price(b1.loc[pos:pos+1])
    #print(len(b1.loc[pos:pos+1]))
    predicted.append(y)
print (predicted)

#plt.scatter(s1.full_sq,predicted, color='red')
plt.scatter(train_df[train_df.sub_area==sub_area_name].full_sq, train_df[train_df.sub_area==sub_area_name].price_doc)
#plt.show()
plt.scatter(b1.full_sq,predicted, color='red')
plt.show()





