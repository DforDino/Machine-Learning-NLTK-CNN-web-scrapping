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



sub_area_name='Savelki'

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


AA=train_df.sub_area.unique()
f=open('aa.txt', 'a+')
for i, j in enumerate(AA):
    line='if sub_area==\'{0}\':\n  model_{1}.predict_price(b1.loc[pos:pos+1])\n\n'.format(j,i+1)
    f.write(line)
f.close()
