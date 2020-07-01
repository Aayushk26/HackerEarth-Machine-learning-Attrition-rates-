#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Train.csv')
df1=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Test.csv')


# In[3]:


df.head()
df=df.fillna(df.mean())
df1=df1.fillna(df1.mean())


# In[4]:


x=df.drop(['Employee_ID', 'Gender', 'Relationship_Status', 'Hometown', 'Unit',
       'Decision_skill_possess', 'Compensation_and_Benefits','Attrition_rate'],axis=1)
y=df.Attrition_rate
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)
xt=df1.drop(['Employee_ID', 'Gender', 'Relationship_Status', 'Hometown', 'Unit',
       'Decision_skill_possess', 'Compensation_and_Benefits'],axis=1)


# In[5]:



x_train,x_test


# In[6]:


x_train.shape,y_train.shape


# In[7]:


x_test.shape


# In[10]:


# Using catboost regression
from catboost import CatBoostRegressor
cat= CatBoostRegressor(learning_rate=0.001,boosting_type='Ordered',depth=2)
cat.fit(x_train,y_train)


# In[178]:


ypred=cat.predict(x_test)


# In[179]:


from sklearn.metrics import mean_squared_error


acc=mean_squared_error(y_test, ypred)

print(np.sqrt(acc))


# In[ ]:


iterations=20000,loss_function='RMSE',eval_metric="RMSE",learning_rate=1,depth=3,boosting_type='Plain'


# In[77]:


y_p=cat.predict(xt)


# In[180]:


my_submission = pd.DataFrame({ 'Attrition_rate':y_p})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[181]:


#Using LightGBM Regressor
from lightgbm import LGBMRegressor
lg=LGBMRegressor(boosting_type='dart', class_weight=None, colsample_bytree=0.25,
              importance_type='split', learning_rate=0.1, max_depth=2,
              min_child_samples=20000, min_child_weight=0.1, min_split_gain=0.0,
              n_estimators=1000, n_jobs=-1, num_leaves=500, objective=None,
              random_state=0, reg_alpha=0.5, reg_lambda=0.0, silent=True,
              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
lg.fit(x_train,y_train)


# In[182]:


yl=lg.predict(x_test)
lga=mean_squared_error(yl,y_test)
print(np.sqrt(lga))


# In[204]:


#using random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=1000,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
rf.fit(x_train,y_train)


# In[267]:


yf=rf.predict(x_test)
accr=mean_squared_error(yf,y_test)
print(np.sqrt(accr))


# In[264]:


# Support vector regression
from sklearn.svm import SVR
sv=SVR(kernel='rbf',C=1.0
       , cache_size=2000, coef0=0.0, degree=2, epsilon=0.05,
    gamma='auto_deprecated', max_iter=-1, shrinking=True,
    tol=0.001)
sv.fit(x_train,y_train)


# In[265]:


ysv=sv.predict(x_test)
accs=mean_squared_error(ysv,y_test)
print(np.sqrt(accs))



# In[284]:


yft=rf.predict(xt)
my_submission = pd.DataFrame({ 'Attrition_rate':yft})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[285]:


xt.shape


# In[293]:


# Multiple regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1000, normalize=True)
lr.fit(x_train,y_train)


# In[294]:


ylr=lr.predict(x_test)
print(np.sqrt(mean_squared_error(ylr,y_test)))


# In[296]:



yrl=lr.predict(xt)
my_submission = pd.DataFrame({ 'Attrition_rate':yft})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[301]:


#Using feature selection
# most correlated fearures 
# Time_since_promotion,growth_rate,Pay_Scale,Work_Life_balance ,VAR3


# In[305]:


xnew=df.drop(['Employee_ID', 'Gender', 'Relationship_Status', 'Hometown', 'Unit',
       'Decision_skill_possess', 'Compensation_and_Benefits','Age', 'Education_Level', 'Time_of_service', 
      'Travel_Rate', 'Post_Level',
        'VAR1', 'VAR2', 'VAR4', 'VAR5', 'VAR6',
       'VAR7'],axis=1)
ynew=df.Attrition_rate


# In[306]:


xnew.shape,ynew.shape


# In[ ]:




