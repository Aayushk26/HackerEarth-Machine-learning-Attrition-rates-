#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Train.csv')
test=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Test.csv')

import seaborn as sns


# In[2]:


train.columns


# In[4]:


train.describe()


# In[5]:


x1=train.drop(['Employee_ID', 'Gender', 'Age', 'Education_Level',
       'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
       'Time_of_service',  'Travel_Rate',
        'Pay_Scale', 'Compensation_and_Benefits',
       'VAR1', 'VAR2',  'VAR4', 'VAR5', 
       'VAR7', 'Attrition_rate'],axis=1)
x=x1.fillna(x1.mean())

y=train.Attrition_rate
xt1=test.drop(['Employee_ID', 'Gender', 'Age', 'Education_Level',
       'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
       'Time_of_service',  'Travel_Rate',
        'Pay_Scale', 'Compensation_and_Benefits',
       'VAR1', 'VAR2',  'VAR4', 'VAR5', 
       'VAR7'],axis=1)
xt=xt1.fillna(xt1.mean())


# In[7]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
a1=lr.predict(xt)


# In[19]:



my_submission= pd.DataFrame({ 'Attrition_rate':a1})
# you could use any filename. We choose submission here
my_submission.to_csv('sub7.csv', index=False)


# In[25]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x,y)


# In[30]:


a2=dt.predict(xt)
my_submission= pd.DataFrame({ 'Attrition_rate':a2})
# you could use any filename. We choose submission here
my_submission.to_csv('sub8.csv', index=False)


# In[32]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x,y)


# In[33]:


a3=rf.predict(xt)
my_submission= pd.DataFrame({ 'Attrition_rate':a3})
# you could use any filename. We choose submission here
my_submission.to_csv('sub10.csv', index=False)


# In[87]:


from catboost import CatBoostRegressor
cb=CatBoostRegressor(iterations=10000,loss_function='RMSE',eval_metric="RMSE",learning_rate=0.02,depth=1,boosting_type='Plain')
cb.fit(x,y)


# In[88]:


a4=cb.predict(xt)
my_submission= pd.DataFrame({ 'Attrition_rate':a4})
# you could use any filename. We choose submission here
my_submission.to_csv('sub11.csv', index=False)


# In[89]:


a4


# In[ ]:


iterations=1000,loss_function='RMSE',eval_metric="RMSE",learning_rate=0.5,depth=3,boosting_type='Plain'


# In[93]:


#SVR 
from sklearn.svm import SVR
reg=SVR(kernel='linear')
reg.fit(x,y)


# In[95]:


r=reg.predict(xt)
my_submission= pd.DataFrame({ 'Attrition_rate':r})
# you could use any filename. We choose submission here
my_submission.to_csv('subr.csv', index=False)


# In[96]:


r


# In[ ]:



from catboost import CatBoostRegressor
cb=CatBoostRegressor(iterations=10000,loss_function='RMSE',eval_metric="RMSE",learning_rate=0.02,depth=1,boosting_type='Plain')
cb.fit(x2,y2)


# In[ ]:




