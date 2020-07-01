#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Train.csv')
test=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Test.csv')
import seaborn as sns


# In[2]:


train.columns


# In[10]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
r=lr.predict(xt)


# In[ ]:


VAR7  Attrition_rate  
Age                   0.008746       -0.015498  
Education_Level       0.012909       -0.008143  
Time_of_service       0.007034       -0.016447  
Time_since_promotion  0.005173        0.013880  
growth_rate          -0.017993        0.014247  
Travel_Rate          -0.006271       -0.012608  
Post_Level           -0.003397        0.016402  
Pay_Scale             0.002381       -0.015236  
Work_Life_balance     0.024534        0.020746  
VAR1                  0.009355       -0.008073  1
VAR2                  0.003003       -0.023991  2
VAR3                  0.008390        0.001245  3
VAR4                 -0.018423       -0.013120  4
VAR5                  0.009436       -0.004770  5
VAR6                  0.011908       -0.003130  6
VAR7                  1.000000       -0.015299  7
Attrition_rate       -0.015299        1.000000  


# In[218]:


x1=train.fillna(train.mean())
x2=x1.drop(['Employee_ID', 'Gender',  'Education_Level',
       'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess','Travel_Rate','Time_of_service',
         'Compensation_and_Benefits','VAR2','VAR6', 'VAR5' ,'VAR1' ,
        'Attrition_rate'],axis=1)
y2=train.Attrition_rate

xt1=test.fillna(test.mean())
xt2=xt1.drop(['Employee_ID', 'Gender',  'Education_Level',
       'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess','Travel_Rate','VAR2','Time_of_service',
              'VAR6','VAR5','VAR1',
              'Compensation_and_Benefits'],axis=1)


# In[219]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x2,y2)
r=lr.predict(xt2)


# In[220]:


x2.shape,y2.shape


# In[221]:


r


# In[161]:



my_submission= pd.DataFrame({ 'Attrition_rate':r})
# you could use any filename. We choose submission here
my_submission.to_csv('sub13.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




