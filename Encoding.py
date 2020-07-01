#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from sklearn.feature_selection import SelectKBest, f_regression,chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.decomposition import KernelPCA
import xgboost as xgb
from sklearn import ensemble
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[3]:


train=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Train.csv')
test=pd.read_csv('C:\\Users\\Aayush Kandpal\\Downloads\\3f488f10aa3d11ea\\Dataset\\Test.csv')


# In[4]:


train.info()


# In[5]:


train=train.drop(['Employee_ID'],axis=1)
train=train.fillna(train.mean())


# In[6]:


test=test.drop(['Employee_ID'],axis=1)
test=test.fillna(test.mean())


# In[7]:


test


# In[8]:


ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
train0 = np.array(ct1.fit_transform(train))
ct2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
test0 = np.array(ct2.fit_transform(test))
train1 = pd.DataFrame(train0)
test1=pd.DataFrame(test0)
train1.head()


# In[9]:


test1.head()


# single/married encoded 

# In[10]:


ct3 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
train2 = np.array(ct3.fit_transform(train1))
ct4 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
test2 = np.array(ct4.fit_transform(test1))
train3= pd.DataFrame(train2)
test3=pd.DataFrame(test2)
train3.head()


# In[11]:


test3.head()


#     1.0,1-gender columns
#     2.2,3-relationship column
#     3.4,5,6,7,8-hometown columns

# In[12]:


ct5 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
train4 = np.array(ct5.fit_transform(train3))
ct6 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
test4 = np.array(ct6.fit_transform(test3))
train5= pd.DataFrame(train4)
test5=pd.DataFrame(test4)
train5.head()


# In[13]:


ct7 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [11])], remainder='passthrough')
train6 = np.array(ct7.fit_transform(train5))
ct8 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [11])], remainder='passthrough')
test6 = np.array(ct8.fit_transform(test5))
train7= pd.DataFrame(train6)
test7=pd.DataFrame(test6)
train7.head()


# unit encoded from column 9 onwards count the number of unique units add them to 9 ,till there unit encoding columns reach

# In[14]:


print(train7[[15,16,17,18,19,20,21,22,23,24]])


# encoding column number 23 now
# 

# In[15]:


ct9 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [23])], remainder='passthrough')
train8 = np.array(ct9.fit_transform(train7))
ct10 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [23])], remainder='passthrough')
test8 = np.array(ct10.fit_transform(test7))
train9= pd.DataFrame(train8)
test9=pd.DataFrame(test8)
train9.head()


# In[17]:


print(train9[range(21,35,1)])


# encoding last column 33

# In[18]:


ct11 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [33])], remainder='passthrough')
train10 = np.array(ct11.fit_transform(train9))
ct12 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [33])], remainder='passthrough')
test10 = np.array(ct12.fit_transform(test9))
train11= pd.DataFrame(train10)
test11=pd.DataFrame(test10)
train11.head()


# In[19]:


test11.head()


# In[20]:


train11.columns=['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','Attrition_rate']


# In[21]:


x=train11.drop('Attrition_rate',axis=1)


# In[22]:


x


# In[23]:


y=train11.Attrition_rate


# In[25]:


y

my_submission= pd.DataFrame({ 'Attrition_rate':y})
# you could use any filename. We choose submission here
my_submission.to_csv('yexcel.csv', index=False)


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[28]:


#Multiple regression
lr=LinearRegression()
lr.fit(x,y)
reg=lr.predict(test11)


# In[32]:


#Decision Tree regression
dt=DecisionTreeRegressor(max_depth=1,splitter='random',random_state=0,criterion='friedman_mse',min_samples_split=20,
                         min_samples_leaf=10)
dt.fit(x,y)
dtr=dt.predict(test11)


# In[ ]:


#SVR
sv=SVR(kernel='linear',gamma='auto')
sv.fit(x_train,y_train)
svc=sv.predict(x_test)
print(np.sqrt(mean_squared_error(svc,y_test)))


# In[36]:


#Random Forest Regressor
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
rfr=rf.predict(x_test)
print(np.sqrt(mean_squared_error(rfr,y_test)))


# In[31]:


#Multiple regression dataset
s1=lr.predict(test11)
my_submission= pd.DataFrame({ 'Attrition_rate':s1})
# you could use any filename. We choose submission here
my_submission.to_csv('sub5.csv', index=False)



# In[33]:



my_submission= pd.DataFrame({ 'Attrition_rate':dtr})
# you could use any filename. We choose submission here
my_submission.to_csv('sub6.csv', index=False)


# In[37]:


rf.predict(test11)
my_submission= pd.DataFrame({ 'Attrition_rate':s1})
# you could use any filename. We choose submission here
my_submission.to_csv('sub3.csv', index=False)


# In[39]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 20000, max_depth = 0.01, min_samples_split = 2,subsample=0.25,
          learning_rate = .25, loss = 'ls',random_state=1,min_samples_leaf=2,alpha=0.2
                                ,warm_start=True )
clf.fit(x_train,y_train)


# In[ ]:


from xgboost import XGBRegressor
xgb=XGBRegressor()
xgb.fit(x_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:




