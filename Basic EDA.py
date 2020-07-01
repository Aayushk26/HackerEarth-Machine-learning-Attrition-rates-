#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')

import seaborn as sns


# In[12]:


train.info()


# In[13]:


train.describe()


# In[14]:


train


# In[15]:


train.count()


# In[16]:


# Splitting categorical data and numerical data for ease of use
categorical_features= train.select_dtypes(include=['object']).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("Attrition_rate")


# In[17]:


# missing values asa percenatge
mv=((7000-train.count())/7000)*100
plt.plot(mv)
plt.xlabel("Features")
plt.ylabel('Percentage of missing value %')


# In[31]:


# Findiig out the missing values using counter and plotting graphs in percentage 

missing_values_count = train.isnull().sum()
print(missing_values_count)
plt.hist(missing_values_count,bins=24)
plt.xlabel('Features')


# In[18]:


# The mean has been found to see its compatibility with the dataset , further in the other section of zip file all the missing 
#  values have been replaced by the mean of the respective columns. I have tried with median and mode but the dataset became more skew.
train.mean()


# In[61]:


#Distribution of the data (Target variable)
plt.figure(figsize=(10, 7))
sns.distplot(train['Attrition_rate'], color='g', bins=100, hist_kws={'alpha': 0.5});
plt.xlabel("Percentage/100(Attrition Rate)")


# In[54]:


# Visualising the skewness of the data using a simple hist plot

print ("Skew is:", train.Attrition_rate.skew())

plt.hist(train.Attrition_rate,bins=100)
plt.xlabel
plt.show()


# In[44]:


# declaring x,y for plotting kde further down 
    
y=train.Attrition_rate
x=train.drop(['Attrition_rate'],axis=1)


# In[58]:


# Visualising the correlation matrix , values ahve been mentioned below

matrix=train.corr()
fig,ax=plt.subplots(figsize=(10,6))
sns.heatmap(matrix,vmax=0.8,square=True, cmap="Blues")
plt.show()


# In[59]:


print(matrix)
#Correlation data


# In[7]:


matrix1 = train.corr() 
  
cg = sns.clustermap(matrix1, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
cg


# In[19]:


numerical_features


# In[11]:


categorical_features


# In[9]:


# Although ths may seem like an unncessary graphs some of the best results have come out by closely examining the following graph
 # The regression line helps visualise and better understand the correlation between different features and target varibale

sns.pairplot(train,kind='reg')


# In[39]:


# A simple KDE plot

y=train.Attrition_rate
sns.kdeplot(y,  shade=True, shade_lowest=True, )


# In[40]:


# the individual values of correlation of different features and the target variable * Attrition_rate*

numerical_features = train.select_dtypes(include=[np.number])
corr = numerical_features.corr()
print (corr['Attrition_rate'].sort_values(ascending=False)[:], '\n')


# In[ ]:





# In[117]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




