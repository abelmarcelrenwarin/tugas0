#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plp
import seaborn as sns
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv('heart.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


plp.figure(figsize= (17,6))
sns.heatmap(df.corr(), annot = True)


# In[5]:


df['sex'].value_counts()


# In[6]:


df['output'].value_counts()


# In[7]:


x = df.drop("output", axis=1)
y = df["output"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state= 1)


# In[8]:


model = LogisticRegression(max_iter = 1000)
model.fit(x_train, y_train)

result = model.predict(x_test)
print(result)


# In[9]:


model.score(x_test,y_test)


# In[10]:


pk.dump(model,open("model0.pkl", "wb"))

