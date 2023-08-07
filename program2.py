#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
get_ipython().system('pip install seaborn')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


mpg = pd.read_csv("C:/Users/DS1/Downloads/mpg_raw.csv")
mpg.head()


# In[3]:


mpg.info()


# In[4]:


mpg.isna().sum()


# In[5]:


mpg.fillna(value=104.469388,inplace=True)


# In[6]:


mpg.isna().sum()


# In[14]:


mpg['displacement']=pd.Categorical(mpg['displacement'])
mpg['cylinders']=pd.Categorical(mpg['cylinders'])
mpg['origin']=pd.Categorical(mpg['origin'])


# In[8]:


plt.hist(mpg['displacement'])


# In[9]:


plt.hist(mpg['horsepower'])


# In[15]:


plt.hist(mpg['origin'])


# In[11]:


plt.scatter(x='horsepower', y='acceleration',data = mpg)


# In[12]:


sns.countplot(x='displacement',data=mpg)


# In[13]:


sns.pointplot(x='horsepower', y='displacement',data = mpg)


# In[ ]:




