#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
import pathlib


# In[1]:


get_ipython().system('pip install seaborn')


# In[68]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import sklearn as sk


# In[3]:


mpg = pd.read_csv("C:/Users/DS1/Downloads/mpg_raw.csv")


# In[4]:


mpg.head()


# In[5]:

#Dimension
mpg.shape


# In[6]:

#Structure
mpg.info()


# In[7]:

#Summary
mpg.describe()


# In[8]:


#part(b)


# In[10]:


mpg.isna().sum()


# In[13]:


mpg['horsepower'] = mpg['horsepower'].fillna(mpg['horsepower'].mean())

# In[14]:


mpg.isna().sum()


# In[ ]:


#part(c)


# In[23]:

#changing Continous variable into categorical variable
mpg['displacement']=pd.Categorical(mpg['displacement'])
mpg['cylinders']=pd.Categorical(mpg['cylinders'])


# In[18]:

#Plotting histogram for continuous variable
plt.hist(mpg['horsepower'])


plt.hist(mpg['weight'])


#part(d)


# In[27]:


sns.violinplot(x='cylinders',y='model_year',data=mpg)


# In[28]:


#part(e)


# In[40]:


sns.boxplot(mpg['acceleration'])


# In[41]:

#Part(v)
Q1 = mpg['acceleration'].quantile(0.25)
Q3 = mpg['acceleration'].quantile(0.75)
IQR = Q3 - Q1
print("Q1: ",Q1,"\nQ3: ",Q3,"\nIQR: ",IQR)
Upper_Whisker = Q3 + (1.5*IQR)
Lower_Whisker = Q1 - (1.5*IQR)
print(Upper_Whisker,Lower_Whisker)


# In[42]:


mpg=mpg[mpg['acceleration']<Upper_Whisker]
mpg=mpg[mpg['acceleration']>Lower_Whisker]


# In[43]:


sns.boxplot(mpg['acceleration'])


# In[44]:


#part(vi)


# In[50]:


#part(vi)
sns.heatmap(mpg.corr())


# In[72]:


#!pip install sklearn (use scikit-learn to install sklearn )

#part(vii)
# First, install scikit-learn if you haven't already
!pip install scikit-learn

# Import the necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the 'horsepower' column
mpg['horsepower'] = scaler.fit_transform(mpg[['horsepower']])



