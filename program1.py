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


mpg.shape


# In[6]:


mpg.info()


# In[7]:


mpg.describe()


# In[8]:


#part(b)


# In[10]:


mpg.isna().sum()


# In[13]:


mpg.fillna(value=104.469388,inplace=True)


# In[14]:


mpg.isna().sum()


# In[ ]:


#part(c)


# In[23]:


mpg['displacement']=pd.Categorical(mpg['displacement'])
mpg['cylinders']=pd.Categorical(mpg['cylinders'])


# In[18]:


plt.hist(mpg['displacement'])


# In[19]:


plt.hist(mpg[''acceleration''])


# In[22]:


mpg.head()


# In[20]:


#part(d)


# In[27]:


sns.violinplot(x='cylinders',y='model_year',data=mpg)


# In[28]:


#part(e)


# In[40]:


sns.boxplot(mpg['acceleration'])


# In[41]:


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


#part(g)


# In[50]:


nums = list(mpg.select_dtypes(exclude = ['object']).columns)
nums


# In[72]:


pip install sklearn


# In[77]:


continous = mpg.select_dtypes(include=['float64','int64']).columns
continous


# In[3]:


from sklearn.preprocessing import StandardScalar

mpg[continous]=StandardScalar().fit_transform(mpg[continous])


# In[2]:


import sklearn.preprocessing


# In[80]:


corr = mpg.cor()
sb.heatmap(corr,annot =True,cmap = "coolwarm")


# In[ ]:




