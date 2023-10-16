#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("C:/Users/DS1/Downloads/breastcancer1.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[11]:

r=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se']
for i in r:
    df[i]=df[i].fillna(df[i].mean())

OR

df['radius_mean'] = df['radius_mean'].fillna(df['radius_mean'].mean())
df['texture_mean'] = df['texture_mean'].fillna(df['texture_mean'].mean())
df['perimeter_mean'] = df['perimeter_mean'].fillna(df['perimeter_mean'].mean())
df['area_mean'] = df['area_mean'].fillna(df['area_mean'].mean())
df['smoothness_mean'] = df['smoothness_mean'].fillna(df['smoothness_mean'].mean())
df['compactness_mean'] = df['compactness_mean'].fillna(df['compactness_mean'].mean())
df['concavity_mean'] = df['concavity_mean'].fillna(df['concavity_mean'].mean())
df['concave points_mean'] = df['concave points_mean'].fillna(df['concave points_mean'].mean())
df['symmetry_mean'] = df['symmetry_mean'].fillna(df['symmetry_mean'].mean())
df['fractal_dimension_mean'] = df['fractal_dimension_mean'].fillna(df['fractal_dimension_mean'].mean())
df['radius_se'] = df['radius_se'].fillna(df['radius_se'].mean())
df['texture_se'] = df['texture_se'].fillna(df['texture_se'].mean())
df['texture_se'] = df['texture_se'].fillna(df['texture_se'].mean())


# In[12]:


df.isnull().sum()


# In[13]:


X = np.array(df.iloc[0:, 1:])
X


# In[56]:


X = df[['texture_mean','perimeter_mean']]


# In[61]:


y = df['diagnosis']


# In[62]:


X.shape


# In[63]:


y.shape


# In[38]:

#install sklearn
!pip install scikit-learn'


# using training and test set 
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,train_size = 42, random_state = 42)


# In[71]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[73]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6,metric = 'minkowski',p=2)
knn.fit(X_train, y_train)



# In[74]:


knn.score(X_test, y_test)


# In[78]:


from sklearn.metrics import confusion_matrix,accuracy_score

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


sns.heatmap(cm,annot=True,cmap = 'rainbow')



#(OPtional)
# In[1]:

knn.predict([[10.38,20.29]])[0]


new_data=np.array([[20.0,15.0]])
result=knn.predict(new_data)
if result[0]==1:
    print("Diagnosis:Malignant")
else:
    print("Diagnosis:Benign")





