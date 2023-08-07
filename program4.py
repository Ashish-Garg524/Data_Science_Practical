#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[2]:


df=pd.read_csv('C:/Users/DS1/Downloads/student_performance_new.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isna().sum()


# In[7]:


df['Compensatory'] = df['Compensatory'].fillna(df['Compensatory'].mean())


# In[8]:


df.isna().sum()


# In[12]:


x=df[df.columns[3:17]]
x


# In[13]:


y = df["Result"]
y


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# In[15]:


clf = DecisionTreeClassifier(criterion='gini',max_depth = 3,random_state =20)


# In[16]:


clf = clf.fit(x_train,y_train)


# In[17]:


classnames = ['Pass','Fail']


# In[18]:


fig = plt.figure(figsize=(50,40))


# In[20]:


from sklearn import tree
tree.plot_tree(clf,feature_names = x_train.columns,class_names = classnames,filled =True)


# In[24]:


train_tree_predict = clf.predict(x_train)
confusionmatrix=confusion_matrix(y_train,train_tree_predict)
sns.heatmap(confusionmatrix,annot=True,yticklabels=classnames,xticklabels=classnames)


# In[26]:


#training accuaccuracy_score
accuracy=accuracy_score(y_train,train_tree_predict)
print(accuracy)


# In[33]:


#test accuaccuracy_score
test_tree_predict = clf.predict(x_test)
accuracy=accuracy_score(y_test,test_tree_predict)
print(accuracy)


# In[27]:


#dcision_report
report = decision_report(y_train,train_tree_predict)


# In[ ]:




