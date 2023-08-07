#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().system('pip install nltk')
import nltk
from nltk import masi_distance
from nltk.probability import FreqDist
import urllib.request
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS 
nltk.download('punkt')


# In[12]:


agatha_novel = open('C:/Users/DS1/Desktop/AgathaChristie.txt','r').read()
agatha_novel[1000:2000]


# In[13]:


print(agatha_novel[1000:2000])


# In[14]:


from nltk import word_tokenize
words = word_tokenize(agatha_novel)
print(words)


# In[15]:


len(words)


# In[16]:


fdist = FreqDist(words)
fdist.most_common(10)


# In[19]:


word_no_punc = []
for word in words:
    if word.isalpha():
        word_no_punc.append(word.lower())
print(len(word_no_punc))


# In[22]:


nltk.download('stopwords')
from nltk.corpus import stopwords


# In[27]:


stopwords_list = stopwords.words('english')
print(stopwords_list)


# In[33]:


clean_word=[]
for word in word_no_punc:
    if word not in stopwords_list:
        clean_word.append(word)
print(clean_word)


# In[34]:


print(len(clean_word))


# In[37]:


fdist = FreqDist(clean_word)
fdist.most_common(10)


# In[40]:


clean_words_string = " ".join(clean_word)

wordcloud = WordCloud(background_color="lightgreen").generate(clean_words_string)

plt.figure(figsize=(12,12))
plt.imshow(wordcloud)

plt.show()


# In[44]:


mask  = np.array(Image.open('C:/Users/DS1/Desktop/cloudImage.jpg'))


# In[53]:


# A similar function, but using the mask
kl= WordCloud(stopwords = STOPWORDS,mask = mask,background_color = "white",max_words = 2000, max_font_size = 500,random_state = 42,width = mask.shape[1],height = mask.shape[0])
    
kl.generate(agatha_novel) 
plt.imshow(kl, interpolation = "None")

plt.axis('off')
 
# Now show the output cloud
plt.show()
    


# In[ ]:




