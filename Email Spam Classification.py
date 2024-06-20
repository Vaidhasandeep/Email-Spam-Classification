#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk


# In[2]:


df = pd.read_csv('C:\\Users\\v.omsai\\Downloads\\SMSSpamCollection', sep='\t', names=['labels','message'])
df.head()


# In[3]:


df.info()


# In[4]:


df['labels'].unique()


# In[5]:


df['labels'].value_counts()


# In[6]:


df['labels'].value_counts(normalize=True)


# In[7]:


import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[8]:


corpus=[]
for i in range(len(df)):
    l = re.sub('[^a-zA-z]',' ', df['message'][i])
    l = l.lower()
    l = l.split()
    l = [ps.stem(word) for word in l if not word in stopwords.words('english')]
    l = ' '.join(l)
    corpus.append(l)


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()


# In[12]:


y = pd.get_dummies(df['labels'],drop_first=True)


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[14]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)


# In[15]:


ypred_test = model.predict(X_test)
ypred_train = model.predict(X_train)


# In[16]:


from sklearn.metrics import accuracy_score
print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))


# In[24]:


from sklearn.model_selection import cross_val_score
cross_val_score(model,X,y,cv=5,scoring = 'accuracy').mean()


# In[ ]:




