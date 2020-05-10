#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv('review_data.csv')


# In[3]:


dataset.head()


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


import nltk
nltk.download('stopwords')


# In[6]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re
ps = PorterStemmer()
corpus = []

for _ in range(len(dataset['review_description'])):
    review = re.sub('[^a-zA-Z]',' ',dataset['review_description'][_])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[17]:


corpus


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
cv = CountVectorizer(max_features = 5000, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


# In[19]:


TfidfTransformer().fit(X)


# In[20]:


y = dataset.iloc[:,-2]


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33, random_state = 0)


# ## Multinonimal NB

# In[22]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha = 0.1)


# ### Model Accuracy Check

# In[23]:


from sklearn import metrics
import itertools
from sklearn.metrics import accuracy_score 


# In[25]:


previous_score = 0
for alpha in np.arange(0,1.1,0.1):
    sub_classifier = MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred = sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test,y_pred)
    if score > previous_score:
        classifier = sub_classifier
        previous_score = score
    print('Alpha: {}   Score: {}%'.format(alpha,score*100))


# ### Saving the Model

# In[26]:


classifier.fit(X,y.astype(int))


# In[27]:


from sklearn.externals import joblib
joblib_file = 'NBclassifier_model'
joblib.dump(classifier,joblib_file)

