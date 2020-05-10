#!/usr/bin/env python
# coding: utf-8

# # Data Analysis

# In[84]:


#Necessary imports for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[85]:


#Loading dataset
dataset = pd.read_csv('train.csv')


# In[86]:


dataset.head()


# In[87]:


dataset.shape


# In[88]:


#Number of varieties or classes of wine
len(dataset['variety'].unique())


# ## Missing Values

# In[89]:


#Check the percentages of missing values
na_features = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 0]
for feature in na_features :
    print(feature,': ',np.round(dataset[feature].isnull().mean(),4), '% missing values')


# ### Removing columns with high nan values

# In[90]:


#Drop columns with high missing values
dataset = dataset.drop(columns={'region_2','region_1','designation','user_name'},axis=1)


# In[91]:


len(dataset['review_title'].unique())


# In[92]:


#review_title contains very high number of unique categories, hence does not provide any useful info.
dataset = dataset.drop(columns='review_title',axis=1)


# In[93]:


#After Column removal
dataset.head()


# In[143]:


#Number of entries for each variety
l=[]
for _ in range(len(dataset['variety'].unique())):
    a = dataset[dataset['variety'] ==dataset['variety'].unique()[_] ].shape[0]
    l.append(a)
    print(' {} : {}'.format(dataset['variety'].unique()[_],a))


# In[146]:


v = dataset['variety'].unique()


# ## Numerical Features

# In[94]:


#Features that contain numerical data
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype != 'O']
numerical_features


# ### Outliers

# In[116]:


#A Visual representation of feature values that fall out of 
for feature in numerical_features:    
    data=dataset.copy()
    if 0 in data[feature].unique():  #log cannot take 0
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        


# The Price has a low median, but there are significant amount of outliers

# ### Missing values in numerical features

# In[96]:


#Percentage of missing values in numerical features
numerical_with_nan = [feature for feature in numerical_features if  dataset[feature].isnull().sum()>1]

for feature in numerical_with_nan:
    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# In[97]:


#Replace missing values in "Price" feature with the median of the data(not mean due to significant number of outliers)
for feature in numerical_with_nan:
        median_value = dataset[feature].median()
        dataset[feature].fillna(median_value,inplace = True)
        
dataset[numerical_with_nan].isnull().sum()


# ## Categorical Features

# In[98]:


categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == 'O' and feature not in ['review_description']]
categorical_features


# In[99]:


#Number of categories in each categorical features
for feature in categorical_features:
    print('The feature is {} and the of categories are {}'.format(feature,len(dataset[feature].unique())))


# In[100]:


#Drop entries with nan values
dataset = dataset.dropna()
dataset.shape
#Only a few samples have been lost due to nan values, so we can proceed


# In[101]:


#reset the indexes to compensate the removal of nan containing entries
dataset.reset_index(inplace=True)


# In[102]:


dataset_2= dataset.copy()


# ### Enumeration of Categorical features

# In[103]:


dataset_2['wine_type'] = dataset_2['variety']


# In[104]:


#Enumeration of categorical_features 

for feature in categorical_features  :
    labels_ordered = dataset_2.groupby(feature)['price'].mean().sort_values().index
    labels_ordered = {k:i for i,k in enumerate(labels_ordered,0)}
    dataset_2[feature] = dataset_2[feature].map(labels_ordered)
    


# In[105]:


labels_ordered = {v: k for k, v in labels_ordered.items()}


# In[106]:


import pickle
f = open("dictionary.pkl","wb")
pickle.dump(labels_ordered,f)
f.close()


# In[107]:


dataset_2.head()


# In[108]:


dataset_2[['review_description','variety','wine_type']].head()


# In[109]:


dataset_2[['review_description','variety','wine_type']].to_csv('review_data.csv',index=False)


# # Data Visualization

# In[110]:


dataset[dataset['price']==dataset['price'].max()]


# In[111]:


dataset[dataset['points']==dataset['points'].max()]


# In[112]:


fig = plt.figure(figsize=(22,8))
plt.scatter(x=dataset['country'],y=dataset['price'])
plt.xticks(rotation='vertical')
plt.xlabel('Country',size=25)
plt.ylabel('Price',size=25)
plt.show()
fig.suptitle('Relation between Country and Price',size=25)

fig.savefig('Country vs Price.jpg')


# In[113]:


fig = plt.figure(figsize=(20,10))
plt.scatter(x=dataset['variety'],y=dataset['price'])
plt.xticks(rotation='vertical')
plt.xlabel('Variety',size=25)
plt.ylabel('Price',size=25)
plt.show()
fig.suptitle('Relation between Variety and Price',size=25)

fig.savefig('Variety vs Price.jpg')


# In[114]:


fig = sns.jointplot(x=dataset['points'],y=dataset['price'],kind = 'reg')
fig.savefig('Points vs Price.jpg')


# In[151]:


fig = plt.figure(figsize=(20,10))
plt.scatter(x=v,y=l)
plt.xticks(rotation='vertical')
plt.xlabel('Variety',size=25)
plt.ylabel('Number of reviews',size=25)
plt.show()
fig.suptitle('Relation between Number of reviews and variety',size=25)

fig.savefig('Variety vs num_reviews.jpg')

