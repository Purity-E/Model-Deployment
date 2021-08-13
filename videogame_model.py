#!/usr/bin/env python
# coding: utf-8

# In[1]:


# loading packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math


# In[2]:


# loading dataset
vidgame_sales = pd.read_csv(r'D:\Purity\Books\MLData\vgsales.csv')
print(vidgame_sales.shape)
vidgame_sales.head()


# In[3]:


#droping year column
vidgame_sales.drop(['Year'], axis = 1, inplace=True)
#filling missing values in publisher with mode
vidgame_sales['Publisher'] = vidgame_sales['Publisher'].fillna(vidgame_sales['Publisher'].mode()[0])


# In[4]:


from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import scipy.stats as ss


# In[5]:


# turning features and label to numpy array
X = np.array(vidgame_sales[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']])
y = np.array(vidgame_sales['Global_Sales'])


# In[6]:


# splitting data to form train & test set
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state = 100)


# In[7]:


# constructing linear model
lin_mod = linear_model.LinearRegression(fit_intercept = False)
lin_mod.fit(X_train, y_train)


# In[8]:


print(lin_mod.intercept_)
print(lin_mod.coef_)


# In[9]:


# testing model
y_score = lin_mod.predict(X_test) 
y_score
# printing scores
print(y_score)


# In[10]:


#saving model to disk (pickling)
import pickle
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(lin_mod, f)


# In[11]:


#loading model to compare results
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.predict([[40, 20, 50, 30]]))


# In[12]:


print(model.predict([[11.27, 8.89, 10.22, 1.00]]))


# In[ ]:




