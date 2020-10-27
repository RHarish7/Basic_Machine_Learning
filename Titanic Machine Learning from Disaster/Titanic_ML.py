#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('train.csv')
X = dataset.iloc[ :, [2,4,6,7]].values
Y = dataset.iloc[ :, 1].values

#taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN' , strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:,[0,2,3]])
X[:,[0,2,3]] = imputer.transform(X[:,[0,2,3]])


#one hot encoding 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers =[('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Creating train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)


#random forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 300, criterion = 'entropy', max_depth=4, random_state=0)
classifier.fit(X_train,Y_train)


y_pred = classifier.predict(X_test)


# In[9]:


#Grid Search
from sklearn.model_selection import GridSearchCV
parameter = { 'n_estimators': [50,100,200,300,500],'max_features': ['auto', 'sqrt', 'log2'],'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}
grid_search = GridSearchCV(estimator= classifier, param_grid=parameter, cv= 10, n_jobs=-1)
grid_search.fit(X_train,Y_train)
best_accuracy = grid_search.best_score_
print(best_accuracy)
best_prameter = grid_search.best_params_
print(best_prameter)


# In[15]:


#Confusion matrix and Acurate score

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,y_pred)
print (cm)
accuracy_score(Y_test,y_pred)


# In[4]:


#To find the missing data
sns.heatmap(dataset.isnull(), yticklabels= False, cbar= False, cmap= 'Blues')


# In[ ]:




