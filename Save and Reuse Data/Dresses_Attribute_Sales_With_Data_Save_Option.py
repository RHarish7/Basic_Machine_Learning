#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#importing the dataset
Attribute_DF = pd.read_excel("Attribute_DataSet.xlsx")
Dress_Sales_DF = pd.read_excel("Dress_Sales.xlsx")
Attribute_DF.head()


# In[3]:


#basic info on dataset
Attribute_DF.info()


# In[5]:


#Describe the data set
Attribute_DF.describe()


# In[6]:


# Null data in the waistline, material, and Pattern Type columns need to be dropped
for i in Attribute_DF: # loops through column names
    count = 0
    for j in Attribute_DF[i]:
        if j is np.nan:
            count += 1
    print('column: '+i+' number of nulls: '+str(count))


# In[8]:


def bargraphs(df,feature):
    totalCountYesRec = df[df['Recommendation']==1].shape[0] # total number of rows for both recommended and not recommended dresses
    totalCountNotRec = df[df['Recommendation'] ==0].shape[0]
    
    resultdf = pd.DataFrame()
    df = df.dropna(axis=0,how='any')
    
    allFeatureValues = df[feature].unique()
    allFeatureValues = np.sort(allFeatureValues)
    resultdf[feature] = allFeatureValues # defining the necessary columns for the temporary data frame
    resultdf['featureRec'] = np.nan
    resultdf['featureNotRec'] = np.nan
    
    # loop will populate the columns with calculated field values that represent the percent of each style makes up of all dresses
    # in either recommended or not recommended dresses
    for col in allFeatureValues:
        resultdf.loc[resultdf[feature] == col,['featureRec']] = (df[(df[feature] == col) & (df['Recommendation'] == 1)][feature].count() / totalCountYesRec) 
        #print(df[df['style'] == col & df['recommendation'] == 1]['style'].count() / totalCountYesRec)
        resultdf.loc[resultdf[feature] == col,['featureNotRec']] =(df[(df[feature] == col) & (df['Recommendation'] == 0)][feature].count() /totalCountNotRec)
        
        
    fig, ax = plt.subplots(1,1,figsize=(9,6))
    
    #resultdf[['pctStyleRec','pctStyleNotRec']].plot(kind='bar',subplots=False)
    print(resultdf)
    
    xTicks = len(allFeatureValues)
    
    ax.bar(np.arange(xTicks),resultdf['featureRec'],width=0.45,color='g',align='edge',label='recommended dress: '+feature+' frequency',tick_label=allFeatureValues)
    ax.bar(np.arange(xTicks),resultdf['featureNotRec'],width=-0.45,color='r',align='edge',label='non-rec dress: '+feature+' frequency',tick_label=allFeatureValues)
    plt.xticks(rotation=70)
    plt.title('percent of dresses of feature: '+feature+' in recommend or non-recommended class')
    plt.legend(loc='best')
    plt.show()


# In[9]:


bargraphs(Attribute_DF,'Style')
bargraphs(Attribute_DF,'Price')
bargraphs(Attribute_DF,'Rating')
bargraphs(Attribute_DF,'Size')
bargraphs(Attribute_DF,'Season')
bargraphs(Attribute_DF,'NeckLine')
bargraphs(Attribute_DF,'SleeveLength')
bargraphs(Attribute_DF,'waiseline')


# In[12]:


df3 = Attribute_DF.drop(['Material','Pattern Type','waiseline','Dress_ID'],axis=1)

df3 = pd.get_dummies(df3) # convert the catagorical columns into dummies

X = df3.drop('Recommendation',axis=1) # split the data into raw data and labels
Y = df3['Recommendation']

# Splitting the data into test and train set
from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[13]:


#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV      # will allow testing for optimal combination of model parameters

paramDict = { 'n_estimators' : [100,500,1000],
             'max_depth' : [3,10,20],
              'random_state' : [0,10,25,32,64]}

gridSearch = GridSearchCV(RandomForestClassifier(),paramDict,cv=5) 
gridSearch.fit(x_train,y_train)
print('best parameters found via grid search %s\n'% gridSearch.best_params_)
print('scoring the test data with the random forest and best paramters %s' % gridSearch.score(x_test,y_test))


# In[15]:


classifier = RandomForestClassifier(n_estimators = 100,max_depth=20 , criterion = 'entropy', random_state = 32)
classifier.fit(x_train, y_train)


# In[17]:


# Fitting a model on the scaled dataset
#in case if you have scaled your data set then you can also save the scaler method to another file in similar way
from pickle import dump
# save the model
dump(classifier, open('model.pkl', 'wb'))


# In[19]:


from pickle import load

# load the model
model = load(open('model.pkl', 'rb'))

# Predicting the Test set results
y_pred = model.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




