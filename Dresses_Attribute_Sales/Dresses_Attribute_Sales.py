#!/usr/bin/env python
# coding: utf-8

# In[11]:


#importing the 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


Attribute_DF = pd.read_excel("Attribute_DataSet.xlsx")
Dress_Sales_DF = pd.read_excel("Dress_Sales.xlsx")
Attribute_DF.head()


# In[13]:


Attribute_DF.info()


# In[14]:


# Null data in the waistline, material, and Pattern Type columns need to be dropped
for i in Attribute_DF: # loops through column names
    count = 0
    for j in Attribute_DF[i]:
        if j is np.nan:
            count += 1
    print('column: '+i+' number of nulls: '+str(count))


# In[125]:



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


# In[128]:


bargraphs(Attribute_DF,'Style')
bargraphs(Attribute_DF,'Price')
bargraphs(Attribute_DF,'Rating')
bargraphs(Attribute_DF,'Size')
bargraphs(Attribute_DF,'Season')
bargraphs(Attribute_DF,'NeckLine')
bargraphs(Attribute_DF,'SleeveLength')
bargraphs(Attribute_DF,'waiseline')


# In[134]:


df3 = Attribute_DF.drop(['Material','Pattern Type','waiseline','Dress_ID'],axis=1)

df3 = pd.get_dummies(df3) # convert the catagorical columns into dummies

feature_data = df3.drop('Recommendation',axis=1) # split the data into raw data and labels
label_data = df3['Recommendation']

# Splitting the data into test and train set
from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(feature_data,label_data,test_size=0.2,random_state=0)


# In[144]:


#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV      # will allow testing for optimal combination of model parameters

paramDict = { 'n_estimators' : [100,500,1000],
             'max_depth' : [3,10,20]}

gridSearch = GridSearchCV(RandomForestClassifier(),paramDict,cv=5) 
gridSearch.fit(x_train,y_train)
print('best parameters found via grid search %s\n'% gridSearch.best_params_)
print('scoring the test data with the random forest and best paramters %s' % gridSearch.score(x_test,y_test))


# In[137]:


classifier = RandomForestClassifier(n_estimators = 1000,max_depth=10 , criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[142]:


#KNN
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[145]:


from sklearn.metrics import roc_curve
 
PosProEstimates = gridSearch.predict_proba(x_test)[:,1] # return the positive class' prediction probability esitmates
fpr, tpr, thresholds = roc_curve(y_test, PosProEstimates)
 
plt.plot(fpr,tpr, label='ROC curve',c='blue')
plt.grid(b=True,alpha=0.7,color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate ")
close_zero = np.argmin(np.abs(np.array(thresholds) - 0.5)) # default classification threshold is 0.5. So finding the index for
# the calculated threshold closest to 0.5 requires subtracting 0.5 and finding the minimum absolute value. 
plt.plot(fpr[close_zero],tpr[close_zero],marker='X',label='zero threshold',ms=15,c='black')
plt.legend(loc='best')


# In[ ]:




