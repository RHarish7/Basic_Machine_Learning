# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:34:03 2021

@author: rhari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
sale_price = df.iloc[:,-1].values
sale_price = pd.DataFrame(sale_price)

df.head()

df.info()

def MissingData(DataFrame):
    missingdata = pd.DataFrame({"Type":DataFrame.dtypes, "Total":DataFrame.isnull().sum(), "Precentage": DataFrame.isnull().sum()/len(DataFrame)})
    missingdata = missingdata.loc[missingdata.Precentage>0]
    return missingdata

MissingData(df)

df.drop(['Alley','PoolQC','Fence','MiscFeature','Id','Street','SalePrice'], axis=1, inplace = True)

sns.heatmap(df.corr())

#taking care of missing data
df['MasVnrType'] = df['MasVnrType'].fillna( value = df['MasVnrType'].mode()[0])
df['BsmtQual'] = df['BsmtQual'].fillna( value = df['BsmtQual'].mode()[0])
df['BsmtCond'] = df['BsmtCond'].fillna( value = df['BsmtCond'].mode()[0])
df['BsmtExposure'] = df['BsmtExposure'].fillna( value = df['BsmtExposure'].mode()[0])
df['BsmtFinType1'] = df['BsmtFinType1'].fillna( value = df['BsmtFinType1'].mode()[0])
df['BsmtFinType2'] = df['BsmtFinType2'].fillna( value = df['BsmtFinType2'].mode()[0])
df['Electrical'] = df['Electrical'].fillna( value = df['Electrical'].mode()[0])
df['FireplaceQu'] = df['FireplaceQu'].fillna( value = df['FireplaceQu'].mode()[0])
df['GarageType'] = df['GarageType'] .fillna( value = df['GarageType'].mode()[0])
df['GarageFinish'] = df['GarageFinish'].fillna( value = df['GarageFinish'].mode()[0])
df['GarageQual'] = df['GarageQual'].fillna( value = df['GarageQual'].mode()[0])
df['GarageCond'] = df['GarageCond'].fillna( value = df['GarageCond'].mode()[0])

df['LotFrontage'] = df['LotFrontage'].fillna( value = df['LotFrontage'].mean())
df['MasVnrArea'] = df['MasVnrArea'].fillna( value = df['MasVnrArea'].mean())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna( value = df['GarageYrBlt'].mean())

#Test Data Processing

MissingData(df_test)

df_test.drop(['Alley','PoolQC','Fence','MiscFeature','Id','Street'], axis=1, inplace = True)

df_test['MSZoning'] = df_test['MSZoning'].fillna( value = df_test['MSZoning'].mode()[0])
df_test['Utilities'] = df_test['Utilities'].fillna( value = df_test['Utilities'].mode()[0])
df_test['Exterior1st'] = df_test['Exterior1st'].fillna( value = df_test['Exterior1st'].mode()[0])
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna( value = df_test['Exterior2nd'].mode()[0])
df_test['MasVnrType'] = df_test['MasVnrType'].fillna( value = df_test['MasVnrType'].mode()[0])
df_test['BsmtQual'] = df_test['BsmtQual'].fillna( value = df_test['BsmtQual'].mode()[0])
df_test['BsmtCond'] = df_test['BsmtCond'].fillna( value = df_test['BsmtCond'].mode()[0])
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna( value = df_test['BsmtExposure'].mode()[0])
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna( value = df_test['BsmtFinType1'].mode()[0])
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna( value = df_test['BsmtFinType2'].mode()[0])
df_test['KitchenQual'] = df_test['KitchenQual'].fillna( value = df_test['KitchenQual'].mode()[0])
df_test['Functional'] = df_test['Functional'].fillna( value = df_test['Functional'].mode()[0])
df_test['FireplaceQu'] = df_test['FireplaceQu'].fillna( value = df_test['FireplaceQu'].mode()[0])
df_test['GarageType'] = df_test['GarageType'].fillna( value = df_test['GarageType'].mode()[0])
df_test['GarageFinish'] = df_test['GarageFinish'].fillna( value = df_test['GarageFinish'].mode()[0])
df_test['GarageQual'] = df_test['GarageQual'] .fillna( value = df_test['GarageQual'].mode()[0])
df_test['GarageCond'] = df_test['GarageCond'].fillna( value = df_test['GarageCond'].mode()[0])
df_test['SaleType'] = df_test['SaleType'].fillna( value = df_test['SaleType'].mode()[0])
df_test['LotFrontage'] = df_test['LotFrontage'].fillna( value = df_test['LotFrontage'].mean())
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna( value = df_test['MasVnrArea'].mean())
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna( value = df_test['BsmtFinSF1'].mean())
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna( value = df_test['BsmtFinSF2'].mean())
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna( value = df_test['TotalBsmtSF'].mean())
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna( value = df_test['BsmtUnfSF'].mean())
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna( value = df_test['BsmtFullBath'].mean())
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna( value = df_test['BsmtHalfBath'].mean())
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna( value = df_test['GarageYrBlt'].mean())
df_test['GarageCars'] = df_test['GarageCars'].fillna( value = df_test['GarageCars'].mean())
df_test['GarageArea'] = df_test['GarageArea'].fillna( value = df_test['GarageArea'].mean())



df.head()


df_cat = df.copy()
df_cat_test = df_test.copy()
df_num = df.copy()
df_num_test = df_test.copy()


df_cat.drop(['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold'], axis=1, inplace = True)
df_cat_test.drop(['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold'], axis=1, inplace = True)
df_num.drop(['MSZoning','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition','FireplaceQu'], axis=1, inplace = True)
df_num_test.drop(['MSZoning','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition','FireplaceQu'], axis=1, inplace = True)


df_cat_merge = df_cat.append(df_cat_test)

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = "all")

df_cat_merge = onehotencoder.fit_transform(df_cat_merge).toarray()

df_cat = df_cat_merge[0:len(df_cat)]
df_cat = pd.DataFrame(df_cat)
df_cat_test = df_cat_merge[len(df_cat):]
df_cat_test = pd.DataFrame(df_cat_test)

df_final = pd.concat([df_num,df_cat], axis=1, join="inner")
df_final_test = pd.concat([df_num_test,df_cat_test], axis=1, join="inner")



#XGBoost Regression
from xgboost import XGBRegressor
regressor = XGBRegressor(base_score=0.5, booster='gbtree',
                                          colsample_bylevel=1,
                                          colsample_bynode=1,
                                          colsample_bytree=1, gamma=0,
                                          gpu_id=-1, importance_type='gain',
                                          interaction_constraints='',
                                          learning_rate=0.300000012,
                                          max_delta_step=0, max_depth=6,
                                          min_child_weight=1, missing=None,
                                          monotone_constraints='()',
                                          n_estimators=100)
regressor.fit(df_final,sale_price)

#Applying Grid Search

'''from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
parameters = [{'kernel':('linear', 'rbf'), 'C':[1, 10]}]
svc = svm.SVC()
grid_search = GridSearchCV(svc, parameters)
grid_search.fit(df_final,sale_price)'''


Y_pred = regressor.predict(df_final_test)


print(Y_pred)

