# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:15:35 2021

@author: rhari
"""

import numpy as np 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import metrics
from scipy import stats

from copy import deepcopy

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score, mean_squared_error
train_df = pd.read_csv('sales_train.csv')
test_df = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')
shops_df = pd.read_csv('shops.csv')
items_df = pd.read_csv('items.csv')
item_categories_df = pd.read_csv('item_categories.csv')


train_df.drop(['date_block_num','item_price'], axis=1, inplace=True)
train_df['date'] = pd.to_datetime(train_df['date'], dayfirst=True)
train_df['date'] = train_df['date'].apply(lambda x: x.strftime('%Y-%m'))
train_df.head()


df = train_df.groupby(['date','shop_id','item_id']).sum()
df = df.pivot_table(index=['shop_id','item_id'], columns='date', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df.head()


test_df = pd.merge(test_df, df, on=['shop_id','item_id'], how='left')
test_df.drop(['ID'], axis=1, inplace=True)
test_df = test_df.fillna(0)
test_df.head()

Y_train = df['2015-10'].values
X_train = df.drop(['2015-10'], axis = 1)
X_test = test_df


x_train, x_test, y_train, y_test = train_test_split( X_train, Y_train, test_size=0.2, random_state=101)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


regression = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=42,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

regression.fit(x_train, y_train)

# Use the forest's predict method on the test data
predictions = regression.predict(x_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

test_df.drop(['2015-10'], axis=1, inplace=True)
preds=pd.Series(regression.predict(test_df))
submission = pd.DataFrame({
    "ID": test_df.index, 
    "item_cnt_month": preds
})
submission.head()

submission.to_csv('xgb_submission.csv', index=False)