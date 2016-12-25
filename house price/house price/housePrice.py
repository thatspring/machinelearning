# -*- coding: utf-8 -*-  

import sklearn as sl
import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib
import matplotlib.pyplot as plt
import PCA


#get data
train_ori_data=pd.read_csv('train.csv')
test_ori_data=pd.read_csv('test.csv')

#print train_ori_data.info()

train_data=train_ori_data.loc[:,'MSSubClass':'SaleCondition']

test_data=test_ori_data.loc[:,'MSSubClass':'SaleCondition']

test_id=test_ori_data.loc[:,'Id']

#feature
train_data.to_csv('feature_train.csv',index=False)

test_data.to_csv('feature_test.csv',index=False)


#label
train_label=train_ori_data.loc[:,'SalePrice']
train_label.to_csv('train_label.csv')

#show price diagram
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train_label, "log(price + 1)":np.log1p(train_label)})
prices.hist()
plt.savefig('label.png')
#plt.show()

#normlize label
train_label=np.log1p(train_label)

#get numeric_feats
#numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index
#print numeric_feats
#
#qualitative to numeric
all_data=pd.concat((train_data,test_data))

#normlize
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data=pd.get_dummies(all_data)
train_data=all_data[:np.shape(train_data)[0]]
test_data=all_data[np.shape(train_data)[0]:]

#fill na
train_data = train_data.fillna(train_data.mean())
print np.shape(train_data)
test_data = test_data.fillna(test_data.mean())
print np.shape(test_data)

'''
	use pca, but not affect
'''
'''
data_arr=np.array(train_data).reshape(np.shape(train_data))
PCA.pca(data_arr)
'''
from sklearn import linear_model

'''
***************************************************************
	#linear regression 
	#ordinary least squares

reg1=linear_model.LinearRegression()
print reg1.fit(train_data,train_label)
test_label=reg1.predict(test_data)
test_label=np.e**test_label-1
test_label=pd.DataFrame({"Id":test_id,"SalePrice":test_label})
test_label.to_csv('test_label.csv',index=False)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(test_label)
plt.savefig('linear_res.png')

'''

'''
*****************************************************************
	#Ridge Regression

reg=linear_model.Ridge(alpha=.5)
reg.fit(train_data,train_label)
test_label=reg.predict(test_data)

test_label=np.e**test_label-1
test_label=pd.DataFrame({"Id":test_id,"SalePrice":test_label})
test_label.to_csv('test_label.csv',index=False)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(test_label)
plt.savefig('ridge_res.png')
'''
'''
******************************************************************
	#RidgeCV

reg=linear_model.RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])
reg.fit(train_data,train_label)
test_label=reg.predict(test_data)

test_label=np.e**test_label-1
test_label=pd.DataFrame({"Id":test_id,"SalePrice":test_label})
test_label.to_csv('test_label.csv',index=False)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(test_label)
plt.savefig('ridgecv_res.png')
'''
'''
*******************************************************************
	#lasso

reg=linear_model.Lasso(alpha=0.1)
reg.fit(train_data,train_label)
test_label=reg.predict(test_data)

test_label=np.e**test_label-1
test_label=pd.DataFrame({"Id":test_id,"SalePrice":test_label})
test_label.to_csv('test_lasso.csv',index=False)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(test_label)
plt.savefig('lasso_res.png')
'''

'''
********************************************************************
	#lassoCV ******best
'''
lassoCVreg=linear_model.LassoCV(cv=5,alphas = [1, 0.1, 0.001, 0.0005])
lassoCVreg.fit(train_data,train_label)
lassoCV_label=lassoCVreg.predict(test_data)
#print lassoCVreg.score(train_data,train_label)
lassoCV_label=np.e**lassoCV_label-1
lassoCV_label=pd.DataFrame({"Id":test_id,"SalePrice":lassoCV_label})
lassoCV_label.to_csv('test_lassoCV.csv',index=False)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(lassoCV_label)
plt.savefig('lassoCV_res.png')


'''
********************************************************************
	#lassoLarsCV

reg=linear_model.LassoLarsCV(cv=5)
reg.fit(train_data,train_label)
test_label=reg.predict(test_data)

test_label=np.e**test_label-1
test_label=pd.DataFrame({"Id":test_id,"SalePrice":test_label})
test_label.to_csv('test_lassoLarsCV.csv',index=False)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(test_label)
plt.savefig('lassoLarsCV_res.png')
'''

'''
*********************************************************************
	#xgboost
'''
import xgboost as xgb

dtrain = xgb.DMatrix(train_data, label = train_label)
dtest = xgb.DMatrix(test_data)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(train_data, train_label)
xgb_label=model_xgb.predict(test_data)
print model_xgb.score(train_data,train_label)
xgb_label=np.e**xgb_label-1
xgb_label=pd.DataFrame({"Id":test_id,"SalePrice":xgb_label})
xgb_label.to_csv('test_xgb.csv',index=False)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(xgb_label)
plt.savefig('xgb_res.png')

fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.scatter(xgb_label,lassoCV_label)
plt.savefig('xgb_lassoCV.png')


preds = 0.6*lassoCV_label.SalePrice + 0.4*xgb_label.SalePrice
print np.shape(preds)
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.rcParams['legend.fontsize'] = 10 
ax.plot(preds)
plt.savefig('fin.png')

solution = pd.DataFrame({"Id":test_id, "SalePrice":preds})
solution.to_csv("xgb_lassoCV.csv", index = False)

