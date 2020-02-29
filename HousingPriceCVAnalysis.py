# -*- coding: utf-8 -*-
"""
Cross validation of regression models on housing data.
	xgboost
	nearest neighbor
	SVR

Created on Tue Dec 24 18:49:02 2019

@author: inphi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# %% load data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# %% Preprocessing
# RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that all features are centered around zero and have variance in the same order

#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)
test = test.drop(test.loc[test['Electrical'].isnull()].index)

#outlier deletion
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

#applying log transformation
from scipy.stats import norm
from scipy import stats
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])
train['GrLivArea'] = np.log(train['GrLivArea'])
#create dummy variable for basement, then log transform
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

#drop Id
train = train.drop(columns = ['Id'])
#get dummy variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)#not same as train
test = test.reindex(columns = train.columns, fill_value = 0)

#get column headers
yDescription = 'SalePrice'
ytrain = train[yDescription]
xtrain = train.drop(columns = [yDescription])
xDescription = list(xtrain.columns)

#scale
from sklearn.preprocessing import StandardScaler
class DummyScaler(StandardScaler):
    def DummyFit(self, X):
        try:
            X = X.values
        except:
            pass
        
        mean = []
        std = []
        for i in range(X.shape[1]):
            x = X[:,i]
            if len(np.unique(x)) > 2:
                mean.append(np.mean(x))
                std.append(np.std(x))
            else: #dummy
                mean.append(0)
                std.append(1)
        self.mean_ = np.array(mean)
        self.scale_ = np.array(std)
        self.var_ = np.array(std)**2
            
xscaler = DummyScaler()
xscaler.DummyFit(xtrain)
xtrain = xscaler.transform(xtrain)

#xscaler = StandardScaler()
#xscaler.fit(xtrain)
#xtrain = xscaler.transform(xtrain)
#yscaler = StandardScaler()
#yscaler.fit(np.matrix(ytrain).T)
#ytrain = yscaler.transform(np.matrix(ytrain).T).flatten()

#train = StandardScaler().fit_transform(train)

plt.close('all')

##don't need imputation
#from sklearn.preprocessing import Imputer
#my_imputer = Imputer()
#xtrain = my_imputer.fit_transform(xtrain)
#xtest = my_imputer.transform(xtest)

# %% LASSO Regression

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate

alphaSearch = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

scores = []
for alpha in alphaSearch:
    lasso = Lasso(alpha = alpha)
    lasso.fit(xtrain, ytrain)
    scores.append(cross_val_score(lasso, xtrain, ytrain, cv = 5).mean())

plt.figure()
plt.plot(alphaSearch, scores, '.-')
plt.show()

lasso = Lasso(alpha = 0.001)
scores = cross_val_score(lasso, xtrain, ytrain, cv = 5)
print(scores.mean())


# %% XGBoost

from xgboost import XGBRegressor
xgbr = XGBRegressor(n_estimators=1000, learning_rate=0.05)
scores = cross_val_score(xgbr, xtrain, ytrain, cv = 5)
print(scores.mean())

# %% Nearest neighbor regression

from sklearn.neighbors import KNeighborsRegressor
knnr = []
nNeighborSpace = np.arange(1, 20, dtype = int)
scores = []
for n in nNeighborSpace:
    knru = KNeighborsRegressor(n_neighbors=n, weights = 'uniform')
    score = cross_val_score(knru, xtrain, ytrain, cv = 5).mean()
    scores.append(score)
plt.figure()
plt.plot(nNeighborSpace, scores)
plt.title('NN Uniform')
plt.grid()
plt.show()

nNeighborSpace = np.arange(1, 20, dtype = int)
scores = []
for n in nNeighborSpace:
    knru = KNeighborsRegressor(n_neighbors=n, weights = 'distance')
    score = cross_val_score(knru, xtrain, ytrain, cv = 5).mean()
    scores.append(score)
plt.figure()
plt.plot(nNeighborSpace, scores)
plt.title('NN Distance')
plt.grid()
plt.show()

knnr = KNeighborsRegressor(n_neighbors=7, weights = 'distance')
score = cross_val_score(knnr, xtrain, ytrain, cv = 5)
print(score.mean())


# %% SVR
#coef0, as p-> inf weirdness around 1, you can add 1-min <x,y>, so no values are smaller than 1 . If you really feel the need for tuning this parameter, I would suggest search in the range of [min(1-min , 0),max(<x,y>)], where max is computed through all the training set.

from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=100, gamma=.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1)

score = cross_val_score(svr_rbf, xtrain, ytrain, cv = 5).mean()
print('SVR RBF: ' + str(score))
score = cross_val_score(svr_lin, xtrain, ytrain, cv = 5).mean()
print('SVR Lin: ' + str(score))
score = cross_val_score(svr_poly, xtrain, ytrain, cv = 5).mean()
print('SVR Poly: ' + str(score))


# %% 

regressionModel = [xgbr, knnr, svr_rbf, svr_lin, svr_poly]
scores = [cross_val_score(r, xtrain, ytrain, cv = 5).mean() for r in regressionModel]













