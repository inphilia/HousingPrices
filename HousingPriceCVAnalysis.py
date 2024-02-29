# -*- coding: utf-8 -*-
"""
Cross validation of regression models on housing data.
	ols
    lasso
    xgboost
	nearest neighbor

Created on Tue Dec 24 18:49:02 2019

@author: inphi
"""

# %% 

!pip install hyperopt

# %% import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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


# %% OLS Regression

from sklearn.linear_model import LinearRegression

# %% LASSO Regression

from sklearn.linear_model import Lasso

alpha = [1e-4, 5e-4, 1e-3, 5e-3, 0.005, 0.01, 0.05, 0.1]
scores = []
for a in alpha:
    lasso = Lasso(alpha = a)
    scores.append(cross_val_score(lasso, xtrain, ytrain, cv = 5).mean())

plt.figure()
plt.plot(alpha, scores, '.-')
plt.xlabel('alpha')
plt.ylabel('score')
plt.xscale('log')
plt.title('Lasso Regression')
plt.grid()
plt.show()


# %% XGBoost

from xgboost import XGBRegressor

xgbr = XGBRegressor(n_estimators=500, learning_rate=0.05)
max_depth = range(1, 8)
scores = []
for md in max_depth:
    xgbr = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=md)
    scores.append(cross_val_score(xgbr, xtrain, ytrain, cv = 5).mean())

plt.figure()
plt.plot(max_depth, scores, '.-')
plt.xlabel('max_depth')
plt.ylabel('score')
plt.title('XGBoost Regression')
plt.grid()
plt.show()

# %% LightGBM

from lightgbm import LGBMRegressor

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
plt.xlabel('Neighbors')
plt.ylabel('Score')
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

# %% Run all models

cv_fold_num = 5

xgb_grid = {'max_depth': range(2, 10),
            #'gamma': range(1, 9),
            #'min_child_weight' : range(10),
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': range(60, 220, 40)}
lightgbm_grid = {'max_depth': range(2, 10),
                'learning_rate': [0.01, 0.05, 0.1, 0.2]}
knn_grid = {'n_neighbors': range(2, 20),
            'weights': ['uniform', 'distance']}
models = [{'name': 'Linear Regression', 'model': LinearRegression(), 'grid': None},
          {'name': 'Lasso', 'model': Lasso(), 'grid': dict(alpha = [1e-4, 5e-4, 1e-3, 5e-3, 0.005, 0.01, 0.05, 0.1])},
          {'name': 'XGBoost', 'model': XGBRegressor(), 'grid': xgb_grid},
          {'name': 'LightGBM', 'model': LGBMRegressor(), 'grid': lightgbm_grid},
          {'name': 'k Nearest Neighbor', 'model': KNeighborsRegressor(), 'grid': knn_grid}]

def run_models(models, xtrain, ytrain, cv_fold_num):
    best_models = []
    for model in models:
        try:
            gridSearch = GridSearchCV(estimator=model['model'], param_grid=model['grid'], n_jobs=-1,
                cv=cv_fold_num, scoring="neg_mean_squared_error")
            searchResults = gridSearch.fit(xtrain, ytrain)
            best_model = searchResults.best_estimator_
        except TypeError:
            best_model = model['model']
            best_model.fit(xtrain, ytrain)
        
        #predictions
        ypred = best_model.predict(xtrain)
        plt.figure()
        plt.plot(ytrain, '.')
        plt.plot(ypred, '.')
        plt.xlabel('Home #')
        plt.ylabel('Price')
        plt.title(model['name'])
        plt.legend(['Actual', 'Predicted'])
        plt.show()

        #scores
        score = best_model.score(xtrain, ytrain)
        print(model['name'], 'score:', score)
        best_models.append(best_model)

        if model['grid'] is not None:
            print(model['name'], 'best parameters:', searchResults.best_params_)

    return best_models

best_models = run_models(models, xtrain, ytrain, cv_fold_num)

# %%
