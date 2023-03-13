import os
import pickle
import io

import numpy as np
import pandas as pd
from matplotlib import pylab as plt

from itertools import product
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from scipy.stats import linregress
import statsmodels.api as sm

def plot_target_vs_pred(target, preds):
    f = plt.figure()
    plt.plot(target, preds, "o", color="black", markerfacecolor='darkgoldenrod', markeredgewidth=0.5)
    reg = linregress(target, preds)
    plt.axline(xy1=(0, reg.intercept), slope=reg.slope, linestyle="-", color="red")
    plt.axline([0, 0], [1, 1], linestyle="--", color='black')
    plt.xlim([target.min(), target.max()])
    plt.ylim([preds.min(), preds.max()])
    plt.xlabel('Age at MRI')
    plt.ylabel('Cardiac-predicted age (years)')
    plt.show()

class BiasCorrectEstimator:
    def __init__(self, est, x_sval_bc, y_sval_bc):
        self.est = est
        x_sval_bc = np.array(x_sval_bc)
        y_sval_bc = np.array(y_sval_bc).reshape(-1)
        
        reg = LinearRegression()
        
        y_train = est.predict(x_sval_bc).reshape(-1) - y_sval_bc
        x_train = y_sval_bc.reshape((-1, 1))
        
        reg.fit(x_train, y_train)
        
        self.intercept = reg.intercept_
        self.coef = reg.coef_.item()

    def predict(self, x_pred, y_pred):
        x_pred = np.array(x_pred)
        y_pred = np.array(y_pred).reshape(-1)
        
        pred = self.est.predict(x_pred).reshape(-1)
        pred -= self.intercept + self.coef * y_pred.reshape(-1)
        return pred
        
class BiasCorrectEstimator2:
    def __init__(self, est, x_sval_bc, y_sval_bc):
        self.est = est
        
        x_sval_bc = np.array(x_sval_bc)
        y_sval_bc = np.array(y_sval_bc).reshape(-1)
        
        reg = LinearRegression()
        reg.fit(y_sval_bc.reshape((-1, 1)), est.predict(x_sval_bc).reshape(-1))
        self.intercept = reg.intercept_
        self.coef = reg.coef_.item()
        try:
            self.y_train_mean = est.y_train_mean
        except AttributeError:
            pass
 
    def predict(self, x_pred, y_pred=None):
        x_pred = np.array(x_pred)
        pred = self.est.predict(x_pred)
        pred = (pred - self.intercept) / self.coef
        return pred

def ecdf(vals, ax=plt, **kwargs):
    x = np.sort(vals)
    y = np.arange(len(x))/float(len(x))
    x = np.hstack([x,[1]])
    y = np.hstack([y,[1]])
    ax.plot(x, y, **kwargs)
    
#np.random.seed(10)

features = ['rf_1', 'rf_2', 'rf_3', 'false_rf_1', 'false_rf_2']
together = pd.DataFrame(columns=features)
separated = pd.DataFrame(columns=features)
for replic_id in tqdm(range(100)):
    n = 1_000_000

    df = pd.DataFrame(dict(
        vol = np.random.random(n)*60, # between 0 and 60
        false_vol = np.random.random(n)*60, # between 0 and 60
        rf_1 = np.empty(n)+np.nan,
        rf_2 = np.empty(n)+np.nan,
        rf_3 = np.random.choice(2, n),
        false_rf_1 = np.random.choice(2, n),
        false_rf_2 = np.empty(n)+np.nan,
        age = np.random.randint(20, 70, n)
    ))

    df.loc[df.age<25, 'rf_1'] = 0
    df.loc[df.age<25, 'rf_2'] = 1
    df.loc[df.age<25, 'false_rf_2'] = 1

    ind = (32 > df.age) & (df.age >= 25)
    df.loc[ind, 'rf_1'] = np.random.choice(2, sum(ind), p=[.9, .1])
    df.loc[ind, 'rf_2'] = 1-np.random.choice(2, sum(ind), p=[.9, .1])
    df.loc[ind, 'false_rf_2'] = 1-np.random.choice(2, sum(ind), p=[.9, .1])

    ind = (40 > df.age) & (df.age >= 32)
    df.loc[ind, 'rf_1'] = np.random.choice(2, sum(ind), p=[.85, .15])
    df.loc[ind, 'rf_2'] = 1-np.random.choice(2, sum(ind), p=[.85, .15])
    df.loc[ind, 'false_rf_2'] = 1-np.random.choice(2, sum(ind), p=[.85, .15])

    ind = (50 > df.age) & (df.age >= 40)
    df.loc[ind, 'rf_1'] = np.random.choice(2, sum(ind), p=[.65, .35])
    df.loc[ind, 'rf_2'] = 1-np.random.choice(2, sum(ind), p=[.65, .35])
    df.loc[ind, 'false_rf_2'] = 1-np.random.choice(2, sum(ind), p=[.65, .35])

    ind = (60 > df.age) & (df.age >= 50)
    df.loc[ind, 'rf_1'] = np.random.choice(2, sum(ind), p=[.45, .55])
    df.loc[ind, 'rf_2'] = 1-np.random.choice(2, sum(ind), p=[.45, .55])
    df.loc[ind, 'false_rf_2'] = 1-np.random.choice(2, sum(ind), p=[.45, .55])

    ind = df.age>=60
    df.loc[ind, 'rf_1'] = np.random.choice(2, sum(ind), p=[.25, .75])
    df.loc[ind, 'rf_2'] = 1-np.random.choice(2, sum(ind), p=[.25, .75])
    df.loc[ind, 'false_rf_2'] = 1-np.random.choice(2, sum(ind), p=[.25, .75])

    df['vol'] = df['age'] - np.random.randn(n)*.2 - np.dot(df[['rf_1', 'rf_2', 'rf_3']], [-2., -3., -2.5])
    #y[df.false_rf==1] = y[df.false_rf==1] * (0.6 + np.random.random(sum(df.false_rf==1)) * 0.4) # make someone sick look between older

    #feature_to_test = 'false_rf_1'

    healthy_ind = (df.rf_1 == 0) & (df.rf_2 == 0) & (df.rf_3 == 0) & (df.false_rf_1 == 0) & (df.false_rf_2 == 0)
    #healthy_ind = df[feature_to_test] == 0
    
    x_train = df.loc[:, df.columns != 'age'][healthy_ind].iloc[:5000]
    x_test = df.loc[:, df.columns != 'age'].iloc[-5000:]
    #x_test = df.loc[:, df.columns != 'age'][~healthy_ind].iloc[:5000]

    y_train = df[healthy_ind].age.iloc[:5000]
    y_test = df.age.iloc[-5000:]
    #y_test = df[~healthy_ind].age.iloc[:5000]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=None)
    param = {
        'iterations': 100_000,
        'early_stopping_rounds': 50,
        #'task_type': 'GPU',
        'verbose': 0,
    }
    creg = CatBoostRegressor(**param)
    vols = ['vol', 'false_vol']
    creg.fit(x_train[vols], y_train, eval_set=(x_val[vols], y_val))

    creg = BiasCorrectEstimator2(creg, x_test, y_test)

    preds = creg.predict(x_test, y_test)
    target = y_test

    # plot_target_vs_pred(target.to_numpy(), preds.to_numpy())

    #mod = sm.OLS(preds - target, sm.add_constant(x_test))
    #res = mod.fit()
    #print(res.summary())

    x_test_wa = x_test.copy()
    x_test_wa['age'] = target
    x_test_wa['age**2'] = target**2
    x_test_wa['age**3'] = target**3
    mod = sm.OLS(preds - target, sm.add_constant(x_test_wa))
    res = mod.fit()
    #print(res.summary())

    for feature_to_test in features:
        together.loc[replic_id, feature_to_test] = res.pvalues[feature_to_test]

    for feature_to_test in features:
        x_test_wa = x_test[['vol', 'false_vol', feature_to_test]].copy()
        x_test_wa['age'] = target
        x_test_wa['age**2'] = target**2
        x_test_wa['age**3'] = target**3
        mod = sm.OLS(preds - target, sm.add_constant(x_test_wa))
        res = mod.fit()
        separated.loc[replic_id, feature_to_test] = res.pvalues[feature_to_test]

print(together)
print(separated)

plt.subplots(figsize=(10, 5))
plt.subplot(121)
for feature in features:
    ecdf(together[feature], label=feature)
plt.axline([0, 0], [1, 1], linestyle="--", color='black')
plt.legend()


plt.subplot(122)
for feature in features:
    ecdf(separated[feature], label=feature)
plt.axline([0, 0], [1, 1], linestyle="--", color='black')
plt.legend()


plt.show()
    
    
    
    
    
    
