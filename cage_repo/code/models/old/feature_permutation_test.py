import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor

df = '/root/cardiac/Ageing/marco/full_df_with_train_and_test.csv'
df = pd.read_csv(df)
train_cols = [c for c in df.columns if c not in ['train', 'age_at_MRI']]
x_train = df.loc[df.train==1, train_cols].to_numpy()
y_train = df.loc[df.train==1, 'age_at_MRI'].to_numpy()
x_test = df.loc[df.train==0, train_cols].to_numpy()
y_test = df.loc[df.train==0, 'age_at_MRI'].to_numpy()

np.random.seed(1)

scores = []

x_perm_train = x_train.copy()
y_perm_train = y_train.copy()
for i in tqdm(range(1000)):
    res = train_test_split(x_perm_train, y_perm_train, test_size=0.1)
    x_perm_train, x_perm_val, y_perm_train, y_perm_val = res
    creg = CatBoostRegressor(
        iterations=10_000,
        early_stopping_rounds=100
    )
    creg.fit(x_perm_train, y_perm_train,
        eval_set=(x_perm_val, y_perm_val),
        verbose=0
    )
    scores.append(creg.score(x_test, y_test))
    
    idxs = np.random.permutation(range(len(x_train)))
    x_perm_train = x_train[idxs].copy()
    y_perm_train = y_train.copy()
    
    if i > 2:
        pvalue = (np.sum(scores[0] <= np.array(scores[1:])) + 1) / len(scores)
        print(pvalue)
