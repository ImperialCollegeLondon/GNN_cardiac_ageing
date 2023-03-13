import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pylab as plt
from tqdm import tqdm

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter

from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit, ShuffleSplit
from catboost import CatBoostRegressor

def ci_calculator(sample, alpha=0.05):
    mean = np.mean(sample)
    std = np.std(sample, ddof=1) / (np.sqrt(len(sample)))
    dist = stats.norm(mean, std)
    p = alpha/2
    return dist.ppf(p), dist.ppf(1-p)

delta_column = 'catb_delta_with_t1_bc_cole'

#df = pd.read_csv('/root/cardiac/Ageing/data/y_Mit_datasets/39kMACEfromUKB_filter_ageing_withdates_FULL.csv')
#df = pd.read_csv('/root/cardiac/Ageing/data/y_Mit_datasets/39kMACE_cvmortalityMACE.csv')
#df = pd.read_csv('/root/cardiac/Ageing/data/y_Mit_datasets/39kMACE_cvdeath.csv')
df = pd.read_csv('/root/cardiac/Ageing/marco/splitframe.csv')
unhealthy_df = pd.read_csv('/root/cardiac/Ageing/marco/predictions_unhealthy.csv')
unhealthy_df.drop_duplicates(subset='eid_40616', inplace=True)
dfj = df.join(unhealthy_df.set_index('eid_40616')[delta_column], on='eid_40616')

dfj = dfj[~ dfj[delta_column].isna()]
dfj = dfj[['diff_in_days', 'composite_event', delta_column, 'quartile']]
dfj.columns = ['time', 'event', 'delta', 'quartile']
dfj['event'] = dfj['event'] - 1
dfj['percentile'] = np.vectorize(lambda x: stats.percentileofscore(dfj.delta.to_numpy().reshape(-1), x))(dfj.delta.to_numpy())

# kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# scores = []
# for train_index, val_index in kf.split(dfj, dfj['event']):
    # cph = CoxPHFitter()
    # cph.fit(dfj.iloc[train_index], duration_col='time', event_col='event')
    # scores.append(cph.score(dfj.iloc[val_index], 'concordance_index'))
# score = np.mean(scores)
# print('3 fold CV concordance index:', score)

cph = CoxPHFitter()
cph.fit(dfj[['time', 'event', 'quartile']][(dfj.quartile==1)|(dfj.quartile==4)], duration_col='time', event_col='event')
print(cph.hazard_ratios_)
print(cph.log_likelihood_ratio_test())


cph = CoxPHFitter()
cph.fit(dfj[['time', 'event', 'percentile', 'delta']], duration_col='time', event_col='event')
print(cph.hazard_ratios_)
print(cph.log_likelihood_ratio_test())

fig = plt.figure()
ax = plt.subplot(111)

for i in range(1, 5):
    f1 = KaplanMeierFitter()
    f1.fit(dfj[dfj.quartile==i]['time'], dfj[dfj.quartile==i]['event'], label=i)
    f1.plot(ax=ax, show_censors=True, ci_show=False, censor_styles={'ms': 6})

fig.savefig(f"/root/cardiac/Ageing/marco/survival_test.pdf", dpi="figure")

x_full = dfj[['delta']].to_numpy()
y_full = dfj[['event', 'time']].to_numpy()
x_full = x_full[y_full[:,0]==1]
y_full = y_full[y_full[:,0]==1,1]
param = {
    #'loss_function': 'Cox',
    #'eval_metric': 'Cox',
    'iterations': 10_000,
    #'early_stopping_rounds': 100,
}
#kf = StratifiedKFold(n_splits=30, shuffle=True, random_state=1)
kf = ShuffleSplit(n_splits=100, test_size=0.4, random_state=1)
conc_inds = []
for train_index, test_index in tqdm(list(kf.split(x_full))):    
    x_trainval = x_full[train_index]
    y_trainval = y_full[train_index]
    x_test = x_full[test_index]
    y_test = y_full[test_index]
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.33, random_state=2,
        shuffle=True)
    x_train, y_train = x_trainval, y_trainval
    creg = CatBoostRegressor(**param)
    creg.fit(x_train, y_train, verbose=0)
    cv_preds = creg.predict(x_test)
    conc_ind = concordance_index(y_test, -cv_preds)
    conc_inds.append(conc_ind)
    print(ci_calculator(conc_inds))
conc_inds

# x_full = dfj[['delta']].to_numpy()
# y_full = dfj[['event', 'time']].to_numpy()
# x_full = x_full[y_full[:,0]==1]
# y_full = y_full[y_full[:,0]==1]
# param = {
    # 'loss_function': 'Cox',
    # 'eval_metric': 'Cox',
    # 'iterations': 10_000,
    # #'early_stopping_rounds': 100,
    # #'learning_rate': 0.5,
    # #'l2_leaf_reg': 64,
    # #'bootstrap_type': 'No',
# }
# kf = StratifiedShuffleSplit(n_splits=30, test_size=0.4, random_state=1)
# #kf = StratifiedKFold(n_splits=30, shuffle=True, random_state=1)
# conc_inds = []
# for train_index, test_index in tqdm(list(kf.split(x_full, y_full[:,0]))):    
    # x_trainval = x_full[train_index]
    # y_trainval = y_full[train_index]
    # x_test = x_full[test_index]
    # y_test = y_full[test_index]
    
    # x_train, x_val, y_train, y_val = train_test_split(
        # x_trainval, y_trainval, test_size=0.33, random_state=2,
        # stratify=y_trainval[:,0], shuffle=True)
    # x_train, y_train = x_trainval, y_trainval
    # creg = CatBoostRegressor(**param)
    
    # catb_y_tf = lambda y: np.where(y[:, 0] == 1, y[:, 1], - y[:, 1])
    # creg.fit(x_train, catb_y_tf(y_train),
        # #eval_set=(x_val, catb_y_tf(y_val)),
        # verbose=0)
    # cv_preds = creg.predict(x_test)
    # conc_ind = concordance_index(y_test[:,1], -cv_preds, y_test[:,0])
    
    # conc_inds.append(conc_ind)
    # print(ci_calculator(conc_inds))
# conc_inds



