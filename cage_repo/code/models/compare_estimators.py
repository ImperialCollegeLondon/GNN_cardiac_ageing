import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import optuna

from matplotlib import pylab as plt
from itertools import product
from tqdm import tqdm

from scipy import stats
import statsmodels.formula.api as smf

from colnames import lv_columns, non_lv_columns, t1_columns

import shap
from sdv.tabular import GaussianCopula
#shap.initjs()

# storage = "sqlite:///"+"/dev/shm/optuna_experiments_cardiac_ageing.db"
# all_studies = optuna.get_all_study_summaries(storage)
# print([x.study_name  for x in all_studies])

home = os.path.expanduser("~")

def se_calculator(sample):
    return np.std(sample, ddof=1) / (np.sqrt(len(sample)))

model_storage_dir = 'cardiac_age_best_models_storage_pkls'

product_list = os.listdir(model_storage_dir)
product_list = [x[:-4].split('_') for x in product_list]
product_list = [(x[0], '_'.join(x[1:])) for x in product_list]
print(product_list)

raw_res = dict()

for esttype, modeln in tqdm(product_list):
    try:
        study_name = f'{esttype}_{modeln}'
        fname = os.path.join(model_storage_dir, study_name) + ".pkl"
        with open(fname, "rb") as f:
            loaded_res = pickle.load(f)
        assert len(loaded_res) == 7
        raw_res[(esttype, modeln)] = loaded_res
    except Exception as e:
        print('Failed to load file ' + fname, end=' ')
        print(e)
        continue

res = pd.DataFrame(columns=['MAE', 'MSE', 'R2'], dtype='O')
for (esttype, modeln), raw_res_i in raw_res.items():
    #if not modeln.startswith('mesh') and not modeln.startswith('lv'):
    #    continue
    preds = raw_res_i['preds']
    target = raw_res_i['test_target']
    mae = np.abs(preds-target)
    raw_res_i['maes'] = mae
    mae = [np.mean(mae), se_calculator(mae)]
    mae = [np.round(x, 3) for x in mae]
    mae = f'{mae[0]} ({mae[1]})'

    mse = (preds-target)**2
    raw_res_i['mses'] = mse
    mse = [np.mean(mse), se_calculator(mse)]
    mse = [np.round(x, 3) for x in mse]
    mse = f'{mse[0]} ({mse[1]})'

    rsq = raw_res_i['err']['rsq']
    err = [mae, mse, rsq]

    #print(raw_res_i['best_params'])
    human_readable_name = f'{esttype} {modeln.replace("_", " ")}'
    #human_readable_name = esttype
    preds, target = preds.reshape(-1), target.reshape(-1)

    res.loc[human_readable_name] = err

    f = plt.figure()
    plt.plot(target, preds, "o", color="black", markerfacecolor='gold', markeredgewidth=0.5)
    reg = stats.linregress(target, preds)
    plt.axline(xy1=(0, reg.intercept), slope=reg.slope, linestyle="-", color="darkgoldenrod")
    plt.axline([0, 0], [1, 1], linestyle="--", color='black')
    plt.xlim([target.min(), target.max()])
    plt.ylim([preds.min(), preds.max()])
    plt.xlabel('Age at MRI')
    plt.ylabel('Cardiac-predicted age (years)')
    f.savefig(home+f"{human_readable_name}.pdf", bbox_inches='tight')
    plt.close()
#plt.show()

print(res)
print(res.to_latex())
