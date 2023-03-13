import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from matplotlib import pylab as plt
from itertools import product
from tqdm import tqdm

from scipy import stats
import statsmodels.formula.api as smf

from utils import lv_columns, non_lv_columns, t1_columns, BiasCorrectEstimatorCole, BiasCorrectEstimatorBeheshti

import shap
from sdv.tabular import GaussianCopula
#shap.initjs()

def se_calculator(sample):
    return np.std(sample, ddof=1) / (np.sqrt(len(sample)))

model_storage_dir = '/scratch/minacio/cardiac_age_best_models_storage_pkls'

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
    if modeln != 'lv_nlv_t1':
        continue
    preds = raw_res_i['preds']
    target = raw_res_i['test_target']
    mae = np.abs(preds-target)
    mae = [np.mean(mae), se_calculator(mae)]
    mae = [np.round(x, 3) for x in mae]
    mae = f'{mae[0]} ({mae[1]})'
    
    mse = (preds-target)**2
    mse = [np.mean(mse), se_calculator(mse)]
    mse = [np.round(x, 3) for x in mse]
    mse = f'{mse[0]} ({mse[1]})'
    
    rsq = raw_res_i['err']['rsq']
    err = [mae, mse, rsq]
    
    #print(raw_res_i['best_params'])
    human_readable_name = f'{esttype} {modeln.replace("_", " ")}'
    human_readable_name = esttype
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
    f.savefig(f"/root/cardiac/Ageing/marco/{human_readable_name}.pdf", bbox_inches='tight')
    plt.close()
#plt.show()

print(res)
print(res.to_latex())

exit
exit()
ddedde

catb_est_no_t1 = raw_res[('catb', 'lv_nlv')]['best_creg']
catb_est_with_t1 = raw_res[('catb', 'lv_nlv_t1')]['best_creg']
lasso_est_no_t1 = raw_res[('lasso', 'lv_nlv')]['best_creg']
lasso_est_with_t1 = raw_res[('lasso', 'lv_nlv_t1')]['best_creg']

# Feature importance extract
feature_importances = catb_est_with_t1.feature_importances_
feature_names = lv_columns + non_lv_columns + t1_columns
fi_df = pd.DataFrame(dict(feature_names=feature_names, feature_importances=feature_importances))
fi_df.to_csv('/root/cardiac/Ageing/marco/feature_importances.csv', index=False)

# Predictions for unhealthy
data_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(data_path, '../../data')
unhealthy_df = os.path.join(data_path, 'nonhealthyageinggroup_t1.csv')
unhealthy_df = pd.read_csv(unhealthy_df)
unhealthy_df = unhealthy_df[np.logical_not(unhealthy_df[feature_names].isna().any(1))]

unhealthy_covariates_no_t1 = unhealthy_df[lv_columns + non_lv_columns].to_numpy()
unhealthy_covariates_with_t1 = unhealthy_df[feature_names].to_numpy()
target = unhealthy_df['age_at_MRI'].to_numpy()
unhealthy_test_df = unhealthy_df[['eid_40616', 'sex']].copy()

catb_est_no_t1_bc_cole = BiasCorrectEstimatorCole(catb_est_no_t1, unhealthy_covariates_no_t1, target)
catb_est_no_t1_bc_beheshti = BiasCorrectEstimatorBeheshti(catb_est_no_t1, unhealthy_covariates_no_t1, target)
catb_est_with_t1_bc_cole = BiasCorrectEstimatorCole(catb_est_with_t1, unhealthy_covariates_with_t1, target)
catb_est_with_t1_bc_beheshti = BiasCorrectEstimatorBeheshti(catb_est_with_t1, unhealthy_covariates_with_t1, target)

lasso_est_no_t1_bc_cole = BiasCorrectEstimatorCole(lasso_est_no_t1, unhealthy_covariates_no_t1, target)
lasso_est_no_t1_bc_beheshti = BiasCorrectEstimatorBeheshti(lasso_est_no_t1, unhealthy_covariates_no_t1, target)
lasso_est_with_t1_bc_cole = BiasCorrectEstimatorCole(lasso_est_with_t1, unhealthy_covariates_with_t1, target)
lasso_est_with_t1_bc_beheshti = BiasCorrectEstimatorBeheshti(lasso_est_with_t1, unhealthy_covariates_with_t1, target)

unhealthy_test_df['age_at_MRI'] = target
unhealthy_test_df['catb_est_no_t1'] = catb_est_no_t1.predict(unhealthy_covariates_no_t1)
unhealthy_test_df['catb_est_no_t1_bc_cole'] = catb_est_no_t1_bc_cole.predict(unhealthy_covariates_no_t1)
unhealthy_test_df['catb_est_no_t1_bc_beheshti'] = catb_est_no_t1_bc_beheshti.predict(unhealthy_covariates_no_t1, target)
unhealthy_test_df['catb_delta_no_t1'] = unhealthy_test_df['catb_est_no_t1'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['catb_delta_no_t1_bc_cole'] = unhealthy_test_df['catb_est_no_t1_bc_cole'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['catb_delta_no_t1_bc_beheshti'] = unhealthy_test_df['catb_est_no_t1_bc_beheshti'] - unhealthy_test_df['age_at_MRI']

unhealthy_test_df['catb_est_with_t1'] = catb_est_with_t1.predict(unhealthy_covariates_with_t1)
unhealthy_test_df['catb_est_with_t1_bc_cole'] = catb_est_with_t1_bc_cole.predict(unhealthy_covariates_with_t1)
unhealthy_test_df['catb_est_with_t1_bc_beheshti'] = catb_est_with_t1_bc_beheshti.predict(unhealthy_covariates_with_t1, target)
unhealthy_test_df['catb_delta_with_t1'] = unhealthy_test_df['catb_est_with_t1'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['catb_delta_with_t1_bc_cole'] = unhealthy_test_df['catb_est_with_t1_bc_cole'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['catb_delta_with_t1_bc_beheshti'] = unhealthy_test_df['catb_est_with_t1_bc_beheshti'] - unhealthy_test_df['age_at_MRI']

unhealthy_test_df['lasso_est_no_t1'] = lasso_est_no_t1.predict(unhealthy_covariates_no_t1)
unhealthy_test_df['lasso_est_no_t1_bc_cole'] = lasso_est_no_t1_bc_cole.predict(unhealthy_covariates_no_t1)
unhealthy_test_df['lasso_est_no_t1_bc_beheshti'] = lasso_est_no_t1_bc_beheshti.predict(unhealthy_covariates_no_t1, target)
unhealthy_test_df['lasso_delta_no_t1'] = unhealthy_test_df['lasso_est_no_t1'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['lasso_delta_no_t1_bc_cole'] = unhealthy_test_df['lasso_est_no_t1_bc_cole'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['lasso_delta_no_t1_bc_beheshti'] = unhealthy_test_df['lasso_est_no_t1_bc_beheshti'] - unhealthy_test_df['age_at_MRI'] 

unhealthy_test_df['lasso_est_with_t1'] = lasso_est_with_t1.predict(unhealthy_covariates_with_t1)
unhealthy_test_df['lasso_est_with_t1_bc_cole'] = lasso_est_with_t1_bc_cole.predict(unhealthy_covariates_with_t1)
unhealthy_test_df['lasso_est_with_t1_bc_beheshti'] = lasso_est_with_t1_bc_beheshti.predict(unhealthy_covariates_with_t1, target)
unhealthy_test_df['lasso_delta_with_t1'] = unhealthy_test_df['lasso_est_with_t1'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['lasso_delta_with_t1_bc_cole'] = unhealthy_test_df['lasso_est_with_t1_bc_cole'] - unhealthy_test_df['age_at_MRI'] 
unhealthy_test_df['lasso_delta_with_t1_bc_beheshti'] = unhealthy_test_df['lasso_est_with_t1_bc_beheshti'] - unhealthy_test_df['age_at_MRI'] 

unhealthy_test_df
unhealthy_test_df.to_csv('/root/cardiac/Ageing/marco/predictions_unhealthy.csv', index=False)

print('Correlation (and pvalue) between delta and ground truth:', stats.pearsonr(unhealthy_test_df['catb_delta_with_t1_bc_cole'], unhealthy_test_df['age_at_MRI']))
print('Correlation (and pvalue) between delta and ground truth:', stats.pearsonr(unhealthy_test_df['catb_delta_with_t1_bc_cole'], unhealthy_test_df['age_at_MRI']))

# Predictions for healthy test
healthy_covariates_no_t1 = raw_res[('catb', 'lv_nlv')]['test_inputv']
healthy_covariates_with_t1 = raw_res[('catb', 'lv_nlv_t1')]['test_inputv']
healthy_target = raw_res[('catb', 'lv_nlv_t1')]['test_target']
healthy_test_df = pd.DataFrame()
healthy_test_df['eid_18545'] = raw_res[('catb', 'lv_nlv_t1')]['test_ids_cp']

# obtain 'eid_40616' and remove 'eid_18545'
bridge_df = os.path.join(data_path, 'bridge_copy.csv')
bridge_df = pd.read_csv(bridge_df)
healthy_test_df = healthy_test_df.join(bridge_df.set_index('eid_18545'), on='eid_18545')
healthy_test_df = healthy_test_df[['eid_40616']]

# obtain sex
healthy_df = os.path.join(data_path, 'healthyageinggroup_t1.csv')
healthy_df = pd.read_csv(healthy_df)
healthy_test_df = healthy_test_df.join(healthy_df[['eid_40616', 'sex']].set_index('eid_40616'), on='eid_40616')

healthy_test_df['age_at_MRI'] = raw_res[('catb', 'lv_nlv_t1')]['test_target']

healthy_test_df['catb_est_no_t1'] = raw_res[('catb', 'lv_nlv')]['preds']
healthy_test_df['catb_est_no_t1_bc_cole'] = catb_est_no_t1_bc_cole.predict(healthy_covariates_no_t1)
healthy_test_df['catb_est_no_t1_bc_beheshti'] = catb_est_no_t1_bc_beheshti.predict(healthy_covariates_no_t1, healthy_target)
healthy_test_df['catb_delta_no_t1'] = healthy_test_df['catb_est_no_t1'] - healthy_test_df['age_at_MRI']
healthy_test_df['catb_delta_no_t1_bc_cole'] = healthy_test_df['catb_est_no_t1_bc_cole'] - healthy_test_df['age_at_MRI']
healthy_test_df['catb_delta_no_t1_bc_beheshti'] = healthy_test_df['catb_est_no_t1_bc_beheshti'] - healthy_test_df['age_at_MRI']

healthy_test_df['catb_est_with_t1'] = raw_res[('catb', 'lv_nlv_t1')]['preds']
healthy_test_df['catb_est_with_t1_bc_cole'] = catb_est_with_t1_bc_cole.predict(healthy_covariates_with_t1)
healthy_test_df['catb_est_with_t1_bc_beheshti'] = catb_est_with_t1_bc_beheshti.predict(healthy_covariates_with_t1, healthy_target)
healthy_test_df['catb_delta_with_t1'] = healthy_test_df['catb_est_with_t1'] - healthy_test_df['age_at_MRI']
healthy_test_df['catb_delta_with_t1_bc_cole'] = healthy_test_df['catb_est_with_t1_bc_cole'] - healthy_test_df['age_at_MRI']
healthy_test_df['catb_delta_with_t1_bc_beheshti'] = healthy_test_df['catb_est_with_t1_bc_beheshti'] - healthy_test_df['age_at_MRI']

healthy_test_df['lasso_est_no_t1'] = raw_res[('lasso', 'lv_nlv')]['preds']
healthy_test_df['lasso_est_no_t1_bc_cole'] = lasso_est_no_t1_bc_cole.predict(healthy_covariates_no_t1)
healthy_test_df['lasso_est_no_t1_bc_beheshti'] = lasso_est_no_t1_bc_beheshti.predict(healthy_covariates_no_t1, healthy_target)
healthy_test_df['lasso_delta_no_t1'] = healthy_test_df['lasso_est_no_t1'] - healthy_test_df['age_at_MRI']
healthy_test_df['lasso_delta_no_t1_bc_cole'] = healthy_test_df['lasso_est_no_t1_bc_cole'] - healthy_test_df['age_at_MRI']
healthy_test_df['lasso_delta_no_t1_bc_beheshti'] = healthy_test_df['lasso_est_no_t1_bc_beheshti'] - healthy_test_df['age_at_MRI']

healthy_test_df['lasso_est_with_t1'] = raw_res[('lasso', 'lv_nlv_t1')]['preds']
healthy_test_df['lasso_est_with_t1_bc_cole'] = lasso_est_with_t1_bc_cole.predict(healthy_covariates_with_t1)
healthy_test_df['lasso_est_with_t1_bc_beheshti'] = lasso_est_with_t1_bc_beheshti.predict(healthy_covariates_with_t1, healthy_target)
healthy_test_df['lasso_delta_with_t1'] = healthy_test_df['lasso_est_with_t1'] - healthy_test_df['age_at_MRI']
healthy_test_df['lasso_delta_with_t1_bc_cole'] = healthy_test_df['lasso_est_with_t1_bc_cole'] - healthy_test_df['age_at_MRI']
healthy_test_df['lasso_delta_with_t1_bc_beheshti'] = healthy_test_df['lasso_est_with_t1_bc_beheshti'] - healthy_test_df['age_at_MRI']

assert (raw_res[('catb', 'lv_nlv')]['test_target'] == raw_res[('lasso', 'lv_nlv')]['test_target']).all()
assert (raw_res[('catb', 'lv_nlv')]['test_ids_cp'] == raw_res[('lasso', 'lv_nlv')]['test_ids_cp']).all()
healthy_test_df.to_csv('/root/cardiac/Ageing/marco/predictions_healthy_test.csv', index=False)

explainer = shap.TreeExplainer(catb_est_with_t1)
shap_values = explainer.shap_values(healthy_covariates_with_t1)

healthy_covariates_with_t1_df = pd.DataFrame(healthy_covariates_with_t1, columns=feature_names)
shap.summary_plot(shap_values, healthy_covariates_with_t1_df, show=False)
plt.savefig(f"/root/cardiac/Ageing/marco/shap_summary_plot.svg")
plt.close()

full_df = unhealthy_df[feature_names + ['age_at_MRI', 'eid_40616']].copy()
for csv_name, col_name in [
    ['COPYcad_2657_unselected.csv', 'ad'],
    ['COPYcdiabetes_2474_unselected.csv', 'diabetes'],
    ['COPYchf_502_unselected.csv', 'hf'],
    ['COPYcHTN_unselected11049.csv', 'htn'],
    ['COPYchyperchol_7484_unselected.csv', 'hyperchol'],
    ['COPYcObese_7089_unselected.csv', 'obese'],
    ]:
    col_name = 'rf_' + col_name
    full_df[col_name] = 0
    rf_df = os.path.join(data_path, csv_name)
    rf_df = np.array(pd.read_csv(rf_df))
    full_df.loc[[x in rf_df for x in full_df.eid_40616], col_name] = 1
full_df.to_csv('/root/cardiac/Ageing/marco/db_risk_factors.csv', index=False)

del full_df['eid_40616']
start_time = time.time()
model = GaussianCopula()
model.fit(full_df)
synthetic_df = model.sample(len(full_df))
synthetic_df.to_csv('/root/cardiac/Ageing/marco/synthetic_db_with_risk_factors.csv', index=False)
print(f'/synthetic_db_with_risk_factors.csv generated in {time.time()-start_time} seconds')

# all_test_df = pd.concat([unhealthy_test_df, healthy_test_df])

# for csv_name, col_name in [
    # ['COPYcad_2657_unselected.csv', 'ad'],
    # ['COPYcdiabetes_2474_unselected.csv', 'diabetes'],
    # ['COPYchf_502_unselected.csv', 'hf'],
    # ['COPYcHTN_unselected11049.csv', 'htn'],
    # ['COPYchyperchol_7484_unselected.csv', 'hyperchol'],
    # ['COPYcObese_7089_unselected.csv', 'obese'],
    # ]:
    # col_name = 'rf_' + col_name
    # all_test_df[col_name] = 0
    # rf_df = os.path.join(data_path, csv_name)
    # rf_df = np.array(pd.read_csv(rf_df))
    # all_test_df.loc[[x in rf_df for x in all_test_df.eid_40616], col_name] = 1



# reg=smf.ols(formula='catb_delta_with_t1 ~ rf_ad*sex + rf_diabetes*sex + rf_hf*sex + rf_htn*sex + rf_hyperchol*sex + rf_obese*sex + rf_ad*sex + age_at_MRI', data=all_test_df).fit()
# reg.summary()

