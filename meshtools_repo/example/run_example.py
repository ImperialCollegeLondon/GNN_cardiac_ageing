import tempfile
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import scipy.stats as stats
import pandas as pd
from sdv.tabular import GaussianCopula

import torch

import optuna

from utils import LitNN, DataModule, non_lv_columns, lv_columns, t1_columns, train_with_hyp_search, mtest

nn_mesh_path = '/scratch/minacio/cardiac_age_mesh_97.0.pkl'
gnn_searchdirs = ''

for instance_idx in tqdm(range(len(dfx_mesh))):
    vols = np.empty(50)
    for frame_idx in range(50):
        #if (dfx_mesh_t[instance_idx, frame_idx]==0).all():
        #    vols[frame_idx] = np.nan
        #    continue
        vols[frame_idx] = ConvexHull(dfx_mesh[idx_part][instance_idx, frame_idx]).volume
    new_dfx_cp[instance_idx, idx_part*4:idx_part*4+4] = min(vols), max(vols), np.argmin(vols), np.argmax(vols)


best_creg, best_params, study_completed, study = train_with_hyp_search(
        train_inputv_or_ids, train_target,
        optuna_sql_path=optuna_sql_path,
        number_of_trials_to_run=n_trials,
        max_number_of_sucessful_trials=n_trials,
        study_name=study_name, esttype=esttype,
        optuna_storage_dir=optuna_storage_dir,
        searchdirs = [
            '/scratch/minacio/anonymized_data/',
        ],
        items_cache_dir = home + '/scratch/minacio/anonymized_cache_geometric_nn_cardiac_age',
        vtk_cache_dir =  '/scratch/minacio/anonymized_cache_vtks_cardiac_age',
        gnn_decimate_level=0, # data is already pre-decimated, so no need to do it again
)

preds = best_creg.predict(test_inputv_or_ids).reshape(-1)
err = mtest(best_creg, test_inputv_or_ids, test_target)
print(err)
