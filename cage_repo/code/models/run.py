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

from colnames import non_lv_columns, lv_columns, t1_columns

from meshtools import LitNN, DataModule, train_with_hyp_search, mtest

optuna_storage_dir = '/scratch/minacio/cardiac_age_optuna_storage_pkls'
optuna_sql_path = '/dev/shm/optuna_experiments_cardiac_ageing.db'
model_storage_dir = '/scratch/minacio/cardiac_age_best_models_storage_pkls'
home = os.path.expanduser("~")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--modeln", dest="modeln", type=str, required=True,
        help="lv_nlv_t1 or lv_nlv or lv or mesh"
    )
    parser.add_argument(
        "--esttype", dest="esttype", type=str, required=True,
        help='gnn or nn or catb'
    )
    parser.add_argument(
        "--n_trials", dest="n_trials", type=int, default=30,
        help='number of optuna trials'
    )
    parser.add_argument(
        "--gnn_decimate_level", dest="gnn_decimate_level", type=float, default=.9,
        help='GNN decimate level'
    )
    parser.add_argument(
        "--subsample_factor", dest="subsample_factor", type=float, default=1.,
        help='Dataset subsample factor'
    )
    args = parser.parse_args()
    modeln = args.modeln
    esttype = args.esttype
    n_trials = args.n_trials
    subsample_factor = args.subsample_factor
    gnn_decimate_level = args.gnn_decimate_level

    #torch.manual_seed(1)

    curdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(curdir, '../../data')
    #mesh_path = os.path.join(datadir, 'downsampled_descriptor_98.9.pkl')
    #if esttype == 'gnn':
    #mesh_path = os.path.join(datadir, 'downsampled_descriptor_97.0.pkl')
    mesh_path = '/scratch/minacio/cardiac_age_mesh_97.0.pkl'
    mesh_path = '/scratch/minacio/cardiac_age_mesh_90.0.pkl'
    mesh_path = '/scratch/minacio/cardiac_age_mesh_98.9.pkl'
    data_path = os.path.join(datadir, 'healthyageinggroup_t1.csv')

    # obtain mesh data and ids
    with open(mesh_path, 'rb') as f:
        inputv_mesh_rel, inputv_mesh_abs, ids_mesh = pickle.load(f)
    if esttype == 'gnn':
        inputv_mesh = inputv_mesh_abs
    else:
        inputv_mesh = inputv_mesh_abs
        #inputv_mesh[:,1:] = inputv_mesh_rel
        inputv_mesh = inputv_mesh.reshape(inputv_mesh.shape[0], -1)
    del inputv_mesh_rel, inputv_mesh_abs
    inputv_mesh = np.array(inputv_mesh, dtype=float)
    ids_mesh = np.array(ids_mesh, dtype=int)

    inputv_mesh = inputv_mesh

    # obtain cp features + id + targets
    df_cp_all = pd.read_csv(data_path)
    df_cp_all.drop_duplicates('eid_18545', inplace=True, ignore_index=True)
    df_cp_test = df_cp_all[df_cp_all['index']==2].copy()
    df_cp_train = df_cp_all[df_cp_all['index']==1].copy()

    # obtain cp features
    train_inputv_nlv = df_cp_train[non_lv_columns].to_numpy()
    test_inputv_nlv = df_cp_test[non_lv_columns].to_numpy()
    train_inputv_lv = df_cp_train[lv_columns].to_numpy()
    test_inputv_lv = df_cp_test[lv_columns].to_numpy()
    train_inputv_t1 = df_cp_train[t1_columns].to_numpy()
    test_inputv_t1 = df_cp_test[t1_columns].to_numpy()

    # obtain cp ids
    train_ids_cp = df_cp_train['eid_18545']
    test_ids_cp = df_cp_test['eid_18545']
    train_ids_cp = np.array(train_ids_cp, dtype=int)
    test_ids_cp = np.array(test_ids_cp, dtype=int)

    # obtain targets
    train_target = df_cp_train['age_at_MRI']
    test_target = df_cp_test['age_at_MRI']
    train_target = np.array(train_target, dtype=int)
    test_target = np.array(test_target, dtype=int)

    # Train/test split for mesh data
    train_inputv_mesh = inputv_mesh[[np.where(id_==ids_mesh)[0].item() for id_ in train_ids_cp if id_ in ids_mesh]]
    test_inputv_mesh = inputv_mesh[[np.where(id_==ids_mesh)[0].item() for id_ in test_ids_cp if id_ in ids_mesh]]

    # filtering to remove data for which there is no mesh data
    if True:
        idx = np.in1d(train_ids_cp, ids_mesh)
        train_target = train_target[idx]
        train_inputv_nlv = train_inputv_nlv[idx]
        train_inputv_lv = train_inputv_lv[idx]
        train_inputv_t1 = train_inputv_t1[idx]
        train_ids_cp = train_ids_cp[idx]

        idx = np.in1d(test_ids_cp, ids_mesh)
        test_target = test_target[idx]
        test_inputv_nlv = test_inputv_nlv[idx]
        test_inputv_lv = test_inputv_lv[idx]
        test_inputv_t1 = test_inputv_t1[idx]
        test_ids_cp = test_ids_cp[idx]

        assert len(train_inputv_mesh) == len(train_target) == len(train_inputv_nlv) == len(train_inputv_lv) == len(train_inputv_t1) == len(train_ids_cp)
        assert len(test_inputv_mesh) == len(test_target) == len(test_inputv_nlv) == len(test_inputv_lv) == len(test_inputv_t1) == len(test_ids_cp)

    if modeln == 'lv_nlv_t1':
        train_inputv = np.hstack((train_inputv_lv, train_inputv_nlv, train_inputv_t1))
        test_inputv = np.hstack((test_inputv_lv, test_inputv_nlv, test_inputv_t1))

    elif modeln == 'lv_nlv':
        train_inputv = np.hstack((train_inputv_lv, train_inputv_nlv))
        test_inputv = np.hstack((test_inputv_lv, test_inputv_nlv))

    elif modeln == 'lv':
        train_inputv = train_inputv_lv
        test_inputv = test_inputv_lv

    elif modeln == 'mesh':
        train_inputv = train_inputv_mesh
        test_inputv = test_inputv_mesh

    # elif modeln == 'nlv_mesh':
    #     train_inputv = np.hstack((train_inputv_mesh, train_inputv_nlv))
    #     test_inputv = np.hstack((test_inputv_mesh, test_inputv_nlv))

    # elif modeln == 'lv_nlv_t1_mesh':
    #     train_inputv = np.hstack((train_inputv_mesh, train_inputv_lv, train_inputv_nlv, train_inputv_t1))
    #     test_inputv = np.hstack((test_inputv_mesh, test_inputv_lv, test_inputv_nlv, test_inputv_t1))

    else:
        raise ValueError('Invalid model type')

    # remove nas
    idx = np.logical_not(np.isnan(train_inputv.reshape(train_inputv.shape[0],-1)).sum(1))
    train_ids_cp = train_ids_cp[idx]
    train_target = train_target[idx]
    train_inputv = train_inputv[idx]

    idx = np.logical_not(np.isnan(test_inputv.reshape(test_inputv.shape[0],-1)).sum(1))
    test_ids_cp = test_ids_cp[idx]
    test_target = test_target[idx]
    test_inputv = test_inputv[idx]

    assert not np.isnan(train_inputv).any()
    assert not np.isnan(train_target).any()
    assert not np.isnan(test_inputv).any()
    assert not np.isnan(test_target).any()

    # Subset train and test
    # train_inputv = train_inputv[:300]
    # train_target = train_target[:300]
    # train_ids_cp = train_ids_cp[:300]
    # idx = np.arange(300)
    # np.random.shuffle(idx)
    # test_inputv = test_inputv[:300]
    # test_target = test_target[:300]
    # test_ids_cp = test_ids_cp[:300]

    # Shuffles test order
    # idx = np.arange(len(test_inputv))
    # np.random.shuffle(idx)
    # test_inputv = test_inputv[idx]
    # test_target = test_target[idx]
    # test_ids_cp = test_ids_cp[idx]

    print(train_inputv.shape, 'train shape')
    print(test_inputv.shape, 'test shape')

    # Export ids
    db_export_path = home + '/cardiac/Ageing/gnn_paper/datasets/healthy_train_ids.csv'
    ids_to_export = df_cp_all[np.isin(df_cp_all.eid_18545, train_ids_cp)]
    ids_to_export = ids_to_export[['eid_40616', 'eid_18545']]
    ids_to_export.to_csv(db_export_path)
    print(ids_to_export.shape[0], 'train length')

    db_export_path = home + '/cardiac/Ageing/gnn_paper/datasets/healthy_test_ids.csv'
    ids_to_export = df_cp_all[np.isin(df_cp_all.eid_18545, test_ids_cp)]
    ids_to_export = ids_to_export[['eid_40616', 'eid_18545']]
    ids_to_export.to_csv(db_export_path)
    print(ids_to_export.shape[0], 'test length')

    if subsample_factor != 1:
        idx = np.arange(train_inputv.shape[0])
        np.random.default_rng(0).shuffle(idx)
        idx = idx[:round(train_inputv.shape[0]*subsample_factor)]
        train_inputv, train_target, train_ids_cp = train_inputv[idx], train_target[idx], train_ids_cp[idx]


    #train_inputv, train_target, train_ids_cp = train_inputv[:120], train_target[:120], train_ids_cp[:120]
    #if input('confirm delete by typing y\n').lower() in ['y', 'yes']:
    #    optuna.delete_study(study_name=f'{esttype}_{modeln}',storage="sqlite:///"+"/dev/shm/optuna_experiments_cardiac_ageing.db")
    #    raise

    if False and esttype == 'catb' and modeln == 'lv_nlv_t1':
        # Export a dataset of the covariates on train set
        train_inputv_df = pd.DataFrame(train_inputv, columns = lv_columns + non_lv_columns + t1_columns)
        train_inputv_df.to_csv(home+'/cardiac/Ageing/marco/original_db.csv', index=False)
        print('original_db.csv generated')

        # Export the filtered dataset with response variable, covariates, train/test indicator and ids
        full_df = pd.DataFrame(np.vstack((train_inputv, test_inputv)),
            columns = lv_columns + non_lv_columns + t1_columns)
        full_df['age_at_MRI'] = np.hstack((train_target, test_target))
        full_df_with_train_and_test = full_df.copy()
        full_df_with_train_and_test['eid_18545'] = np.hstack((train_ids_cp, test_ids_cp))
        full_df_with_train_and_test = full_df_with_train_and_test.join(df_cp_all.set_index('eid_18545')[['eid_40616']], on='eid_18545')
        full_df_with_train_and_test['train'] = 0
        full_df_with_train_and_test.iloc[:len(train_inputv), -1] = 1
        full_df_with_train_and_test.to_csv(home+'/cardiac/Ageing/marco/full_df_with_train_and_test.csv', index=False)

        # Create and export a synthetic dataset
        model = GaussianCopula()
        model.fit(full_df)
        synthetic_df = model.sample(len(full_df))
        synthetic_df.to_csv('/root/cardiac/Ageing/marco/synthetic_db_healthy.csv', index=False)
        print('/synthetic_db_healthy.csv generated')

    study_completed = False
    study_name = f'{esttype}_{modeln}'
    if esttype == 'gnn':
        study_name += f'_{gnn_decimate_level}'
    if subsample_factor != 1:
        study_name += f'_subsample_factor_{subsample_factor}'
    while not study_completed:
        if esttype == 'gnn':
            train_inputv_or_ids = train_ids_cp
            test_inputv_or_ids = test_ids_cp
        else:
            train_inputv_or_ids = train_inputv
            test_inputv_or_ids = test_inputv

        kwargs = dict(
                x_trial=train_inputv_or_ids, y_trial=train_target,
                optuna_sql_path=optuna_sql_path,
                number_of_trials_to_run=n_trials,
                max_number_of_sucessful_trials=n_trials,
                study_name=study_name, esttype=esttype,
                optuna_storage_dir=optuna_storage_dir,
        )
        if esttype == 'gnn':
            kwargs.update(dict(
                    searchdirs = [
                        home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022',
                        home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_02_2022',
                    ],
                    items_cache_dir = '/scratch/minacio/cache_items_cardiac_age',
                    vtk_cache_dir =   '/scratch/minacio/cache_vtks_cardiac_age',
                    gnn_decimate_level = gnn_decimate_level,
            ))
        best_creg, best_params, study_completed, study = train_with_hyp_search(**kwargs)

        preds = best_creg.predict(test_inputv_or_ids).reshape(-1)

        err = mtest(best_creg, test_inputv_or_ids, test_target)

        print(err)

    os.makedirs(model_storage_dir, exist_ok=True)
    with open(f'{model_storage_dir}/{study_name}.pkl', 'wb') as f:
        pickle.dump(dict(best_creg=best_creg, best_params=best_params, preds=preds, test_target=test_target, test_ids_cp=test_ids_cp, test_inputv=test_inputv, err=err), f)
