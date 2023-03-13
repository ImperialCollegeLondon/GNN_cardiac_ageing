import tempfile
import os
import pickle
import uuid
import itertools
import hashlib
import shutil
import time
import tempfile

import numpy as np
import scipy.stats as stats
import pandas as pd
import optuna
import sklearn
from catboost import CatBoostRegressor as CatBoostRegressorOriginal
from tqdm import tqdm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader, BatchSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import torch_geometric as tgeo
import torch_geometric.nn as nngeo
from torch_geometric.loader import DataLoader as DataLoaderGeo

from .nn import LitNN
from .gnn import LitNNGeometric

def strace():
    from ipdb import set_trace; set_trace()

def cox_nnl(event_indicators, risks, cph_loss_penalizer=0):
    logcumsumexp_risk = torch.logcumsumexp(risks, dim=0)
    neg_likelihood = risks - logcumsumexp_risk
    neg_likelihood = neg_likelihood * event_indicators
    neg_likelihood = - torch.sum(neg_likelihood)
    
    # see section 3.3 https://jmlr.org/papers/volume20/18-424/18-424.pdf
    if cph_loss_penalizer:
        cumsum_risk = torch.abs(torch.cumsum(risks, dim=0))
        penalization = torch.sum(cumsum_risk * event_indicators) * cph_loss_penalizer
        neg_likelihood = neg_likelihood + penalization

    return neg_likelihood


class CatBoostRegressor(CatBoostRegressorOriginal):
    def __getstate__(self):
        dict_ = super().__getstate__()
        if hasattr(self, 'y_train_mean'):
            dict_['y_train_mean'] = self.y_train_mean
        return dict_
    def __setstate__(self, dict_):
        super().__setstate__(dict_)
        self.y_train_mean = dict_['y_train_mean']

def mtest(est, x_test, y_test):
    try:
        output = est.predict(x_test).reshape(-1)
    except TypeError:
        output = est.predict(x_test, y_test).reshape(-1)        
    target = y_test.reshape(-1)

    diff = output - target
    se = diff**2
    ss_res = se.sum()
    ss_total = ((target - est.y_train_mean)**2).sum()

    rsq = 1 - ss_res / ss_total
    rmse = np.sqrt(se.mean())
    mae = np.abs(diff).mean()

    return dict(mae=mae, rmse=rmse, rsq=rsq)

class BiasCorrectEstimatorCole:
    def __init__(self, prediction, ground_truth):
        prediction = np.array(prediction).reshape(-1)
        ground_truth = np.array(ground_truth).reshape((-1, 1))
        
        reg = LinearRegression()
        reg.fit(ground_truth, prediction)
        self.intercept = reg.intercept_
        self.coef = reg.coef_.item()
 
    def correct_prediction(self, prediction):
        prediction = np.array(prediction).reshape(-1)
        return (prediction - self.intercept) / self.coef


def train_with_hyp_search(x_trial, y_trial, optuna_sql_path,
    esttype='nn',
    number_of_trials_to_run=100, max_number_of_sucessful_trials=100,
    study_name=None, optuna_storage_dir=None, gnn_decimate_level=0.9,
    **kwargs):
    #x_trial, y_trial = x_trial[:101], y_trial[:101] # fast test for debugging

    x_trial_train, x_trial_test, y_trial_train, y_trial_test = train_test_split(x_trial, y_trial, test_size=0.1, random_state=0)
    if esttype == 'catb':
        x_trial_val = x_trial_test
        y_trial_val = y_trial_test
        number_of_trials_to_run = 1
        max_number_of_sucessful_trials = 1
    else:
        x_trial_train, x_trial_val, y_trial_train, y_trial_val = train_test_split(x_trial_train, y_trial_train, test_size=0.1, random_state=1)

    def objective(trial):
        successful_trials_so_far = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        if successful_trials_so_far >= max_number_of_sucessful_trials:
            study.stop()
            print('Maximum number of trials reached, prunning')
            raise optuna.TrialPruned()
        print(f"Running trial {trial.number}")

        #bias_correct = trial.suggest_int("bias_correct", 0, 1)
        #bias_correct = 1
        #if bias_correct:
        #    x_trial_train_bc, x_trial_sval_bc, y_trial_train_bc, y_trial_sval_bc = train_test_split(x_trial_train, y_trial_train, test_size=0.2, random_state=2)
            #x_trial_train_bc, x_trial_sval_bc, y_trial_train_bc, y_trial_sval_bc = x_trial_train, x_trial_train, y_trial_train, y_trial_train
        # x_trial_train_bc, y_trial_train_bc = x_trial_train, y_trial_train

        if esttype == 'nn':
            hsize1 = trial.suggest_int("hsize1", 100, 2000)
            hsize2 = trial.suggest_int("hsize2", 100, 2000)
            lr = 10**trial.suggest_float("log_lr", -5, 1)
            batch_size = trial.suggest_int("batch_size", 50, 110)
            weight_decay = 10.**trial.suggest_float("weight_decay_exp", -10, -1)

            creg = LitNN(
                       hsizes = [hsize1, hsize2],
                       lr=lr,
                       weight_decay=weight_decay, batch_size=batch_size)
            creg.mtrain(x_trial_train, y_trial_train, x_trial_val, y_trial_val)

# 'conv2_input_size1': 204, 'conv2_input_size2': 234, 'decimate_level': 0.95,
# 'log_lr': -3.5672524005796458, 'transforms_names': 'position_fpfh',
# 'weight_decay_exp': -9.1647873024483} done

        elif esttype == 'gnn':
            params = dict(
                batch_size = trial.suggest_int("batch_size", 10, 20),
                weight_decay = 10.**trial.suggest_float("weight_decay_exp", -15, -5),
                conv2_input_size1 = trial.suggest_int("conv2_input_size1", 10, 50),
                #conv2_input_size2 = trial.suggest_int("conv2_input_size2", 200, 250),
                ll_input_multp_size1 = trial.suggest_int("ll_input_multp_size1", 10, 50),
                ll_input_size2 = trial.suggest_int("ll_input_size2", 100, 1000),
                #transforms_names = trial.suggest_categorical("transforms_names", ['constant', 'position', 'fpfh', 'position_fpfh']),
                transforms_names = trial.suggest_categorical("transforms_names", ['position_fpfh']),
                #decimate_level = trial.suggest_categorical("decimate_level", [0.85, 0.90]),
                decimate_level = gnn_decimate_level,
                heads = 1,#trial.suggest_int("heads", 3, 6),
                dropout = trial.suggest_float("dropout", 0., 0.3),
                lr = 10**trial.suggest_float("log_lr", -5, 1),
            )
            print(params)
            creg = LitNNGeometric(**params, **kwargs)
            creg.mtrain(x_trial_train, y_trial_train)

        elif esttype == 'catb':
            params = {
                'iterations': 100_000,
                'early_stopping_rounds': 100,
                #'task_type': 'GPU',
                'verbose': 1000,
            }
            creg = CatBoostRegressor(**params, **kwargs)
            creg.fit(x_trial_train, y_trial_train,
                eval_set=(x_trial_val, y_trial_val)
            )

        elif esttype == 'lasso':
            alpha = trial.suggest_float("alpha", 0, 10)

            creg = sklearn.linear_model.Lasso(alpha=alpha, **kwargs)
            creg.fit(x_trial_train, y_trial_train)

        elif esttype == 'ridge':
            alpha = trial.suggest_float("alpha", 0, 10)

            creg = sklearn.linear_model.Ridge(alpha=alpha, **kwargs)
            creg.fit(x_trial_train, y_trial_train)

        elif esttype == 'enet':
            alpha = trial.suggest_float("alpha", 0, 10)
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1)

            creg = sklearn.linear_model.ElasticNet(alpha=alpha,
                l1_ratio=l1_ratio, **kwargs)
            creg.fit(x_trial_train, y_trial_train)

        elif esttype == 'rf':
            n_estimators = trial.suggest_int("alpha", 100, 1000)

            creg = sklearn.ensemble.RandomForestRegressor(
                n_estimators=n_estimators, **kwargs)
            creg.fit(x_trial_train, y_trial_train)

        else:
            raise ValueError('Invalid estimator type')
        
        creg.y_train_mean = y_trial_train.mean()

        #if bias_correct:
        #    creg = BiasCorrectEstimator(creg, x_trial_sval_bc, y_trial_sval_bc)

        err = mtest(creg, x_trial_test, y_trial_test)
        trial.set_user_attr("err", err)
        print('error on hyp search validation:', err)
        err = err['mae']

        best_estimator_performance = np.inf
        try:
            best_estimator_performance = study.best_trial.values[0]
        except Exception:
            pass

        if best_estimator_performance >= err:
            print('Best estimator so far. Saving estimator, started.')
            with open(f"{os.path.join(optuna_storage_dir, str(trial.number))}.pkl", "wb") as f:
                pickle.dump(creg, f)
            print('Saving estimator, done.')
        else:
            print('Not the best estimator so far.')

        print(f"Finished trial {trial.number}")
        return err

    if study_name is None:
        study_name = 'model_' + str(uuid.uuid1())
    curdir = os.path.dirname(os.path.realpath(__file__))
    if optuna_storage_dir is None:
        optuna_storage_dir = os.path.join(curdir, 'cache', 'optuna_pkls')
    optuna_storage_dir = os.path.join(optuna_storage_dir, study_name)
    os.makedirs(optuna_storage_dir, exist_ok=True)

    for exception_counter in itertools.count(1,1):
        try:
            optuna_storage = optuna.storages.RDBStorage(url="sqlite:///"+optuna_sql_path, engine_kwargs={"connect_args": {"timeout": 600}})
            study = optuna.create_study(storage=optuna_storage,
                study_name=study_name, direction="minimize",
                load_if_exists=True)
            study.optimize(objective, n_trials=number_of_trials_to_run)
            break
        except optuna.exceptions.StorageInternalError as e:
            if exception_counter >= 10:
                raise e
            time.sleep(5)


    best_creg = None
    #print("Number of finished trials: {}".format(len(study.trials)))
    print("Loading best trial:", study.best_params, 'started')
    with open(f"{os.path.join(optuna_storage_dir, str(study.best_trial.number))}.pkl", "rb") as f:
        best_creg = pickle.load(f)
    print("Loading best trial:", study.best_params, 'done')

    study_completed = len([t for t in study.trials if t.state.name == 'COMPLETE']) >= max_number_of_sucessful_trials
    return best_creg, study.best_params, study_completed, study
