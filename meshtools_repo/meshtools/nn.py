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
from torch.utils.data import random_split, TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import torch_geometric as tgeo
import torch_geometric.nn as nngeo
from torch_geometric.loader import DataLoader as DataLoaderGeo

dl_num_workers = 0

class LitNN(pl.LightningModule):
    def __init__(self, hsizes = [50, 10],
                 lr=0.01, weight_decay=0, batch_size=50, dropout=0.5,
                 es_patience=20, num_workers=dl_num_workers):
        super().__init__()

        self.hsizes = hsizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.es_patience = es_patience
        self.num_workers = num_workers
                
        self.high_loss_count = 0
        
    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


    def _initialize_layer(self, layer):
        nn.init.constant_(layer.bias, 0)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(layer.weight, gain=gain)
        return layer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self, batch, batch_idx):
        return self.forward(batch[0]).squeeze()

    def training_step(self, train_batch, batch_idx):
        inputv, target = train_batch
        output = self.forward(inputv)
        mse = F.mse_loss(output, target)
        self.log('train_rmse', torch.sqrt(mse).item(), prog_bar=True, batch_size=target.shape[0])
        
        if self.current_epoch >= 0:
            if np.sqrt(mse.item()) > 20:
                self.high_loss_count += 1
            else:
                self.high_loss_count = 0
                
            if self.high_loss_count > 10:
                raise optuna.TrialPruned('Trial pruning due to high loss')
                
        return mse

    def test_validation_step(self, batch, batch_idx, name):
        inputv, target = batch
        output = self.forward(inputv)
        loss_mse = F.mse_loss(output, target).item()
        loss_mae = F.l1_loss(output, target).item()
        self.log(f'{name}_loss_rmse', np.sqrt(loss_mse))
        self.log(f'{name}_loss_mae', loss_mae)

    def validation_step(self, val_batch, batch_idx):
        self.test_validation_step(val_batch, batch_idx, 'val')

    def test_step(self, test_batch, batch_idx):
        self.test_validation_step(test_batch, batch_idx, 'test')

    def mtrain(self, x_train, y_train, x_val=None, y_val=None):
        input_size = x_train.shape[1]
        out_size = 1
        self.y_train_mean = y_train.mean()
        if len(y_train.shape) == 1:
            y_train = y_train[:, None]
        if y_val is not None and len(y_val.shape) == 1:
            y_val = y_val[:, None]

        train_size = 2*len(y_train)//3
        if train_size < self.batch_size:
            self.batch_size = train_size

        modules_list = [nn.BatchNorm1d(input_size)]
        for hsize in self.hsizes:
            modules_list.extend([
                nn.Linear(input_size, hsize),
                nn.ELU(),
                nn.BatchNorm1d(hsize),
                nn.Dropout(self.dropout),
            ])
            input_size = hsize

        out_size = 1
        modules_list.append(nn.Linear(input_size, out_size))
        self.modules_list = nn.ModuleList(modules_list)

        datamodule = DataModule(x_train, y_train, x_val, y_val, batch_size=self.batch_size, num_workers=self.num_workers)
        self.batch_size = datamodule.batch_size

        trainer = self.get_trainer()

        # find "best" lr
        # trainer.tune(self, datamodule = datamodule)

        trainer.fit(self, datamodule = datamodule)

        #self.trainer = None

        return self

    def get_trainer(self):
        callbacks = [EarlyStopping(
           monitor='val_loss_rmse',
           min_delta=0.00,
           patience=self.es_patience,
           verbose=False,
           mode='min'
        )]
        #if trial is not None:
        #    callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss_rmse"))

        try:
            from pytorch_lightning.loggers import MLFlowLogger
            logger = MLFlowLogger(
                experiment_name="nn",
                tracking_uri="file:./mlruns"
            )
        except ImportError:
            logger = True

        return pl.Trainer(
                             precision=32,
                             gpus=int(torch.cuda.is_available()),
                             tpu_cores=None,
                             logger=logger,
                             log_every_n_steps=self.batch_size,
                             val_check_interval=0.25, # do validation check 4 times for each epoch
                             #auto_lr_find=True,
                             callbacks=callbacks,
                             max_epochs = 1000,
                             num_sanity_val_steps=0,
                            )

    def mtest(self, x_test, y_test):
        if len(y_test.shape) == 1:
            y_test = y_test[:, None]
        y_pred = self.predict(x_test)
        diff = y_pred - y_test
        se = diff**2
        ss_res = se.sum()
        ss_total = ((y_test - self.y_train_mean)**2).sum()
        rsq = 1 - ss_res / ss_total
        rmse = np.sqrt(se.mean())
        mae = np.abs(diff).mean()
        return dict(mae=mae, rmse=rmse, rsq=rsq)

    def predict(self, x_pred):
        #trainer = self.get_trainer()
        x_pred = torch.as_tensor(x_pred, dtype=torch.float32)
        dataset = TensorDataset(x_pred)
        dataloader = DataLoader(dataset, batch_size=min(100, len(dataset)), num_workers=self.num_workers)
        try:
            preds = self.trainer.predict(self, dataloader)
        except Exception:
            with torch.no_grad():
                self.eval()
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.to(device=device)
                preds = []
                for idx, batch in enumerate(tqdm(dataloader)):
                    preds.append(self.predict_step([x.to(device=device) for x in batch], idx).cpu())
                    del batch
                    torch.cuda.empty_cache()
        res = [x.cpu().numpy().reshape(-1) for x in preds]
        return np.hstack(res)

class DataModule(pl.LightningDataModule):
    def __init__(self, train_inputv, train_target,
                 val_inputv=None, val_target=None,
                 batch_size = 50,
                 num_workers=dl_num_workers, train_val_split_seed=0):
        super().__init__()

        self.batch_size = min(batch_size, 8*len(train_target)//10)

        y_dtype = torch.float32

        self.train_inputv = torch.as_tensor(train_inputv, dtype=torch.float32)
        self.train_target = torch.as_tensor(train_target, dtype=y_dtype)

        self.val_inputv = val_inputv
        self.val_target = val_target
        if val_inputv is not None:
            self.val_inputv = torch.as_tensor(val_inputv, dtype=torch.float32)
        if val_target is not None:
            self.val_target = torch.as_tensor(val_target, dtype=y_dtype)

        self.num_workers = num_workers
        self.train_val_split_seed = train_val_split_seed

    def setup(self, stage):
        if stage == 'fit':
            if self.val_inputv is None:
                fdataset = TensorDataset(self.train_inputv, self.train_target)
                generator = torch.Generator().manual_seed(self.train_val_split_seed)
                partitions = [len(fdataset) - len(fdataset)//10, len(fdataset) // 10]
                fdataset = torch.utils.data.random_split(fdataset, partitions,
                    generator=generator)
                self.train_dataset, self.val_dataset = fdataset
            else:
                self.train_dataset = TensorDataset(self.train_inputv, self.train_target)
                self.val_dataset = TensorDataset(self.val_inputv, self.val_target)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers = self.num_workers)

    def test_dataloader(self):
        raise RuntimeError("Test data not set")
