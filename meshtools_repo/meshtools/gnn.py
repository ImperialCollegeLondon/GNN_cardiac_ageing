import tempfile
import os
import pickle
import uuid
import itertools
import hashlib
import shutil
import time
import tempfile
import numbers

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

import pyvista as pv

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import PyTorchProfiler
from optuna.integration import PyTorchLightningPruningCallback

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import torch_geometric as tgeo
import torch_geometric.nn as nngeo
from torch_geometric.loader import DataLoader as DataLoaderGeo

from .gnn_transformers import FPFHFeatures, DisplacementFeatures, PositionFeatures
from .nn import LitNN

dl_num_workers = 5

class GetCPUUsage():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.process_time = time.process_time()
        self.time = time.time()
        
    def get(self):
        cpuusage = (time.process_time() - self.process_time) / (time.time() - self.time)
        #self.reset()
        return cpuusage

class LitNNGeometric(LitNN):
    def __init__(self, *args, transforms_names='constant',
        decimate_level=0.95, searchdirs=None,
        vtk_cache_dir=None, items_cache_dir=None,
        conv2_input_size1=30, #conv2_input_size2=30,
        ll_input_multp_size1=30, ll_input_size2=1000,
        dropout=0.0, heads=1,
        num_workers=dl_num_workers, **kwargs):
            
        self.transforms_names = transforms_names
        self.decimate_level = decimate_level
        self.searchdirs = searchdirs
        self.vtk_cache_dir = vtk_cache_dir
        self.items_cache_dir = items_cache_dir
        self.conv2_input_size1 = conv2_input_size1
        #self.conv2_input_size2 = conv2_input_size2
        self.ll_input_multp_size1 = ll_input_multp_size1
        self.ll_input_size2 = ll_input_size2
        self.dropout = dropout
        self.heads = heads
        self.num_workers = num_workers
                
        self.high_loss_count = 0
        
        super().__init__(*args, **kwargs)
    
    def forward(self, data_list, edge_weight=None):
        xl = []
        for i, data in enumerate(data_list):
            frame_edge_weight = None
            if edge_weight is not None:
                frame_edge_weight = edge_weight[i]
                
            #print(i)
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

            #x = F.elu(self.conv1[i](x, frame_edge_weight, edge_attr))
            #x = self.conv2[i](x, frame_edge_weight, edge_attr)
            
            x = F.elu(self.conv1[i](x, edge_index, frame_edge_weight))
            x = self.conv2[i](x, edge_index, frame_edge_weight)
            
            #x = F.elu(self.conv2[i](x, edge_index, frame_edge_weight))
            #x = self.conv3[i](x, edge_index, frame_edge_weight)

            # 2. Readout layer
            x = nngeo.global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            xl.append(x)

        xl = torch.cat(xl, 1)

        # 3. Apply a final layer
        xl = F.elu(self.fc1(xl))

        return self.fc2(xl)

    def _initialize_layer(self, layer):
        nn.init.constant_(layer.bias, 0)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(layer.weight, gain=gain)
        return layer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self, batch, batch_idx):
        return self.forward(batch).squeeze()

    def training_step(self, batch, batch_idx):
        out = self.forward(batch).squeeze()
        mse = F.mse_loss(out, batch[0].y)
        self.log('train_rmse', torch.sqrt(mse).item(), prog_bar=True, batch_size=batch[0].x.shape[0])
        
        if self.current_epoch >= 0:
            if np.sqrt(mse.item()) > 20:
                self.high_loss_count += 1
            else:
                self.high_loss_count = 0
                
            if self.high_loss_count > 10:
                raise optuna.TrialPruned('Trial pruning due to high loss')
        
        return mse

    def test_validation_step(self, batch, batch_idx, name):
        out = self.forward(batch).squeeze()
        mse = F.mse_loss(out, batch[0].y)
        rmse = torch.sqrt(mse)
        mae = F.l1_loss(out, batch[0].y)

        self.log(f'{name}_loss_rmse', rmse.item(), batch_size=batch[0].x.shape[0])
        self.log(f'{name}_loss_mae', mae.item(), batch_size=batch[0].x.shape[0])

    def validation_step(self, batch, batch_idx):
        self.test_validation_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.test_validation_step(batch, batch_idx, 'test')

    def mtrain(self, x_mesh, targets, x_extra=None):
        #input_size = x_train.shape[1]
        targets = targets.reshape(-1)
        self.y_train_mean = targets.mean()

        train_size = 2*len(targets)//3
        if train_size < self.batch_size:
            self.batch_size = train_size

        out_size = 1
        if self.transforms_names == 'constant':
            init_size = 1
        elif self.transforms_names == 'position':
            init_size = 4
        elif self.transforms_names == 'fpfh':
            init_size = 34
        elif self.transforms_names == 'position_fpfh':
            init_size = 37
        #self.conv1 = nn.ModuleList([nngeo.GraphConv(init_size, self.conv2_input_size1) for _ in range(50)])
        #self.conv2 = nn.ModuleList([nngeo.GraphConv(self.conv2_input_size1, self.conv2_input_size2) for _ in range(50)])
        #self.conv3 = nn.ModuleList([nngeo.GraphConv(self.conv2_input_size2, self.ll_input_multp_size1) for _ in range(50)])
        
        self.conv1 = nn.ModuleList([nngeo.GCNConv(init_size, self.conv2_input_size1) for _ in range(50)])
        self.conv2 = nn.ModuleList([nngeo.GCNConv(self.conv2_input_size1, self.ll_input_multp_size1) for _ in range(50)])
        
        #self.conv1 = nn.ModuleList([nngeo.SplineConv(init_size, self.conv2_input_size1, dim=3, kernel_size=5, aggr='add') for _ in range(50)])
        #self.conv2 = nn.ModuleList([nngeo.SplineConv(self.conv2_input_size1, self.ll_input_multp_size1, dim=3, kernel_size=5, aggr='add') for _ in range(50)])
        
        #self.conv1 = nn.ModuleList([nngeo.HypergraphConv(init_size, self.conv2_input_size1, use_attention=True, dropout=self.dropout, heads=self.heads) for _ in range(50)])
        #self.conv2 = nn.ModuleList([nngeo.HypergraphConv(self.conv2_input_size1, self.ll_input_multp_size1, use_attention=False, dropout=self.dropout, heads=self.heads) for _ in range(50)])
        
        #self.conv3 = nn.ModuleList([nngeo.SplineConv(self.conv2_input_size2, self.ll_input_multp_size1, dim=3, kernel_size=5, aggr='add') for _ in range(50)])
        
        #self.conv1 = nn.ModuleList([nngeo.GATConv(init_size, self.conv2_input_size1, dropout=self.dropout, heads=self.heads) for _ in range(50)])
        #self.conv2 = nn.ModuleList([nngeo.GATConv(self.conv2_input_size1*self.heads, self.ll_input_multp_size1, dropout=self.dropout, heads=self.heads) for _ in range(50)])
        #self.conv3 = nn.ModuleList([nngeo.GATConv(self.conv2_input_size2*self.heads, self.ll_input_multp_size1, dropout=self.dropout, heads=self.heads) for _ in range(50)])
        
        #self.conv1 = nn.ModuleList([nngeo.GATv2Conv(init_size, self.conv2_input_size1, dropout=self.dropout, heads=self.heads, edge_dim=3) for _ in range(50)])
        #self.conv2 = nn.ModuleList([nngeo.GATv2Conv(self.conv2_input_size1*self.heads, self.ll_input_multp_size1, dropout=self.dropout, heads=self.heads, edge_dim=3) for _ in range(50)])
        
        #self.conv3 = nn.ModuleList([nngeo.GATConv(self.conv2_input_size2*self.heads, self.ll_input_multp_size1, dropout=self.dropout, heads=self.heads) for _ in range(50)])
        self.fc1 = nn.Linear(self.ll_input_multp_size1*self.heads*50, self.ll_input_size2)
        self.fc2 = nn.Linear(self.ll_input_size2, out_size)

        datamodule = DataModuleGeometric(x_mesh, targets,
            batch_size=self.batch_size,
            transforms_names=self.transforms_names,
            decimate_level=self.decimate_level,
            searchdirs=self.searchdirs,
            vtk_cache_dir=self.vtk_cache_dir,
            items_cache_dir=self.items_cache_dir,
            num_workers=self.num_workers,
        )
        self.batch_size = datamodule.batch_size

        trainer = self.get_trainer()
        
        # from pytorch_lightning.loops import FitLoop
        # class CustomFitLoop(FitLoop):
            # def advance(self):
                # dataloader = self.trainer.train_dataloader
                # gcu = GetCPUUsage()
                # print('start', flush=True)
                # for i, batch in enumerate(dataloader):
                    # print('got batch', gcu.get(), flush=True)
                    # self.run_a_batch(batch, i)
                    # print('finished run_a_batch', gcu.get(), flush=True)

            # def run_a_batch(self, batch, i):
                # self.trainer.model.to(device='cuda')
                # print('model to cuda', flush=True)
                # [x.to(device='cuda') for x in batch]
                # print('batch to cuda', flush=True)
                # loss = self.trainer.model.training_step(batch, i)
                # print('got loss', flush=True)
                # optimizer = self.trainer.optimizers[0]
                # optimizer.zero_grad()
                # loss.backward()
                # print('backwarded', flush=True)
                # optimizer.step()
                # print('steped', flush=True)
        # trainer.fit_loop = CustomFitLoop()

        trainer.fit(self, datamodule = datamodule)
        
        #self.trainer = None
        #self.mtrainer = trainer

        return self

    def get_trainer(self):
        callbacks = [EarlyStopping(
           monitor='val_loss_rmse',
           min_delta=0.00,
           patience=self.es_patience,
           verbose=False,
           mode='min'
        )]
        #callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss_mae"))

        try:
            raise ImportError
            from pytorch_lightning.loggers import MLFlowLogger
            logger = MLFlowLogger(
                experiment_name="gnn",
                tracking_uri="file:./mlruns"
            )
        except ImportError:
            logger = True
                
        return pl.Trainer(   
                             #profiler=PyTorchProfiler(),
                             precision=32,
                             accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             logger=logger,
                             log_every_n_steps=self.batch_size,
                             val_check_interval=0.25, # do validation check 4 times for each epoch
                             auto_lr_find=True,
                             callbacks=callbacks,
                             max_epochs = 1000,
                             #max_epochs = 1,
                             num_sanity_val_steps=0,
                            )
                            
    def mtest(self, x_mesh, targets, x_extra=None):
        pred = self.predict(x_mesh)
        targets = targets.reshape(-1)
        loss_rmse = np.sqrt(((pred - targets)**2).mean())
        loss_mae = np.abs(pred - targets).mean()
        return dict(mae=loss_mae, rmse=loss_rmse)

    def predict(self, x_mesh, x_extra=None):
        #trainer = self.get_trainer()
        dataset = DatasetGeometric(x_mesh,
            transforms_names=self.transforms_names,
            decimate_level=self.decimate_level,
            searchdirs=self.searchdirs,
            vtk_cache_dir=self.vtk_cache_dir,
            items_cache_dir=self.items_cache_dir,
        )
        dataloader = DataLoaderGeo(dataset, batch_size=min(self.batch_size, len(dataset)), num_workers=self.num_workers)
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

class DataModuleGeometric(pl.LightningDataModule):
    def __init__(self, train_x_mesh, train_targets,
                 test_x_mesh=None, test_targets=None,
                 batch_size = 40,
                 num_workers=dl_num_workers, train_val_split_seed=0,
                 transforms_names=None,
                 decimate_level=0.95, searchdirs=None,
                 vtk_cache_dir=None, items_cache_dir=None):
        super().__init__()

        self.batch_size = min(batch_size, 8*len(train_targets)//10)
        self.num_workers = num_workers
        self.train_val_split_seed = train_val_split_seed

        self.train_x_mesh = train_x_mesh
        self.train_targets = train_targets

        self.test_x_mesh = test_x_mesh
        self.test_targets = test_targets
        
        self.transforms_names = transforms_names
        self.decimate_level = decimate_level
        self.searchdirs = searchdirs
        self.vtk_cache_dir = vtk_cache_dir
        self.items_cache_dir = items_cache_dir

    def setup(self, stage):
        if stage == 'fit':
            fdataset = DatasetGeometric(
                    self.train_x_mesh,
                    self.train_targets,
                    transforms_names=self.transforms_names,
                    decimate_level=self.decimate_level,
                    searchdirs=self.searchdirs,
                    vtk_cache_dir=self.vtk_cache_dir,
                    items_cache_dir=self.items_cache_dir
            )
            #train_idx, test_idx = train_test_split(len(fdataset), shuffle=True, test_size=0.1, random_state=self.train_val_split_seed)
            #self.train_dataset = fdataset[train_idx]
            #self.val_dataset = fdataset[test_idx]
            generator = torch.Generator().manual_seed(self.train_val_split_seed)
            partitions = [len(fdataset) - len(fdataset)//10, len(fdataset) // 10]
            fdataset = torch.utils.data.random_split(fdataset, partitions, generator=generator)
            self.train_dataset, self.val_dataset = fdataset

        if stage == 'test':
            if self.test_x_mesh is not None:
                self.test_dataset = DatasetGeometric(
                    test_x_mesh, test_targets,
                    transforms_names=self.transforms_names,
                    decimate_level=self.decimate_level,
                    searchdirs=self.searchdirs,
                    vtk_cache_dir=self.vtk_cache_dir,
                    items_cache_dir=self.items_cache_dir
                )
    
    def train_dataloader(self):
        return DataLoaderGeo(self.train_dataset, batch_size=self.batch_size, drop_last=True,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          )

    def val_dataloader(self):
        return DataLoaderGeo(self.val_dataset, batch_size=self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          )

    def test_dataloader(self):
        if self.test_inputv is None:
            raise RuntimeError("Test data not set")
        return DataLoaderGeo(self.test_dataset, batch_size=self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          )

class DatasetGeometric(torch.utils.data.Dataset):
    def __init__(self, x_ids, targets=None,
        transforms_names='position_fpfh',
        decimate_level=0.95, searchdirs=None,
        vtk_cache_dir=None, items_cache_dir=None,
        ):
        home = os.path.expanduser("~")
        if searchdirs is None:
            raise ValueError('searchdirs must be set!')
        if isinstance(searchdirs, str):
            raise ValueError('searchdirs must be an list-like object')
        
        self.x_ids = np.array(x_ids, dtype=str)
        self.cache_hash_id = []
        self.decimate_level = decimate_level
        self.searchdirs = searchdirs
        self.vtk_cache_dir = vtk_cache_dir
        self.items_cache_dir = items_cache_dir
        
        for i in tqdm(range(len(self.x_ids))):
            hash_ = hashlib.sha256()
            hash_.update(b'x')
            hash_.update(str(self.x_ids[i].item()).encode())
            if targets is not None:
                hash_.update(b'y')
                hash_.update(str(targets[i].item()).encode())
            hash_.update(b'decimation')
            hash_.update(str(self.decimate_level).encode())
            hash_.update(b'transforms')
            hash_.update(transforms_names.encode())
            hash_ = hash_.hexdigest()
            self.cache_hash_id.append(hash_)

        self.targets = targets
        if targets is not None:
            self.targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)

        T = tgeo.transforms
        self.transforms_names = transforms_names
        
        if self.transforms_names == 'constant':
            self.transform = []
            
        elif self.transforms_names == 'position':
            self.transform = [PositionFeatures()]
            
        elif self.transforms_names == 'fpfh':
            self.transform = [FPFHFeatures()]
            
        elif self.transforms_names == 'position_fpfh':
            self.transform = [PositionFeatures(), FPFHFeatures()]
            
        else:
            raise ValueError('Invalid transforms_names')
            
        self.transform = T.Compose([T.FaceToEdge(), T.Constant(value=1), *self.transform, T.Spherical()])
        self.faces = None
        self.tdata_cache = dict()
        
        if self.items_cache_dir is None:    
            self.items_cache_dir = os.path.join(home, '.cache', 'meshtools_cache_geodataset_items')
        os.makedirs(self.items_cache_dir, exist_ok=True)
        
        if self.vtk_cache_dir is None:    
            self.vtk_cache_dir = os.path.join(home, '.cache', 'meshtools_cache_geodataset_vtks')
        os.makedirs(self.vtk_cache_dir, exist_ok=True)   
        
        # get the folders of the searchdirs
        self.subdirs = []
        for searchdir in self.searchdirs:
            self.subdirs.append(next(iter(os.walk(searchdir)))[1])
            
    def get_meshes_from_vtk_cache(self, id_):
        for i, (searchdir, subdir) in enumerate(zip(self.searchdirs, self.subdirs)):
            if id_ in subdir:
                break
            if i == len(self.searchdirs)-1: # not found in any dir
                raise ValueError
            
        meshes = []
        for frame in range(50):
            original_file = f"{searchdir}/{id_}/VTK/LV_endo/LVendo_fr{frame:02d}.vtk"
            cache_dir = f"{self.vtk_cache_dir}/{id_}/VTK/LV_endo"
            cache_file = f"{cache_dir}/LVendo_fr{frame:02d}.vtk"
            os.makedirs(cache_dir, exist_ok=True)
            if not os.path.exists(cache_file):
                shutil.copyfile(original_file, cache_file+'_tmp')
                shutil.move(cache_file+'_tmp', cache_file)
            mesh = pv.read(cache_file)
            if self.decimate_level:
                mesh = mesh.decimate(self.decimate_level)
            meshes.append(mesh)
            
        return meshes

    def __getitem__(self, idx):
        if not isinstance(idx, numbers.Integral):
            idxn = np.arange(len(self))[idx]
            if isinstance(idxn, np.ndarray):
                return [self[i] for i in idxn]

        cache_file = self.items_cache_dir + '/' + str(self.cache_hash_id[idx])
        if False:#idx in self.tdata_cache:
            tdata = self.tdata_cache[idx]
        else:
            try:
                with open(cache_file, 'rb') as f:
                    tdata = pickle.load(f)
                #self.tdata_cache[idx] = tdata
            except Exception:
                meshes = []
                id_ = self.x_ids[idx]
                print(f'Generating data for {id_}')
                meshes = self.get_meshes_from_vtk_cache(id_)
                    
                volumes = []
                for frame in range(50):
                    volume = ConvexHull(meshes[frame].points).volume
                    volumes.append(volume)
                
                min_volume = np.argmin(volumes)
            
                tdata = []
                for frame in [*range(min_volume, 50), *range(min_volume)]:
                    # path = self.full_path.format(id_=self.ids[idx], frame=frame)
                    # polydata =  pv.read(path)
                    polydata = meshes[frame]
                        
                    pos = polydata.points
                    faces = polydata.faces.reshape(-1, 4)[:, 1:].T
                    data = tgeo.data.Data(
                        pos=torch.tensor(pos),
                        normal=torch.tensor(polydata.point_normals),
                        face=torch.tensor(faces),
                    )
                    idata = self.transform(data)
                    if self.targets is not None:
                        idata.y = self.targets[idx]
                    tdata.append(idata)

                #self.tdata_cache[idx] = tdata
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(tdata, f)
                except Exception:
                    pass
        return tdata

    def __len__(self):
        return len(self.x_ids)
