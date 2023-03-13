import os
import pickle
import pyvista as pv

import numpy as np
import pandas as pd
from meshtools import DatasetGeometric
from vaecompare import VAE
from sklearn.preprocessing import StandardScaler
from itertools import chain

decimate_level = 0.95
n_synthetic_instances = 100

vtk_cache_dir = '/scratch/minacio/cache_vtks_cardiac_age'
data_path = 'original_dataset.csv'
df_cp_train = pd.read_csv(data_path)
home = os.path.expanduser("~")

searchdirs = [
    home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022',
    home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_02_2022',
]
ids_with_motion = [os.listdir(x) for x in searchdirs]
ids_with_motion = list(chain(*ids_with_motion))
df_cp_train = df_cp_train[np.isin(df_cp_train['eid_18545'], ids_with_motion)]
ids = df_cp_train['eid_18545'].to_numpy()
target = df_cp_train['age_at_MRI'].to_numpy()


# Get faces from some instance
faces = f"{vtk_cache_dir}/{ids[0].item()}/VTK/LV_endo/LVendo_fr00.vtk"
faces = pv.read(faces).decimate(decimate_level).faces

# shuffles data
idx = np.random.choice(len(ids), len(ids))
ids = ids[idx]
target = target[idx]

dataset = DatasetGeometric(
    ids,
    searchdirs = searchdirs,
    items_cache_dir = '/scratch/minacio/cache_items_cardiac_age',
    vtk_cache_dir = vtk_cache_dir,
    decimate_level=decimate_level,
    transforms_names='constant'
)

# true_data_raw.shape == (n_instances, n_frames, n_features, 3)
true_data_raw = np.array([np.array([frame.pos.numpy() for frame in frames]) for frames in dataset])

true_data = true_data_raw.reshape(len(true_data_raw), -1)
true_data = np.column_stack((true_data, target))
vae = VAE(
    dataloader_workers=1,
    verbose=2,
    batch_initial=10,
    batch_max_size=10,
    num_layers_decoder=3,
    hidden_size_decoder=1000,
    dropout_decoder=0.3,
    num_layers_encoder=3,
    hidden_size_encoder=1000,
    dropout_encoder=0.3,
    latent_dim = 200,
)
scaler = StandardScaler().fit(true_data)
vae.fit(scaler.transform(true_data))
synthetic_data = scaler.inverse_transform(vae.sample_y(n_synthetic_instances))
synthetic_age = synthetic_data[:,-1]
synthetic_data = synthetic_data[:,:-1]
synthetic_data = synthetic_data.reshape([len(synthetic_data), *true_data_raw.shape[1:]])

dir_to_save = home + "/cardiac/Ageing/gnn_paper/datasets/synthetic_data/vtk"
for instance_id in range(synthetic_data.shape[0]):
    dir_ = f'{dir_to_save}/{instance_id}/VTK/LV_endo'
    os.makedirs(dir_, exist_ok=True)    
    for frame_id in range(synthetic_data.shape[1]):
        polydata = pv.PolyData(synthetic_data[instance_id, frame_id], faces)
        polydata.save(f"{dir_}/LVendo_fr{frame_id:02d}.vtk")

