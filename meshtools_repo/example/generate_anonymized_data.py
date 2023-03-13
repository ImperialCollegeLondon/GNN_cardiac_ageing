import tempfile
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial import ConvexHull
from tqdm import tqdm
import pyvista as pv

#searchdir = os.path.expanduser("~") + '/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022'
searchdir = '/scratch/minacio/cache_vtks_cardiac_age'
data_path = '/root/cardiac/minacio/healthyageinggroup_t1.csv'
output_path = '/scratch/minacio/anonymized_data'
matched_points_path = '/scratch/minacio/anonymized_data/matched_points.txt'
n_frames = 50
decimate_level = 0.9

df_cp_train = pd.read_csv(data_path)#.iloc[:20]
df_cp_train.sort_values('age_at_MRI', inplace=True)
true_targets = df_cp_train['age_at_MRI']
ids = df_cp_train['eid_18545']
match_subset = np.loadtxt(str(matched_points_path), dtype=int)[:, 1]

count = 0
faces = None
mesh_dataset = []
target_dataset = []

#ids = ids[:30]
for idx, id_ in enumerate(tqdm(ids)):
    if count == 0:
        average_mesh = []
        average_target = []

    try:
        instance_mesh = []
        for frame in range(n_frames):
            fname = f"{searchdir}/{id_}/VTK/LV_endo/LVendo_fr{frame:02d}.vtk"
            polydata = pv.read(fname)
            if faces is None:
                faces = polydata.faces
            instance_mesh.append(polydata.points)
        instance_mesh = np.stack(instance_mesh)
        instance_target = true_targets[idx]
        count += 1
        #print(f'Success for id {id_}')
        
    except Exception as e:
        #print(f'Failed for id {id_}:', e)
        continue
        
    if count < 10:
        average_mesh.append(instance_mesh)
        average_target.append(instance_target)
    else:
        mesh_dataset.append(np.stack(average_mesh).mean(0))
        average_target.append(np.round(np.mean(instance_target)))
        count = 0   
    
np.stack(mesh_dataset)
targets_dataset = pd.DataFrame(enumerate(average_target), columns=['id', 'age'])
os.makedirs(output_path, exist_ok=True)
targets_dataset.to_csv(f'{output_path}/targets.csv', index=False)


# (n_instances, n_frames, n_vertices, 3)
decimated_meshes = len(mesh_dataset), n_frames, len(match_subset), 3 
decimated_meshes = np.empty(decimated_meshes) + np.nan

volumes_from_meshes = len(mesh_dataset), n_frames
volumes_from_meshes = np.empty(volumes_from_meshes) + np.nan

for idx, mesh in enumerate(tqdm(mesh_dataset)):
    folder = f'{output_path}/vtks/{idx}/VTK/LV_endo'
    os.makedirs(folder, exist_ok=True)
    volumes = []
    for frame in range(n_frames):
        fname = f"{folder}/LVendo_fr{frame:02d}.vtk"
        points = mesh_dataset[idx][frame]
        polydate = pv.PolyData(points, faces)
        polydate = polydate.decimate(decimate_level)
        polydate.save(fname)
        decimated_meshes[idx, frame, :] = points[match_subset]
        volumes.append(ConvexHull(points).volume)
        
    volumes = np.array(volumes)
    min_volume = np.argmin(volumes)
    volumes = volumes[[*range(min_volume, 50), *range(min_volume)]]
    volumes_from_meshes[idx] = volumes

assert not np.isnan(decimated_meshes).any()
assert not np.isnan(volumes_from_meshes).any()

with open(f'{output_path}/decimated_meshes.pkl', 'wb') as f:
    pickle.dump(decimated_meshes, f)

with open(f'{output_path}/volumes_from_meshes.pkl', 'wb') as f:
    pickle.dump(volumes_from_meshes, f)
