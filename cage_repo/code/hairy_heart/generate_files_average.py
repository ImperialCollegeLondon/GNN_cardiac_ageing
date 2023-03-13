from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import pickle
import pyvista as pv
from scipy.spatial import ConvexHull

home = os.path.expanduser("~")
cache_folder = '/scratch/minacio/cache_hairyheart'
# cache_folder = home + '/.cache/cache_hairyheart'

matched_points = np.array(np.loadtxt(f"../../data/downsample_99.8/matched_points.txt")[:,1], dtype=int)

ages = [40, 50, 60, 70, 'any']

os.makedirs(cache_folder, exist_ok=True)
for healthy in [False, True]:
    for age in ages:
        print(f'healthy: {healthy}, age: {age}')
        if healthy:
            ids = pd.read_csv('../../data/healthyageinggroup.csv')[['eid_18545', 'age_at_MRI']]
        else:
            ids = pd.read_csv('../../data/COPYcHTN_unselected11049_bridged.csv')[['eid_18545', 'eid_40616']]
            to_join = pd.read_csv('../../data/nonhealthyageinggroup.csv')[['eid_40616', 'age_at_MRI']]
            ids = ids.join(to_join.set_index('eid_40616'), on='eid_40616')[['eid_18545', 'age_at_MRI']]

        ids = ids[ids.eid_18545!=5489035]
        ids = ids[ids.eid_18545!=1577121]
        if age != 'any':
            ids = ids[(ids.age_at_MRI>=age)&(ids.age_at_MRI<age+10)]
        ids = np.array(ids.iloc[:, 0].to_numpy(), dtype=str)
        np.random.shuffle(ids)

        os.makedirs('/scratch/minacio/cache_vtks_cardiac_age', exist_ok=True)
        searchdirs = [
            '/scratch/minacio/cache_vtks_cardiac_age',
            home + '/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_diseasegroups/htn',
            home + '/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022',
            home + '/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_02_2022',
        ]
        # get the folders of the searchdirs
        subdirs = []
        for searchdir in searchdirs:
            subdirs.append(next(iter(os.walk(searchdir)))[1])

        average_coordinates = None
        n = 0
        ids_that_worked = []
        faces = None
        for id_ in tqdm(ids):
            cachefile = os.path.join(cache_folder, id_)
            if os.path.isfile(cachefile) and faces is not None:
                with open(cachefile, 'rb') as f:
                    subject_coordinates = pickle.load(f)
            else:
                continue_signal = False

                for i, (searchdir, subdir) in enumerate(zip(searchdirs, subdirs)):
                    if id_ in subdir:
                        break
                    if i == len(searchdirs)-1: # not found in any dir
                        #print('Warning: subject', id_, 'not found')
                        continue_signal = True
                        break
                if continue_signal:
                    continue

                subject_coordinates = []
                subject_volumes = []
                for frame in range(50):
                    try:
                        frame_coordinates = f"{searchdir}/{id_}/VTK/LV_endo/LVendo_fr{frame:02d}.vtk"
                        if os.path.getsize(frame_coordinates) == 0:
                            raise Exception
                        if faces is None:
                            faces = pv.read(frame_coordinates).faces
                        frame_coordinates = np.array(pv.read(frame_coordinates).points)
                        volume = ConvexHull(frame_coordinates).volume
                    except Exception:
                        #print('Warning: problem processing:', frame_coordinates)
                        continue_signal = True
                        break
                    subject_coordinates.append(frame_coordinates)
                    subject_volumes.append(volume)
                if continue_signal:
                    continue
                min_volume = np.argmin(subject_volumes)
                subject_coordinates = np.array([
                    *subject_coordinates[min_volume:],
                    *subject_coordinates[:min_volume]
                ])
                with open(cachefile+'_tmp', 'wb') as f:
                    pickle.dump(subject_coordinates, f)
                os.rename(cachefile+'_tmp', cachefile)
            if average_coordinates is None:
                average_coordinates = subject_coordinates
            else:
                average_coordinates += subject_coordinates
            ids_that_worked.append(id_)
            n += 1

        #path_to_save = home + '/cardiac/minacio/ids_that_worked_cardiac_age.csv'
        #ids_that_worked = pd.DataFrame(ids_that_worked, columns=['eid_18545'])
        #ids_that_worked.to_csv(path_to_save, index=False)

        print('Percentage of subject read failure:', 1-n/len(ids))

        average_coordinates /= n

        healthy_str = 'healthy' if healthy else 'unhealthy'
        os.makedirs(f'average_hearts_{age}_{healthy_str}/VTK/LV_endo', exist_ok=True)
        for frame in range(50):
            poly = pv.wrap(average_coordinates[frame])
            poly = pv.PolyData(poly.points, faces)
            poly.save(f'average_hearts_{age}_{healthy_str}/VTK/LV_endo/LVendo_fr{frame:02d}.vtk')

        average_coordinates = average_coordinates.transpose(1,0,2)[matched_points]
        average_coordinates = average_coordinates.reshape((-1,3))
        np.savetxt(f'loop_coordinates_{age}_{healthy_str}.txt', average_coordinates)
