import numpy as np
import torch
import pyvista as pv
import open3d as o3d

# Author of these functions:
# "Mohamed, Nairouz H M" <n.mohamed16@imperial.ac.uk>

class DisplacementFeatures():
    def __call__(self, data):
        ref_pvpolydata = data.pos
        npts = len(ref_pvpolydata)
        t_dist = np.zeros(npts)
        for frame in range(50):
            pvpolydata = self.x_mesh[idx, frame]
            pvpolydata = pv.wrap(self.x_mesh[idx, frame])

            p_dists=[]
            for p in range(npts):
                p_dists = np.append(p_dists, np.linalg.norm(pvpolydata.points[p] - ref_pvpolydata.points[p]))

            t_dist = np.vstack( (t_dist, p_dists) )

        if data.x is None:
            data.x = torch.tensor(t_dist.transpose(), dtype=torch.float)
        else:
            data.x = torch.cat((data.x, torch.tensor(t_dist.transpose(), dtype=torch.float)), dim = 1).float()
        return data
        
class FPFHFeatures():
    def __call__(self, data, voxel_size=2):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.pos)

        radius_normal = voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(
                                                                   radius=radius_feature, max_nn=100))

        fpfh_features = torch.tensor(pcd_fpfh.data.transpose(), dtype=torch.float)

        if data.x is None:
            data.x = fpfh_features
        else:
            data.x = torch.cat((data.x, fpfh_features), dim = 1).float()
        return data
    
class PositionFeatures():
    def __call__(self, data):
        pos_features = data.pos.detach().clone()
        if data.x is None:
            data.x = pos_features.float()
        else:
            data.x = torch.cat((data.x, pos_features), dim = 1).float()
        return data       
