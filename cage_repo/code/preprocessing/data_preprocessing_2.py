# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:43:37 2020
@author: gbello, sjadhav
"""

import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
import mirtk
from argparse import ArgumentParser
import shutil
import vtk
from tqdm import tqdm

"""
This code is based on the output of 'DataProcessing-1_2.py', and will carry out further processing on the mesh motion data
and the survival outcome data:
(1) Mesh downsampling: full mesh has 21510 vertices, but to prevent the analysis from being too computationally intensive,
    we will select only a subset of these vertices (n=146)
(2) Flattening: Mesh motion data has 4 dimensions: # of patients x # of frames x # of vertices x # of coordinates.
(3) For use with our network, we will flatten it to 2 dimensions: # of patients x # of motion features, where # of motion
    features is the product of # of frames by # of vertices by # of coordinates
"""


def main(motion_descriptor_path: Path, output_path: Path, matched_points_path: Path, mesh_path: Path,
         downsample: bool = False, downsample_rate: float = 98.9):
    # Load LV mesh data created in 'DataProcessing-1_2.py'
    # path = '/mnt/storage/home/sjadhav/' # Use PWD
    # with open(path+'cardiac/SJ/work/data/mesh_motion_descriptor_full.pkl', 'rb') as f:
    with open(str(motion_descriptor_path), 'rb') as f:
        Xflatall, ID_list = pickle.load(f)

    with open(Xflatall[0], 'rb') as f:
        first_x, _ = pickle.load(f)
        print(tuple([len(Xflatall), *first_x.shape]), "initial shape")
        del first_x

    print(len(ID_list), "number of ids")

    """
    For downsampling, first generate below text file using mirtk library and the commands:
    (1)mirtk decimate-surface,(2)snreg,(3)mirtk transform-points,(4)mirtk match-points
    mirtk decimate-surface RV_ED.vtk RV_u.vtk -reduceby 98.9 -preservetopology on
    """
    if downsample or not matched_points_path.exists():
        assert mesh_path is not None, "To decimate surface, mesh path has to be provided. "
        downsample_dir = matched_points_path.parent
        downsample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(mesh_path), str(downsample_dir))
        mesh_path = downsample_dir.joinpath(mesh_path.name)
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(str(mesh_path))
        reader.Update()

        polydata = reader.GetOutput()
        print("Mesh number of points: ", polydata.GetPoints().GetNumberOfPoints())
        print('Decimating')
        mirtk.decimate_surface(
            str(mesh_path),
            str(downsample_dir.joinpath("downsampled.vtk")),
            reduceby=downsample_rate,
            preservetopology="on",
        )
        print('Registering')
        mirtk.register(
            str(downsample_dir.joinpath("downsampled.vtk")),
            str(mesh_path),
            "-par", "Point set distance correspondence", "CP",
            model="FFD",
            dofout=str(downsample_dir.joinpath("downsample.dof.gz")),
        )
        print('Transforming points')
        mirtk.transform_points(
            str(downsample_dir.joinpath("downsampled.vtk")),
            str(downsample_dir.joinpath("decimated.vtk")),
            dofin=str(downsample_dir.joinpath("downsample.dof.gz")),
        )
        print('Matching points')
        mirtk.match_points(
            str(downsample_dir.joinpath("decimated.vtk")),
            str(mesh_path),
            corout=str(matched_points_path),
        )
        print('mirtk done')

    # Now downsample the mesh using this text file, i.e. select subset of point cloud (146 of 21510).
    # The txt file contains x,y,z coordinates for the 0-145 points.
    # PatintID_mpts = np.loadtxt(path+'cardiac/sj/IHD/data/matchedpointsapril.txt', dtype=int)
    PatintID_mpts = np.loadtxt(str(matched_points_path), dtype=int)

    x_orig_rel = []
    x_orig_abs = []
    for x_ in tqdm(Xflatall):
        with open(x_, 'rb') as f:
            subject_x_rel, subject_x_abs = pickle.load(f)
        x_orig_rel.append(subject_x_rel[:, PatintID_mpts[:, 1], :])
        x_orig_abs.append(subject_x_abs[:, PatintID_mpts[:, 1], :])
    x_orig_rel = np.stack(x_orig_rel, axis=0)
    x_orig_abs = np.stack(x_orig_abs, axis=0)
    del Xflatall

    print(x_orig_abs.shape, 'abs decimated shape')
    print(x_orig_rel.shape, 'rel decimated shape')

    # Store motion, outcome and IDs in one file. Use this file as input data file to the 4DS network
    plistout = [x_orig_rel, x_orig_abs, ID_list]
    # with open(path + 'cardiac/SJ/work/data/mesh_motiondescriptor_453_april.pkl', 'wb') as f:
    with open(str(output_path), 'wb') as f:
        pickle.dump(plistout, f, protocol=pickle.HIGHEST_PROTOCOL)
    # This pkl file contains:
    #1) mesh motion data for patients (i.e. motion descriptors) [For IHD data: 199*10512 vector]
    #2) censoring status and survival times for patients [For IHD data: 199*2]
    #3) list of patient IDs # Used only at the end in 4DS, while writing the final output file [For IHD data: 199]
    #########


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mesh", dest="mesh_path", type=str, required=True,
                        help="Path pointing to a subject's LVendo_ED mesh. (vtk file)")
    parser.add_argument("--motion", dest="motion_descriptor_path", type=str, default=None,
                        help="Path pointing to a pkl file containing mesh motion and survival outcome. "
                             "Output of the data_processing_1.py. Default is looking at data folder of the repo.")
    parser.add_argument("--matched", dest="matched_points_path", type=str, default=None,
                        help="Path to a txt file of downsampled points correspondence. "
                             "If it is not provided, then it will be generated using mirtk")
    parser.add_argument("--output", dest="output_path", type=str, default=None,
                        help="Output downsampled pkl file. Default is the parent directory of motion descriptor.")
    parser.add_argument("--downsample", dest="downsample", action="store_true",
                        help="Whether to perform downsampling or use exisiting matched points. "
                             "If matched points file does not exist, downsample will be performed. ")
    parser.add_argument("--downsample-rate", dest="downsample_rate", type=float, default=98.9,
                        help="Downsampling rate.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.motion_descriptor_path is None:
        motion_descriptor_path = Path(__file__).parent.parent.parent.joinpath(
            "data", "mesh_descriptor.pkl"
        )
    else:
        motion_descriptor_path = Path(args.motion_descriptor_path)

    if args.matched_points_path is None:
        matched_points_path = motion_descriptor_path.parent.joinpath(
            "downsample_{}".format(args.downsample_rate), "matched_points.txt"
        )
    else:
        matched_points_path = Path(args.matched_points_path)

    if args.output_path is None:
        output_path = motion_descriptor_path.parent.joinpath("downsampled_descriptor_{}.pkl".format(args.downsample_rate))
    else:
        output_path = Path(args.output_path)
    main(
        motion_descriptor_path=motion_descriptor_path,
        output_path=output_path,
        matched_points_path=matched_points_path,
        mesh_path=Path(args.mesh_path) if args.mesh_path is not None else None,
        downsample=args.downsample,
        downsample_rate=args.downsample_rate,
    )
