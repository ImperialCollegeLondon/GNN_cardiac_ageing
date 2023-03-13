import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from matplotlib import pylab as plt
from itertools import product
from tqdm import tqdm
import pyvista as pv

from scipy import stats
import statsmodels.formula.api as smf

from colnames import lv_columns, non_lv_columns, t1_columns
from meshtools import DatasetGeometric
from captum.attr import Saliency, IntegratedGradients

model_storage_dir = '/scratch/minacio/cardiac_age_best_models_storage_pkls'
home = os.path.expanduser("~")


def model_forward(mask, input_):
    out = model(data)
    return out

def model_forward(edge_mask, data):
    #print(edge_mask.shape)
    #from ipdb import set_trace; set_trace()
    return model(data_list=data, edge_weight=edge_mask)

def standardize(x):
    shape = x.shape
    x = x.reshape(-1)
    x = stats.boxcox(x)[0]
    x = x - x.mean()
    x = x / x.std()
    x = x.reshape(shape)
    return x

product_list = os.listdir(model_storage_dir)
product_list = [x[:-4].split('_') for x in product_list]
product_list = [(x[0], '_'.join(x[1:])) for x in product_list]
print(product_list)

raw_res = dict()

esttype, modeln = ('gnn', 'mesh')
study_name = f'{esttype}_{modeln}'
fname = os.path.join(model_storage_dir, study_name) + ".pkl"
with open(fname, "rb") as f:
    loaded_res = pickle.load(f)
assert len(loaded_res) == 7

model = loaded_res['best_creg'].to('cuda')

ages = ['any']

theme = pv.themes.DefaultTheme()
theme.background = 'white'
theme.font.color = 'black'
#theme.transparent_background = True
decimate_levels = [.0, .7, .8, .91, .93, .95, .96, .97, .98]
decimate_levels = [.98]
np.random.shuffle(decimate_levels)
for decimate_level in decimate_levels:
    for age in ages:
        for healthy in ['healthy', 'unhealthy']:
            dataset = DatasetGeometric(
                [f'average_hearts_{age}_{healthy}'],
                transforms_names=model.transforms_names,
                searchdirs=['../hairy_heart'],
                decimate_level=decimate_level)

            input_ = dataset[0]
            input_ = [x.to('cuda') for x in input_]
            masks = torch.randn(50, input_[0].edge_index.shape[1]).requires_grad_(True).to('cuda')*0.01
            saliency = Saliency(model_forward)
            masks = saliency.attribute(masks, target=0, additional_forward_args=(input_,))
            masks = masks.cpu().detach().numpy()
            if masks.max() > 0:
                masks = masks / (masks.max() + 1e-10)
                masks = masks + 1e-10
            masks = standardize(masks)

            for frame in [0, 10, 20, 30, 40]:
                mask = np.abs(masks[frame])
                #if mask.max() > 0:
                #    mask = mask / mask.max()
                edge_index = input_[frame].edge_index.data.cpu().numpy()
                points = input_[frame].pos.data.cpu().numpy()

                lines = np.repeat(2, edge_index.shape[1]).reshape(-1,1)
                lines = np.hstack([lines,edge_index.T]).reshape(-1)

                polydata = pv.PolyData(points, lines=lines)

                #polydata.plot(scalars=standardize(mask), line_width=10, cmap='hot_r')
                plotter = pv.Plotter(off_screen=True, theme=theme)
                plotter.add_mesh(polydata, scalars=mask, line_width=10,
                    cmap='hot_r')
                #plotter.add_text(f'{age}s', position='upper_left')
                folder = f"{home}/cardiac/Ageing/gnn_paper/figures/saliency/decimation_{decimate_level}"
                os.makedirs(folder, exist_ok=True)
                plotter.show(
                    cpos=[-1, -1, 0.3], screenshot=
                    f"{folder}/saliency_age_{age}s_{healthy}_frame_{frame}.png",
                )
# home = os.path.expanduser("~")
# dataset = DatasetGeometric(
    # [f'1006041'],
    # transforms_names=model.transforms_names,
    # searchdirs=[home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_diseasegroups/htn'],
    # decimate_level=0)

# input_ = dataset[0]
# input_ = [x.to('cuda') for x in input_]
# masks = torch.randn(50, input_[0].edge_index.shape[1]).requires_grad_(True).to('cuda')*0.01
# saliency = Saliency(model_forward)
# masks = saliency.attribute(masks, target=0,
                    # additional_forward_args=(input_,))
# mask = masks[0].cpu().detach().numpy()
# # mask = np.abs(mask)
# # if mask.max() > 0:
    # # mask = mask / mask.max()
# mask = mask - mask.min()
# mask = mask / mask.max()
# # mask = standarlize(max)
# edge_index = input_[0].edge_index.data.cpu().numpy()
# points = input_[0].pos.data.cpu().numpy()

# lines = np.repeat(2, edge_index.shape[1]).reshape(-1,1)
# lines = np.hstack([lines,edge_index.T]).reshape(-1)

# polydata = pv.PolyData(points, lines=lines)

# #polydata.plot(scalars=mask, line_width=10, cmap='hot_r')

# plotter = pv.Plotter(off_screen=True, theme=theme)
# plotter.add_mesh(polydata, scalars=mask, line_width=10,
    # cmap='hot_r')
# plotter.add_text(f'unhealthy', position='upper_left')

# rv_polydata = pv.read(home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_diseasegroups/htn/1006041/VTK/RV/RV_fr00.vtk')
# plotter.add_mesh(rv_polydata)

# plotter.show(screenshot=f'saliency_unhealthy.png', cpos=[1, -1, 0.3])
