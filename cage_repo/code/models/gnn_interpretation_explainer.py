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

from torch_geometric.explain import Explainer, GNNExplainer
import torch
import torch_geometric

def standardize(x):
    shape = x.shape
    x = x.reshape(-1)
    x = stats.boxcox(x)[0]
    x = x - x.mean()
    x = x / x.std()
    x = x.reshape(shape)
    return x

model_storage_dir = '/scratch/minacio/cardiac_age_best_models_storage_pkls'
home = os.path.expanduser("~")

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

for i in range(len(model.conv1)):
    model.conv1[i].aggr_module = torch_geometric.nn.aggr.basic.SumAggregation()
    model.conv2[i].aggr_module = torch_geometric.nn.aggr.basic.SumAggregation()

ages = ['any']

theme = pv.themes.DefaultTheme()
theme.background = 'white'
theme.font.color = 'black'
#theme.transparent_background = True
#decimate_levels = [.9]
decimate_levels = [.0, .7, .8, .91, .93, .95, .96, .97, .98]
np.random.shuffle(decimate_levels)
for frame in [0, 10, 20, 30, 40]:
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
                def forward(self, x, edge_index):
                    data_list = self.input_.copy()
                    data_list[frame] = data_list[frame].clone()
                    data_list[frame].x = x
                    data_list[frame].edge_index = edge_index
                    return model.forward2(data_list=data_list)
                try:
                    model.forward2
                except Exception:
                    model.forward2 = model.forward
                model.forward = forward.__get__(model)
                model.input_ = input_

                explainer_phenomenon = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=200),
                    explainer_config=dict(
                        explanation_type='phenomenon',
                        node_mask_type='object',
                        edge_mask_type='object',
                    ),
                    model_config=dict(
                        mode='regression',
                        task_level='graph',
                        return_type='raw',
                    ),
                )
                explainer_model = Explainer(
                    model=model,
                    algorithm=GNNExplainer(epochs=200),
                    explainer_config=dict(
                        explanation_type='model',
                        node_mask_type='object',
                        edge_mask_type='object',
                    ),
                    model_config=dict(
                        mode='regression',
                        task_level='graph',
                        return_type='raw',
                    ),
                )
                expl_p = explainer_phenomenon(input_[frame].x, input_[frame].edge_index, target=torch.Tensor([[50]]).cuda())
                expl_m = explainer_model(input_[frame].x, input_[frame].edge_index)

                for explainer_model in ['phenomenon', 'model']:
                    for mask_type in ['edge', 'node']:
                        if explainer_model == 'phenomenon':
                            expl = expl_p
                        else:
                            expl = expl_m

                        if mask_type == 'edge':
                            mask = expl.edge_mask.cpu().numpy()
                        else:
                            mask = expl.node_mask.cpu().numpy()

                        edge_index = input_[frame].edge_index.data.cpu().numpy()
                        points = input_[frame].pos.data.cpu().numpy()

                        lines = np.repeat(2, edge_index.shape[1]).reshape(-1,1)
                        lines = np.hstack([lines,edge_index.T]).reshape(-1)

                        polydata = pv.PolyData(points, lines=lines)

                        #polydata.plot(scalars=standardize(mask), line_width=10, cmap='hot_r')
                        plotter = pv.Plotter(off_screen=True, theme=theme)
                        plotter.add_mesh(polydata, scalars=mask, line_width=10,
                            cmap='hot_r', clim=[0, 1])
                        plotter.add_text(f'{age}s', position='upper_left')
                        folder = f"{home}/cardiac/Ageing/gnn_paper/figures/explainer/v2/plots/decimation_{decimate_level}/explainer_{explainer_model}/mask_type_{mask_type}"
                        os.makedirs(folder, exist_ok=True)
                        plotter.show(
                            cpos=[-1, -1, 0.3], screenshot=
                            f"{folder}/explainer_age_{age}s_{healthy}_frame_{frame}.png",
                        )

                folder = f"{home}/cardiac/Ageing/gnn_paper/figures/explainer/v2/vtks/decimation_{decimate_level}"
                os.makedirs(folder, exist_ok=True)
                polydata['node_mask_p'] = expl.node_mask.cpu().detach().numpy()
                polydata['edge_mask_p'] = expl.edge_mask.cpu().detach().numpy()
                polydata['node_mask_m'] = expl.node_mask.cpu().detach().numpy()
                polydata['edge_mask_m'] = expl.edge_mask.cpu().detach().numpy()
                polydata.save(f"{folder}/explainer_age_{age}s_{healthy}_frame_{frame}.vtk")
