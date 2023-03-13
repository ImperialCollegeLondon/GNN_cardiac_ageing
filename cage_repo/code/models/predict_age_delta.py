import os
import pickle
import time
import tempfile

import numpy as np
import pandas as pd
import torch

from matplotlib import pylab as plt
from itertools import chain
from tqdm import tqdm
import pyvista as pv

from scipy import stats
import statsmodels.formula.api as smf

from colnames import lv_columns, non_lv_columns, t1_columns
from meshtools import DatasetGeometric, BiasCorrectEstimatorCole
from torch_geometric.loader import DataLoader as DataLoaderGeo

import torch

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

home = os.path.expanduser("~")

# Load the trained model
model_storage_dir = '/scratch/minacio/cardiac_age_best_models_storage_pkls'
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

# Get ids, age and sex of unhealthy the patients
df = pd.read_csv('../../data/COPYcHTN_unselected11049_bridged.csv')
dfn = pd.read_csv('../../data/nonhealthyageinggroup_t1.csv')
df = df.join(dfn.set_index('eid_40616')[['age_at_MRI', 'sex']], on='eid_40616')
df['healthy'] = 0

# Get ids, age and sex of healthy the patients
dfh = pd.read_csv('../../data/healthyageinggroup_t1.csv')
ids = pd.read_csv(home + '/cardiac/Ageing/gnn_paper/datasets/healthy_test_ids.csv')
dfh = dfh[dfh['eid_18545'].isin(ids['eid_18545'])]
assert len(dfh) == len(ids)
dfh = dfh[['eid_40616', 'eid_18545', 'age_at_MRI', 'sex']]
dfh['healthy'] = 1

df = pd.concat((dfh, df))

searchdirs = [
    home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_diseasegroups/htn',
    home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_01_2022',
    home+'/cardiac/UKBB_40616/UKBB_motion_analysis/results/UKBB_02_2022',
]

# remove patients without motion files
ids_with_motion = [os.listdir(x) for x in searchdirs]
ids_with_motion = list(chain(*ids_with_motion))
df = df[np.isin(df['eid_18545'], ids_with_motion)]
list_of_patients = df['eid_18545'].to_numpy()

# Load dataset
dataset = DatasetGeometric(
    list_of_patients,
    transforms_names=model.transforms_names,
    searchdirs=searchdirs,
    items_cache_dir = '/scratch/minacio/cache_items_cardiac_age',
    vtk_cache_dir = '/scratch/minacio/cache_vtks_cardiac_age',
    decimate_level=.9)

# Predict using trained model
res = []
with torch.no_grad():
    model.eval()
    for batch in tqdm(DataLoaderGeo(dataset, batch_size=100)):
        batch = [x.to('cuda') for x in batch]
        res.append(model(batch).cpu().numpy())
res = np.vstack(res).reshape(-1)
df['predicted_age'] = res

# Bias correction
df['delta_not_corrected'] = df['predicted_age'] - df['age_at_MRI']
correcter = BiasCorrectEstimatorCole(df.loc[df['healthy'] == 0, 'predicted_age'], df.loc[df['healthy'] == 0, 'age_at_MRI'])
df['predicted_age_cole_bias_corrected'] = correcter.correct_prediction(df['predicted_age'])
df['delta_cole_bias_corrected'] = df['predicted_age_cole_bias_corrected'] - df['age_at_MRI']

print(stats.pearsonr(df['age_at_MRI'][df['healthy'] == 0], df['delta_not_corrected'][df['healthy'] == 0]))
print(stats.pearsonr(df['age_at_MRI'][df['healthy'] == 0], df['delta_cole_bias_corrected'][df['healthy'] == 0]))

# Save data
db_path = home+'/cardiac/Ageing/gnn_paper/datasets/predicted_age.csv'
df.to_csv(db_path, index=False)

# Plot predicted age vs chronological age
# def plot_gen(pred_or_delta_age, chronological_age, ylabel, filename,
#     add_diagonal_line):
#     f = plt.figure(figsize=[4.4, 4.4])
#     plt.plot(chronological_age, pred_or_delta_age, "o", color="black", markerfacecolor='gold', markeredgewidth=0.5)
#     reg = stats.linregress(chronological_age, pred_or_delta_age)
#     plt.axline(xy1=(0, reg.intercept), slope=reg.slope, linestyle="-",
#         color="green")
#     if add_diagonal_line:
#         plt.axline([0, 0], [1, 1], linestyle="--", color='black')
#     plt.xlim([chronological_age.min(), chronological_age.max()])
#     plt.ylim([pred_or_delta_age.min(), pred_or_delta_age.max()])
#     plt.xlabel('Chronological age')
#     plt.ylabel(ylabel)
#     f.savefig(home + f"/cardiac/Ageing/gnn_paper/figures/pred_age_delta/{filename}.pdf", bbox_inches='tight')
#     plt.close()

# plot_gen(
# pred_or_delta_age = df['predicted_age_cole_bias_corrected'][df['healthy'] == 0],
# chronological_age = df['age_at_MRI'][df['healthy'] == 0],
# ylabel = 'Predicted age',
# filename = "pred_age_vs_chron_age.pdf",
# add_diagonal_line=True,
# )

# plot_gen(
# pred_or_delta_age = df['delta_cole_bias_corrected'][df['healthy'] == 0],
# chronological_age = df['age_at_MRI'][df['healthy'] == 0],
# ylabel = 'Age delta',
# filename = "age_delta_vs_chron_age.pdf",
# add_diagonal_line=False,
# )

importr('fastmap')
importr('ggplot2')
importr('ggExtra')
importr('tidyverse')
importr('cowplot')
for healthy_indicator, healthy_str in enumerate(['unhealthy', 'healthy']):
  ro.r(f"""
  theme_update(
    panel.border = element_blank(),
    axis.line.y = element_line(colour="black"),
    axis.text = element_text(colour="black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x  = element_text(size=16),
    axis.text.y  = element_text(size=16),
    axis.title.x  = element_text(size=16, vjust=0.3, face="plain"),
    axis.title.y  = element_text(size=16, face = "plain", vjust=0.9, angle = 90),
    axis.line = element_line(size = 1.2, linetype = "solid"),
    axis.ticks = element_line(size = 1),
    legend.position="none",
  )
  #theme(plot.title = element_text(vjust = - 10)) # Change position downwards
  df <- read_csv("{db_path}")
  p1 <- ggplot(df[df$healthy=={healthy_indicator},], aes(x=age_at_MRI, y=predicted_age_cole_bias_corrected)) +
    geom_point(alpha=0.1, colour="dodgerblue1", pch=19, position = position_jitter(width = .5)) +
    stat_density2d(geom="density2d", aes(alpha=..level..), colour="dodgerblue4", size=1.5, contour=TRUE) +
    #geom_smooth(method="lm", colour="black", se=TRUE, size=1.5, level=0.99)+
    labs(x = "\nChronological age (years)", "/n")+
    labs(y = "Bias-adjusted predicted age (years)\n")+
    geom_abline()
  res <- ggExtra::ggMarginal(
    p = p1,
    type = 'density',
    margins = 'y',
    size = 5,
    colour = 'black',
    fill = 'grey'
  )
  ggsave("{home + f"/cardiac/Ageing/gnn_paper/figures/pred_age_delta/pred_age_vs_chron_age_ggplot_{healthy_str}.pdf"}", res)
  p1 <- ggplot(df[df$healthy=={healthy_indicator},], aes(x=age_at_MRI, y=delta_cole_bias_corrected)) +
    geom_point(alpha=0.1, colour="dodgerblue1", pch=19, position = position_jitter(width = .5)) +
    stat_density2d(geom="density2d", aes(alpha=..level..), colour="dodgerblue4", size=1.5, contour=TRUE) +
    #geom_smooth(method="lm", colour="black", se=TRUE, size=1.5, level=0.99)+
    labs(x = "\nChronological age (years)", "/n")+
    labs(y = "Bias-adjusted age delta (years)\n")
  res <- ggExtra::ggMarginal(
    p = p1,
    type = 'density',
    margins = 'y',
    size = 5,
    colour = 'black',
    fill = 'grey'
  )
  ggsave("{home + f"/cardiac/Ageing/gnn_paper/figures/pred_age_delta/age_delta_vs_chron_age_ggplot_{healthy_str}.pdf"}", res)
  """)

# Propensity matching
importr('MatchIt')
pandas2ri.activate()
df_matched = ro.r(f"""
dfr <- read.table("{db_path}", header = TRUE, sep = ",")
mod_match <- matchit(I(1-healthy) ~ age_at_MRI + sex,
                     method = "nearest", data = dfr)
dta_m <- match.data(mod_match)
""")

db_path_matched = home+'/cardiac/Ageing/gnn_paper/datasets/predicted_age_propensity_matched.csv'
df_matched.to_csv(db_path_matched, index=False)


# Regressions
reg = smf.ols(formula='delta_cole_bias_corrected ~ I(1-healthy) + sex', data = df_matched).fit(use_t=False)
print(reg.summary())
print(reg.summary().as_latex())

#reg = smf.ols(formula='delta_not_corrected ~ I(1-healthy) + sex + age_at_MRI', data = df).fit(use_t=False)
#print(reg.summary())
#print(reg.summary().as_latex())
