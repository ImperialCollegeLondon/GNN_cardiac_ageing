import numpy as np
import pandas as pd
from joypy import joyplot
import matplotlib.pylab as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA

cnames_groups = [
['WTG (mm)', 'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVCO', 'LVM'],
['Ahron. age',  "BMI (kg/m2)", "DBP (mmHg)", "PR (bpm)", "CBP (mmHg)"],
['rad_2', 'rad_3', 'rad_4', 'rad_5', 'rad_6', 'rad_7', 'rad_8', 'rad_9', 'rad_10', 'rad_11', 'rad_12', 'rad_13', 'rad_14', 'rad_15', 'rad_16', 'rad_17', 'rad_18', 'rad_19', 'rad_20', 'rad_21', 'rad_22', 'rad_23', 'rad_24', 'rad_25', 'rad_26', 'rad_27', 'rad_28', 'rad_29', 'rad_30', 'rad_31', 'rad_32', 'rad_33', 'rad_34', 'rad_35', 'rad_36', 'rad_37', 'rad_38', 'rad_39', 'rad_40', 'rad_41', 'rad_42', 'rad_43', 'rad_44', 'rad_45', 'rad_46', 'rad_47', 'rad_48', 'rad_49', 'rad_50', 'long_2', 'long_3', 'long_4', 'long_5', 'long_6', 'long_7', 'long_8', 'long_9', 'long_10', 'long_11', 'long_12', 'long_13', 'long_14', 'long_15', 'long_16', 'long_17', 'long_18', 'long_19', 'long_20', 'long_21', 'long_22', 'long_23', 'long_24', 'long_25', 'long_26', 'long_27', 'long_28', 'long_29', 'long_30', 'long_31', 'long_32', 'long_33', 'long_34', 'long_35', 'long_36', 'long_37', 'long_38', 'long_39', 'long_40', 'long_41', 'long_42', 'long_43', 'long_44', 'long_45', 'long_46', 'long_47', 'long_48', 'long_49', 'long_50'],
]

curdir = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(curdir, '../../data')
home = os.path.expanduser("~")

df_healthy = os.path.join(datadir, 'healthyageinggroup_t1.csv')
df_healthy = pd.read_csv(df_healthy)

df_unhealthy = os.path.join(datadir, 'nonhealthyageinggroup_t1.csv')
df_unhealthy = pd.read_csv(df_unhealthy)

ids_healthy_train = 'healthy_train_ids.csv'
ids_healthy_train = pd.read_csv(ids_healthy_train)

ids_healthy_test = 'healthy_test_ids.csv'
ids_healthy_test = pd.read_csv(ids_healthy_test)

ids_unhealthy = 'predicted_age.csv'
ids_unhealthy = pd.read_csv(ids_unhealthy)

df_healthy = df_healthy[df_healthy['eid_18545'].isin(ids_healthy_train['eid_18545']) | df_healthy['eid_18545'].isin(ids_healthy_test['eid_18545'])]

df_unhealthy = df_unhealthy[df_unhealthy['eid_40616'].isin(ids_unhealthy['eid_40616'])]

df_healthy['healthy'] = 1
df_unhealthy['healthy'] = 0
dfu = pd.concat((df_healthy, df_unhealthy), join='inner')

dfe1 = 'phenotype_ukbb_all_imaging_instance.csv'
dfe1 = pd.read_csv(dfe1)
dfe1 = dfe1[["eid_40616", "SBP", "DBP", "Pulse"]]
dfe1.columns = ["eid_40616", "SBP (mmHg)", "DBP (mmHg)", "PR (bpm)"]

dfe2 = 'ukbbMetaData_Kathryn_Copy.csv'
dfe2 = pd.read_csv(dfe2)
dfe2 = dfe2[["eid_40616", "bmi"]]
dfe2.columns = ["eid_40616", "BMI (kg/m2)"]

dfu = dfu.join(dfe1.set_index('eid_40616'), on='eid_40616')
dfu = dfu.join(dfe2.set_index('eid_40616'), on='eid_40616')

dfu.rename(inplace=True, columns={
    'age_at_MRI': 'Ahron. age',
    'WT_Global_mm': 'WTG (mm)',
    'SBP (mmHg)': 'CBP (mmHg)',
})

for cgid, cnames in enumerate(cnames_groups):
    df = dfu.copy()

    if cgid == 2:
        df = df[['healthy']+cnames]
        pca = PCA(n_components=3)
        df1 = pca.fit_transform(df[[c for c in df.columns if c.startswith('rad')]])
        df2 = pca.fit_transform(df[[c for c in df.columns if c.startswith('long')]])
        df = pd.concat((df[['healthy']].reset_index(drop=True), pd.DataFrame(df1, columns=['Rad SR PC1', 'Rad SR PC2', 'Rad SR PC3']), pd.DataFrame(df2, columns=['Long SR PC1', 'Long SR PC2', 'Long SR PC3'])), axis=1)
        cnames = df.columns[1:]

    #cnames = [c for c in df.columns if c != 'sex']
    scaler = StandardScaler()
    df[cnames] = scaler.fit_transform(df[cnames])
    for cname in cnames:
        df.loc[df[cname] < -2, cname] = np.nan
        df.loc[df[cname] > 2, cname] = np.nan
    #df[cnames] = scaler.inverse_transform(df[cnames])

    df_expanded = np.empty((len(df)*len(cnames), 3))+np.nan
    df_expanded = np.array(df_expanded, dtype=object)

    count = 0
    for i in tqdm(range(len(df))):
        for cname in cnames:
            df_expanded[count, 2] = cname

            if df.healthy.iloc[i] == 0:
                df_expanded[count, 0] = df[cname].iloc[i]
            elif df.healthy.iloc[i] == 1:
                df_expanded[count, 1] = df[cname].iloc[i]
            else:
                raise ValueError

            count += 1
    df_expanded = pd.DataFrame(df_expanded, columns=['Unhealthy', 'Healthy', 'cname'])
    #df_expanded = df_expanded.convert_dtypes()
    df_expanded = df_expanded.astype({'Unhealthy': float, 'Healthy': float, })
    ax, fig = joyplot(
        data=df_expanded,
        alpha=0.85,
        figsize=(12, 8),
        column = ['Unhealthy', 'Healthy'],
        by='cname',
        xlabelsize=25,
        ylabelsize=25,
        color=['#CD2626', '#BABABA'],
        linewidth=0.0,
    )

    plt.title('', fontsize=20)
    plt.savefig(f"joyplot_v4_{cgid}.pdf")
    plt.close()

# idea from https://stackoverflow.com/a/54870776
plt.subplots()
colors = ['#CD2626', '#BABABA']
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(2)]
labels = ['Unhealthy', 'Healthy',]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, ncol=2)

fig  = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(f"joyplot_legend_v4.pdf", dpi="figure", bbox_inches=bbox)
