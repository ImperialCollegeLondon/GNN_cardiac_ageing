import numpy as np
import pandas as pd
from joypy import joyplot
import matplotlib.pylab as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

cnames_groups = [
['Asc aorta dist', 'Asc aorta max. area', 'Asc aorta min. area', 'Desc aorta dist', 'Desc aorta max. area', 'Desc aorta min. area', 'L atrial max. vol', 'L atrial min. vol'],
['LAEF', 'LASV', 'LVCO', 'LVEDV', 'LVEF', 'LVESV', 'LVM', 'LVSV'],
['T1 (septum)', 'Long SR PC1', 'Long SR PC2', 'R atrium max. volume', 'RA min. vol', 'Radial SR PC1', 'Radial SR PC2'],
['RAEF', 'RASV', 'RVEDV', 'RVEF', 'RVESV', 'RVSV']
]

for cgid, cnames in enumerate(cnames_groups):
        
    df = pd.read_csv('/root/cardiac/Ageing/marco/original_db_with_principal_components.csv')
    #cnames = [c for c in df.columns if c != 'sex']
    scaler = StandardScaler()
    df[cnames] = scaler.fit_transform(df[cnames])
    for cname in cnames:
        df.loc[df[cname] < -2, cname] = np.nan
        df.loc[df[cname] > 2, cname] = np.nan
    #df[cnames] = scaler.inverse_transform(df[cnames])
    
    if cgid == 2:
        df.rename(inplace=True, columns={
            'T1 (septum)': 'AT1 (septum)',
            'Long SR PC1': 'BLong SR PC1' ,
            'Long SR PC2': 'BLong SR PC2' ,
            'Radial SR PC1': 'BRadial SR PC1' ,
            'Radial SR PC2': 'BRadial SR PC2'
        })
        cnames = ['AT1 (septum)',  'BLong SR PC1', 'BLong SR PC2', 'R atrium max. volume', 'RA min. vol', 'BRadial SR PC1', 'BRadial SR PC2']

    df_expanded = np.empty((len(df)*len(cnames), 3))+np.nan
    df_expanded = np.array(df_expanded, dtype=object)

    count = 0
    for i in tqdm(range(len(df))):
        for cname in cnames:
            df_expanded[count, 2] = cname
            
            if df.sex.iloc[i] == 1:
                df_expanded[count, 0] = df[cname].iloc[i]
            elif df.sex.iloc[i] == 0:
                df_expanded[count, 1] = df[cname].iloc[i]
            else:
                raise ValueError
                
            count += 1
    df_expanded = pd.DataFrame(df_expanded, columns=['Male', 'Female', 'cname'])
    df_expanded = df_expanded.convert_dtypes()

    ax, fig = joyplot(
        data=df_expanded,
        alpha=0.85,
        figsize=(12, 8),
        column = ['Male', 'Female'],
        by='cname',
        xlabelsize=25,
        ylabelsize=25,
        color=['#7FBCC5', '#9F9FBF'],
    )

    plt.title('', fontsize=20)
    plt.savefig(f"/root/cardiac/Ageing/marco/joyplot_{cgid}.pdf")
    plt.close()

# idea from https://stackoverflow.com/a/54870776
plt.subplots()
colors = ['#7FBCC5', '#9F9FBF']
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(2)]
labels = ['Male', 'Female']
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, ncol=2)

fig  = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(f"/root/cardiac/Ageing/marco/joyplot_legend.pdf", dpi="figure", bbox_inches=bbox)


