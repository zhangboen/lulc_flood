from src.plot_utils import *
from pathlib import Path
import json
from scipy import stats

dir_Qmax7 = Path('../results/run_Qmax7_onlyUrban_0506_1359_seed824956/')
dir_Qmin7 = Path('../results/run_Qmin7_onlyUrban_0506_1357_seed220973/')

Qmin7Fname = '../data/Qmin7_final_dataset_seasonal4.pkl'
Qmax7Fname = '../data/Qmax7_final_dataset_seasonal4.pkl'

par_map = pd.read_csv('../data/predictors.csv')

with open(dir_Qmax7 / 'cfg.json', 'r') as fp:
    cfg = json.load(fp)
    
mode = cfg['mode']
predictors = cfg['meteo_name'] + cfg['lulc_name'] + cfg['attr_name']
log = cfg['log']
m3s = cfg['m3s']
feature = cfg['feature']
model = cfg['model']
featureName = par_map.loc[par_map.par==feature,'name'].values[0]

with open(dir_Qmin7 / 'cfg.json', 'r') as fp:
    cfg_Qmin7 = json.load(fp)
cfg_Qmax7 = cfg

Qmin7Fname = dir_Qmin7 / f'{model}_{mode}_real_sensitivity_func_{feature}_diff_in_percentage.csv'
Qmax7Fname = dir_Qmax7 / f'{model}_{mode}_real_sensitivity_func_{feature}_diff_in_percentage.csv'

diff_Qmin7 = pd.read_csv(Qmin7Fname)
diff_Qmax7 = pd.read_csv(Qmax7Fname)

diff_Qmin7['Qmin7date'] = pd.to_datetime(diff_Qmin7['Qmin7date'])
diff_Qmax7['Qmax7date'] = pd.to_datetime(diff_Qmax7['Qmax7date'])

diff_Qmin7 = pd.read_pickle(cfg_Qmin7['fname']).merge(diff_Qmin7, on = ['ohdb_id','Qmin7date'])
diff_Qmax7 = pd.read_pickle(cfg_Qmax7['fname']).merge(diff_Qmax7, on = ['ohdb_id','Qmax7date'])

# calculate diff
for ssp in ['ssp1','ssp2','ssp3','ssp4','ssp5']:
    diff_Qmin7['diff_'+ssp] = ( diff_Qmin7['future_'+ssp] - diff_Qmin7['base'] ) / diff_Qmin7['base'] * 100
    diff_Qmax7['diff_'+ssp] = ( diff_Qmax7['future_'+ssp] - diff_Qmax7['base'] ) / diff_Qmax7['base'] * 100

cols = ['diff_'+b for b in ['ssp1','ssp2','ssp3','ssp4','ssp5']]
diff_Qmin7_ave = diff_Qmin7.groupby(['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']).agg(
    SSP1 = ('diff_ssp1', lambda x:x.mean()),
    SSP2 = ('diff_ssp2', lambda x:x.mean()),
    SSP3 = ('diff_ssp3', lambda x:x.mean()),
    SSP4 = ('diff_ssp4', lambda x:x.mean()),
    SSP5 = ('diff_ssp5', lambda x:x.mean()),
    p_ssp1 = ('diff_ssp1', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp2 = ('diff_ssp2', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp3 = ('diff_ssp3', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp4 = ('diff_ssp4', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp5 = ('diff_ssp5', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
).reset_index()
diff_Qmax7_ave = diff_Qmax7.groupby(['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']).agg(
    SSP1 = ('diff_ssp1', lambda x:x.mean()),
    SSP2 = ('diff_ssp2', lambda x:x.mean()),
    SSP3 = ('diff_ssp3', lambda x:x.mean()),
    SSP4 = ('diff_ssp4', lambda x:x.mean()),
    SSP5 = ('diff_ssp5', lambda x:x.mean()),
    p_ssp1 = ('diff_ssp1', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp2 = ('diff_ssp2', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp3 = ('diff_ssp3', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp4 = ('diff_ssp4', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp5 = ('diff_ssp5', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
).reset_index()

# change due to urbanization under SSP5
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7, 8), subplot_kw = {'projection':ccrs.EqualEarth()})
plt.subplots_adjust(hspace = .2)
ax1.set_facecolor('none')
ax2.set_facecolor('none')
for i,name in enumerate(['Qmin7','Qmax7']):
    ax = eval('ax'+str(i+1))
    df = eval('diff_'+name+'_ave')
    lons = df.ohdb_longitude.values
    lats = df.ohdb_latitude.values
    vals = df['SSP5'].values

    if i == 0:
        vmin, vmax, vind = -40, 40, 4
    else:
        vmin, vmax, vind = -20, 20, 2
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    if i == 0:
        title = f'Potential low flow change from realistic urbanization under SSP5'
    else:
        title = f'Potential high flow change from realistic urbanization under SSP5'
    label = f'{name} in $\Delta$%'
    _, ras = plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, norm = norm, fontSize = 11, size = 3)
    # add colorbar
    cax = ax.inset_axes([.35, .02, 0.25, .03])
    cbar = plt.colorbar(ras, cax = cax, orientation = 'horizontal', extend = 'both')
    cax.tick_params(labelsize = 10)
    cax.set_title(label, size = 10, pad = 5)
    # add boxplot to show the impact for dry (AI<1) and wet (AI>1) catchments
    df['tmp'] = np.where(df['aridity']>0.65, 'wet', 'dry')
    
    ttest = stats.ttest_ind(df.loc[df.tmp=='wet','SSP5'].values, df.loc[df.tmp=='dry','SSP5'].values)

    print('fraction of significant gauges:', df.loc[df.p_ssp5<=0.05,:].shape[0] / df.shape[0] * 100)
    print('average diff of significant gauges:', df.loc[df.p_ssp5<=0.05,'SSP5'].mean())
    print(df.loc[df.tmp=='wet','SSP5'].mean(), df.loc[df.tmp=='wet','SSP5'].std())
    print(df.loc[df.tmp=='dry','SSP5'].mean(), df.loc[df.tmp=='dry','SSP5'].std())
    
    axin = ax.inset_axes([0.08, .05, .1, .3])
    sns.boxplot(df.loc[df.p_ssp5<=0.01,:], 
                x = 'tmp', y = 'SSP5', ax = axin, 
                showfliers = False, width = .7, 
                whis = [5, 95],
                color = '#c2dcea',
                showmeans = True,
                capprops = {'linewidth': 0},
                boxprops={'edgecolor': 'none'},  # No edge line
                meanprops={'marker': 'o',
                    'markerfacecolor': 'white',
                    'markeredgecolor': 'black',
                    'markersize': '3'},
                medianprops={'color': 'black', 'linewidth': 1},  # Black median line
                whiskerprops={'color': 'black', 'linewidth': 1},  # Black whiskers
            )
    axin.set_facecolor('none')
#     axin.set_yticks([0,9])
    axin.set_xlabel(None)
    axin.set_ylabel(f'{name} in $\Delta$%', fontsize = 8)
    axin.tick_params(labelsize = 8,  which='both', top = False, right = False)
    axin.xaxis.set_minor_locator(ticker.NullLocator())
    axin.spines["top"].set_visible(False) 
    axin.spines["right"].set_visible(False) 

# changes under different SSPs
ax3 = ax1.inset_axes([1.1, .1, .4, .9])
ax4 = ax2.inset_axes([1.1, .1, .4, .9])
for i,name in enumerate(['Qmin7','Qmax7']):
    ax0 = [ax3,ax4][i]
    df0 = eval('diff_'+name+'_ave')
    
    df0 = df0.melt(id_vars = ['ohdb_id','aridity'], value_vars = ['SSP1','SSP2','SSP3','SSP4','SSP5'])

    palette = {'SSP1':'#169ac8','SSP2':'#17266d','SSP3':'#d8c036','SSP4':'#d87b36','SSP5':'#a01d09'}
    sns.scatterplot(data = df0, x = 'aridity', y = 'value', hue = 'variable', ax = ax0, palette = palette, markerscale = 2, fontsize = 11)
    sns.move_legend(ax0, 'upper right', title = None)
    ax0.set_xlabel('Catchment aridity', fontsize = 10)
    ax0.set_ylabel(f'{name} in $\Delta$%', fontsize = 10)
    ax0.tick_params(axis = 'both', labelsize = 10)

# add subplot order
fig.text(.15, .9, 'a', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
fig.text(.88, .9, 'b', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
fig.text(.15,  .5, 'c', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
fig.text(.88,  .5, 'd', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)

fig.savefig(dir_Qmax7 / 'fig3.png', dpi = 600)