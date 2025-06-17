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
    diff_Qmin7['Qmin7_diff_'+ssp] = ( diff_Qmin7['Qmin7_'+ssp] - diff_Qmin7['Qmin7_base'] ) / diff_Qmin7['Qmin7_base'] * 100
    diff_Qmax7['Qmax7_diff_'+ssp] = ( diff_Qmax7['Qmax7_'+ssp] - diff_Qmax7['Qmax7_base'] ) / diff_Qmax7['Qmax7_base'] * 100

cols = ['diff_'+b for b in ['ssp1','ssp2','ssp3','ssp4','ssp5']]
diff_Qmin7_ave = diff_Qmin7.groupby(['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']).agg(
    Qmin7_1 = ('Qmin7_diff_ssp1', lambda x:x.mean()),
    Qmin7_2 = ('Qmin7_diff_ssp2', lambda x:x.mean()),
    Qmin7_3 = ('Qmin7_diff_ssp3', lambda x:x.mean()),
    Qmin7_4 = ('Qmin7_diff_ssp4', lambda x:x.mean()),
    Qmin7_5 = ('Qmin7_diff_ssp5', lambda x:x.mean()),
    urban1 = ('urban_diff_ssp1', lambda x:x.mean()),
    urban2 = ('urban_diff_ssp2', lambda x:x.mean()),
    urban3 = ('urban_diff_ssp3', lambda x:x.mean()),
    urban4 = ('urban_diff_ssp4', lambda x:x.mean()),
    urban5 = ('urban_diff_ssp5', lambda x:x.mean()),
    p_ssp1 = ('Qmin7_diff_ssp1', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp2 = ('Qmin7_diff_ssp2', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp3 = ('Qmin7_diff_ssp3', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp4 = ('Qmin7_diff_ssp4', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp5 = ('Qmin7_diff_ssp5', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
).reset_index()
diff_Qmax7_ave = diff_Qmax7.groupby(['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']).agg(
    Qmax7_1 = ('Qmax7_diff_ssp1', lambda x:x.mean()),
    Qmax7_2 = ('Qmax7_diff_ssp2', lambda x:x.mean()),
    Qmax7_3 = ('Qmax7_diff_ssp3', lambda x:x.mean()),
    Qmax7_4 = ('Qmax7_diff_ssp4', lambda x:x.mean()),
    Qmax7_5 = ('Qmax7_diff_ssp5', lambda x:x.mean()),
    urban1 = ('urban_diff_ssp1', lambda x:x.mean()),
    urban2 = ('urban_diff_ssp2', lambda x:x.mean()),
    urban3 = ('urban_diff_ssp3', lambda x:x.mean()),
    urban4 = ('urban_diff_ssp4', lambda x:x.mean()),
    urban5 = ('urban_diff_ssp5', lambda x:x.mean()),
    p_ssp1 = ('Qmax7_diff_ssp1', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp2 = ('Qmax7_diff_ssp2', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp3 = ('Qmax7_diff_ssp3', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp4 = ('Qmax7_diff_ssp4', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
    p_ssp5 = ('Qmax7_diff_ssp5', lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue),
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
    vals = df[f'{name}_5'].values

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
    
    ttest = stats.ttest_ind(df.loc[df.tmp=='wet',f'{name}_5'].values, df.loc[df.tmp=='dry',f'{name}_5'].values)

    print('fraction of significant gauges:', df.loc[df.p_ssp5<=0.05,:].shape[0] / df.shape[0] * 100)
    print('average diff of significant gauges:', df.loc[df.p_ssp5<=0.05,f'{name}_5'].mean())
    print(df.loc[df.tmp=='wet',f'{name}_5'].mean(), df.loc[df.tmp=='wet',f'{name}_5'].std())
    print(df.loc[df.tmp=='dry',f'{name}_5'].mean(), df.loc[df.tmp=='dry',f'{name}_5'].std())
    
    axin = ax.inset_axes([0.08, .05, .1, .3])
    sns.boxplot(df.loc[df.p_ssp5<=0.01,:], 
                x = 'tmp', y = f'{name}_5', ax = axin, 
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

# scatters between urban expansion fractions and streamflow extremes change under SSP5
ax3 = ax1.inset_axes([1.1, .1, .5, .8])
ax4 = ax2.inset_axes([1.1, .1, .5, .8])
for i,name in enumerate(['Qmin7','Qmax7']):
    ax0 = [ax3,ax4][i]
    df0 = eval('diff_'+name+'_ave')
    df0['catch'] = np.where(df0['aridity']>0.65, 'Catchments in wet climate', 'Catchments in dry climate')
    palette0 = {'Catchments in wet climate':"#65C2A5",'Catchments in dry climate':'#C7B18A'}
    sns.scatterplot(data = df0, 
                    x = 'urban5', 
                    y = f'{name}_5', 
                    hue = 'catch', 
                    alpha = .3,
                    palette = palette0,
                    legend = False,
                    ax = ax0)
    for catch0 in df0.catch.unique():
        sns.regplot(df0.loc[df0.catch==catch0,:],
                    x = 'urban5',
                    y = f'{name}_5', 
                    scatter = False,
                    color = palette0[catch0],
                    ax = ax0, 
                    line_kws={"lw": 2},
                    robust=True
                   )
    
    # legend
    line1 = Line2D([], [], color=palette0['Catchments in dry climate'], ls="-", linewidth=1.5)
    line2 = Line2D([], [], color=palette0['Catchments in wet climate'], ls="-", linewidth=1.5)
    sc1 = plt.scatter([],[], s=15, facecolors=palette0['Catchments in dry climate'], edgecolors=palette0['Catchments in dry climate'])
    sc2 = plt.scatter([],[], s=15, facecolors=palette0['Catchments in wet climate'], edgecolors=palette0['Catchments in wet climate'])
    ax0.legend([(sc1,line1), (sc2,line2)], ['Catchments in dry climate','Catchments in wet climate'], numpoints=1, handlelength = 1, fontsize = 9)
    
    ax0.set_xlabel('Urban expansion (%)', fontsize = 10)
    ax0.set_ylabel(f'{name} in $\Delta$%', fontsize = 10)
    ax0.tick_params(axis = 'both', labelsize = 10)
    ax0.set_ylim(-50, df0[f'{name}_5'].quantile(.999))

    # add upper and right axes to plot data distribution
    ax_upper = ax0.inset_axes([0, 1.01, 1, .1])
    ax_right = ax0.inset_axes([1.01, 0, .1, 1])
    sns.boxplot(data = df0, 
                x = 'urban5', 
                ax=ax_upper, 
                hue = 'catch', 
                legend = False, 
                palette = palette0, 
                showfliers = False, 
                whis = (2.5, 97.5),
                capprops = dict(linewidth=0), 
                width = 1)
    sns.boxplot(data = df0, 
                y = f'{name}_5', 
                ax=ax_right, 
                hue = 'catch', 
                legend = False, 
                whis = (2.5, 97.5),
                palette = palette0, 
                showfliers = False, 
                capprops = dict(linewidth=0), 
                width = 1)

    ax_right.set_ylim(ax0.get_ylim())
    ax_upper.set_xlim(ax0.get_xlim())
    ax_upper.xaxis.label.set_visible(False)
    ax_right.yaxis.label.set_visible(False)
    for side in ['top','right','bottom','left']:
        ax_right.spines[side].set_visible(False)
        ax_upper.spines[side].set_visible(False)
    ax_right.tick_params(axis='both',which='both',labelleft=False,bottom=False,left=False,right=False,top=False)
    ax_upper.tick_params(axis='both',which='both',labelbottom=False,bottom=False,left=False,right=False,top=False)

# add subplot order
fig.text(.15, .9, 'a', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
fig.text(.88, .9, 'b', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
fig.text(.15,  .5, 'c', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
fig.text(.88,  .5, 'd', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)

fig.savefig(dir_Qmax7 / 'fig3.png', dpi = 600)