from src.plot_utils import *
import pickle
import pymannkendall as mk
from scipy.interpolate import interp1d
from scipy import stats
from tqdm import tqdm
import multiprocessing as mp
import statsmodels.api as sm
from parallel_pandas import ParallelPandas
from pathlib import Path
import json
import statsmodels.api as sm
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

if mode == 'noLULC':
    predictors1 = [item for item in predictors if item not in ['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland']]
elif  mode == 'onlyUrban':
    predictors1 = [item for item in predictors if item not in ['forest', 'crop', 'grass', 'water', 'wetland']]

par_map = pd.read_csv('../data/predictors.csv')

df_Qmin7 = pd.read_pickle(cfg_Qmin7['fname'])
df_Qmax7 = pd.read_pickle(cfg_Qmax7['fname'])

df_Qmin7 = df_Qmin7[["ohdb_id", "Qmin7date", "ohdb_longitude", "ohdb_latitude", 'climate_label', 'aridity', feature]]
df_Qmax7 = df_Qmax7[["ohdb_id", "Qmax7date", "ohdb_longitude", "ohdb_latitude", 'climate_label', 'aridity', feature]]

df_Qmin7 = df_Qmin7.rename(columns={'ohdb_longitude':'lon','ohdb_latitude':'lat', feature:feature+'_val'})
df_Qmax7 = df_Qmax7.rename(columns={'ohdb_longitude':'lon','ohdb_latitude':'lat', feature:feature+'_val'})

def load_shap(fname):
    shap0 = pickle.load(open(fname,'rb'))
    df0 = pd.DataFrame(data = shap0, columns = predictors1)
    df0[feature+'_rank'] = df0.abs().apply(lambda x: x.rank(ascending=False,method='min').loc[feature], axis = 1)
    df0 = df0[[feature,feature+'_rank']].rename(columns={feature:feature+'_shap'})
    if 'Qmax7' in str(fname):
        df0 = pd.concat([df_Qmax7, df0], axis = 1)
    else:
        df0 = pd.concat([df_Qmin7, df0], axis = 1)
    return df0

# load shap and predictors
Qmax7_fnames = dir_Qmax7.glob(f'{model}_{mode}_shap_values_explain_[0-9][0-9]*.pkl')
with mp.Pool(processes=8) as pool:
    df_Qmax7 = tqdm(pool.map(load_shap, Qmax7_fnames))
df_Qmax7 = pd.concat(df_Qmax7)

Qmin7_fnames = dir_Qmin7.glob(f'{model}_{mode}_shap_values_explain_[0-9]*.pkl')
with mp.Pool(processes=8) as pool:
    df_Qmin7 = tqdm(pool.map(load_shap, Qmin7_fnames))
df_Qmin7 = pd.concat(df_Qmin7)

# calculate average
ParallelPandas.initialize(n_cpu=24, split_factor=24)
df_Qmax7 = df_Qmax7.groupby(['ohdb_id', "Qmax7date", 'lon','lat','climate_label','aridity', feature+'_val'], group_keys=False).mean().reset_index()
df_Qmin7 = df_Qmin7.groupby(['ohdb_id', "Qmin7date", 'lon','lat','climate_label','aridity', feature+'_val'], group_keys=False).mean().reset_index()

df_Qmin7['catch'] = np.where(df_Qmin7.aridity<=0.65, 'dry', 'wet')
df_Qmax7['catch'] = np.where(df_Qmax7.aridity<=0.65, 'dry', 'wet')

df_Qmin7_ave = df_Qmin7.groupby(['ohdb_id','lon','lat','climate_label','aridity'], group_keys=False)[[feature+'_shap',feature+'_rank']].p_apply(
    lambda x:pd.Series(
        [x[feature+'_shap'].abs().mean(), x[feature+'_rank'].mean()], index = [feature+'_shap',feature+'_rank']
    )
).reset_index()
df_Qmax7_ave = df_Qmax7.groupby(['ohdb_id','lon','lat','climate_label','aridity'], group_keys=False)[[feature+'_shap',feature+'_rank']].p_apply(
    lambda x:pd.Series(
        [x[feature+'_shap'].abs().mean(), x[feature+'_rank'].mean()], index = [feature+'_shap',feature+'_rank']
    )
).reset_index()

def plot_shap_dependence(ax1, ax2, df_Qmin7, df_Qmax7, title = False):
    palette0 = {'dry':'#C7B18A', 'wet':"#65C2A5"}
    sns.scatterplot(df_Qmin7, x = feature+'_val', y = feature+'_shap', hue = 'catch', ax = ax1, alpha = .2, palette = palette0)
    sns.scatterplot(df_Qmax7, x = feature+'_val', y = feature+'_shap', hue = 'catch', ax = ax2, alpha = .2, palette = palette0)

    for i,name in enumerate(['Qmin7','Qmax7']):
        ax = eval('ax'+str(i+1))
        ax.set_xlabel(featureName.capitalize()+' (%)', fontsize = 9)
        ax.set_ylabel('SHAP value', fontsize = 9)
        sns.move_legend(ax, 'lower right', bbox_to_anchor = (.98, .02), title = None, fontsize = 9)
        if title:
            if i == 0:
                ax.set_title('Dependence of low flow to '+featureName, fontsize = 9)
            else:
                ax.set_title('Dependence of high flow to '+featureName, fontsize = 9)
        ax.tick_params(axis = 'both', labelsize = 9)

        # estimate LOWESS
        df0 = eval('df_'+name)
        xvals = np.linspace(df0[feature+'_val'].min(), df0[feature+'_val'].max(), 100)
        ps = []
        for catch in ['dry','wet']:
            lowess0 = sm.nonparametric.lowess(
                df0.loc[df0.catch==catch,feature+'_shap'].values, 
                df0.loc[df0.catch==catch,feature+'_val'].values, 
                xvals=xvals, 
                frac=0.1, 
                return_sorted = True)   
            p0 = ax.plot(xvals, lowess0, color = palette0[catch], lw = 1, label = catch + ' climate')
            ps.append(p0)
        
        # legend
        line1 = Line2D([], [], color=palette0['dry'], ls="-", linewidth=1.5)
        line2 = Line2D([], [], color=palette0['wet'], ls="-", linewidth=1.5)
        sc1 = plt.scatter([],[], s=15, facecolors=palette0['dry'], edgecolors=palette0['dry'])
        sc2 = plt.scatter([],[], s=15, facecolors=palette0['wet'], edgecolors=palette0['wet'])
        ax.legend([(sc1,line1), (sc2,line2)], ['Catchments in dry climate','Catchments in wet climate'], numpoints=1, handlelength = 1, fontsize = 9)

def plot_shap_rank(ax1, ax2, df_Qmin7_ave, df_Qmax7_ave, title = False):
    for i,name in enumerate(['Qmin7','Qmax7']):
        ax = eval('ax'+str(i+1))
        df = eval('df_'+name+'_ave')
        lons = df.lon.values
        lats = df.lat.values
        vals = df[feature+'_rank'].values

        vmin, vmax, vind = 1, 20, 1
        cmap = plt.cm.plasma
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
        if title:
            if i == 0:
                title = f'Importance ranking of {featureName.lower()} in ML of low flows'
            else:
                title = f'Importance ranking of {featureName.lower()} in ML of high flows'
        else:
            title = None
        label = f'SHAP ranking'
        _, ras = plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, norm = norm, fontSize = 9, size = 3, addHist = False)
        # add colorbar
        cax = ax.inset_axes([.3, .02, 0.2, .03])
        cbar = plt.colorbar(ras, cax = cax, orientation = 'horizontal', extend = 'both')
        cax.tick_params(labelsize = 9)
        cax.set_title(label, size = 9, pad = 5)
        # add boxplot to show the impact for dry (AI<1) and wet (AI>1) catchments
        df['tmp'] = np.where(df['aridity']>0.65, 'wet', 'dry')

        axin = ax.inset_axes([0.71, .05, .08, .3])
        sns.boxplot(df, 
                    x = 'tmp', y = feature+'_rank', ax = axin, 
                    showfliers = False, width = .6, 
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
        axin.set_ylabel(f'SHAP ranking', fontsize = 9)
        axin.tick_params(labelsize = 9,  which='both', top = False, right = False)
        axin.xaxis.set_minor_locator(ticker.NullLocator())
        axin.spines["top"].set_visible(False) 
        axin.spines["right"].set_visible(False) 

def plot_ale(ax1, ax2, cfg_Qmin7, cfg_Qmax7, title = False):

    model = cfg_Qmin7['model']
    mode = cfg_Qmin7['mode']
    feature = cfg_Qmin7['feature']
    purpose = 'ale'
    min_interval = 0
    Qmin7Fname = dir_Qmin7 / f'{model}_{mode}_{purpose}_{feature}_min_interval_{min_interval:.1f}.csv'
    Qmax7Fname = dir_Qmax7 / f'{model}_{mode}_{purpose}_{feature}_min_interval_{min_interval:.1f}.csv'

    df_eff_Qmin7 = pd.read_csv(Qmin7Fname)
    df_eff_Qmax7 = pd.read_csv(Qmax7Fname)

    for i,name in enumerate(['Qmin7','Qmax7']):
        df = eval('df_eff_'+name)
        ax = eval('ax'+str(i+1))
        for climate in df.climate_label.unique():
            tmp = df.loc[df.climate_label==climate,:]

            # Find common x-axis range
            min_x = tmp['bins'].min()
            max_x = tmp['bins'].max()

            groupName = 'n_explain'

            n_sample = tmp.groupby(groupName)['eff'].count().min()
            x_range = np.linspace(min_x, max_x, n_sample)  # Adjust number of points as needed

            # Interpolate each group to common x-axis
            interpolated_data = {}
            for group in tmp[groupName].unique():
                group_df =tmp[tmp[groupName] == group]
                f = interp1d(group_df['bins'], group_df['eff'], kind='linear', fill_value='extrapolate')
                interpolated_data[group] = pd.DataFrame({'x': x_range, 'y': f(x_range), 'group': group})

            # Concatenate interpolated data
            interpolated_df = pd.concat(interpolated_data.values())

            # Calculate average y values for each x
            df_avg = interpolated_df.groupby('x').apply(
                lambda x: pd.Series([x.y.mean(), x.y.quantile(.025), x.y.quantile(.975)], index = ['ave','low','upp'])
            ).reset_index()

            # Create the lineplot with individual lines
            ax.plot(df_avg.x.values, df_avg.ave.values, color = palette[climate], lw = 2, label = climate, zorder = 3)
            ax.fill_between(
                df_avg.x.values, 
                df_avg.low.values, 
                df_avg.upp.values, 
                color = palette[climate], 
                ec = 'none',
                alpha = .3)

            ax.legend(fontsize = 9)

        ax.set_xlabel(f'{featureName} (%)', fontsize = 9)
        ax.set_ylabel(f'{name} in $\Delta$%', fontsize = 9)
        if title:
            if i == 0:
                ax.set_title(f'Effects of {featureName.lower()} on low flow', fontsize = 9)
            else:
                ax.set_title(f'Effects of {featureName.lower()} on high flow', fontsize = 9)
        ax.tick_params(labelsize=9)

def plot_scatter_shap_aridity(ax1, ax2, df_Qmin7_ave, df_Qmax7_ave, title = False):
    sns.scatterplot(df_Qmin7_ave, x = 'aridity', y = feature+'_shap', ax = ax1)
    sns.scatterplot(df_Qmax7_ave, x = 'aridity', y = feature+'_shap', ax = ax2)

    ax1.set_xlabel('Catchment aridity', fontsize = 9)
    ax2.set_xlabel('Catchment aridity', fontsize = 9)

    ax1.set_ylabel('Absolute SHAP values', fontsize = 9)
    ax2.set_ylabel('Absolute SHAP values', fontsize = 9)

    ax1.tick_params(labelsize=9)
    ax2.tick_params(labelsize=9)

if __name__ == '__main__':
    fig, axes = plt.subplots(2, 3, figsize = (10, 6))
    plt.subplots_adjust(wspace = .3, hspace = .25)
    
    plot_ale(axes[0,0], axes[1,0], cfg_Qmin7, cfg_Qmax7)
    
    plot_shap_dependence(axes[0,1], 
                         axes[1,1], 
                         df_Qmin7, 
                         df_Qmax7,
                        )
    
    plot_scatter_shap_aridity(axes[0,2], axes[1,2], df_Qmin7_ave, df_Qmax7_ave)

    ax1 = inset_axes(
        axes[1,0],
        width = "140%",
        height = "130%",
        loc='lower left',
        axes_class=GeoAxes,
        bbox_transform=axes[1,0].transAxes,
        axes_kwargs=dict(projection=ccrs.EqualEarth()),
        bbox_to_anchor=(-.28, -2, 1.5, 1.5),
    )
    ax1.set_facecolor('none')
    ax2 = ax1.inset_axes([.9, 0, 1, 1], projection =  ccrs.EqualEarth())
    ax2.set_facecolor('none')

    plot_shap_rank(ax1, ax2, df_Qmin7_ave, df_Qmax7_ave, title = True)
    
    import string
    for i,ax in enumerate(axes[:2,:].ravel(order='F')):
        ax.text(-.2, 1, string.ascii_letters[i], weight = 'bold', transform = ax.transAxes, fontsize = 10)
    
    ax1.text(0, 1, 'g', weight = 'bold', transform = ax1.transAxes, fontsize = 10)
    ax2.text(0, 1, 'h', weight = 'bold', transform = ax2.transAxes, fontsize = 10)
    
    # add arrow to indicate dry/wet direction
    for ax2 in [axes[0,2],axes[1,2]]:
        ax2.annotate("Dry", xy=(.65, .75), xytext=(.55, .75),
                    arrowprops=dict(arrowstyle = '<|-', color = 'k'), va = 'center', ha = 'right',
                    textcoords = 'axes fraction', xycoords = 'axes fraction', fontsize = 9, color = '#d88228')
        ax2.annotate("Wet", xy=(.66, .75), xytext=(.76, .75),
                    arrowprops=dict(arrowstyle = '<|-', color = 'k'), va = 'center', ha = 'left',
                    textcoords = 'axes fraction', xycoords = 'axes fraction', fontsize = 9, color = '#287ed8')
    
    fig.savefig(dir_Qmax7 / 'fig1.png', dpi = 600)