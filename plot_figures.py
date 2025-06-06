#!/usr/bin/env python
# coding: utf-8

from src.plot_utils import *
import pickle
import pymannkendall as mk
from scipy.interpolate import interp1d
from scipy import stats
from tqdm import tqdm
import multiprocessing as mp
import statsmodels.api as sm
from parallel_pandas import ParallelPandas
ParallelPandas.initialize(n_cpu=24, split_factor=24)
from pathlib import Path
import json,sys

dir_Qmax7 = Path(sys.argv[1])
dir_Qmin7 = Path(sys.argv[2])

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

def plot_cv():
    Qmin7Fname = dir_Qmin7 / f'{model}_{mode}_cv10_raw_result.csv'
    Qmax7Fname = dir_Qmax7 / f'{model}_{mode}_cv10_raw_result.csv'

    try:
        df_Qmin7 = pd.read_csv(Qmin7Fname)
        df_Qmax7 = pd.read_csv(Qmax7Fname)
    except:
        print(f"Error: Cannot open {str(Qmax7Fname)}")
        return

    df_Qmin7['Qmin7date'] = pd.to_datetime(df_Qmin7['Qmin7date'])
    df_Qmax7['Qmax7date'] = pd.to_datetime(df_Qmax7['Qmax7date'])

    df_Qmin7_obs = pd.read_pickle(cfg_Qmin7['fname'])
    df_Qmax7_obs = pd.read_pickle(cfg_Qmax7['fname'])

    df_Qmin7 = df_Qmin7.merge(df_Qmin7_obs, on = ['ohdb_id','Qmin7date'])
    df_Qmax7 = df_Qmax7.merge(df_Qmax7_obs, on = ['ohdb_id','Qmax7date'])

    Qmin7Fname = dir_Qmin7 / f'{model}_{mode}_cv10_station_based_result.csv'
    Qmax7Fname = dir_Qmax7 / f'{model}_{mode}_cv10_station_based_result.csv'

    df_Qmin7_sta = pd.read_csv(Qmin7Fname)
    df_Qmax7_sta = pd.read_csv(Qmax7Fname)

    df_Qmin7_sta['R2'] = df_Qmin7_sta.r.values ** 2
    df_Qmax7_sta['R2'] = df_Qmax7_sta.r.values ** 2

    fig, ax1 = plt.subplots(figsize = (5,3), subplot_kw = {'projection':ccrs.EqualEarth()},dpi=200)
    ax2 = ax1.inset_axes([.9, 0, 1, 1], projection =  ccrs.EqualEarth())
    ax2.set_facecolor('none')
    ax1_1 = ax1.inset_axes([0, -1.2, 1, 1], projection = ccrs.EqualEarth())
    ax2_1 = ax1_1.inset_axes([.9, 0, 1, 1], projection =  ccrs.EqualEarth())
    ax2_1.set_facecolor('none')
    for i,name in enumerate(['Qmin7','Qmax7']):
        if i == 0:
            df = df_Qmin7.copy()
            legend = False
            name1 = 'low flow'
        else:
            df = df_Qmax7.copy()
            legend = True
            name1 = 'high flow'
        df_sta = eval('df_'+name+'_sta')
        
        title = f'10-fold cross-validation for {name1}'
        lons = df_sta.ohdb_longitude.values
        lats = df_sta.ohdb_latitude.values
        
        # R2
        ax = eval('ax'+str(i+1))
        vals = df_sta.R2.values
        label = '$\mathregular{R^2}$'
        cmap = plt.cm.RdBu_r
        _,ras = plot_map(ax, lons, lats, vals, 0, 1, 0.1, cmap, title, label, marker = "$\circ$", size = 1, fontSize = 9, norm = None, addHist = False)
        cax = ax.inset_axes([.55, 0.05, .3, .03])
        fig.colorbar(ras, cax = cax, orientation = 'horizontal')
        cax.set_title(label, fontsize = 8)
        
        # nRMSE
        ax = eval('ax'+str(i+1)+'_1')
        vals = df_sta.nRMSEminmax.values
        label = 'nRMSE (%)'
        cmap = plt.cm.viridis
        _,ras = plot_map(ax, lons, lats, vals, 0, 100, 10, cmap, title, label, marker = "$\circ$", size = 1, fontSize = 9, norm = None, addHist = False)
        cax = ax.inset_axes([.55, 0.05, .3, .03])
        fig.colorbar(ras, cax = cax, orientation = 'horizontal')
        cax.set_title(label, fontsize = 8)
        
        print(name, df_sta.loc[df_sta.R2>=0.3,:].shape[0] / df_sta.shape[0] * 100, 
            df_sta.loc[df_sta.KGE>=0.3,:].shape[0] / df_sta.shape[0] * 100)
        
    ax3 = ax1_1.inset_axes([0, -1.2, 1, 1])
    ax4 = ax2_1.inset_axes([0, -1.2, 1, 1])
    df_Qmin70 = df_Qmin7
    plot_scatter(df_Qmin70.Q.values, 
                df_Qmin70['pred'].values, 
                df_Qmin70.climate_label.values, 
                'Observed Qmin7' + '($\mathregular{m^3/s}$)', 
                'Predicted Qmin7' +  '($\mathregular{m^3/s}$)', 
                normColor = 'norm',
                palette = palette,
                fontsize = 9,
                size = 1,
                ax = ax3)
    ax3.set_title('10-fold cross-validation for low flow', fontsize = 9)

    df_Qmax70 = df_Qmax7
    plot_scatter(df_Qmax70.Q.values, 
                df_Qmax70['pred'].values, 
                df_Qmax70.climate_label.values, 
                'Observed Qmax7' + '($\mathregular{m^3/s}$)', 
                'Predicted Qmax7' +  '($\mathregular{m^3/s}$)', 
                normColor = 'norm',
                palette = palette,
                fontsize = 9,
                legend = False,
                size = 1,
                ax = ax4)
    sns.move_legend(ax3, 'upper left', bbox_to_anchor = (0, .99), markerscale = 5, fontsize = 8, handletextpad = .1, borderaxespad = 0)
    ax4.set_title('10-fold cross-validation for high flow', fontsize = 9)

    for i,ax in enumerate([ax1, ax2, ax1_1, ax2_1]):
        ax.text(0, 1, string.ascii_letters[i], weight = 'bold', ha = 'right', va = 'center', fontsize = 10, transform = ax.transAxes)
    for i,ax in enumerate([ax3, ax4]):
        ax.text(-.2, 1, string.ascii_letters[i+4], weight = 'bold', ha = 'right', va = 'center', fontsize = 10, transform = ax.transAxes)
        # set axis limits
        ax.set_xlim(-.1, 1e6)
        ax.set_ylim(-.1, 1e6)

    fig.savefig(dir_Qmax7 / f'figure_scatter_and_R2_map_{mode}.png', dpi = 600)

def plot_shap_importance():
    # load shap and predictors
    Qmax7_fname = dir_Qmax7 / f'{model}_{mode}_shap_values.pkl'
    Qmin7_fname = dir_Qmin7 / f'{model}_{mode}_shap_values.pkl'

    try:
        shap_Qmin7 = pickle.load(open(Qmin7_fname, 'rb'))
        shap_Qmax7 = pickle.load(open(Qmax7_fname, 'rb'))
    except:
        print(f"Error: Cannot open {str(Qmax7_fname)}")
        return

    if mode == 'noLULC':
        predictors1 = [item for item in predictors if item not in ['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland']]
    elif mode == 'onlyUrban':
        predictors1 = [item for item in predictors if item not in ['forest', 'crop', 'grass', 'water', 'wetland']]

    df_shap_Qmax7 = pd.DataFrame(data = shap_Qmax7, columns = predictors1)
    df_shap_Qmin7 = pd.DataFrame(data = shap_Qmin7, columns = predictors1)

    fig, axes = plt.subplots(1, 2, figsize = (8, 10), sharey = True)
    for i,name in enumerate(['Qmin7','Qmax7']):
        shap = eval('df_shap_'+name)
        shap = shap.abs().mean()
        shap.name = 'SHAP'
        shap = shap.reset_index()
        shap['name'] = shap['index'].apply(lambda x:par_map.loc[par_map.par==x,'name'].values[0])
        shap['type'] = shap['index'].apply(lambda x:par_map.loc[par_map.par==x,'type'].values[0])
        shap = shap.sort_values(['type','name'])
        # reoder Antecedent conditions
        shap1 = shap.loc[shap.type=='Antecedent conditions',:]
        shap1['tmp'] = shap1.name.apply(lambda x:int(x.split('-')[0]) if '-day' in x else 99)
        shap1 = shap1.sort_values('tmp').drop(columns=['tmp'])
        shap2 = shap.loc[shap.type!='Antecedent conditions',:]
        shap = pd.concat([shap1, shap2])
        shap['rank'] = shap['SHAP'].rank(ascending = False).astype(int).astype(str)
        shap = shap.reset_index()
        if i == 1:
            legend = True
        else:
            legend = False
        sns.barplot(shap, x = 'SHAP', y = 'name', hue = 'type', ax = axes[i], legend = legend, palette = 'tab10')
        if name == 'Qmin7':
            name2 = 'Low river flow'
        else:
            name2 = 'High river flow'
        axes[i].set_title(name2, fontsize = 11)
        axes[i].set_ylabel('Explanatory variable', fontsize = 11)
        axes[i].set_xlabel('Absolute SHAP value', fontsize = 11)
        axes[i].set_xscale('symlog', linthresh=0.1)
        axes[i].yaxis.set_minor_locator(ticker.NullLocator())

        patches = axes[i].patches
        handles, labels = axes[i].get_legend_handles_labels()
        patches = [s for s in patches if s not in handles]
        tmp = []
        for s,a in enumerate(patches):
            rank0 = shap.iloc[s,:]['rank']
            axes[i].annotate(rank0, (a.get_width(), a.get_y()+a.get_height()/2),
                            ha='left', va='center', xytext=(5, 0),
                            textcoords='offset points', fontsize=8,)
            tmp.append(a.get_width())
        axes[i].set_xlim(0, np.max(tmp)*1.5)
        axes[i].tick_params(axis='both', labelsize = 9)
    sns.move_legend(axes[1], loc = 'upper left', bbox_to_anchor = (.32, .7), title = None, fontsize = 10)

    fig.savefig(dir_Qmax7 / f'figure_shap_feature_importance_{mode}.png', dpi = 600)

def plot_shap_dependence():
    # load shap and predictors
    Qmax7_fname = dir_Qmax7 / f'{model}_{mode}_shap_values.pkl'
    Qmin7_fname = dir_Qmin7 / f'{model}_{mode}_shap_values.pkl'

    try:
        shap_Qmin7 = pickle.load(open(Qmin7_fname, 'rb'))
        shap_Qmax7 = pickle.load(open(Qmax7_fname, 'rb'))
    except:
        print(f"Error: Cannot open {str(Qmax7_fname)}")
        return

    if mode == 'noLULC':
        predictors1 = [item for item in predictors if item not in ['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland']]
    elif  mode == 'onlyUrban':
        predictors1 = [item for item in predictors if item not in ['forest', 'crop', 'grass', 'water', 'wetland']]

    df_shap_Qmax7 = pd.DataFrame(data = shap_Qmax7, columns = predictors1)
    df_shap_Qmin7 = pd.DataFrame(data = shap_Qmin7, columns = predictors1)

    par_map = pd.read_csv('../data/predictors.csv')

    df_Qmin7 = pd.read_pickle(cfg_Qmin7['fname'])
    df_Qmax7 = pd.read_pickle(cfg_Qmax7['fname'])
    df_Qmin7['shap'] = df_shap_Qmin7['ImperviousSurface'].copy()
    df_Qmax7['shap'] = df_shap_Qmax7['ImperviousSurface'].copy()

    df_Qmin7['catch'] = np.where(df_Qmin7.aridity<=0.65, 'dry', 'wet')
    df_Qmax7['catch'] = np.where(df_Qmax7.aridity<=0.65, 'dry', 'wet')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 3), dpi = 200)
    palette0 = {'dry':'#C7B18A', 'wet':"#65C2A5"}
    sns.scatterplot(df_Qmin7, x = feature, y = 'shap', hue = 'catch', ax = ax1, alpha = .2, palette = palette0)
    sns.scatterplot(df_Qmax7, x = feature, y = 'shap', hue = 'catch', ax = ax2, alpha = .2, palette = palette0)

    import statsmodels.api as sm
    for i,name in enumerate(['Qmin7','Qmax7']):
        ax = eval('ax'+str(i+1))
        featureName = par_map.loc[par_map.par==feature,'name'].values[0]
        ax.set_xlabel(featureName.capitalize()+' (%)', fontsize = 11)
        ax.set_ylabel('SHAP value', fontsize = 11)
        sns.move_legend(ax, 'lower right', bbox_to_anchor = (.98, .02), title = None, fontsize = 11)
        if i == 0:
            ax.set_title('Dependence of low flow to '+featureName, fontsize = 11)
        else:
            ax.set_title('Dependence of high flow to '+featureName, fontsize = 11)
        ax.tick_params(axis = 'both', labelsize = 11)

        # estimate LOWESS
        df0 = eval('df_'+name)
        xvals = np.linspace(df0[feature].min(), df0[feature].max(), 100)
        ps = []
        for catch in ['dry','wet']:
            lowess0 = sm.nonparametric.lowess(
                df0.loc[df0.catch==catch,'shap'].values, 
                df0.loc[df0.catch==catch,feature].values, 
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
        ax.legend([(sc1,line1), (sc2,line2)], ['Catchments in dry climate','Catchments in wet climate'], numpoints=1, handlelength = 1)

        ax.text(-.1, 1.1, ['a','b'][i], weight = 'bold', ha = 'center', va = 'center', transform = ax.transAxes, fontsize = 11)

    fig.savefig(dir_Qmax7 / f'figure_shap_dependence_{feature}_{mode}.png', dpi = 600)

def plot_shap_ranking_map():
    # load shap and predictors
    Qmax7_fname = dir_Qmax7 / f'{model}_{mode}_shap_values.pkl'
    Qmin7_fname = dir_Qmin7 / f'{model}_{mode}_shap_values.pkl'

    try:
        shap_Qmin7 = pickle.load(open(Qmin7_fname, 'rb'))
        shap_Qmax7 = pickle.load(open(Qmax7_fname, 'rb'))
    except:
        print(f"Error: Cannot open {str(Qmax7_fname)}")
        return

    if mode == 'noLULC':
        predictors1 = [item for item in predictors if item not in ['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland']]
    elif mode == 'onlyUrban':
        predictors1 = [item for item in predictors if item not in ['forest', 'crop', 'grass', 'water', 'wetland']]

    shap_Qmax7 = pd.DataFrame(data = shap_Qmax7, columns = predictors1)
    shap_Qmin7 = pd.DataFrame(data = shap_Qmin7, columns = predictors1)

    df_Qmin7 = pd.read_pickle(cfg_Qmin7['fname'])
    df_Qmax7 = pd.read_pickle(cfg_Qmax7['fname'])

    df_Qmin7 = df_Qmin7[["ohdb_id", "ohdb_longitude", "ohdb_latitude", 'climate_label', 'aridity']]
    df_Qmax7 = df_Qmax7[["ohdb_id", "ohdb_longitude", "ohdb_latitude", 'climate_label', 'aridity']]

    df_Qmin7 = df_Qmin7.rename(columns={'ohdb_longitude':'lon','ohdb_latitude':'lat'})
    df_Qmax7 = df_Qmax7.rename(columns={'ohdb_longitude':'lon','ohdb_latitude':'lat'})

    shap_Qmin7 = pd.concat([df_Qmin7, shap_Qmin7], axis = 1)
    shap_Qmax7 = pd.concat([df_Qmax7, shap_Qmax7], axis = 1)

    shap_Qmin7_ave = shap_Qmin7.groupby(['ohdb_id','lon','lat','climate_label','aridity'], group_keys=False).p_apply(
        lambda x: pd.Series(
            [x.iloc[:,5:].abs().mean(0).rank(ascending=False, method='min').loc[feature]],
            index = [feature]
        )
    ).reset_index()
    shap_Qmax7_ave = shap_Qmin7.groupby(['ohdb_id','lon','lat','climate_label','aridity'], group_keys=False).p_apply(
        lambda x: pd.Series(
            [x.iloc[:,5:].abs().mean(0).rank(ascending=False, method='min').loc[feature]],
            index = [feature]
        )
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7, 8), subplot_kw = {'projection':ccrs.EqualEarth()})
    plt.subplots_adjust(hspace = .2)
    ax1.set_facecolor('none')
    ax2.set_facecolor('none')
    for i,name in enumerate(['Qmin7','Qmax7']):
        ax = eval('ax'+str(i+1))
        df = eval('shap_'+name+'_ave')
        lons = df.lon.values
        lats = df.lat.values
        vals = df[feature].values

        vmin, vmax, vind = 1, shap_Qmin7.shape[1]-5, 1
        cmap = plt.cm.RdBu_r
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
        if i == 0:
            title = f'Importance ranking of {featureName.lower()} in ML of low flows'
        else:
            title = f'Importance ranking of {featureName.lower()} in ML of high flows'
        label = f'SHAP ranking'
        _, ras = plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, norm = norm, fontSize = 11, size = 3)
        # add colorbar
        cax = ax.inset_axes([.35, .02, 0.25, .03])
        cbar = plt.colorbar(ras, cax = cax, orientation = 'horizontal', extend = 'both')
        cax.tick_params(labelsize = 10)
        cax.set_title(label, size = 10, pad = 5)
        # add boxplot to show the impact for dry (AI<1) and wet (AI>1) catchments
        df['tmp'] = np.where(df['aridity']>0.65, 'wet', 'dry')

        axin = ax.inset_axes([0.08, .05, .1, .3])
        sns.boxplot(df, 
                    x = 'tmp', y = feature, ax = axin, 
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
        axin.set_ylabel(f'SHAP ranking', fontsize = 8)
        axin.tick_params(labelsize = 8,  which='both', top = False, right = False)
        axin.xaxis.set_minor_locator(ticker.NullLocator())
        axin.spines["top"].set_visible(False) 
        axin.spines["right"].set_visible(False) 

    ax3 = ax1.inset_axes([1.1, .1, .4, .9])
    ax4 = ax2.inset_axes([1.1, .1, .4, .9])
    for i,name in enumerate(['Qmin7','Qmax7']):
        ax0 = [ax3,ax4][i]
        df0 = eval('shap_'+name+'_ave')
        sns.boxplot(data = df0, x = 'climate_label', y = feature, hue = 'climate_label', 
                    showfliers = False, showmeans = True, width = .5, 
                    whis = [5, 95],
                    meanprops={'marker': 'o',
                            'markerfacecolor': 'white',
                            'markeredgecolor': 'black',
                            'markersize': '8'},
                    ax = ax0, palette = palette)
        ax0.set_xlabel('Climate region', fontsize = 10)
        ax0.set_ylabel(f'SHAP ranking', fontsize = 10)
        ax0.tick_params(axis = 'both', labelsize = 10)
        # sns.move_legend(ax2, 'upper left', title = None, fontsize = 10)
        ax0.set_title(f'SHAP ranking of urban area', fontsize = 10)

    # add subplot order
    fig.text(.15, .9, 'a', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
    fig.text(.88, .9, 'b', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
    fig.text(.15,  .5, 'c', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
    fig.text(.88,  .5, 'd', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)

    fig.savefig(dir_Qmax7 / f'figure_shap_ranking_map_bobxplot_{mode}.png', dpi = 600)

def plot_correlation_between_urban_and_other():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8,8), sharey = True)
    for i,name in enumerate(['Qmin7','Qmax7']):
        df0 = pd.read_pickle(f'../data/{name}_final_dataset_seasonal4.pkl')
        corr0 = df0[predictors].corr(method = 'spearman')
        corr0 = corr0.loc[corr0.index!=feature,feature].reset_index()
        corr0['index'] = corr0['index'].apply(lambda x:par_map.loc[par_map.par==x,'name'].values[0])
        corr0['index'] = corr0['index'].apply(lambda x:re.sub(r'(\d+)-day','',x))
        corr0['index'] = corr0['index'].apply(lambda x:x.strip().capitalize())
        ax = eval('ax'+str(i+1))
        sns.barplot(corr0, y = 'index', x = feature, ax = ax)
        if name == 'Qmin7':
            ax.set_title('Low river flow', fontsize = 11)
        else:
            ax.set_title('High river flow', fontsize = 11)
        ax.set_xlabel(f'Spearman correlation\nbetween {featureName.lower()} and other features', fontsize = 10)
        ax.set_ylabel('Explanatory variable', fontsize = 10)
        ax.tick_params(axis='both', labelsize = 9)
    fig.savefig(dir_Qmax7 / f'spearman_correlation_between_{feature}_and_others.png', dpi = 600)

def plot_pdp_ale(purpose, min_interval = 0):
    Qmin7Fname = dir_Qmin7 / f'{model}_{mode}_{purpose}_{feature}_min_interval_{min_interval:.1f}.csv'
    Qmax7Fname = dir_Qmax7 / f'{model}_{mode}_{purpose}_{feature}_min_interval_{min_interval:.1f}.csv'

    try:
        df_eff_Qmin7 = pd.read_csv(Qmin7Fname)
        df_eff_Qmax7 = pd.read_csv(Qmax7Fname)
    except:
        print(f"Error: Cannot open {str(Qmax7Fname)}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4))
    for i,name in enumerate(['Qmin7','Qmax7']):
        df = eval('df_eff_'+name)
        ax = eval('ax'+str(i+1))
        for climate in df.climate.unique():
            tmp = df.loc[df.climate==climate,:]
            if monte:
                # Find common x-axis range
                min_x = tmp[feature].min()
                max_x = tmp[feature].max()

                groupName = 'fold' if purpose.startswith('cv') else 'monte'
                n_sample = tmp.groupby([groupName])['eff'].count().min()
                x_range = np.linspace(min_x, max_x, n_sample)  # Adjust number of points as needed

                # Interpolate each group to common x-axis
                interpolated_data = {}
                for group in tmp[groupName].unique():
                    group_df =tmp[tmp[groupName] == group]
                    f = interp1d(group_df[feature], group_df['eff'], kind='linear', fill_value='extrapolate')
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
            else:
                ax.plot(tmp[feature].values, tmp.eff.values, color = palette[climate], lw = 2, label = climate, zorder = 3)
            ax.legend(fontsize = 11)
        ax.set_xlabel(f'{featureName} (%)', fontsize = 10)
        ax.set_ylabel(f'{name} in $\Delta$%', fontsize = 10)
        ax.set_xlim(0, 8)
        if i == 0:
            ax.set_title(f'Effects of {featureName.lower()} on low flow', fontsize = 11)
        else:
            ax.set_title(f'Effects of {featureName.lower()} on high flow', fontsize = 11)
        ax.tick_params(labelsize=10)
        ax.text(-.1, 1.1, ['a','b'][i], weight = 'bold', ha = 'center', va = 'center', transform = ax.transAxes, fontsize = 12)
    fig.tight_layout()

    fig.savefig(outName, dpi = 600)

def compute_effsize(expGroup, controlGroup, eftype = 'cohen'):
    n1, n2 = len(expGroup), len(controlGroup)
    MAD1 = 1.4826 * np.median(np.abs(expGroup-np.median(expGroup)))
    MAD2 = 1.4826 * np.median(np.abs(controlGroup-np.median(controlGroup)))
    pooled_std = np.sqrt(((n1 - 1) * MAD1 ** 2 + (n2 - 1) * MAD2 ** 2) / (n1 + n2 - 2))
    if eftype == 'cohen':
        # d = np.mean(expGroup - controlGroup) / np.sqrt((MAD1**2+MAD2**2)/2)
        # d = np.mean(expGroup - controlGroup) / pooled_std
        d = (np.mean(expGroup) - np.mean(controlGroup)) / np.std(expGroup)
    elif eftype == 'glass':
        d = (np.mean(expGroup) - np.mean(controlGroup)) / MAD2
    return d

def plot_sensitivity(delta_feature):
    delta_feature = int(delta_feature)
    Qmin7Fname = dir_Qmin7 / f'{model}_{mode}_sensitivity_func_+0{delta_feature}{feature}_diff_in_percentage.csv'
    Qmax7Fname = dir_Qmax7 / f'{model}_{mode}_sensitivity_func_+0{delta_feature}{feature}_diff_in_percentage.csv'

    try:
        diff_Qmin7 = pd.read_csv(Qmin7Fname)
        diff_Qmax7 = pd.read_csv(Qmax7Fname)
    except:
        print(f"Error: Cannot open {str(Qmax7Fname)}")
        return

    diff_Qmin7['Qmin7date'] = pd.to_datetime(diff_Qmin7['Qmin7date'])
    diff_Qmax7['Qmax7date'] = pd.to_datetime(diff_Qmax7['Qmax7date'])

    diff_Qmin7 = pd.read_pickle(cfg_Qmin7['fname']).merge(diff_Qmin7, on = ['ohdb_id','Qmin7date'])
    diff_Qmax7 = pd.read_pickle(cfg_Qmax7['fname']).merge(diff_Qmax7, on = ['ohdb_id','Qmax7date'])

    diff_Qmin7_ave = diff_Qmin7.groupby(['ohdb_longitude','ohdb_latitude','climate_label','aridity'])['diff'].agg(
        diff = lambda x:x.mean(),
        p = lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue
    ).reset_index()
    diff_Qmax7_ave = diff_Qmax7.groupby(['ohdb_longitude','ohdb_latitude','climate_label','aridity'])['diff'].agg(
        diff = lambda x:x.mean(),
        p = lambda x: stats.ttest_1samp(x, 0, alternative='two-sided').pvalue
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7, 8), subplot_kw = {'projection':ccrs.EqualEarth()})
    plt.subplots_adjust(hspace = .2)
    ax1.set_facecolor('none')
    ax2.set_facecolor('none')
    for i,name in enumerate(['Qmin7','Qmax7']):
        ax = eval('ax'+str(i+1))
        df = eval('diff_'+name+'_ave')
        lons = df.ohdb_longitude.values
        lats = df.ohdb_latitude.values
        vals = df['diff'].values

        if i == 0:
            vmin, vmax, vind = -40, 40, 4
        else:
            vmin, vmax, vind = -20, 20, 2
        cmap = plt.cm.RdBu_r
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
        if i == 0:
            title = f'Impact of +10pp {featureName.lower()} on low flow'
        else:
            title = f'Impact of +10pp {featureName.lower()} on high flow'
        label = f'{name} in $\Delta$%'
        _, ras = plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, norm = norm, fontSize = 11, size = 3)
        # add colorbar
        cax = ax.inset_axes([.35, .02, 0.25, .03])
        cbar = plt.colorbar(ras, cax = cax, orientation = 'horizontal', extend = 'both')
        cax.tick_params(labelsize = 10)
        cax.set_title(label, size = 10, pad = 5)
        # add boxplot to show the impact for dry (AI<1) and wet (AI>1) catchments
        df['tmp'] = np.where(df['aridity']>0.65, 'wet', 'dry')
        
        ttest = stats.ttest_ind(df.loc[df.tmp=='wet','diff'].values, df.loc[df.tmp=='dry','diff'].values)
        D = compute_effsize(df.loc[df.tmp=='wet','diff'].values, df.loc[df.tmp=='dry','diff'].values)
        
        print('fraction of significant gauges:', df.loc[df.p<=0.05,:].shape[0] / df.shape[0] * 100)
        print('average diff of significant gauges:', df.loc[df.p<=0.05,'diff'].mean())
        print(df.loc[df.tmp=='wet','diff'].mean(), df.loc[df.tmp=='wet','diff'].std())
        print(df.loc[df.tmp=='dry','diff'].mean(), df.loc[df.tmp=='dry','diff'].std())
        print(ttest, D)
        
        axin = ax.inset_axes([0.08, .05, .1, .3])
        sns.boxplot(df.loc[df.p<=0.01,:], 
                    x = 'tmp', y = 'diff', ax = axin, 
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

    # add title
    ax1.set_title(f'Impact of +10pp {featureName.lower()} on low flow', fontsize = 10)
    ax2.set_title(f'Impact of +10pp {featureName.lower()} on low flow', fontsize = 10)   

    ax3 = ax1.inset_axes([1.1, .1, .4, .9])
    ax4 = ax2.inset_axes([1.1, .1, .4, .9])
    for i,name in enumerate(['Qmin7','Qmax7']):
        ax0 = [ax3,ax4][i]
        df0 = eval('diff_'+name+'_ave')
        sns.boxplot(data = df0, x = 'climate_label', y = 'diff', hue = 'climate_label', 
                    showfliers = False, showmeans = True, width = .5, 
                    whis = [5, 95],
                    meanprops={'marker': 'o',
                            'markerfacecolor': 'white',
                            'markeredgecolor': 'black',
                            'markersize': '8'},
                    ax = ax0, palette = palette)
        ax0.set_xlabel('Climate region', fontsize = 10)
        ax0.set_ylabel(f'{name} in $\Delta$%', fontsize = 10)
        ax0.tick_params(axis = 'both', labelsize = 10)
        # sns.move_legend(ax2, 'upper left', title = None, fontsize = 10)
        ax0.set_title(f'Impact on {name} across climate regions', fontsize = 10)

    # add subplot order
    fig.text(.15, .9, 'a', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
    fig.text(.88, .9, 'b', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
    fig.text(.15,  .5, 'c', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)
    fig.text(.88,  .5, 'd', weight = 'bold', va = 'top', ha = 'center', fontsize = 12)

    fig.savefig(dir_Qmax7 / f'figure_map_sensitivity_delta+0{delta_feature}{feature}_bobxplot_{mode}.png', dpi = 600)

    # ecdfplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9, 4))
    sns.ecdfplot(data = diff_Qmin7_ave, x = 'diff', hue = 'climate_label', ax = ax1, palette = palette, lw = 2)
    sns.ecdfplot(data = diff_Qmax7_ave, x = 'diff', hue = 'climate_label', ax = ax2, palette = palette, lw = 2)
    for i,name in enumerate(['Qmin7','Qmax7']):
        ax = eval('ax'+str(i+1))
        ax.set_xlabel('Catchment-based average $\Delta$'+name+' (%)', fontsize = 11)
        sns.move_legend(ax, 'upper left', bbox_to_anchor = (.02, .98), title = None, fontsize = 11)
        ax.set_ylabel('Proportion of catchments', fontsize = 11)
        ax.tick_params(axis = 'both', labelsize = 11)
        ax.set_xscale('symlog')
        ax.text(-.2, 1, string.ascii_letters[i], weight = 'bold', fontsize = 12, transform = ax.transAxes)
        ax.axvline(x = 0, ls = 'dashed', color = 'k', lw = .5)
        if i == 0:
            ax.set_title(f'Impact of +10pp {featureName.lower()} on low flow', fontsize = 12)
        else:
            ax.set_title(f'Impact of +10pp {featureName.lower()} on high flow', fontsize = 12)
    fig.tight_layout()
    fig.savefig(dir_Qmax7 / f'figure_ecdfplot_sensitivity_delta+0{delta_feature}{feature}_{mode}.png', dpi = 600)

def plot_urban_importance():
    # read model performance with and without urban area
    df_Qmin7 = pd.read_csv('../results/importance_of_urban_run10_Qmin7_seasonal4_multi_MSWX_meteo_MSWEP_GLEAM_simple3.csv')
    df_Qmax7 = pd.read_csv('../results/importance_of_urban_run10_Qmax7_seasonal4_multi_MSWX_meteo_MSWEP_GLEAM_simple3.csv')

    # calculate average values of indices
    indices = df_Qmin7.columns[df_Qmin7.columns.str.contains('1')]
    indices = [item[:-1] for item in indices]
    for name in ['Qmin7','Qmax7']:
        for index in indices:
            eval('df_'+name)[index+'_mean'] = eval('df_'+name).loc[:,eval('df_'+name).columns.str.contains(index)].mean(axis=1)
        
    df_Qmin7 = df_Qmin7.pivot_table(
        index = ['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label'],
        columns = 'mode',
        values = [item+'_mean' for item in indices]
    )
    df_Qmax7 = df_Qmax7.pivot_table(
        index = ['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label'],
        columns = 'mode',
        values = [item+'_mean' for item in indices]
    )

    # calculate difference of indices between noLULC and onlyUrban
    for name in ['Qmin7','Qmax7']:
        for index in [item+'_mean' for item in indices]:
            df0 = eval('df_'+name)
            df0[index+'_diff'] = df0.loc[:,(index, 'onlyUrban')] - df0.loc[:,(index, 'noLULC')]

    df_Qmin7 = df_Qmin7.reset_index()
    df_Qmax7 = df_Qmax7.reset_index()

    sss = 'nRMSEmean'

    if sss == 'nRMSEmean':
        vmin = -20; vmax = 20; index = 'nRMSE (%)'
    else:
        vmin = -0.2; vmax = 0.2; index = sss

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (5, 6), subplot_kw={'projection':ccrs.EqualEarth()})
    for i,name in enumerate(['Qmin7','Qmax7']):
        if i == 0:
            df_sta = df_Qmin7.copy()
            ax = ax1
            legend = False
            title = f'Importance of urban area in ML of low flow'
        else:
            df_sta = df_Qmax7.copy()
            ax = ax2
            legend = True
            title = f'Importance of urban area in ML of high flow'

        label = '$\Delta$' + index
        
        x = df_sta['ohdb_longitude'].values
        y = df_sta['ohdb_latitude'].values
        val = df_sta[sss+'_mean_diff'].values
        cmap = plt.cm.RdBu_r
        ras = ax.scatter(x, y, c = val, cmap = cmap, vmin = vmin, vmax = vmax, 
                        marker = "$\circ$", ec = "face", s = .3, 
                        transform = ccrs.PlateCarree())
        cax = ax.inset_axes([0.4, 0.05, 0.4, 0.03])
        fig.colorbar(ras, cax = cax, orientation = 'horizontal', extend = 'both')
        cax.set_title(label, fontsize = 9)
        
        ax.set_global()
        ax.set_ylim([-6525154.6651, 8625154.6651]) 
        ax.set_xlim([-12662826, 15924484]) 
        ax.spines['geo'].set_linewidth(0)
        ax.coastlines(linewidth = .2, color = '#707070')
        ax.add_feature(cf.BORDERS, linewidth = .2, color = '#707070')
        ax.text(0, 1, ['a','b'][i], transform = ax.transAxes, ha = 'center', va = 'top', fontsize = 10, weight = 'bold')
        ax.set_title(title, fontsize = 9)

        print(name, df_sta.loc[df_sta[f'{sss}_mean_diff']>=0.1,:].shape[0] / df_sta.shape[0] * 100)
        
    ax3 = ax1.inset_axes([1.1, .15, .6, .85])
    ax4 = ax2.inset_axes([1.1, .15, .6, .85])

    sns.boxplot(df_Qmin7, y = sss+'_mean_diff', x = 'climate_label', hue = 'climate_label', width = .5,
                ax = ax3, palette = palette, legend = True, showfliers = False)
    ax3.set_title('Importance of urban area in ML of low flow', fontsize = 9)
    sns.move_legend(ax3, 'upper left', title = None)
    ax3.set_xlabel(label, fontsize = 9)
    ax3.tick_params(axis = 'both', labelsize = 9)
    ax3.set_ylabel('$\Delta$'+index, fontsize = 9)

    sns.boxplot(df_Qmax7, y = sss+'_mean_diff', x = 'climate_label', hue = 'climate_label', width = .5,
                ax = ax4, palette = palette, legend = True, showfliers = False)
    ax4.set_title('Importance of urban area in ML of high flow', fontsize = 9)
    sns.move_legend(ax4, 'upper left', title = None)
    ax4.tick_params(axis = 'both', labelsize = 9)
    ax4.set_xlabel(label, fontsize = 9)
    ax4.set_ylabel('$\Delta$'+index, fontsize = 9)

    ax3.text(-.2, 1, 'c', weight = 'bold', ha = 'right', va = 'center', fontsize = 10, transform = ax3.transAxes)
    ax4.text(-.2, 1, 'd', weight = 'bold', ha = 'right', va = 'center', fontsize = 10, transform = ax4.transAxes)

    fig.savefig(f'../picture/importance_of_{feature}_{sss}_{outName}.png', dpi = 600)

if __name__ == '__main__':
    plot_cv()
    plot_shap_importance()
    plot_shap_dependence()
    plot_shap_ranking_map()
    plot_correlation_between_urban_and_other()
    plot_pdp_ale('ale', min_interval = 0)
    plot_pdp_ale('pdp', min_interval = 0)
    plot_sensitivity(10)