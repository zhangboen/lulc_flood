import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import hydroeval as he
import matplotlib.ticker as ticker
import scienceplots
import cartopy
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
import warnings
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import os,glob,re,sys,cmaps,string
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import multiprocessing as mp
from multiprocessing import Process
import cartopy.feature as cf
from matplotlib.collections import LineCollection
import matplotlib.font_manager as font_manager
warnings.filterwarnings("ignore")
font_manager.fontManager.addfont(os.environ['DATA']+'/fonts/Helvetica/Helvetica.ttf')
font_manager.fontManager.addfont(os.environ['DATA']+'/fonts/Helvetica/Helvetica-Bold.ttf')
plt.style.use(['science','nature','no-latex']) # require install SciencePlots
plt.rcParams.update({"font.size":12, 'font.family':'Helvetica'}) 
from parallel_pandas import ParallelPandas
ParallelPandas.initialize(n_cpu=24, split_factor=12)

palette = {'tropical':'#F8D347',
           'dry':'#C7B18A',
           'temperate':"#65C2A5",
           'cold':"#a692b0",
           'polar':"#B3B3B3"
          }

def add_histogram(ax, vals, extent, vmin, vmax, vind, cmap):
    ax2 = ax.inset_axes(extent)
    cm = plt.cm.get_cmap(cmap)
    bins = [-np.inf] + np.arange(vmin, vmax+vind, vind).tolist() + [np.inf]
    n, bins, patches = ax2.hist(vals, bins=bins)
    bins[0] = vmin - vind
    bins[-1] = vmax + vind
    vcenter=(vmin+vmax)/2
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    bin_center = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_center, patches):
        plt.setp(p, 'facecolor', cm(norm(c)))
    ax2.get_yaxis().set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(top=False, which='both', labelsize  = 8)
    ax2.xaxis.label.set_visible(False)
    ax2.xaxis.set_minor_locator(ticker.NullLocator())
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axvline(x=vcenter,color='k')

def plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, marker = "$\circ$", size = 1, fontSize = 9, norm = None, addHist = True):
    ax.set_global()
    ax.set_ylim([-6525154.6651, 8625154.6651]) 
    ax.set_xlim([-12662826, 15924484]) 
    ax.spines['geo'].set_linewidth(0)
    ax.coastlines(linewidth = .2, color = '#707070')
    ax.add_feature(cf.BORDERS, linewidth = .2, color = '#707070')
    # plot scatter setting
    if norm is None:
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    ras = ax.scatter(lons, lats, c = vals, norm = norm, cmap = cmap, s = size, marker = marker, 
                    ec = "face", transform = ccrs.PlateCarree(), zorder = 3, linewidths = 0)
    # add histograms
    if addHist:
        add_histogram(ax, vals, [.62, .25, .15, .2], vmin, vmax, vind, cmap)
    # set plot title
    ax.set_title(title, size = fontSize, pad = 7)
    return ax, ras

def plot_scatter(x, y, climate, xlabel, ylabel, metrics = True, log = True, normColor = 'log', addDiagnol = True, palette = None, ax = None, legend = True, fontsize = 10, size = 1):
    df0 = pd.DataFrame({'x':x, 'y':y,'climate':climate})
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(4, 4))
    else:
        ax1 = ax
    # y vs y_pred
    if normColor == 'log':
        g = sns.scatterplot(data = df0, x = 'x', y = 'y', hue = 'climate', ax = ax1, hue_norm = LogNorm(), alpha = .6, palette = palette, legend = legend, s = size)
    else:
        g = sns.scatterplot(data = df0, x = 'x', y = 'y', hue = 'climate', ax = ax1, alpha = .6, palette = palette, legend = legend, s = size)
    if legend:
        g.legend_.set_title(None)
        sns.move_legend(ax1, "lower right")
    max0 = max(np.nanmax(x[~np.isinf(x)]), np.nanmax(y))
    min0 = min(np.nanmin(x[~np.isinf(x)]), np.nanmin(y))
    if addDiagnol:
        ax1.axline((min0,min0), (max0,max0), color = 'k', ls = (0,(5,5)))
    if log:
        if (x==0).any() or (y==0).any():
            ax1.set_xscale('symlog')
            ax1.set_yscale('symlog')
        else:
            ax1.set_yscale('log')
            ax1.set_xscale('log')
    ax1.set_xlim(min0*0.99, max0*1.01)
    ax1.set_ylim(min0*0.99, max0*1.01)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(xlabel, fontsize = fontsize)
    ax1.set_ylabel(ylabel, fontsize = fontsize)
    if metrics:
        kge, r, beta, alpha = he.evaluator(he.kge, y, x).squeeze()
        nse0 = he.evaluator(he.nse, y, x)[0]
        nRMSE = np.sqrt(np.mean((y-x)**2)) / (np.max(x)-np.min(x)) * 100
        kge = np.round(kge, 2)
        r = np.round(r, 2)
        beta = np.round(beta, 2)
        alpha = np.round(alpha, 2)
        nse0 = np.round(nse0, 2)
        ax1.text(.95, .05, 
                'r = {:1.2f}\nβ = {:1.2f}\nα = {:1.2f}\nNSE = {:1.2f}\nKGE = {:1.2f}\nnRMSE = {:1.1f}%'.format(r, beta, alpha, nse0, kge, nRMSE),
                linespacing = 1.5, 
                transform=ax1.transAxes, 
                size = fontsize, 
                va = 'bottom', 
                ha = 'right')
    return ax1

def plot_folium(df, lon_name, lat_name, radius_name = None, popup_name = None, radius_scale = 0.1):
    import folium
    # Create a map centered on the mean of your coordinates
    map_center = [df[lat_name].mean(), df[lon_name].mean()]
    m = folium.Map(location=map_center, zoom_start=4)
    # Add scatter points
    for index, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_name], row[lon_name]],
            radius=abs(row[radius_name])*radius_scale if radius_name is not None else 1,  # Adjust radius based on your data
            popup='%s'%(row[popup_name]) if popup_name is not None else None,
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)
    return m