import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import hydroeval as he
import matplotlib.ticker as ticker
import scienceplots
import cartopy
import warnings
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import os,glob,re,sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import multiprocessing as mp
from multiprocessing import Process
import cartopy.feature as cf
from matplotlib.collections import LineCollection
import matplotlib.font_manager as font_manager
warnings.filterwarnings("ignore")

# set custom font (not a good idea)
font_manager.fontManager.addfont(os.environ['DATA']+'/fonts/Helvetica/Helvetica.ttf')
font_manager.fontManager.addfont(os.environ['DATA']+'/fonts/Helvetica/Helvetica-Bold.ttf')

plt.style.use(['science','nature','no-latex']) # require install SciencePlots
plt.rcParams.update({"font.size":12, 'font.family':'Helvetica'}) 

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

def plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, size = 1, fontSize = 9, norm = None):
    ax.set_global()
    ax.set_ylim([-6525154.6651, 8625154.6651]) 
    ax.set_xlim([-12662826, 15924484]) 
    ax.spines['geo'].set_linewidth(0)
    ax.coastlines(linewidth = .2, color = '#707070')
    ax.add_feature(cf.BORDERS, linewidth = .2, color = '#707070')
    # plot scatter setting
    if norm is None:
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    # add colorbar
    ras = ax.scatter(lons, lats, c = vals, norm = norm, cmap = cmap, s = size, transform = ccrs.PlateCarree(), zorder = 3, linewidths = 0)
    ax2 = ax.inset_axes([.35, .03, 0.25, .03])
    plt.colorbar(ras, cax = ax2, orientation = 'horizontal', extend = 'both')
    ax2.tick_params(labelsize = fontSize)
    ax2.set_title(label, size = fontSize, pad = 2)
    # add histograms
    add_histogram(ax, vals, [.62, .2, .15, .2], vmin, vmax, vind, cmap)
    # set plot title
    ax.set_title(title, size = fontSize, pad = 7)
    return ax

def plot_scatter(x, y, climate, xlabel, ylabel, metrics = True, log = True, normColor = 'log', addDiagnol = True, palette = None, ax = None, legend = True, fontsize = 10):
    df0 = pd.DataFrame({'x':x, 'y':y,'climate':climate})
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(4, 4))
    else:
        ax1 = ax
    # y vs y_pred
    if normColor == 'log':
        g = sns.scatterplot(data = df0, x = 'x', y = 'y', hue = 'climate', ax = ax1, hue_norm = LogNorm(), alpha = .6, palette = palette, legend = legend)
    else:
        g = sns.scatterplot(data = df0, x = 'x', y = 'y', hue = 'climate', ax = ax1, alpha = .6, palette = palette, legend = legend)
    if legend:
        g.legend_.set_title(None)
        sns.move_legend(ax1, "lower right")
    max0 = max(np.nanmax(x[~np.isinf(x)]), np.nanmax(y))
    min0 = min(np.nanmin(x[~np.isinf(x)]), np.nanmin(y))
    if addDiagnol:
        ax1.axline((min0,min0), (max0,max0), color = 'k', ls = (0,(5,5)))
    if log:
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
        nRMSE = np.sum((y-x)**2) / np.sum(x**2) * 100
        kge = np.round(kge, 2)
        r = np.round(r, 2)
        beta = np.round(beta, 2)
        alpha = np.round(alpha, 2)
        nse0 = np.round(nse0, 2)
        nRMSE = int(nRMSE)
        ax1.text(.03, .97, 
                'NSE = {:1.2f}\nnRMSE = {:1.0f}%\nKGE = {:1.2f}\nr = {:1.2f}\nβ = {:1.2f}\nα = {:1.2f}'.format(nse0, nRMSE, kge, r, beta, alpha),
                linespacing = 1.5, 
                transform=ax1.transAxes, 
                size = fontsize, 
                va = 'top', 
                ha = 'left')
    return ax1