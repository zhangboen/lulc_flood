from src.plot_utils import *
from parallel_pandas import ParallelPandas
ParallelPandas.initialize(n_cpu=24, split_factor=24)

Qmin7Fname = '../results/xgb_cv10_Qmin7_seasonal4_multi_MSWX_meteo_MSWEP_GLEAM_simple_raw_result.csv'
Qmax7Fname = '../results/xgb_cv10_Qmax7_seasonal4_multi_MSWX_meteo_MSWEP_GLEAM_simple_raw_result.csv'

df_Qmin7 = pd.read_csv(Qmin7Fname)
df_Qmax7 = pd.read_csv(Qmax7Fname)

df_Qmin7_sta = df_Qmin7.groupby(['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label']).p_apply(
        lambda x: pd.Series(
            [np.corrcoef(x.pred.values,x.Q.values)[0,1]**2,
            np.sum((x.pred.values-x.Q.values)**2) / np.sum(x.Q.values**2) * 100], index = ['R2','nRMSE'])
    ).reset_index()
df_Qmax7_sta = df_Qmax7.groupby(['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label']).p_apply(
        lambda x: pd.Series(
            [np.corrcoef(x.pred.values,x.Q.values)[0,1]**2,
            np.sum((x.pred.values-x.Q.values)**2) / np.sum(x.Q.values**2) * 100], index = ['R2','nRMSE'])
    ).reset_index()

outName = re.sub('Qmax7_','', os.path.basename(Qmax7Fname)).split('.')[0]

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
    vals = df_sta.nRMSE.values
    label = 'nRMSE (%)'
    cmap = plt.cm.viridis
    _,ras = plot_map(ax, lons, lats, vals, 0, 100, 10, cmap, title, label, marker = "$\circ$", size = 1, fontSize = 9, norm = None, addHist = False)
    cax = ax.inset_axes([.55, 0.05, .3, .03])
    fig.colorbar(ras, cax = cax, orientation = 'horizontal')
    cax.set_title(label, fontsize = 8)
    
    print(name, df_sta.loc[df_sta.R2>=0.3,:].shape[0] / df_sta.shape[0] * 100, df_sta.loc[df_sta.nRMSE<=10,:].shape[0] / df_sta.shape[0] * 100)
    
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

fig.savefig(f'../picture/scatter_and_R2_map_{outName}.png', dpi = 600)