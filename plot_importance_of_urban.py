from src.plot_utils import *

# read model performance with and without urban area
df_Qmin7 = pd.read_csv('../results/importance_of_urban_run100_Qmin7.csv')
df_Qmax7 = pd.read_csv('../results/importance_of_urban_run100_Qmax7.csv')

# calculate difference of KGE and nRMSE
for name in ['Qmin7','Qmax7']:
    for index in ['KGE','nRMSE','r','alpha','beta']:
        eval('df_'+name)[index] = eval('df_'+name).loc[:,eval('df_'+name).columns.str.contains(index)].mean(axis=1)
    
df_Qmin7 = df_Qmin7.pivot_table(
    index = ['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label'],
    columns = 'mode',
    values = ['nRMSE','KGE','r','alpha','beta']
)
df_Qmax7 = df_Qmax7.pivot_table(
    index = ['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label'],
    columns = 'mode',
    values = ['nRMSE','KGE','r','alpha','beta']
)

for name in ['Qmin7','Qmax7']:
    for index in ['KGE','nRMSE','r','alpha','beta']:
        eval('df_'+name)[index+'_diff'] = eval('df_'+name).loc[:,(index, 'onlyUrban')] - eval('df_'+name).loc[:,(index, 'noLULC')]

df_Qmax7 = df_Qmax7.reset_index()
df_Qmin7 = df_Qmin7.reset_index()

sss = 'beta'

# plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (5, 6), subplot_kw={'projection':ccrs.EqualEarth()})
for i,name in enumerate(['Qmin7','Qmax7']):
    if i == 0:
        df_sta = df_Qmin7.copy()
        ax = ax1
        legend = False
    else:
        df_sta = df_Qmax7.copy()
        ax = ax2
        legend = True

    title = f'Importance of urban area in ML of {name}'
    label = '$\Delta$' + sss
    
    x = df_sta['ohdb_longitude'].values
    y = df_sta['ohdb_latitude'].values
    val = df_sta[f'{sss}_diff'].values
    cmap = plt.cm.RdBu_r
    ras = ax.scatter(x, y, c = val, cmap = cmap, vmin = -0.3, vmax = 0.3, 
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

    print(name, df_sta.loc[df_sta[f'{sss}_diff']>=0.1,:].shape[0] / df_sta.shape[0] * 100)
    
ax3 = ax1.inset_axes([1.1, .15, .6, .85])
ax4 = ax2.inset_axes([1.1, .15, .6, .85])

sns.boxplot(df_Qmin7, y = f'{sss}_diff', x = 'climate_label', hue = 'climate_label', ax = ax3, palette = palette, legend = True, showfliers = False)
ax3.set_title('Qmin7', fontsize = 9)
sns.move_legend(ax3, 'upper left', title = None)
ax3.set_xlabel(label, fontsize = 9)

sns.boxplot(df_Qmax7, y = f'{sss}_diff', x = 'climate_label', hue = 'climate_label', ax = ax4, palette = palette, legend = True, showfliers = False)
ax4.set_title('Qmax7', fontsize = 9)
sns.move_legend(ax4, 'upper left', title = None)
ax4.set_xlabel(label, fontsize = 9)

ax3.text(-.2, 1, 'c', weight = 'bold', ha = 'right', va = 'center', fontsize = 10, transform = ax3.transAxes)
ax4.text(-.2, 1, 'd', weight = 'bold', ha = 'right', va = 'center', fontsize = 10, transform = ax4.transAxes)
# ax3.set_xlim(-1, 1e6)
# ax3.set_ylim(-1, 1e6)
# ax4.set_xlim(-1, 1e6)
# ax4.set_ylim(-1, 1e6)
# fig.tight_layout()
fig.savefig(f'../picture/importance_of_urban_in_model_{sss}.png', dpi = 600)