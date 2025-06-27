import string
from src.plot_utils import *

attrs = [   'BDTICM', 'elevation', 'slope', 
            'aridity', 'sedimentary', 'plutonic', 
            'volcanic', 'metamorphic', 'clay', 
            'sand', 'silt', 'Porosity_x', 
            'logK_Ice_x',  'form_factor', 'LAI', 
            'res_darea_normalize', 'Year_ave', 'climate_label',
            'basin_id', 'Main_Purpose_id', 'Main_Purpose',
        ]

df = pd.read_pickle('../data/Qmax7_final_dataset_seasonal4.pkl')
df = df.groupby(['ohdb_id','ohdb_longitude','ohdb_latitude'])[attrs].apply(lambda x:x.iloc[0,:]).reset_index()
con = pd.read_csv('../data/predictors.csv')

fig, ax0 = plt.subplots(figsize = (4, 3), subplot_kw = {'projection':ccrs.EqualEarth()})
axes = []
for j,attr in enumerate(attrs[:-1]):
    ax = ax0
    if j > 0:
        if j%3 != 0:
            ax = ax0.inset_axes([.9,0,1,1], projection = ccrs.EqualEarth())
        else:
            ax = axes[-3].inset_axes([0, -1.23, 1, 1], projection = ccrs.EqualEarth())
    ax0 = ax
    axes.append(ax)
    ax.set_facecolor('none')
    lons = df.ohdb_longitude.values
    lats = df.ohdb_latitude.values
    vals = df[attr].values
    if attr in ['res_darea_normalize','FAPAR']:
        vals = vals * 100
    if attr == 'climate_label':
        ax.set_global()
        ax.set_ylim([-6525154.6651, 8625154.6651]) 
        ax.set_xlim([-12662826, 15924484]) 
        ax.spines['geo'].set_linewidth(0)
        ax.coastlines(linewidth = .2, color = '#707070')
        ax.add_feature(cf.BORDERS, linewidth = .2, color = '#707070')
        tmp = df.groupby('climate_label').ohdb_id.count().astype(str).reset_index().rename(columns={'ohdb_id':'count'})
        df = df.merge(tmp, on = 'climate_label')
        df['climate_label1'] = df['climate_label'] + ' (' + df['count'] + ')'
        palette1 = pd.DataFrame.from_dict({
            'climate_label':['tropical','dry','temperate','cold'],
            'color':['#F8D347', '#C7B18A', "#65C2A5", "#a692b0"]})
        df = df.merge(palette1, on = 'climate_label')
        palette1 = df[['climate_label1', 'color']].set_index('climate_label1').color.to_dict()
        sns.scatterplot(df, 
                        x = 'ohdb_longitude', 
                        y = 'ohdb_latitude',
                        hue = 'climate_label1',
                        ax = ax,
                        edgecolor = 'none',
                        palette = palette1,
                        transform = ccrs.PlateCarree(),
                        s = .5, 
                        legend = True)
        sns.move_legend(ax, 'upper center', title = None, bbox_to_anchor = (.5, 0.07), ncols = 2, markerscale = 2, fontsize = 9)
        df = df.drop(columns=['count'])
        ax.set_title('Koppen climate classification', fontsize = 9)
    elif attr == 'basin_id':
        ax.set_global()
        ax.set_ylim([-6525154.6651, 8625154.6651]) 
        ax.set_xlim([-12662826, 15924484]) 
        ax.spines['geo'].set_linewidth(0)
        ax.coastlines(linewidth = .2, color = '#707070')
        ax.add_feature(cf.BORDERS, linewidth = .2, color = '#707070')
        df['basin_id'] = df['basin_id'].astype(str)
        sns.scatterplot(df, 
                        x = 'ohdb_longitude', 
                        y = 'ohdb_latitude',
                        hue = 'basin_id',
                        edgecolor = 'none',
                        palette = 'tab20',
                        ax = ax,
                        transform = ccrs.PlateCarree(),
                        s = .5, 
                        legend = False)
        ax.set_title('HydroBASINS level-12 watershed', fontsize = 9)
    else:
        vmin = np.quantile(vals, .1)
        vmax = np.quantile(vals, .9)
        if attr == 'FAPAR':
            vmin = 0; vmax = 100
        if attr == 'LAI':
            vmin = 0; vmax = 5
        vind = np.round((vmax-vmin)/10, 1)
        cmap = 'viridis'
        title = con.loc[con.par==attr,'name'].values[0]
        if attr == 'FAPAR':
            title = 'Fraction of absorbed\nphotosynthetically active radiation'
        if attr == 'climate_label':
            title = 'Koppen climate classification'
        label = con.loc[con.par==attr,'unit'].values[0]
        title = title + f' ({label})'
        label = None
        _,ras = plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, marker = '.',
                            size = 1, fontSize = 9, norm = None, addHist = False)
        # add colorbar
        if attr != 'Main_Purpose_id':
            cax = ax.inset_axes([.5, .08, 0.3, .03])
            cbar = plt.colorbar(ras, cax = cax, orientation = 'horizontal', extend = 'both')
            cax.tick_params(labelsize = 9)
            cax.set_title(label, size = 9, pad = 5)
        else:
            ras.remove()
            df.loc[df.Main_Purpose=='nores','Main_Purpose'] = 'zero dam'
            tmp = df.groupby('Main_Purpose')[attr].count().astype(str).reset_index().rename(columns={attr:'count'})
            df = df.merge(tmp, on = 'Main_Purpose')
            df['Main_Purpose'] = df['Main_Purpose'] + ' (' + df['count'] + ')'
            sns.scatterplot(df, 
                            x = 'ohdb_longitude', 
                            y = 'ohdb_latitude',
                            hue = 'Main_Purpose',
                            ax = ax,
                            edgecolor = 'none',
                            transform = ccrs.PlateCarree(),
                            s = .5, 
                            legend = True)
            sns.move_legend(ax, 'center left', title = None, bbox_to_anchor = (1, 0.35), ncols = 1, markerscale = 4, fontsize = 9, columnspacing = 0.8)
    ax.text(0, 1, string.ascii_letters[j], weight = 'bold', fontsize = 9, transform = ax.transAxes)
fig.savefig(f'../picture/map_of_basin_attr.png', dpi = 600)