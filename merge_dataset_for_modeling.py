import os,glob,cmaps,sys,re,pickle
import numpy as np
import pandas as pd
import xarray as xr

target = sys.argv[1]
if target not in ['Qmax7','Qmin7','Qmax']:
    raise Exception('target must be Qmax7 or Qmin7 or Qmax')

###################################################################
# read meteo dataset
fname = f'../data/{target}_seasonal4_multi_MSWX_MSWEP_GLEAM.csv'
df_meteo = pd.read_csv(fname)

if target == 'Qmax7':
    df_meteo = df_meteo[['ohdb_id', target+'date'] + df_meteo.columns[df_meteo.columns.str.contains('_7')].tolist()]
    df_meteo = df_meteo.rename(columns=lambda x:re.sub('_7','',x))
else:
    df_meteo = df_meteo[['ohdb_id', target+'date'] + df_meteo.columns[df_meteo.columns.str.contains('_30')].tolist()]
    df_meteo = df_meteo.rename(columns=lambda x:re.sub('_30','',x))

# merge with discharge
df_dis = pd.read_csv('../data/dis_OHDB_seasonal4_Qmin7_Qmax7_1982-2023.csv')
df_dis = df_dis.loc[df_dis.winter_year>=1982,:]

df_meteo[target+'date'] = pd.to_datetime(df_meteo[target+'date'])
df_dis[target+'date'] = pd.to_datetime(df_dis[target+'date'])

df_meteo = df_meteo.merge(df_dis[[target+'date','season','countDay', 'winter_year', target, 'ohdb_id']], on = ['ohdb_id', target+'date'])
df_meteo = df_meteo.rename(columns={target:'Q'})
df_meteo = df_meteo.rename(columns={'winter_year':'year'})

# add month variable
df_meteo['month'] = df_meteo[target+'date'].dt.month

# read lulc dataset
df_lulc = pd.concat([pd.read_csv(fname).set_index('ohdb_id') for fname in glob.glob('../ee_lulc/*csv')], axis = 1).reset_index()
df_lulc = df_lulc.melt(id_vars = 'ohdb_id')
df_lulc['year'] = df_lulc.variable.str[:4].astype(int)
df_lulc['var'] = df_lulc.variable.str[5:]
df_lulc = df_lulc.pivot_table(index = ['ohdb_id','year'], columns = 'var', values = 'value').reset_index()

# change lulc unit to %
df_lulc[['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland']] = df_lulc[['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland']] / 100
df_lulc[['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland']] = df_lulc[['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland']].round(6)

df_attr = pd.read_csv('../data/basin_attributes_koppen.csv')

# add LAI
tmp = pd.read_csv('../data/basin_attributes.csv')
df_attr = df_attr.merge(tmp[['ohdb_id','LAI']], on = 'ohdb_id')

# calculate form factor
from pyogrio import read_dataframe
gdf = read_dataframe('../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg')
gdf['form_factor'] = gdf.area / (gdf.length)**2
df_attr = df_attr.merge(gdf[['ohdb_id','form_factor']], on = 'ohdb_id')

# merge soil texture fraction
df_attr['sedimentary'] = df_attr[['su', 'ss', 'sm', 'sc']].fillna(0).sum(1).mul(100)
df_attr['plutonic'] = df_attr[['pa', 'pb', 'pi']].fillna(0).sum(1).mul(100)
df_attr['volcanic'] = df_attr[['va', 'vi', 'vb']].fillna(0).sum(1).mul(100)
df_attr['metamorphic'] = df_attr['mt'].copy().mul(100)

# change clay units
df_attr['BDTICM'] = df_attr['BDTICM']/100
df_attr['BDTICM'] = df_attr['BDTICM'].round(3)
for name in ['clay','sand','silt']:
    df_attr[df_attr.columns[df_attr.columns.str.contains(name)]] = df_attr[df_attr.columns[df_attr.columns.str.contains(name)]] / 10
    df_attr[df_attr.columns[df_attr.columns.str.contains(name)]] = df_attr[df_attr.columns[df_attr.columns.str.contains(name)]].round(3)

df_attr = df_attr.drop(columns=[
    'ev', 'ig', 'mt', 'nd', 'pa', 'pb', 'pi', 'py', 'sc', 'sm', 'ss', 'su', 'va', 'vb', 'vi', 'wb', 
    'ohdb_altitude', 'ohdb_catchment_area', 'ohdb_is_public', 'ohdb_start_year', 'ohdb_end_year',
    'ohdb_data_availability', 'ohdb_post1983_data_availability', 'ohdb_duplicated_country', 'ohdb_river', 'ohdb_station_name',
    'domain', 'ohdbDarea0', 'ohdb_source_id', 
], errors = 'ignore')

# add climate 
ds = xr.open_dataset('../../data/koppen_5class_0p5.nc')
df_attr['climate'] = df_attr.apply(lambda x:float(ds.koppen.sel(lon=x.ohdb_longitude, lat=x.ohdb_latitude, method='nearest').values), axis = 1)
df_attr['climate_label'] = df_attr.climate.map({1:'tropical',2:'dry',3:'temperate',4:'cold',5:'polar'})
ds.close()

# create average fraction of sand, silt, and clay across different layers of soil
df_attr['clay'] = df_attr.loc[:,df_attr.columns.str.contains('clay_layer')].mean(axis = 1)
df_attr['sand'] = df_attr.loc[:,df_attr.columns.str.contains('sand_layer')].mean(axis = 1)
df_attr['silt'] = df_attr.loc[:,df_attr.columns.str.contains('silt_layer')].mean(axis = 1)

df_attr = df_attr.drop(columns=df_attr.columns[df_attr.columns.str.contains('_layer')])
df_attr = df_attr.drop(columns=df_attr.columns[df_attr.columns.str.contains('koppen')])

df_attr = df_attr.drop(columns=['logK_Ferr_','grwl_width_median'])

# merge
df = df_meteo.merge(df_lulc, on = ['ohdb_id','year']).merge(df_attr, on = 'ohdb_id')

# # add hydrologic signatures
# hs = pd.read_csv('../data/hydrologic_signatures.csv').rename(columns={'index':'ohdb_id'})
# df = df.merge(hs, on = 'ohdb_id')

# add dam impact: res_darea_normalize, Year_ave, and Main_Purpose_mode
dam = pd.read_csv('../data/dam_impact.csv')[['res_darea_normalize', 'Year_ave', 'Main_Purpose_mode', 'ohdb_id']].rename(columns={'Main_Purpose_mode':'Main_Purpose'})
df = df.merge(dam, on = 'ohdb_id', how = 'left')
df['res_darea_normalize'] = df['res_darea_normalize'].fillna(0)
df['Year_ave'] = df['Year_ave'].fillna(2024)
df['Main_Purpose'] = df['Main_Purpose'].fillna('NoRes').str.lower()
df.loc[df.Main_Purpose.str.contains('not specified'),'Main_Purpose'] = 'other'

df['climate_label'] = df.climate.map({1:'tropical',2:'dry',3:'temperate',4:'cold',5:'polar'})

# exclude polar gauges
df = df.loc[df.climate_label!='polar',:].reset_index(drop=True)

# create label-encoding variable for gauge id
x = pd.DataFrame({'ohdb_id':df.ohdb_id.unique(),'gauge_id':np.arange(1, df.ohdb_id.unique().shape[0]+1)})
df = df.merge(x, on = 'ohdb_id')

# create label-encoding variable for basin id
x = pd.DataFrame({'HYBAS_ID':df.HYBAS_ID.unique(),'basin_id':np.arange(df.HYBAS_ID.unique().shape[0])})
df = df.merge(x, on = 'HYBAS_ID')

# create label-encoding variable for dam purpose
x = pd.DataFrame({'Main_Purpose':df.Main_Purpose.unique(),'Main_Purpose_id':np.arange(df.Main_Purpose.unique().shape[0])})
df = df.merge(x, on = 'Main_Purpose')

# create label-encoding variable for season 
x = pd.DataFrame({'season':df.season.unique(),'season_id':np.arange(df.season.unique().shape[0])})
df = df.merge(x, on = 'season').reset_index(drop=True)
df.year = df.year.astype(np.float32)

print(df.ohdb_id.unique().shape)

# # add GDP
# df_GDP = pd.read_csv('../data_gdp/GDP_1km_catch_ave_1982-2023_linear_interp.csv')
# df = df.merge(df_GDP, on = ['ohdb_id', 'year'])

# # add population
# df_pop = pd.read_csv('../data_population/GHS_population_catch_ave_1982-2023_cubic_interp.csv')
# df_pop = df_pop.melt(id_vars = 'ohdb_id', var_name = 'year', value_name = 'population')
# df_pop.loc[df_pop.population<1e-3,'population'] = 0
# df_pop['year'] = df_pop['year'].astype(int)
# df = df.merge(df_pop, on = ['ohdb_id','year'])
# print(df.ohdb_id.unique().shape)

# round streamflow to three decimal place
df.loc[(df.Q>0)&(df.Q<0.0005),'Q'] = 0
df['Q'] = df['Q'].round(3)

# exclude reservoir gauges by identifying the name that contains the substring of 'RESERVOIR'
df_attr = pd.read_csv('../../data/OHDB/OHDB_v0.2.3/OHDB_metadata/OHDB_metadata.csv')
df_attr = df_attr.loc[~df_attr.ohdb_station_name.isna(),:]
ohdb_ids = df_attr.loc[df_attr.ohdb_station_name.str.contains('RESERVOIR'),'ohdb_id'].values
df = df.loc[~df.ohdb_id.isin(ohdb_ids),:]

if target == 'Qmax7':
    df = df.loc[df.Q>0,:]

# limit to those catchment area less than 100,000 km2
df = df.loc[(df.gritDarea<=100000),:]
print(df.ohdb_id.unique().shape)

# limit to catchments with at least 80 seasonal samples
tmp = df.groupby('ohdb_id').Q.count()
df = df.loc[df.ohdb_id.isin(tmp.loc[tmp>=80].index),:]

df.to_pickle(f'../data/{target}_final_dataset_seasonal4.pkl')

print(df.shape, df.ohdb_id.unique().shape)