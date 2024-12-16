import os,glob,cmaps,sys
import numpy as np
import pandas as pd
import xarray as xr

target = sys.argv[1]
if target not in ['Qmax7','Qmin7']:
    raise Exception('target must be Qmax7 or Qmin7')

###################################################################
# read meteo dataset
fname = f'../data/{target}_seasonal_multi_MSWX_meteo.csv'
df_meteo = pd.read_csv(fname)

# merge with discharge
df_dis = pd.read_csv('../data/dis_OHDB_seasonal_Qmin7_Qmax7_1982-2023.csv')

df_meteo[target+'date'] = pd.to_datetime(df_meteo[target+'date'])
df_dis[target+'date'] = pd.to_datetime(df_dis[target+'date'])

df_meteo = df_meteo.merge(df_dis[[target+'date','season','year',target, 'ohdb_id']], on = ['ohdb_id', target+'date'])
df_meteo = df_meteo.rename(columns={target:'Q'})

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

df_attr = pd.read_csv('../data/basin_attributes.csv')

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

# merge
df = df_meteo.merge(df_lulc, on = ['ohdb_id','year']).merge(df_attr, on = 'ohdb_id')

# add hydrologic signatures
hs = pd.read_csv('../data/hydrologic_signatures.csv').rename(columns={'index':'ohdb_id'})
df = df.merge(hs, on = 'ohdb_id')

# add dam impact: res_darea_normalize, Year_ave, and Main_Purpose_mode
dam = pd.read_csv('../data/dam_impact.csv')[['res_darea_normalize', 'Year_ave', 'Main_Purpose_mode', 'ohdb_id']].rename(columns={'Main_Purpose_mode':'Main_Purpose'})
df = df.merge(dam, on = 'ohdb_id', how = 'left')
df['res_darea_normalize'] = df['res_darea_normalize'].fillna(0)
df['Year_ave'] = df['Year_ave'].fillna(2024)
df['Main_Purpose'] = df['Main_Purpose'].fillna('NoRes').str.lower()
df.loc[df.Main_Purpose.str.contains('not specified'),'Main_Purpose'] = 'other'

# round small values to 0.001
df.loc[(df.Q>0)&(df.Q<0.001),'Q'] = 0.001

if 'seasonal4' in fname:
    df.to_csv(f'../data/{target}_final_dataset_seasonal4_multi_MSWX_meteo.csv', index = False)
else:
    df.to_csv(f'../data/{target}_final_dataset_seasonal_multi_MSWX_meteo.csv', index = False)
print(df.shape)