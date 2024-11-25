import os,glob,cmaps,sys
import numpy as np
import pandas as pd
import xarray as xr

target = sys.argv[1]
if target not in ['Qmax7','Qmin7']:
    raise Exception('target must be Qmax7 or Qmin7')

###################################################################
# read dataset
if target == 'Qmax7':
    df_meteo = pd.concat([pd.read_csv(fname) for fname in glob.glob('../data/Qmax7_meteo*csv')])
    df_meteo = df_meteo.rename(columns={'Qmax7':'Q'})
else:
    df_meteo = pd.concat([pd.read_csv(fname) for fname in glob.glob('../data/Qmin7_meteo30*csv')])
    df_meteo = df_meteo.rename(columns={'Qmin7':'Q'})

df_lulc = pd.concat([pd.read_csv(fname).set_index('ohdb_id') for fname in glob.glob('../ee_lulc/*csv')], axis = 1).reset_index()
df_lulc = df_lulc.melt(id_vars = 'ohdb_id')
df_lulc['year'] = df_lulc.variable.str[:4].astype(int)
df_lulc['var'] = df_lulc.variable.str[5:]
df_lulc = df_lulc.pivot_table(index = ['ohdb_id','year'], columns = 'var', values = 'value').reset_index()

df_attr = pd.read_csv('../data/basin_attributes.csv')

# merge soil texture fraction
df_attr['sedimentary'] = df_attr[['su', 'ss', 'sm', 'sc']].fillna(0).sum(1)
df_attr['plutonic'] = df_attr[['pa', 'pb', 'pi']].fillna(0).sum(1)
df_attr['volcanic'] = df_attr[['va', 'vi', 'vb']].fillna(0).sum(1)
df_attr['metamorphic'] = df_attr['mt'].copy()

# add climate 
ds = xr.open_dataset('../../data/koppen_5class_0p5.nc')
df_attr['climate'] = df_attr.apply(lambda x:float(ds.koppen.sel(lon=x.ohdb_longitude, lat=x.ohdb_latitude, method='nearest').values), axis = 1)
df_attr['climate_label'] = df_attr.climate.map({1:'tropical',2:'dry',3:'temperate',4:'cold',5:'polar'})
ds.close()

# merge
df = df_meteo.merge(df_lulc, on = ['ohdb_id','year']).merge(df_attr, on = 'ohdb_id')

# Q cannot be too small
df = df.loc[(df.Q==0)|(df.Q>1e-6),:]

df.to_csv(f'../data/{target}_final_dataset.csv', index = False)