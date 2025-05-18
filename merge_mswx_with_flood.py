from pyogrio import read_dataframe,write_dataframe
import geopandas as gpd
import os,glob,sys,time,re
import pandas as pd
import numpy as np
import multiprocessing as mp
from parallel_pandas import ParallelPandas
from tqdm.auto import tqdm
ParallelPandas.initialize(n_cpu=16, split_factor=16)

# transform all the meteo files into separate csv files for each gauge
def read(year, meteo = 'MSWX'):
    if meteo == 'ERA5':
        fname = f'../ee_era5_land/ERA5_Land_daily_meteorology_for_OHDB_10717_stations_{year}.csv'
    elif meteo == 'MSWX':
        fname = f'../data_mswx/MSWX_daily_meteorology_for_OHDB_10717_stations_{year}.csv'
    df = pd.read_csv(fname).set_index('ohdb_id')
    print(year)
    return df
pool = mp.Pool(8)
df_meteo = pool.map(read, np.arange(1982, 2024).tolist())
df_meteo = pd.concat(df_meteo, axis = 1)
df_meteo = df_meteo.rename(columns=lambda x:x.lower())
df_meteo = df_meteo.round(6)
df_meteo.loc[:,df_meteo.columns.str.endswith(('_p','_tmax','_tmin','_wind'))] = df_meteo.loc[:,df_meteo.columns.str.endswith(('_p','_tmax','_tmin','_wind'))].round(2)
df_meteo = df_meteo.reset_index()

def func_main(x, meteo = 'MSWX'):
    ohdb_id = x.ohdb_id
    if os.path.exists(f'../data_mswx/mswx_each_basin/{ohdb_id}.csv'):
        try:
            a = pd.read_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}.csv')
            return
        except:
            os.remove(f'../data_mswx/mswx_each_basin/{ohdb_id}.csv')
    x = x.drop(index=['ohdb_id'])
    x.name = 'value'
    y = []
    for name in ['p','tmax','tmin','lwd','pres','relhum','spechum','swd','wind']:
        y0 = x.loc[x.index.str.endswith('_'+name)]
        y0.name = name
        y0.index = y0.index.str[:8]
        y.append(y0)
    y = pd.concat(y, axis = 1)
    # x = x.pivot_table(index = 'date', columns = 'meteo', values = 'value').rename(columns=lambda x:x.lower()).reset_index()
    # x['date'] = pd.to_datetime(x.date.values)
    # x.to_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}.csv', index = False)
    y['date'] = pd.to_datetime(y.index.values, format = '%Y%m%d')
    y.to_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}.csv', index = False)

df_meteo.p_apply(func_main, axis = 1)

# ------------------------------------------------------------------------------------------------------------------------------------------------
# connect Qmin7 and Qmax7 records with meteo records
# func to calculate averages in a parallel manner
def func_meteo(x, name = 'Qmax7date'):
    ohdb_id = x.xxx.values[0]
    x = x.drop(columns=['xxx'])
    df_meteo = pd.read_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}.csv')
    df_meteo['date'] = pd.to_datetime(df_meteo['date'])
    df_meteo['ohdb_id'] = ohdb_id
    x = x[['ohdb_id',name]].merge(df_meteo,on = 'ohdb_id')
    x['tmp'] = (x.date - x[name]).dt.days
    x3 = x.loc[(x.tmp>-3)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_3')
    x7 = x.loc[(x.tmp>-7)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_7')
    x15 = x.loc[(x.tmp>-15)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_15')
    x30 = x.loc[(x.tmp>-30)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_30')
    x365 = x.loc[(x.tmp>-365)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_365')
    x = pd.concat([x3,x7,x15,x30,x365], axis = 1).reset_index()
    return x

fname = '../data/dis_OHDB_seasonal_Qmin7_Qmax7_1982-2023.csv'
df_flood = pd.read_csv(fname)
df_flood['Qmax7date'] = pd.to_datetime(df_flood['Qmax7date'])
df_flood['Qmin7date'] = pd.to_datetime(df_flood['Qmin7date'])

df_flood['xxx'] = df_flood['ohdb_id'].values

tmp = os.path.basename(fname).split('_')[2]

df2 = df_flood.groupby('ohdb_id').p_apply(lambda x: func_meteo(x, name = 'Qmax7date')).reset_index().drop(columns = ['level_1'])
print(df2.shape)
df2.to_csv(f'../data/Qmax7_{tmp}_multi_MSWX_meteo.csv', index = False)
del df2

df2 = df_flood.groupby('ohdb_id').p_apply(lambda x: func_meteo(x, name = 'Qmin7date')).reset_index().drop(columns = ['level_1'])
print(df2.shape)
df2.to_csv(f'../data/Qmin7_{tmp}_multi_MSWX_meteo.csv', index = False)