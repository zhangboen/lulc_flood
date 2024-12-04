from pyogrio import read_dataframe,write_dataframe
import geopandas as gpd
import os,glob,sys,time,re
import pandas as pd
import numpy as np
import multiprocessing as mp
from parallel_pandas import ParallelPandas
from tqdm.auto import tqdm
ParallelPandas.initialize(n_cpu=24, split_factor=24)

# func to calculate averages in a parallel manner
def func_meteo(x):
    x['tmp'] = (x.date - x.Qdate).dt.days
    x3 = x.loc[(x.tmp>-3)&(x.tmp<=0),:].drop(columns=['date','Qdate','ohdb_id','tmp']).mean().rename(index=lambda x:x+'_3')
    x7 = x.loc[(x.tmp>-7)&(x.tmp<=0),:].drop(columns=['date','Qdate','ohdb_id','tmp']).mean().rename(index=lambda x:x+'_7')
    x15 = x.loc[(x.tmp>-15)&(x.tmp<=0),:].drop(columns=['date','Qdate','ohdb_id','tmp']).mean().rename(index=lambda x:x+'_15')
    x30 = x.loc[(x.tmp>-30)&(x.tmp<=0),:].drop(columns=['date','Qdate','ohdb_id','tmp']).mean().rename(index=lambda x:x+'_30')
    x = pd.concat([x3,x7,x15,x30])
    return x

df_flood = pd.read_csv('../data/dis_OHDB_seasonal4_Qmin7_Qmax7_1982-2023.csv')
df_flood['Qmax7date'] = pd.to_datetime(df_flood['Qmax7date'])
df_flood['Qmin7date'] = pd.to_datetime(df_flood['Qmin7date'])

meteo = 'MSWX'
for year in range(1982, 2024):
    if meteo == 'ERA5':
        fname = f'../ee_era5_land/ERA5_Land_daily_meteorology_for_OHDB_10717_stations_{year}.csv'
    elif meteo == 'MSWX':
        fname = f'../data_mswx/MSWX_daily_meteorology_for_OHDB_10717_stations_{year}.csv'
    df = pd.read_csv(fname)
    df = df.melt(id_vars = 'ohdb_id')
    df['date'] = df.variable.apply(lambda x:x[:4]+'-'+x[4:6]+'-'+x[6:8])
    df['meteo'] = df.variable.str[9:]
    df = df.drop(columns=['variable'])
    df = df.pivot_table(index = ['ohdb_id','date'], columns = 'meteo', values = 'value').reset_index()
    df['date'] = pd.to_datetime(df.date.values)

    if meteo == 'ERA5':
        # change meteo unit
        df[['snow_depth_water_equivalent','snowmelt_sum','total_evaporation_sum','total_precipitation_sum']] = df[['snow_depth_water_equivalent','snowmelt_sum','total_evaporation_sum','total_precipitation_sum']] * 1000
        df[['snow_depth_water_equivalent','snowmelt_sum','total_evaporation_sum','total_precipitation_sum']] = df[['snow_depth_water_equivalent','snowmelt_sum','total_evaporation_sum','total_precipitation_sum']].round(3)
        df[['temperature_2m_min','temperature_2m_max']] = df[['temperature_2m_min','temperature_2m_max']] - 273.15
        df[['temperature_2m_min','temperature_2m_max']] = df[['temperature_2m_min','temperature_2m_max']].round(3)
        df['surface_net_solar_radiation_sum'] = df['surface_net_solar_radiation_sum'] * 1e-6
        df['surface_net_solar_radiation_sum'] = df['surface_net_solar_radiation_sum'].round(3)
        # change meteo name
        df = df.rename(columns = {
            'snow_depth_water_equivalent':'swe',
            'snowmelt_sum':'swmelt',
            'total_evaporation_sum':'evap',
            'total_precipitation_sum':'pr',
            'temperature_2m_min':'t2min',
            'temperature_2m_max':'t2max',
            'surface_net_solar_radiation_sum':'srad'
        })
    elif meteo == 'MSWX':
        df = df.rename(columns = lambda x:x.lower())

    df_flood0 = df_flood.loc[df_flood.year==year,:]

    df_Qmax7 = df.merge(df_flood0[['ohdb_id','Qmax7date']].rename(columns={'Qmax7date':'Qdate'}), on = 'ohdb_id')
    print(df_Qmax7.shape)
    df_Qmax7 = df_Qmax7.groupby('ohdb_id').p_apply(func_meteo).reset_index().assign(year=year)
    df_Qmax7 = df_Qmax7.merge(df_flood0[['ohdb_id','Qmax7','Qmax7date']], on = 'ohdb_id')
    df_Qmax7.to_csv(f'../data/Qmax7_seasonal4_MSWX_meteo_multi_{year}.csv', index = False)

    df_Qmin7 = df.merge(df_flood0[['ohdb_id','Qmin7date']].rename(columns={'Qmin7date':'Qdate'}), on = 'ohdb_id')
    print(df_Qmin7.shape)
    df_Qmin7 = df_Qmin7.groupby('ohdb_id').p_apply(func_meteo).reset_index().assign(year=year)
    df_Qmin7 = df_Qmin7.merge(df_flood0[['ohdb_id','Qmin7','Qmin7date']], on = 'ohdb_id')
    df_Qmin7.to_csv(f'../data/Qmin7_seasonal4_MSWX_meteo_multi_{year}.csv', index = False)