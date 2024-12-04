import pandas as pd
import os,glob
import multiprocessing as mp
import numpy as np
from parallel_pandas import ParallelPandas
ParallelPandas.initialize(n_cpu=24, split_factor=12)

def func(x):
    ohdb_id = x.ohdb_id.values[0]
    name = f'lstm/data/mswx_forcing/{ohdb_id}_daily_meteo_mswx_1982-2023.csv'
    if os.path.exists(name):
        return
    df0 = x.drop(columns=['ohdb_id']).T.reset_index()
    df0.columns = ['tmp','value']
    df0[['date','name']] = df0['tmp'].str.split('_',n=2,expand=True)
    df1 = df0.loc[df0.date.str.len()==4,:]
    df0 = df0.loc[df0.date.str.len()==8,:]
    df0['date'] = pd.to_datetime(df0['date'],format='%Y%m%d')
    df0['year'] = df0.date.dt.year
    df0['month'] = df0.date.dt.month
    df0['day'] = df0.date.dt.day
    df0 = df0.drop(columns=['date'])
    df0 = df0.pivot_table(index = ['year','month','day'], columns = 'name', values = 'value')
    df0 = df0.reset_index()
    df0 = df0.rename(columns={
        'LWd':'LWd(W/m)',
        'P':'P(mm/day)',
        'Pres':'Pres(Pa)',
        'Tmin':'Tmin(C)',
        'Tmax':'Tmax(C)',
        'Wind':'Wind(m/s)',
        'SpecHum':'SpecHum(g/g)',
        'SWd':'SWd(W/m)',
        'RelHum':'RelHum(%)',
    })
    df0['LWd(W/m)'] = df0['LWd(W/m)'].round(3)
    df0['P(mm/day)'] = df0['P(mm/day)'].round(1)
    df0['Pres(Pa)'] = df0['Pres(Pa)'].round(3)
    df0['Tmin(C)'] = df0['Tmin(C)'].round(1)
    df0['Tmax(C)'] = df0['Tmax(C)'].round(1)
    df0['Wind(m/s)'] = df0['Wind(m/s)'].round(1)
    df0['SpecHum(g/g)'] = df0['SpecHum(g/g)'].round(5)
    df0['SWd(W/m)'] = df0['SWd(W/m)'].round(3)
    df0['RelHum(%)'] = df0['RelHum(%)'].round(1)

    df1 = df1.rename(columns={'date':'year'})
    df1['year'] = df1.year.astype(int)
    df1['value'] = df1['value'] / 100
    df1['value'] = df1['value'].round(2)
    df1 = df1.pivot_table(index='year', columns = 'name', values = 'value').rename(columns=lambda x:x+'(%)').reset_index()
    df0 = df0.merge(df1, on = 'year')

    f = open(name, 'w')
    f.write(' %.2f\n'%(df_attr.loc[df_attr.ohdb_id==ohdb_id,'ohdb_latitude'].values[0]))  # write latitude
    f.write('%.2f\n'%(df_attr.loc[df_attr.ohdb_id==ohdb_id,'elevation'].values[0]))  # write elevation
    f.write(' %d\n'%(df_attr.loc[df_attr.ohdb_id==ohdb_id,'gritDarea'].values[0]))  # write catchment area in km2
    df0.to_csv(f, index = False, sep = ' ')
    f.close()

def f(x):
    df = pd.read_csv(x).set_index('ohdb_id')
    return df

fnames = glob.glob('../data_mswx/*meteo*csv')
pool = mp.Pool(8)
df = pool.map(f, fnames)
df = pd.concat(df, axis = 1)
print(df.shape)

# merge with lulc
df_lulc = pd.concat([pd.read_csv(fname).set_index('ohdb_id') for fname in glob.glob('../ee_lulc/*csv')], axis = 1)
print(df_lulc.shape)
df = pd.concat([df, df_lulc], axis = 1)
df = df.reset_index()

df_attr = pd.read_csv('../data/basin_attributes.csv')

df.groupby('ohdb_id').p_apply(func)