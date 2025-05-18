import os,glob,sys,time,re
import pandas as pd
import numpy as np
import multiprocessing as mp
from parallel_pandas import ParallelPandas
ParallelPandas.initialize(n_cpu=16, split_factor=16)

##############################################################################################################
# NOTE: this script is used to transform GRIT_catch_ave_{***}_{year}.csv to separate csv for each gauge
##############################################################################################################

# # transform all the meteo files into separate csv files for each gauge
# def read(year):
#     df2 = pd.read_csv(f'../gleam_data/GRIT_catch_ave_E_{year}_GLEAM_v4.2a.csv').assign(a='evap')
#     df3 = pd.read_csv(f'../gleam_data/GRIT_catch_ave_SMrz_{year}_GLEAM_v4.2a.csv').assign(a='smrz')
#     try:
#         df1 = pd.read_csv(f'../data_mswx/GRIT_catch_ave_pr_MSWEP_{year}.csv').assign(a='pr_mswep')
#         df1.iloc[:,:-2] = df1.iloc[:,:-2] / 10
#         df = pd.concat([df1, df2, df3])
#     except:
#         df = pd.concat([df2, df3])
#     print(year)
#     return df
# pool = mp.Pool(8)
# df_meteo = pool.map(read, np.arange(1981, 2024).tolist())
# df_meteo = pd.concat(df_meteo)

# df_meteo1 = df_meteo.loc[df_meteo.a=='pr_mswep',:].drop(columns=['a']).melt(id_vars = 'time', var_name = 'ohdb_id', value_name = 'pr_mswep')
# df_meteo2 = df_meteo.loc[df_meteo.a=='evap',:].drop(columns=['a']).melt(id_vars = 'time', var_name = 'ohdb_id', value_name = 'evap')
# df_meteo3 = df_meteo.loc[df_meteo.a=='smrz',:].drop(columns=['a']).melt(id_vars = 'time', var_name = 'ohdb_id', value_name = 'smrz')
# df_meteo = df_meteo1.merge(df_meteo2, on = ['time', 'ohdb_id'], how = 'right').merge(df_meteo3, on = ['time', 'ohdb_id'])
# df_meteo = df_meteo.set_index(['time','ohdb_id'])
# df_meteo = df_meteo.round(6)
# df_meteo = df_meteo.reset_index()

# df_meteo.groupby('ohdb_id').p_apply(lambda x:x.drop(columns=['ohdb_id']).rename(columns={'time':'date'}).to_csv(f'../data_mswx/mswx_each_basin/{x.ohdb_id.values[0]}_mswep_gleam.csv', index = False))

# ------------------------------------------------------------------------------------------------------------------------------------------------
# connect Qmin7 and Qmax7 records with meteo records
# func to calculate averages in a parallel manner
def func_meteo(x, name = 'Qmax7date'):
    ohdb_id = x.xxx.values[0]
    x = x.drop(columns=['xxx'])
    df_mswx = pd.read_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}.csv')
    df_mswx['date'] = pd.to_datetime(df_mswx['date'])
    df_meteo = pd.read_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}_mswep_gleam.csv')
    df_meteo['date'] = pd.to_datetime(df_meteo['date'])
    df_meteo['ohdb_id'] = ohdb_id
    df_meteo = df_meteo.merge(df_mswx, on = 'date')
    x = x[['ohdb_id',name]].merge(df_meteo, on = 'ohdb_id')
    # x['p_meanXstd'] = x['p_mean'] * x['p_std']
    x['tmp'] = (x.date - x[name]).dt.days
    x3 = x.loc[(x.tmp>-3)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_3')
    x7 = x.loc[(x.tmp>-7)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_7')
    x15 = x.loc[(x.tmp>-15)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_15')
    x30 = x.loc[(x.tmp>-30)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_30')
    x365 = x.loc[(x.tmp>-365)&(x.tmp<=0),:].drop(columns=['date','ohdb_id','tmp']).groupby(name).mean().rename(columns=lambda x:x+'_365')
    x = pd.concat([x3,x7,x15,x30,x365], axis = 1).reset_index()
    return x

fname = '../data/dis_OHDB_seasonal4_Qmin7_Qmax7_1982-2023.csv'
df_flood = pd.read_csv(fname)
df_flood['Qmax7date'] = pd.to_datetime(df_flood['Qmax7date'])
df_flood['Qmin7date'] = pd.to_datetime(df_flood['Qmin7date'])

df_flood['xxx'] = df_flood['ohdb_id'].values

tmp = os.path.basename(fname).split('_')[2]

df2 = df_flood.groupby('ohdb_id').p_apply(lambda x: func_meteo(x, name = 'Qmax7date')).reset_index().drop(columns = ['level_1'])
print(df2.shape)
df2.to_csv(f'../data/Qmax7_{tmp}_multi_MSWX_MSWEP_GLEAM.csv', index = False)
del df2

df2 = df_flood.groupby('ohdb_id').p_apply(lambda x: func_meteo(x, name = 'Qmin7date')).reset_index().drop(columns = ['level_1'])
print(df2.shape)
df2.to_csv(f'../data/Qmin7_{tmp}_multi_MSWX_MSWEP_GLEAM.csv', index = False)