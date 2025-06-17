import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas
ParallelPandas.initialize(n_cpu=24, split_factor=12)

def read_meteo(ohdb_id):
    df = pd.read_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}_1981-2023.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date')[['p','tmax','tmin','swd','snowmelt','smrz','snowfall']].apply(lambda x:x.iloc[0,:]).reset_index()
    df = df.sort_values('date').set_index('date')
    df['rain'] = df.p - df.snowfall
    df = df[['p','rain','tmax','tmin','swd','snowmelt','smrz','snowfall']].rename(columns={'p':'total'})
    return df

def classify_high_flow(par):
    '''
    Classify seasonal high-flow events into:
        (1) rain/snow flood
        (2) snowmelt flood
        (3) excess rain flood
        (4) heavy rain flood
        (5) others: stratiform rain flood
    '''
    ohdb_id = par.ohdb_id.values[0]
    dates = par.date.values
    df = read_meteo(ohdb_id)

    # some thresholds
    p90d7 = df.rain.rolling(7).mean().quantile(.9)
    p7_mean = df.rain.rolling(7).mean().mean()
    sm90d7 = df.snowmelt.rolling(7).mean().quantile(.9)
    sw_thres = df.smrz.rolling(7).mean().mean() * 1.28 # Zhang et al., 2022

    types = []
    for date in dates:
        df0 = df.loc[(df.index>=date-pd.Timedelta(days=6))&(df.index<=date),:].mean(0)
        if df0.snowmelt > 1/3 * df0.total and df0.snowmelt < 2/3 * df0.total:
            type0 = 'rain/snow flood'
        elif df0.snowmelt > sm90d7:
            type0 = 'snowmelt flood'
        elif df0.smrz > sw_thres and df0.rain > p7_mean:
            type0 = 'excess rain flood'
        elif df0.rain > p90d7:
            type0 = 'heavy rain flood'
        else:
            type0 = 'stratiform rain flood'
        types.append(type0)
    df_out = pd.DataFrame({
        'ohdb_id': ohdb_id,
        'date': dates,
        'classification': types
    })
    return df_out

def classify_low_flow(par):
    '''
    Classify seasonal low-flow events into:
        (1) 
    '''
    ohdb_id = par.ohdb_id.values[0]
    dates = par.date.values
    df = read_meteo(ohdb_id)
    
    # identify precipitation deficits by applying a variable threshold at the 20th percentile, using a moving window of 30 days
    df30 = df.rain.rolling(30).mean()
    df30['doy'] = df30.index.dayofyear
    df30['year'] = df30.index.year
    df30 = df30.pivot_table(index='doy', columns = 'year', values = 'rain')
    df30 = df30.reindex(np.arange(df30.index.values[0], df30.index.values[-1]+1, 1))
    min_idx = df30.index.values[0]
    thres = df30.apply(
        lambda x:df30.loc[(max(0,int(x.name)-15)):(int(x.name)+16)].stack().quantile(q), 
        axis = 1)
    df30 = df30.apply(lambda x:np.where(x<=thres, x, np.nan), axis = 0)
    rain_deficit = df30.stack().reset_index().rename(
        columns={0:'rain30'}).sort_values(['year','doy']).reset_index(drop=True)
    del df30

    types = []
    for date in dates:
        df0 = df.loc[(df.index>=date-pd.Timedelta(days=6))&(df.index<=date),:].mean(0)
        if df0.snowmelt > 1/3 * df0.total and df0.snowmelt < 2/3 * df0.total:
            type0 = 'rain/snow flood'
        elif df0.snowmelt > sm90d7:
            type0 = 'snowmelt flood'
        elif df0.smrz > sw_thres and df0.rain > p7_mean and df0.rain < p90d7:
            type0 = 'excess rain flood'
        elif df0.rain > p90d7:
            type0 = 'heavy rain flood'
        else:
            type0 = 'stratiform rain flood'
        types.append(type0)
    df_out = pd.DataFrame({
        'ohdb_id': ohdb_id,
        'date': dates,
        'classification': types
    })
    return df_out

# classify high-flow events
df = pd.read_pickle('../data/Qmax7_final_dataset_seasonal4.pkl')
df = df.rename(columns={'Qmax7date':'date'})
df_out = df[['ohdb_id','date']].groupby('ohdb_id').p_apply(classify_high_flow).reset_index(drop=True)
df_out = df_out.merge(df[['ohdb_id','ohdb_longitude','ohdb_latitude','aridity','climate_label']].drop_duplicates(), on = 'ohdb_id')
df_out.to_csv('../results/high_flow_classification.csv', index = False)

# # classify low-flow events
# df = pd.read_pickle('../data/Qmin7_final_dataset_seasonal4.pkl')
# df = df.rename(columns={'Qmin7date':'date'})



