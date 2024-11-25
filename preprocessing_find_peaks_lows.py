from pyogrio import read_dataframe,write_dataframe
import geopandas as gpd
import os,glob,sys,time,re
import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy.signal import find_peaks
from parallel_pandas import ParallelPandas
from tqdm.auto import tqdm
ParallelPandas.initialize(n_cpu=24, split_factor=24)

def cleanQ(df):
    # eliminate invalid records
    df1 = df.loc[df.Q.apply(lambda x: not isinstance(x, str)),:]
    df2 = df.loc[df.Q.apply(lambda x: isinstance(x, str)),:]
    try:
        df2 = df2.loc[df2.Q.str.match('\d+'),:]
    except:
        pass
    df = pd.concat([df1, df2])
    df['Q'] = df.Q.astype(np.float32)
    return df

def del_unreliableQ(df):
    '''observations less than 0 were flagged as
        suspected, and (b) observations with more than ten consecutive
        equal values greater than 0 were flagged as suspected'''
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    index = pd.date_range(df.index[0], df.index[-1], freq = 'D')
    df = df.reindex(index).fillna(0)
    df1 = df.diff()
    df1 = df1.where(df1==0, 1).diff()
    start = np.where(df1.values==-1)[0]
    end = np.where(df1.values==1)[0]
    if len(start) == 0 or len(end) == 0:
        # must no less than zero
        df = df.loc[df.Q>=0,:]
        return (df)
    if start[0] > end[0]:
        start = np.array([0]+start.tolist())
    if start[-1] > end[-1]:
        end = np.array(end.tolist()+[df1.shape[0]+10])
    duration = end - start
    start = start[duration>=10]
    end = end[duration>=10]
    del_idx = np.array([item for a,b in zip(start,end) for item in np.arange(a+1,b+2).tolist()])
    del_idx = del_idx[del_idx<df.shape[0]]
    if len(del_idx) > 0:
        df.drop(df.index[del_idx], inplace = True)
    # must no less than zero
    df = df.loc[df.Q>=0,:]
    return (df)

def get_peaks(df, Darea):
    Q = df.Q.values
    peak_idx, _ = find_peaks(Q)
    peak_idx = np.array([s for s in peak_idx if Q[s]>0])
    # thres for Qmax7
    thres = 5 + np.log(Darea * 0.386102) 
    peak_idx = np.array([peak_idx[0]] + peak_idx[1:][np.diff(peak_idx)>=thres].tolist())
    while True:
        to_delete = []
        for i,peak_idx0 in enumerate(peak_idx[:-1]):
            peak1 = Q[peak_idx0]
            peak2 = Q[peak_idx[i+1]]
            minQ = Q[peak_idx0:(peak_idx[i+1])].min()
            min_peak = min(peak1, peak2)
            if minQ / min_peak < 0.75:
                continue
            if peak1 < peak2:
                to_delete.append(peak_idx0)
            else:
                to_delete.append(peak_idx[i+1])
        if len(to_delete) == 0:
            break
        peak_idx = np.array(list(set(peak_idx)-set(to_delete)))
        peak_idx.sort()
    return peak_idx

def get_valleys(df, ohdb_id, thres = 30):
    valley_idx, _ = find_peaks(-1 * df.Q.values)
    if len(valley_idx) == 0:
        return []
    valley_idx = np.array([valley_idx[0]] + valley_idx[1:][np.diff(valley_idx)>=thres].tolist())
    Q = df.Q.values
    while True:
        to_delete = []
        for i,valley_idx0 in enumerate(valley_idx[:-1]):
            valley1 = Q[valley_idx0]
            valley2 = Q[valley_idx[i+1]]
            minQ = Q[valley_idx0:(valley_idx[i+1])].min()
            if minQ / min(valley1, valley2) < 0.75:
                continue
            if valley1 < valley2:
                to_delete.append(valley_idx0)
            else:
                to_delete.append(valley_idx[i+1])
        if len(to_delete) == 0:
            break
        valley_idx = np.array(list(set(valley_idx)-set(to_delete)))
        valley_idx.sort()
    return valley_idx

def main(par):
    ohdb_id, Darea = par
    print(ohdb_id)
    df = pd.read_csv(os.environ['DATA']+f'/data/OHDB/OHDB_v0.2.3/OHDB_data/discharge/daily/{ohdb_id}.csv')
    # read
    df = cleanQ(df)
    # quality check
    df = del_unreliableQ(df)
    # only retain records with at least 328 observations (90%) are required
    tmp = df.resample('Y')['Q'].agg(countDay = lambda x:x.shape[0])
    if tmp.loc[tmp.countDay>=300,:].shape[0] == 0:
        return
    years = tmp.loc[(tmp.countDay>=300)&(tmp.index.year>=1982),:].index.year.tolist()
    df = df.loc[df.index.year.isin(years),:]
    # only retain gauge with at least 20 years of AMS during 1982-2023
    if len(years) < 20:
        return

    # reindex
    newidx = pd.date_range(df.index.values[0], df.index.values[-1], freq = 'D')
    df = df.reindex(newidx)

    # 7-day moving average
    df = df.rolling(7).mean().dropna()
    df['year'] = df.index.year

    # # Type 1: calculate Qmax7 and Qmin7 for each year
    # df1 = df.groupby('year')['Q'].agg(countDay = lambda x:x.shape[0], 
    #                                 Qmax7 = lambda x:x.max(),
    #                                 Qmin7 = lambda x:x.min(),
    #                                 Qmax7date = lambda x:x.idxmax(),
    #                                 Qmin7date = lambda x:x.idxmin(),
    #                                 )
    
    # # Type 2: calculate Qmax7 and Qmin7 for each of summer (5-10) and winter (11-4) seasons
    # df['season'] = 'winter'
    # df.loc[(df.index.month>=5)&(df.index.month<=10),'season'] = 'summer'
    # df1 = df.groupby(['year','season'])['Q'].agg(countDay = lambda x:x.shape[0], 
    #                                             Qmax7 = lambda x:x.max(),
    #                                             Qmin7 = lambda x:x.min(),
    #                                             Qmax7date = lambda x:x.idxmax(),
    #                                             Qmin7date = lambda x:x.idxmin(),
    #                                             ).reset_index()

    # Type 3: use scipy find_peaks
    peaks = get_peaks(df, Darea)
    df_Qmax = df.iloc[peaks,:].reset_index(names='date')
    df_Qmax['type'] = 'Qmax7'
    valleys = get_valleys(df, ohdb_id)
    df_Qmin = df.iloc[valleys,:].reset_index(names='date')
    df_Qmin['type'] = 'Qmin7'
    df1 = pd.concat([df_Qmax, df_Qmin])

    df1['ohdb_id'] = ohdb_id
    return (df1)

if __name__ == '__main__':
    # if not os.path.exists('../data/OHDB_metadata_subset.csv'):
    #     # select gauges that have good basin boundary
    #     df = pd.read_csv('../../data/OHDB/OHDB_v0.2.3/OHDB_metadata/OHDB_metadata.csv')
    #     df1 = []
    #     for fname in glob.glob('../basin_boundary/GRIT*8857*'):
    #         gdf = read_dataframe(fname, read_geometry = False)
    #         print(gdf.shape, gdf.ohdb_id.unique().shape)
    #         df1.append(df.loc[df.ohdb_id.isin(gdf.ohdb_id.unique()),:])
    #     df1 = pd.concat(df1)
    #     df1.to_csv('../OHDB_metadata_subset.csv', index = False)
    # else:
    df1 = pd.read_csv('../data/basin_attributes.csv')
    print(df1.shape)
    ohdb_ids = df1.ohdb_id.values
    pool = mp.Pool(48)
    pars = df1[['ohdb_id','gritDarea']].values.tolist()
    df = pool.map(main, pars)
    df = pd.concat(df)
    df.to_csv('../data/dis_OHDB_local_Qmin7_Qmax7_1982-2023.csv', index = False)
# X = df.loc[df.type=='Qmax7',:].groupby('ohdb_id')['Q'].count()
# Y = df.loc[df.type=='Qmin7',:].groupby('ohdb_id')['Q'].count()
# print(X.loc[X>36].shape)
# print(Y.loc[Y>36].shape)

# df['Qmax7date'] = pd.to_datetime(df['Qmax7date'])
# df['Qmin7date'] = pd.to_datetime(df['Qmin7date'])

# def func(x):
#     xx = []
#     for name in ['Qmax7','Qmin7']:
#         x1 = x[['year','season','countDay',name,name+'date','ohdb_id']]
#         x1 = x1.sort_values(name+'date').reset_index(drop = True)
#         x1['tmp'] = x1[name+'date'].diff().dt.days
#         to_del = []
#         while True:
#             if x1.tmp.min() > 30:
#                 break
#             idx = 
#             for i in 

# df.groupby('ohdb_id').apply(lambda x:x.sort_values('Qmax7date').Qmax7date.diff().dt.days.min()).min()


# sce = 'diff_245'
# p = 'p_245'
# df33['sss'] = 'None'
# df33.loc[(df33[sce]<0)&(df33[p]>=15),'sss'] = 'sig -'
# df33.loc[(df33[sce]>0)&(df33[p]>=15),'sss'] = 'sig +'
# df33.loc[(df33[sce]<0)&(df33[p]>=10)&(df33[p]<15),'sss'] = 'med -'
# df33.loc[(df33[sce]>0)&(df33[p]>=10)&(df33[p]<15),'sss'] = 'med +'
# df33.loc[(df33[sce]<0)&(df33[p]<10),'sss'] = 'low -'
# df33.loc[(df33[sce]>0)&(df33[p]<10),'sss'] = 'low +'
