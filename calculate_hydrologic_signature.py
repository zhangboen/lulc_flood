import pandas as pd
import numpy as np
import multiprocessing as mp
import os,glob
from scipy.signal import find_peaks

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

def Eckhardt(Q, alpha=.98, BFI=.80, re=1):
    """
    Recursive digital filter for baseflow separation. Based on Eckhardt, 2004.\n
    Q : array of discharge measurements\n
    alpha : filter parameter\n
    BFI : BFI_max (maximum baseflow index)\n
    re : number of times to run filter
    """
    Q = np.array(Q)
    f = np.zeros(len(Q))
    f[0] = Q[0]
    for t in np.arange(1,len(Q)):
        # algorithm
        f[t] = ((1 - BFI) * alpha * f[t-1] + (1 - alpha) * BFI * Q[t]) / (1 - alpha * BFI)
        if f[t] > Q[t]:
            f[t] = Q[t]
    # calls method again if multiple passes are specified
    return np.nansum(f)/np.nansum(Q)

# Function to identify and delete the smaller sample within 5 days
def delete_smaller_sample(df):
    while True:
        intervals = (df.index[1:] - df.index[:-1]).days
        if (intervals > 5).all():
            break
        indices1 = df.index[:-1][intervals<=5]
        indices2 = df.index[1:][intervals<=5]
        for int1,int2 in zip(indices1,indices2):
            if int1 not in df.index:
                continue
            if df.loc[int1] < df.loc[int2]:
                df.drop(int1, inplace=True)
            else:
                df.drop(int2, inplace=True)
    return df

# Function to calculate lag time between rainfall peak and discharge peak
def calc_lagT(pr_dis, p = 95):
    tmp = pr_dis.unstack(level=-1)
    # use scipy to find local discharge peaks
    R95p = tmp.loc[tmp.pr>=0.1,'pr'].quantile(p/100)
    peaks_pr, _ = find_peaks(tmp.pr.values, height=R95p) # find R95p rainfall peaks
    peaks_dis, _ = find_peaks(tmp.dis.values, height=0)
    # remove peaks within five days
    df_peaks_pr = tmp.iloc[peaks_pr,:].pr
    df_peaks_pr = delete_smaller_sample(df_peaks_pr)
    df_peaks_dis = tmp.iloc[peaks_dis,:].dis
    df_peaks_dis = delete_smaller_sample(df_peaks_dis)
    # calculate lag time bewteen rainfall peaks and discharge peaks
    lagT = []
    noResponse = 0
    for index_pr in df_peaks_pr.index:
        index_dis = df_peaks_dis.loc[df_peaks_dis.index>=index_pr]
        if index_dis.shape[0] == 0:
            noResponse += 1
            continue
        index_dis = index_dis.iloc[[0]].index[0]
        if df_peaks_pr.loc[(df_peaks_pr.index>index_pr)&(df_peaks_pr.index<=index_dis)].shape[0] > 0:
            noResponse += 1
            continue
        lagT.append((index_dis - index_pr).days)
    if df_peaks_pr.shape[0] == 0:
        noResRatio = 1
    else:
        noResRatio = noResponse/df_peaks_pr.shape[0]
    return (pd.Series([np.mean(lagT), noResRatio], index = ['lagT','noResRatio',]))

def calc_hs(df):
    '''df should be a dataframe, include four columns: date, pr, dis, lat, and darea'''
    if len(set(['date','pr','dis','lat','darea'])-set(df.columns.tolist())) != 0:
        raise Exception ('the input df misses columns: date, pr, dis, and darea') 

    # transform streamflow to specific discharge
    df['dis'] = df.dis.values / df.darea.values
    lat = df.lat.values[0]
    df = df.drop(columns=['darea','lat']).set_index('date')

    newtime = pd.date_range(df.index.values[0], df.index.values[-1], freq = 'D')
    df = df.reindex(newtime)

    # discharge quantile
    Q = df.loc[df.dis>0,'dis'].quantile([.05, .1, .5, .95])

    # event duration
    tmp1 = (df[['dis']] > Q.loc[0.5] * 9) * 1
    tmp2 = (df[['dis']] < Q.loc[0.5] * 0.2) * 1
    def func(x):
        y = (np.diff(x) != 0).astype('int').cumsum()
        y = np.hstack([np.nan, y])
        y = pd.DataFrame({'x':x,'y':y})
        y = y.loc[y.x==1,:].groupby('y').size().mean()
        return np.array([y])
    high_q_dur = tmp1.agg(func); high_q_dur = high_q_dur.fillna(0).squeeze()
    low_q_dur = tmp2.agg(func); low_q_dur = low_q_dur.fillna(0).squeeze()

    # calculate some hydrologic signatures
    HS = pd.Series([
        # Mean daily runoff
        df['dis'].mean(), 
        # runoff ratio
        df['dis'].mean() / df['pr'].mean(),  
        # slope of the flow duration curve 
        (np.log(df.loc[df.dis>0,'dis'].quantile(.33)) - np.log(df.loc[df.dis>0,'dis'].quantile(.66))) / (0.66-0.33), 
        # runoff Q5 and Q95
        Q.loc[0.05],
        Q.loc[0.95],
        # ratio of Q10 to Q50 to indicate groundwater
        Q.loc[0.1] / Q.loc[0.5],
        # frequency of high flows, low flows, and zero flows
        (df['dis']>Q.loc[0.5]*9).sum(),
        (df['dis']<Q.loc[0.5]*.2).sum(),
        (df['dis']==0).sum(),
        # variability coefficient
        df['dis'].std() / df['dis'].mean(),
        # event duration
        high_q_dur, low_q_dur,
        # BFI
        df.dis.agg(Eckhardt),
    ], index = [
        'q_mean', 'runoff_ratio', 'slope_fdc', 'Q5', 'Q95', 'Q10_50', 'high_q_freq', 'low_q_freq', 'zero_q_freq', 'cv', 'high_q_dur', 'low_q_dur', 'BFI',
    ])

    # calculate lag time between peak rainfall and peak discharge 
    df_combine = df.reset_index().rename(columns={'index':'date'}).melt(id_vars = 'date')
    df_combine = df_combine.rename(columns={'variable':'name'})
    df_combine = df_combine.set_index(['date','name'])
    df_lagT = df_combine.agg(calc_lagT).T.reset_index()
    HS = pd.concat([HS, df_lagT[['lagT','noResRatio']].squeeze()])

    # transform dataframe to hydrological-year cycle and then calculate some HS
    # generally, 1 October to 30 September in the Northern Hemisphere, 1 July to 30 June in the Southern Hemisphere (https://glossary.ametsoc.org/wiki/Water_year)
    
    if lat >= 0:
        start,end = 10,9
    else:
        start,end = 7,6
    year1 = df.index.year.values[0]
    year2 = df.index.year.values[-1]
    hy1 = pd.to_datetime('%d-%02d-01'%(year1,start))
    hy2 = pd.to_datetime('%d-%02d-30'%(year2,end))
    
    # transform time to hydrologic-cycle time
    df = df.loc[(df.index>=hy1)&(df.index<=hy2),:]
    df.index = pd.date_range('%d-1-1'%year1, periods = df.shape[0], freq = 'D')

    # calculate mean annual rainfall and flashiness index
    p_mean0 = df.groupby(df.index.year)['pr'].sum().mean()
    def funcs(x):
        a = np.abs(x.diff()).sum()
        b = x.sum()
        if b == 0 or np.isinf(b):
            return np.nan
        else:
            return a/b
    FI0 = df.groupby(df.index.year)['dis'].apply(funcs).mean()
    HS = pd.concat([HS, pd.Series([FI0,p_mean0], index = ['FI','p_mean'])])

    # calculate other hydrologic signatures
    # 1. stream_elas
    mq_tot = df.dis.mean()
    mp_tot = df.pr.mean()
    mq = df.groupby(df.index.year)['dis'].mean()
    mp = df.groupby(df.index.year)['pr'].mean()
    dp = mp - mp_tot
    dq = mq - mq_tot
    stream_elas0 = ((dq/mq_tot)/(dp/mp_tot)).median()
    HS = pd.concat([HS, pd.Series([stream_elas0], index = ['stream_elas'])])

    # 2. hfd_mean
    hfd_mean0 = df.groupby(df.index.year)['dis'].apply(lambda x: np.abs(x.cumsum() - x.sum()*0.5).argmin())
    hfd_mean0 = hfd_mean0.where(hfd_mean0<365)
    hfd_mean0 = hfd_mean0.mean(skipna=True)
    HS = pd.concat([HS, pd.Series([hfd_mean0], index = ['hfd_mean'])])

    return (HS)

# read rainfall
def readPr(fname):
    df = pd.read_csv(fname)
    df = df.set_index('ohdb_id')
    df = df.loc[:,df.columns.str.match('\d+_P$')].T.reset_index()
    df['date'] = pd.to_datetime(df['index'].str[:8], format = '%Y%m%d')
    df = df.drop(columns=['index'])
    print(fname)
    return (df)
fnames = glob.glob('../data_mswx/*daily_meteo*csv')
pool = mp.Pool(8)
df_pr = pool.map(readPr, fnames)
df_pr = pd.concat(df_pr)
print(df_pr.shape)

df_attr = pd.read_csv('../data/basin_attributes.csv')

def main(ohdb_id):
    df_pr0 = df_pr[['date',ohdb_id]].rename(columns={ohdb_id:'pr'})
    df_dis = pd.read_csv(f'../../data/OHDB/OHDB_v0.2.3/OHDB_data/discharge/daily/{ohdb_id}.csv')
    df_dis['date'] = pd.to_datetime(df_dis['date'])
    # read
    df_dis = cleanQ(df_dis)
    # quality check
    df_dis = del_unreliableQ(df_dis).reset_index().rename(columns={'index':'date'})
    df = df_pr0.merge(df_dis, on = 'date')
    df = df.sort_values('date',ascending=True).rename(columns={'Q':'dis'})
    df['darea'] = df_attr.loc[df_attr.ohdb_id==ohdb_id,'gritDarea'].values[0]
    df['lat'] = df_attr.loc[df_attr.ohdb_id==ohdb_id,'ohdb_latitude'].values[0]
    HS = calc_hs(df)
    HS.name = ohdb_id
    print(ohdb_id)
    return (HS)

if __name__ == '__main__':
    pool = mp.Pool(12)
    HS = pool.map(main, df_attr.ohdb_id.values)
    HS = pd.concat(HS, axis = 1)
    HS = HS.T.reset_index()
    HS.to_csv('../data/hydrologic_signatures.csv', index = False)