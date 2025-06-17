from src.plot_utils import *
from scipy.stats import linregress
from tqdm import tqdm
import multiprocessing as mp

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
    '''
    all records are rounded to three decimal places
    observations less than 0 were flagged as suspected
    observations with more than ten consecutive equal values greater than 0 were flagged as suspected
    '''
    df = df.loc[df.Q>=0,:].reset_index()
    df['Q'] = df['Q'].round(3)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    index = pd.date_range(df.index[0], df.index[-1], freq = 'D')
    df = df.reindex(index)
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
     # identical values should be greater than zero
    del_idx = [s for s in del_idx if df.iloc[s,:].Q > 0]
    if len(del_idx) > 0:
        df.drop(df.index[del_idx], inplace = True)
    # must no less than zero
    df = df.loc[df.Q>=0,:]
    return (df)

def del_outlierQ(df):
    '''
        Based on a previously suggested approach for evaluating temperature series (Klein Tank et al., 2009), 
        daily streamflow values are declared as outliers if values of log (Q+0.01) are larger or smaller than 
        the mean value of log (Q+0.01) plus or minus 6 times the standard deviation of log (Q+0.01) computed for 
        that calendar day for the entire length of the series. The mean and standard deviation are computed for 
        a 5-day window centred on the calendar day to ensure that a sufficient amount of data is considered. 
        The log-transformation is used to account for the skewness of the distribution of daily streamflow values 
        and 0.01 was added because the logarithm of zero is undefined. Outliers are flagged as suspect. 
        The rationale underlying this rule is that unusually large or small values are often associated with observational issues. 
        The 6 standard-deviation threshold is a compromise, aiming at screening out outliers that could come from 
        instrument malfunction, while not flagging extreme floods or low flows.
    '''
    df['logQ'] = np.log(df['Q']+0.01)
    df['doy'] = df.index.dayofyear
    df['year'] = df.index.year
    df = df.pivot_table(index = 'doy', columns = 'year', values = 'logQ').reset_index()
    def tmp(x0):
        x = np.arange(x0-2, x0+3) 
        x = np.where(x <= 0, x + 366, x)
        x = np.where(x > 366, x - 366, x)
        s = df.loc[df.doy.isin(x),:].drop(columns=['doy']).values.flatten()
        ave = np.nanmean(s)
        std = np.nanstd(s)
        low = ave - std * 6
        upp = ave + std * 6
        return (x0, low, upp)
    thres = list(map(tmp, np.arange(1, 367)))
    thres = pd.DataFrame(data = np.array(thres), columns = ['doy','low','upp'])
    df = df.merge(thres, on = 'doy').set_index('doy')
    df.iloc[:,:(df.shape[1]-2)] = df.iloc[:,:(df.shape[1]-2)].where(df.iloc[:,:(df.shape[1]-2)].lt(df['upp'], axis=0))
    df.iloc[:,:(df.shape[1]-2)] = df.iloc[:,:(df.shape[1]-2)].where(df.iloc[:,:(df.shape[1]-2)].gt(df['low'], axis=0))
    df = df.drop(columns = ['low','upp']).stack().reset_index(name='logQ')
    df['Q'] = np.exp(df['logQ']) - 0.01
    df['Q'] = np.where(df['Q'].abs()<1e-6, 0, df['Q'])
    df['date'] = pd.to_datetime(df['level_1'].astype(str) + '-' + df['doy'].astype(str), format='%Y-%j')
    df = df[['date','Q']].sort_values('date').set_index('date')
    return df

def remove_periodic_zero(df):
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear
    df1 = df.pivot_table(index = 'year', columns = 'doy', values = 'Q').fillna(0)
    s = (df1==0).all()
    s.name = 'periodic'
    df = df.merge(pd.DataFrame(s).reset_index(), on = 'doy')
    df.loc[df.periodic==True,'Q'] = np.nan
    df['date'] = pd.to_datetime(df['year'] * 1000 + df['doy'], format='%Y%j')
    df = df[['date','Q']].set_index('date')
    return df

def read_ohdb(ohdb_id):
    df = pd.read_csv(os.environ['DATA']+f'/data/OHDB/OHDB_v0.2.3/OHDB_data/discharge/daily/{ohdb_id}.csv')
    # read
    df = cleanQ(df)
    # quality check
    df = del_unreliableQ(df)
    # delete outliers
    df = del_outlierQ(df)
    # periodic zero not included
    df = remove_periodic_zero(df)
    return df

def Eckhardt(df, alpha=.98, BFI=.80, re=1, warm = 30):
    """
    Recursive digital filter for baseflow separation. Based on Eckhardt, 2004.\n
    Q : array of discharge measurements\n
    alpha : filter parameter\n
    BFI : BFI_max (maximum baseflow index)\n
    re : number of times to run filter
    
    split Q time series into consecutive segments without NaN and the length of segments should be greater than 'warm'
    Baseflow index would be calculated for each segment
    and the baseflow timeseries only outside of the warm period will be considered
    
    """
    Q = df.Q_mmday.values
    df['baseflow'] = np.nan
    
    # split Q into consecutive segments without NaN
    diff0 = np.where(np.isnan(Q), 0, 1)
    if not np.isnan(Q).any():
        start = [0]
        end = [None]
    else:
        start = (np.where(np.diff(diff0) == 1)[0] + 1).tolist()
        end = np.where(np.diff(diff0) == -1)[0].tolist()
        if start[0] > end[0]:
            start = [0] + start
        if end[-1] < start[-1]:
            end = end + [None]
    
    baseflow = np.zeros(df.shape[0]) * np.nan
    for start0,end0 in zip(start, end):
        if start0 is None:
            num = end0
        elif end0 is None:
            num = len(Q) - start0
        else:
            num = end0 - start0
        if num < warm:
            continue
        Q0 = Q[start0:end0]
        f = np.zeros(len(Q0))
        flag = np.zeros(len(Q0))
        f[0] = Q0[0]
        for t in np.arange(1,len(Q0)):
            # algorithm
            f[t] = ((1 - BFI) * alpha * f[t-1] + (1 - alpha) * BFI * Q0[t]) / (1 - alpha * BFI)
            if f[t] > Q0[t]:
                f[t] = Q0[t]
        baseflow[(start0+warm):end0] = f[warm:]
    df['baseflow'] = baseflow
    return df

df_raw = pd.read_pickle('../data/Qmax7_final_dataset_seasonal4.pkl')

# only keep catchments with changes in urban areas greater than 0.1%
df_raw = df_raw.groupby(['ohdb_id','year']).apply(lambda x:x.iloc[0,:]).reset_index(drop=True)
df0_gauge = df_raw[['ohdb_id','ImperviousSurface']].groupby('ohdb_id').apply(lambda x:x.ImperviousSurface.max()-x.ImperviousSurface.min())
df0_gauge = df0_gauge.loc[df0_gauge>=0.1]
df_raw = df_raw.loc[df_raw.ohdb_id.isin(df0_gauge.index),:]

df_lulc = []
for name in ['ImperviousSurface','forest']:
    df_urban = pd.read_csv(f'../ee_lulc/lulc_{name}_1982-2023_all.csv')
    df_urban = df_urban.melt(id_vars = 'ohdb_id')
    df_urban['year'] = df_urban['variable'].str[:4].astype(int)
    df_urban = df_urban[['ohdb_id','year','value']].rename(columns={'value':name})
    df_urban = df_urban.set_index(['ohdb_id','year']) / 100
    df_lulc.append(df_urban)
df_lulc = pd.concat(df_lulc, axis = 1).reset_index()

def main(par):
    ohdb_id, darea, aridity, climate_label = par

    # read runoff 
    df_runoff = read_ohdb(ohdb_id)

    # read meteorology
    df_meteo = pd.read_csv(f'../data_mswx/mswx_each_basin/{ohdb_id}_1981-2023.csv')
    df_meteo['date'] = pd.to_datetime(df_meteo['date'])
    df_meteo = df_meteo.groupby('date').apply(lambda x:x.iloc[0,:]).reset_index(drop=True)
    df_meteo = df_meteo.set_index('date')

    # merge
    df0 = pd.concat([df_runoff, df_meteo], axis = 1)

    # make a continous dataframe by adding NaN rows
    df0 = df0[['Q','p']].dropna()
    index = pd.date_range(df0.index.values[0], df0.index.values[-1], freq = 'D')
    df0 = df0.reindex(index)

    df0['darea'] = darea
    df0['Q_mmday'] = df0.Q / df0.darea * 86.4
    df0['year'] = df0.index.year

    # calculate baseflow
    df0 = Eckhardt(df0)

    # calculate stream_elas
    df0['delta_P'] = df0['p'].pct_change()
    df0['delta_Q'] = df0['Q'].pct_change()
    df0['elas'] = df0['delta_Q'] / df0['delta_P']

    # calculate annual change
    df0_year = df0.groupby('year').agg(
        Q_mmday = ('Q_mmday', lambda x:x.mean()),
        p = ('p', lambda x:x.mean()),
        count = ('Q', lambda x:len(x)),
        stream_elas = ('elas', lambda x:x.median()),
        baseflow_mean = ('baseflow', lambda x:x.mean()),
        baseflow_sum = ('baseflow', lambda x:x.sum()),
        streamflow_sum = ('Q_mmday', lambda x:x.sum()),
        baseflow_count = ('baseflow', lambda x:len(x[x>=0]))
    ).reset_index()
    df0_year['runoff_ratio'] = df0_year.Q_mmday / df0_year.p
    df0_year['BFI'] = df0_year.baseflow_sum / df0_year.streamflow_sum
    df0_year['aridity'] = aridity
    df0_year['climate_label'] = climate_label
    df0_year['ohdb_id'] = ohdb_id
    return df0_year

pars = df_raw[['ohdb_id','gritDarea','aridity','climate_label']].drop_duplicates().values.tolist()
with mp.Pool(processes=48) as pool:
    results = list(tqdm(pool.imap(main, pars), total=len(pars)))
results = pd.concat(results)

# merge with urban and forest
results = results.merge(df_lulc, on = ['ohdb_id','year'])

results.to_csv('../results/lulcTrend_vs_hsTrend.csv', index = False)