import logging
import subprocess
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import t,kstest,uniform

LOGGER = logging.getLogger(__name__)

def LogTrans(y, darea = None, addition = 0.1, log = True):
    '''transform predictand from m3/s to log(mm/day)'''
    y = y + addition
    y = y / darea * 86.4
    if log:
        y = np.log(y)
    return y

def InvLogTrans(y, darea = None, addition = 0.1, log = True):
    '''transform predictand from log(mm/day) to m3/s'''
    if log:
        y = np.exp(y)
    y = y * darea / 86.4
    y = y - addition
    return y

def check_GPU():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        GPU = True
        if result.returncode != 0:
            GPU = False
    except FileNotFoundError:
        GPU = False
    return GPU

def quantile_ied(x_vec, q):
    """
    Inverse of empirical distribution function (quantile R type 1).

    More details in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    https://en.wikipedia.org/wiki/Quantile

    Arguments:
    x_vec -- A pandas series containing the values to compute the quantile for
    q -- An array of probabilities (values between 0 and 1)
    """

    x_vec = x_vec.sort_values()
    n = len(x_vec) - 1
    m = 0
    j = (n * q + m).astype(int)  # location of the value
    g = n * q + m - j

    gamma = (g != 0).astype(int)
    quant_res = (1 - gamma) * x_vec.shift(1, fill_value=0).iloc[j] + gamma * x_vec.iloc[
        j
    ]
    quant_res.index = q
    # add min at quantile zero and max at quantile one (if needed)
    if 0 in q:
        quant_res.loc[0] = x_vec.min()
    if 1 in q:
        quant_res.loc[1] = x_vec.max()
    return quant_res

def check_uniform(data):
    # Define the parameters of the uniform distribution
    data_min = np.min(data)
    data_max = np.max(data)
    scale = data_max - data_min

    # Perform the Kolmogorov–Smirnov test
    # We need to standardize the uniform distribution to match the data
    # A greater ks_stat indicates a greater imbalance
    ks_stat, p_value = kstest(data, 'uniform', args=(data_min, scale))
    return ks_stat, p_value

def undersample(df, feature, percentiles=(0.05, 0.95)):
    low = df[feature].quantile(.05)
    upp = df[feature].quantile(.95)
    # downsampling would be only conducted for dataframe within 5th-95th percentiles
    df0 = df.loc[(df[feature]>=low)&(df[feature]<=upp),:].reset_index(drop=True)
    
    df0['group'] = pd.cut(df0[feature], 10)
    sample_num = df0.groupby('group').Q.count().min()
    df0 = df0.groupby('group').apply(lambda x:x.sample(n=sample_num, replace = False)).reset_index(drop=True).drop(columns=['group'])
    return df0

def ale_func(df, predictors, model, feature, grid_size=20, log=True, m3s=False, min_interval = 0.1):
    """Compute the accumulated local effect of a numeric continuous feature.

    Arguments:
    df -- A pandas DataFrame to pass to the model for prediction.
    predictors -- predictors columns
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    grid_size -- An integer indicating the number of intervals into which the
    feature range is divided.
    log -- target variables is log-transformed before modeling, so must back transform when calculating ALE
    m3s -- back transform specific discharge to m3/s
    min_interval -- if the interval is less than min_interval for a given bin, then merge it with next bin

    Return: A pandas DataFrame containing for each bin: the size of the sample in it
    and the accumulated centered effect of this bin.
    """
    X = df[predictors]

    if feature not in predictors:
        raise Exception('Feature is not in the predictors!')
    
    quantiles = np.linspace(0, 1, grid_size + 1, endpoint=True)
    # use customized quantile function to get the same result as
    # type 1 R quantile (Inverse of empirical distribution function)
    bins = [X[feature].min()] + quantile_ied(X[feature], quantiles).to_list()
    bins = np.unique(bins)

    # merge small-interval bins
    # Initialize the new list of bins with the first bin from the original list
    new_bins = [bins[0]]
    # Iterate through the rest of the bins
    for i in range(1, len(bins)):
        # If the difference between the current bin and the last bin added to new_bins
        # is greater than or equal to the minimum required interval, add the current bin.
        if bins[i] - new_bins[-1] >= min_interval:
            new_bins.append(bins[i])
    # Ensure the last original bin is included if it extends beyond the last new_bin,
    # especially if the last interval would be very small.
    # This part is optional but can be useful to retain the full range of the data.
    if bins[-1] > new_bins[-1] and (bins[-1] - new_bins[-1] < min_interval):
        new_bins[-1] = bins[-1] # Replace the last bin to ensure the upper bound
    bins = np.array(new_bins)
    
    feat_cut = pd.cut(X[feature], bins, include_lowest=True)

    bin_codes = feat_cut.cat.codes
    bin_codes_unique = np.unique(bin_codes)

    X1 = X.copy()
    X2 = X.copy()
    X1[feature] = [bins[i] for i in bin_codes]
    X2[feature] = [bins[i + 1] for i in bin_codes]
    try:
        y_1 = model.predict(X1).ravel()
        y_2 = model.predict(X2).ravel()
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    if log:
        y_1 = np.exp(y_1)
        y_2 = np.exp(y_2)
    if m3s:
        darea = df['gritDarea'].values
        y_1 = y_1 * darea / 86.4 - 0.1
        y_2 = y_2 * darea / 86.4 - 0.1

    delta_df = pd.DataFrame({feature: bins[bin_codes + 1], "Delta": y_2 - y_1})
    res_df = delta_df.groupby([feature], observed=False).Delta.agg(
        [("eff", "mean"), "size"]
    )
    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[min(bins), :] = 0
    
    # subtract the total average of a moving average of size 2
    mean_mv_avg = (
        (res_df["eff"] + res_df["eff"].shift(1, fill_value=0)) / 2 * res_df["size"]
    ).sum() / res_df["size"].sum()
    res_df = res_df.sort_index().assign(eff=res_df["eff"] - mean_mv_avg)

    # divide by average prediction to make the ALE in percentage terms
    ave0 = model.predict(X).ravel()
    if log:
        ave0 = np.exp(ave0)
    if m3s:
        ave0 = ave0 * darea / 86.4 - 0.1
    ave0 = np.mean(ave0)
    res_df = res_df / ave0 * 100

    return res_df

def pdp_func(df, predictors, model, feature, grid_size=20, log=True, m3s=False, min_interval = 0.1):
    """Compute the partial dependence effect of a numeric continuous feature.

    Arguments:
    df -- A pandas DataFrame to pass to the model for prediction.
    predictors -- predictors columns
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    grid_size -- An integer indicating the number of intervals into which the
    feature range is divided.
    log -- target variables is log-transformed before modeling, so must back transform when calculating ALE
    m3s -- back transform specific discharge to m3/s
    min_interval -- if the interval is less than min_interval for a given bin, then merge it with next bin

    Return: A pandas DataFrame containing for each bin: the size of the sample in it
    and the accumulated centered effect of this bin.
    """
    X = df[predictors]

    if feature not in predictors:
        raise Exception('Feature is not in the predictors!')

    quantiles = np.linspace(0.05, 0.95, grid_size + 1, endpoint=True)
    # use customized quantile function to get the same result as
    # type 1 R quantile (Inverse of empirical distribution function)
    bins = np.quantile(X[feature], quantiles).tolist()
    bins = np.unique(bins)

    # merge small-interval bins
    # Initialize the new list of bins with the first bin from the original list
    new_bins = [bins[0]]
    # Iterate through the rest of the bins
    for i in range(1, len(bins)):
        # If the difference between the current bin and the last bin added to new_bins
        # is greater than or equal to the minimum required interval, add the current bin.
        if bins[i] - new_bins[-1] >= min_interval:
            new_bins.append(bins[i])
    # Ensure the last original bin is included if it extends beyond the last new_bin,
    # especially if the last interval would be very small.
    # This part is optional but can be useful to retain the full range of the data.
    if bins[-1] > new_bins[-1] and (bins[-1] - new_bins[-1] < min_interval):
        new_bins[-1] = bins[-1] # Replace the last bin to ensure the upper bound
    bins = np.array(new_bins)
    
    pdp0 = []
    for i,bin0 in enumerate(bins):
        X1 = X.copy()
        X1[feature] = bin0
    
        y_1 = model.predict(X1).ravel()
        
        if log:
            y_1 = np.exp(y_1)
        if m3s:
            darea = df['gritDarea'].values
            y_1 = y_1 * darea / 86.4 - 0.1
        y_1 = np.mean(y_1)
        pdp0.append(y_1)
    
    res_df = pd.DataFrame({feature:bins,'eff':pdp0})
    
    # subtract average
    res_df['eff'] = res_df['eff'] - res_df['eff'].mean()

    # divide by average prediction to make the PDP in percentage terms
    ave0 = model.predict(X).ravel()
    if log:
        ave0 = np.exp(ave0)
    if m3s:
        ave0 = ave0 * darea / 86.4 - 0.1
    ave0 = np.mean(ave0)
    res_df['eff'] = res_df['eff'] / ave0 * 100

    return res_df

def aleplot_1D_continuous(df, predictors, model, feature, grid_size=100, log=True, m3s=False, monte_carlo = None, monte_ratio = None, min_interval = 0.1):
    if monte_carlo is not None:
        if monte_carlo is None:
            raise Warning('since monte_carlo is not None, monte_ratio should be specified, default is 0.1')
            monte_ratio = 0.1
        df_ale = []
        df_ale0 = ale_func(df, predictors, model, feature, grid_size = grid_size, log = log, m3s = m3s, min_interval = min_interval)
        df_ale0['monte'] = 0
        df_ale.append(df_ale0)
        for monte in range(1, monte_carlo+1):
            df0 = df.iloc[np.random.choice(df.shape[0], int(df.shape[0]*monte_ratio), replace = False), :]
            df_ale0 = ale_func(df0, predictors, model, feature, grid_size=grid_size, log=log, m3s=m3s, min_interval = min_interval)
            df_ale0['monte'] = monte
            df_ale.append(df_ale0)
        df_ale = pd.concat(df_ale)
        return df_ale
    else:
        df_ale0 = ale_func(df, predictors, model, feature, grid_size = grid_size, log = log, m3s = m3s, min_interval = min_interval)
        df_ale0['monte'] = 0
        return df_ale0
    
def pdpplot_1D_continuous(df, predictors, model, feature, grid_size=100, log=True, m3s=False, monte_carlo = None, monte_ratio = None, min_interval = 0.1):
    if monte_carlo is not None:
        if monte_carlo is None:
            raise Warning('since monte_carlo is not None, monte_ratio should be specified, default is 0.1')
            monte_ratio = 0.1
        df_pdp = []
        df_pdp0 = pdp_func(df, predictors, model, feature, grid_size = grid_size, log = log, m3s = m3s, min_interval = min_interval)
        df_pdp0['monte'] = 0
        df_pdp.append(df_pdp0)
        for monte in range(1, monte_carlo+1):
            df0 = df.iloc[np.random.choice(df.shape[0], int(df.shape[0]*monte_ratio), replace = False), :]
            df_pdp0 = pdp_func(df0, predictors, model, feature, grid_size=grid_size, log=log, m3s=m3s, min_interval = min_interval)
            df_pdp0['monte'] = monte
            df_pdp.append(df_pdp0)
        df_pdp = pd.concat(df_pdp)
        return df_pdp
    else:
        df_pdp0 = pdp_func(df, predictors, model, feature, grid_size = grid_size, log = log, m3s = m3s, min_interval = min_interval)
        df_pdp0['monte'] = 0
        return df_pdp0