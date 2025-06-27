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

def create_ale_bin(df, feature, grid_size=20, min_interval = 0):
    quantiles = np.linspace(0, 1, grid_size + 1, endpoint=True)

    # use customized quantile function to get the same result as
    # type 1 R quantile (Inverse of empirical distribution function)
    bins = [df[feature].min()] + quantile_ied(df[feature], quantiles).to_list()
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
    
    feat_cut = pd.cut(df[feature], bins, include_lowest=True)

    bin_codes = feat_cut.cat.codes

    df1 = df.copy()
    df2 = df.copy()
    df1[feature] = [bins[i] for i in bin_codes]
    df1['bins'] = bins[bin_codes + 1]
    df1['bins_low'] = bins[bin_codes]
    df2[feature] = [bins[i + 1] for i in bin_codes]

    return df1, df2

def ale_func(df, predictors, model, feature, grid_size=20, group=None, log=True, m3s=False, min_interval = 0.1):
    """Compute the accumulated local effect of a numeric continuous feature.

    Arguments:
    df -- A pandas DataFrame to pass to the model for prediction.
    predictors -- predictors columns
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    grid_size -- An integer indicating the number of intervals into which the
    feature range is divided.
    group -- group feature used to calculate ALE for each group
    log -- target variables is log-transformed before modeling, so must back transform when calculating ALE
    m3s -- back transform specific discharge to m3/s
    min_interval -- if the interval is less than min_interval for a given bin, then merge it with next bin

    Return: A pandas DataFrame containing for each bin: the size of the sample in it
    and the accumulated centered effect of this bin.
    """

    if feature not in predictors:
        raise Exception('Feature is not in the predictors!')
    
    if group is None:
        df1, df2 = create_ale_bin(df, feature, grid_size=grid_size)
    else:
        # create dict to save objects
        df1 = []
        df2 = []
        for group0 in df[group].unique():
            df0 = df.loc[df[group]==group0,:].reset_index()
            df11, df22 = create_ale_bin(df0, feature, grid_size=grid_size)
            df1.append(df11)
            df2.append(df22)
        df1 = pd.concat(df1)
        df2 = pd.concat(df2)

    X1 = df1[predictors]
    X2 = df2[predictors]

    if model.get_params()['device'] in ['cuda','gpu']:
        import cupy as cp
        X1 = cp.array(X1.values)
        X2 = cp.array(X2.values)

    try:
        y_1 = model.predict(X1).ravel()
        y_2 = model.predict(X2).ravel()
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    df1['y_1'] = y_1
    df2['y_2'] = y_2
    
    delta_df = pd.concat([df1[['bins','bins_low','gritDarea','y_1']], df2[['y_2']]], axis = 1)
    if group is not None:
        delta_df[group] = df1[group].copy()

    if log:
        delta_df[['y_1','y_2']] = np.exp(delta_df[['y_1','y_2']])

    if m3s:
        delta_df[['y_1','y_2']] = delta_df[['y_1','y_2']].mul(df['gritDarea'], axis = 0) / 86.4 - 0.1
    
    delta_df['Delta'] = delta_df['y_2'] - delta_df['y_1']

    if group is None:
        res_df = delta_df.groupby(['bins'], observed=False).apply(
            lambda x: pd.Series([
                x.Delta.mean(), x.bins_low.min(), x.shape[0]
            ], index = ['eff', 'bins_low', 'size'])
        ).reset_index()
        res_df["eff"] = res_df["eff"].cumsum()
        
        bin_min = res_df.bins_low.min()
        bin_min = pd.DataFrame({'bins':[bin_min],'eff':[0]})
        res_df = pd.concat([res_df, bin_min]).fillna(0)

        # subtract the total average of a moving average of size 2
        mean_mv_avg = (
            (res_df["eff"] + res_df["eff"].shift(1, fill_value=0)) / 2 * res_df["size"]
        ).sum() / res_df["size"].sum()
        res_df['eff'] = res_df["eff"] - mean_mv_avg

    else:
        res_df = delta_df.groupby(['bins',group], observed=False).apply(
            lambda x: pd.Series([
                x.Delta.mean(), x.bins_low.min(), x.shape[0]
            ], index = ['eff', 'bins_low', 'size'])
        ).reset_index()
        eff = res_df.groupby(group).apply(lambda x:x.set_index('bins').eff.cumsum()).reset_index()
        res_df = res_df.drop(columns=['eff']).merge(eff, on = ['bins',group])
        
        bin_min = res_df.groupby(group).bins_low.min()
        bin_min = bin_min.reset_index().rename(columns={'bins_low':'bins'})
        bin_min['eff'] = 0
        res_df = pd.concat([res_df, bin_min]).fillna(0)
    
        # subtract the total average of a moving average of size 2
        mean_mv_avg = res_df.groupby(group).apply(lambda x: pd.Series(
            [(
            (x["eff"] + x["eff"].shift(1, fill_value=0)) / 2 * x["size"]
        ).sum() / x["size"].sum()], index = ['ave_eff']
        )).reset_index()
        res_df = res_df.merge(mean_mv_avg, on = group)
        res_df['eff'] = res_df["eff"] - res_df['ave_eff']

    # divide by average prediction to make the ALE in percentage terms
    X = df[predictors]
    if model.get_params()['device'] in ['cuda','gpu']:
        X = cp.array(X.values)
    ave0 = model.predict(X).ravel()
    if log:
        ave0 = np.exp(ave0)
    if m3s:
        ave0 = ave0 * df['gritDarea'] / 86.4 - 0.1
    
    if group is None:
        ave0 = np.mean(ave0)
        res_df['eff'] = res_df['eff'] / ave0 * 100
    else:
        df['ave0'] = ave0
        res_df = res_df.merge(df.groupby(group).ave0.mean().reset_index(), on = group)
        res_df['eff'] = res_df['eff'] / res_df['ave0'] * 100

    return res_df

def ale2_func(df, predictors, model, features, grid_size=20, group=None, log=True, m3s=False, min_interval = 0.1):
    """Compute the two dimentional accumulated local effect of a two numeric continuous features.

    This function divides the space of the two features into a grid of size
    grid_size*grid_size and computes the difference in prediction between the four
    edges (corners) of each bin in the grid and then centers the results.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    features -- A list of twos strings indicating the names of the columns
    holding the features in question.
    grid_size -- An integer indicating the number of intervals into which each
    feature range is divided.

    Return: A pandas DataFrame containing for each bin in the grid
    the accumulated centered effect of this bin.
    """

    # reset index to avoid index missmatches when replacing the values with the codes (lines 50 - 73)
    df = df.reset_index(drop=True)
    X = df[predictors]

    quantiles = np.append(0, np.arange(1 / grid_size, 1 + 1 / grid_size, 1 / grid_size))
    bins_0 = [X[features[0]].min()] + quantile_ied(X[features[0]], quantiles).to_list()
    bins_0 = np.unique(bins_0)
    feat_cut_0 = pd.cut(X[features[0]], bins_0, include_lowest=True)
    bin_codes_0 = feat_cut_0.cat.codes
    bin_codes_unique_0 = np.unique(bin_codes_0)

    bins_1 = [X[features[1]].min()] + quantile_ied(X[features[1]], quantiles).to_list()
    bins_1 = np.unique(bins_1)
    feat_cut_1 = pd.cut(X[features[1]], bins_1, include_lowest=True)
    bin_codes_1 = feat_cut_1.cat.codes
    bin_codes_unique_1 = np.unique(bin_codes_1)

    X11 = X.copy()
    X12 = X.copy()
    X21 = X.copy()
    X22 = X.copy()

    X11[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i] for i in bin_codes_0],
            features[1]: [bins_1[i] for i in bin_codes_1],
        }
    )
    X12[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i] for i in bin_codes_0],
            features[1]: [bins_1[i + 1] for i in bin_codes_1],
        }
    )
    X21[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i + 1] for i in bin_codes_0],
            features[1]: [bins_1[i] for i in bin_codes_1],
        }
    )
    X22[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i + 1] for i in bin_codes_0],
            features[1]: [bins_1[i + 1] for i in bin_codes_1],
        }
    )

    y_11 = model.predict(X11).ravel()
    y_12 = model.predict(X12).ravel()
    y_21 = model.predict(X21).ravel()
    y_22 = model.predict(X22).ravel()

    delta_df = pd.DataFrame(
        {
            features[0]: bin_codes_0 + 1,
            features[1]: bin_codes_1 + 1,
            "Delta": (y_22 - y_21) - (y_12 - y_11),
        }
    )
    index_combinations = pd.MultiIndex.from_product(
        [bin_codes_unique_0 + 1, bin_codes_unique_1 + 1], names=features
    )

    delta_df = delta_df.groupby([features[0], features[1]]).Delta.agg(["size", "mean"])

    sizes_df = delta_df["size"].reindex(index_combinations, fill_value=0)
    sizes_0 = sizes_df.groupby(level=0).sum().reindex(range(len(bins_0)), fill_value=0)
    sizes_1 = sizes_df.groupby(level=1).sum().reindex(range(len(bins_0)), fill_value=0)

    eff_df = delta_df["mean"].reindex(index_combinations, fill_value=np.nan)

    # ============== fill in the effects of missing combinations ================= #
    # ============== use the kd-tree nearest neighbour algorithm ================= #
    if impute_empty_cells is True:
        row_na_idx = np.where(eff_df.isna())[0]
        feat0_code_na = eff_df.iloc[row_na_idx].index.get_level_values(0)
        feat1_code_na = eff_df.iloc[row_na_idx].index.get_level_values(1)

        row_notna_idx = np.where(eff_df.notna())[0]
        feat0_code_notna = eff_df.iloc[row_notna_idx].index.get_level_values(0)
        feat1_code_notna = eff_df.iloc[row_notna_idx].index.get_level_values(1)

        if len(row_na_idx) != 0:
            range0 = bins_0.max() - bins_0.min()
            range1 = bins_1.max() - bins_1.min()

            feats_at_na = pd.DataFrame(
                {
                    features[0]: (bins_0[feat0_code_na - 1] + bins_0[feat0_code_na])
                    / (2 * range0),
                    features[1]: (bins_1[feat1_code_na - 1] + bins_1[feat1_code_na])
                    / (2 * range1),
                }
            )
            feats_at_notna = pd.DataFrame(
                {
                    features[0]: (bins_0[feat0_code_notna - 1] + bins_0[feat0_code_notna])
                    / (2 * range0),
                    features[1]: (bins_1[feat1_code_notna - 1] + bins_1[feat1_code_notna])
                    / (2 * range1),
                }
            )
            # fit the algorithm with the features where the effect is not missing
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(feats_at_notna)
            # find the neighbours of the features where the effect is missing
            distances, indices = nbrs.kneighbors(feats_at_na)
            # fill the missing effects with the effects of the nearest neighbours
            eff_df.iloc[row_na_idx] = eff_df.iloc[
                row_notna_idx[indices.flatten()]
            ].to_list()

    # ============== cumulative sum of the difference ================= #
    eff_df = eff_df.groupby(level=0).cumsum().groupby(level=1).cumsum()

    # ============== centering with the moving average ================= #
    # subtract the cumulative sum of the mean of 1D moving average (for each axis)
    eff_df_0 = eff_df - eff_df.groupby(level=1).shift(periods=1, fill_value=0)
    fJ0 = (
        (
            sizes_df
            * (
                eff_df_0.groupby(level=0).shift(periods=1, fill_value=0)
                + eff_df_0
            )
            / 2
        )
        .groupby(level=0)
        .sum()
        .div(sizes_0)
        .fillna(0)
        .cumsum()
    )

    eff_df_1 = eff_df - eff_df.groupby(level=0).shift(periods=1, fill_value=0)
    fJ1 = (
        (
            sizes_df
            * (eff_df_1.groupby(level=1).shift(periods=1, fill_value=0) + eff_df_1)
            / 2
        )
        .groupby(level=1)
        .sum()
        .div(sizes_1)
        .fillna(0)
        .cumsum()
    )

    all_combinations = pd.MultiIndex.from_product(
        [[x for x in range(len(bins_0))], [x for x in range(len(bins_1))]],
        names=features,
    )
    eff_df = eff_df.reindex(all_combinations, fill_value=0)
    eff_df = eff_df.subtract(fJ0, level=0).subtract(fJ1, level=1)

    # subtract the total average of a 2D moving average of size 2 (4 cells)
    idx = pd.IndexSlice
    eff_df = (
        eff_df
        - (
            sizes_df
            * (
                eff_df.loc[
                    idx[0 : len(bin_codes_unique_0) - 1],
                    idx[0 : len(bin_codes_unique_1) - 1],
                    :,
                ].values
                + eff_df.loc[
                    idx[1 : len(bin_codes_unique_0)],
                    idx[1 : len(bin_codes_unique_1)],
                    :,
                ].values
                + eff_df.loc[
                    idx[0 : len(bin_codes_unique_0) - 1],
                    idx[1 : len(bin_codes_unique_1)],
                    :,
                ].values
                + eff_df.loc[
                    idx[1 : len(bin_codes_unique_0)],
                    idx[0 : len(bin_codes_unique_1) - 1],
                    :,
                ].values
            )
            / 4
        ).sum()
        / sizes_df.sum()
    )

    # renaming and preparing final output
    eff_df = eff_df.reset_index(name="eff")
    eff_df[features[0]] = bins_0[eff_df[features[0]].values]
    eff_df[features[1]] = bins_1[eff_df[features[1]].values]
    eff_grid = eff_df.pivot_table(columns=features[1], values="eff", index=features[0])

    return eff_grid

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

        if model.get_params()['device'] in ['cuda','gpu']:
            X1 = cp.array(X1.values)

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
    if model.get_params()['device'] in ['cuda','gpu']:
        X = cp.array(X.values)
    ave0 = model.predict(X).ravel()
    if log:
        ave0 = np.exp(ave0)
    if m3s:
        ave0 = ave0 * darea / 86.4 - 0.1
    ave0 = np.mean(ave0)
    res_df['eff'] = res_df['eff'] / ave0 * 100

    return res_df

def aleplot_1D_continuous(df, predictors, model, feature, grid_size=100, group=None, log=True, m3s=False, monte_carlo = None, monte_ratio = None, min_interval = 0.1):
    if monte_carlo is not None:
        if monte_carlo is None:
            raise Warning('since monte_carlo is not None, monte_ratio should be specified, default is 0.1')
            monte_ratio = 0.1
        df_ale = []
        df_ale0 = ale_func(df, predictors, model, feature, grid_size = grid_size, group=group, log = log, m3s = m3s, min_interval = min_interval)
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
        df_ale0 = ale_func(df, predictors, model, feature, grid_size = grid_size, group=group, log = log, m3s = m3s, min_interval = min_interval)
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

