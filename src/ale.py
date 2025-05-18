import numpy as np
from scipy.stats import t
import pandas as pd

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

def aleplot_1D_continuous(df, predictors, model, feature, grid_size=20, log=True, m3s=False):
    """Compute the accumulated local effect of a numeric continuous feature.

    This function divides the feature in question into grid_size intervals (bins)
    and computes the difference in prediction between the first and last value
    of each interval and then centers the results.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    grid_size -- An integer indicating the number of intervals into which the
    feature range is divided.
    include_CI -- A boolean, if True the confidence interval
    of the effect is returned with the results.
    C -- A float the confidence level for which to compute the confidence interval
    log -- target variables is log-transformed before modeling, so must back transform when calculating ALE
    m3s -- back transform specific discharge to m3/s

    Return: A pandas DataFrame containing for each bin: the size of the sample in it
    and the accumulated centered effect of this bin.
    """
    X = df[predictors]

    quantiles = np.linspace(0, 1, grid_size + 1, endpoint=True)
    # use customized quantile function to get the same result as
    # type 1 R quantile (Inverse of empirical distribution function)
    bins = [X[feature].min()] + quantile_ied(X[feature], quantiles).to_list()
    bins = np.unique(bins)
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
        darea = df['grit_darea'].values
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

def CI_estimate(x_vec, C=0.95):
    """Estimate the size of the confidence interval of a data sample.

    The confidence interval of the given data sample (x_vec) is
    [mean(x_vec) - returned value, mean(x_vec) + returned value].
    """
    alpha = 1 - C
    n = len(x_vec)
    stand_err = x_vec.std() / np.sqrt(n)
    critical_val = 1 - (alpha / 2)
    z_star = stand_err * t.ppf(critical_val, n - 1)
    return z_star