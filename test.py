import os,glob,sys,re
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import hydroeval as he
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer
import pickle
import shap
import tqdm
import warnings
import subprocess
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

fname = '../data/Qmax7_final_dataset_annual_multi_MSWX_meteo.csv'
target = os.path.basename(fname).split('_')[0]
model = 'xgb'

# read dataset
df = pd.read_csv(fname)

# add new rainfall forcing
df_1 = pd.read_csv(f'../data/{target}_seasonal4_multi_MSWX_rain_onlyGridGreaterThan0.1_AveStd.csv')
df = df.merge(df_1, on = ['ohdb_id', target+'7date'])
del df_1

# # limit to RP >= 1
# tmp = pd.read_csv(glob.glob(f'../data/{target}*seasonal4*rp.csv')[0])
# df = df.merge(tmp[['ohdb_id',f'{target}date','rp']], on = ['ohdb_id',f'{target}date'])
# df = df.loc[df.rp>=1,:].reset_index(drop=True)

# limit to those catchment area less than 100,000 km2
df = df.loc[(df.gritDarea<=100000),:]
print(df.ohdb_id.unique().shape)

# # limit to those catchments with at least 0.1% changes in the urban area over the study period (1982-2023)
# tmp = df.groupby('ohdb_id').ImperviousSurface.apply(lambda x:x.max()-x.min()).reset_index()
# df = df.loc[df.ohdb_id.isin(tmp.loc[tmp.ImperviousSurface>=0.1,'ohdb_id'].values),:]
# print(df.ohdb_id.unique().shape)

# define outName to better name the ouput files
outName = re.sub('final_dataset_', '', os.path.basename(fname).split('.')[0])

# # limit gauges to those with minimal influences of dams:
#     # 2. percentage of reservoir area to catchment area less than 10
# connect = pd.read_csv('../data/basin_reservoir_darea_ratio.csv')
# connect = connect.loc[(connect.ratio>=10),:]
# df = df.loc[~df.ohdb_id.isin(connect.ohdb_id),:].reset_index(drop=True)

# # limit to upstream gauges
# gdf = pd.read_csv('../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset_onlyUpstream.csv')
# df = df.loc[df.ohdb_id.isin(gdf.ohdb_id.values),:].reset_index(drop=True)

print(df.shape, df.ohdb_id.unique().shape[0])

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 'aridity',
                'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                'clay', 'sand', 'silt',
                'Porosity_x', 'logK_Ice_x',
                'ohdb_latitude', 'ohdb_longitude', 
                'year', 
                'climate', 
                'basin_id', 
                'p_mean_3', 'tmax_3', 'tmin_3', 'swd_3', 'snowmelt_3', 'snowfall_3',
                'p_mean_7', 'tmax_7', 'tmin_7', 'swd_7', 'snowmelt_7', 'snowfall_7', 
                'p_mean_15', 'tmax_15', 'tmin_15', 'swd_15', 'snowmelt_15', 'snowfall_15', 
                'p_mean_30', 'tmax_30', 'tmin_30', 'swd_30', 'snowmelt_30', 'snowfall_30',
#               'annSumP','annMaxP','annAveTmax','annAveTmin','annSumSnowfall','annMaxSnowfall','annSumSnowmelt','annMaxSnowmelt',
                'slope_fdc', 
                'Q10_50', 
                'high_q_freq', 'low_q_freq', 'zero_q_freq', 'high_q_dur', 'low_q_dur', 
                'runoff_ratio', 'stream_elas', 
                'cv', 'BFI', 
                # 'noResRatio', 'lagT', 
                'FI', 'hfd_mean',
                'p_mean', 'tmax_ave', 'tmax_std',
                'ImperviousSurface', 'forest', 
                'crop', 'grass', 'water', 'wetland',
                'res_darea_normalize', 'Year_ave', 'Main_Purpose_id',
                'GDP', 'population'
            ]

# print predictors
i = 0
s = ''
for p in predictors:
    s = s + ' ' + p
    i += 1
    if i % 6 == 0:
        print(s)
        s = ''
print(s)

# transform predictand
X = df[predictors].astype(np.float32)
y = df['Q'].astype(np.float32).values
y = y / df['gritDarea'].values * 86.4
y = y + 0.1
y = np.log(y)

# get feature dtype
feature_types = X.agg(lambda x: 'q').to_dict()
for a in ['climate', 'season_id', 'Main_Purpose_id', 'basin_id', 'gauge_id']:
    if a in predictors:
        feature_types[a] = 'c'
        X[a] = X[a].astype(np.int16)
# feature_types['freeze'] = 'i'
feature_types = list(feature_types.values())

device = 'cpu'

from econml.dml import CausalForestDML

X = pd.DataFrame(data = X, columns = predictors)

# define covariates  (X), treatment (T)
T0 = X['ImperviousSurface'].values
X0 = X.drop(columns = ['ImperviousSurface']).values

# train XGBoost for outcome_model and treatment_model
# exhaustively search for the optimal hyperparameters
# set up our search grid
param_dist = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 800, 1000],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

feature_types.pop(predictors.index('ImperviousSurface'))

# custom function to avoid overestimate small values
def my_assymetric_error_wrapper(tau, delta):
    def my_assymetric_error(y_pred, y_true):
        error = (y_pred - y_true)
        grad = np.where(((y_true<tau)&(error>0)), delta*2*error, 2*error)
        hess = np.where(((y_true<tau)&(error>0)), delta*2, 2)
        return grad, hess
    return my_assymetric_error

def huber_obj(y_pred, y_true, delta=1.0):
    """Huber loss objective function."""
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    
    grad = np.where(abs_diff < delta, diff, delta * np.sign(diff))
    hess = np.where(abs_diff < delta, 1.0, 0.0)
    
    return grad, hess

ml=xgb.XGBRegressor(
    eval_metric='rmsle', 
    tree_method="hist", 
    device=device, 
    objective = huber_obj,
    # objective = my_assymetric_error_wrapper(tau = 0, delta = 9),
    enable_categorical = True,
    feature_types = feature_types,
)

random_search = RandomizedSearchCV(
    ml, 
    param_distributions = param_dist, 
    n_iter = 100, 
    cv = 3, 
    scoring='neg_root_mean_squared_error',
    # scoring = kge_scorer,
    verbose = 5, 
    random_state = 42, 
    n_jobs = 8
)

# tune outcome model
random_search.fit(X0, y)
outcome_model = random_search.best_estimator_

pred_outcome = cross_val_predict(outcome_model, X0, y, cv = 10, n_jobs = 3)
pred_outcome = np.exp(pred_outcome)
pred_outcome = pred_outcome - 0.1
pred_outcome = pred_outcome / 86.4 * df['gritDarea'].values
df['pred_outcome'] = pred_outcome

df['bias'] = (df.pred_outcome - df.Q) / df.Q * 100
df0 = df.loc[df.gritDarea<=1000,:]
df1 = df.loc[df.gritDarea>=1000,:]
print('Small', df0.bias.mean(), df0.bias.median())
print('Large', df1.bias.mean(), df1.bias.median())

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize = (5,3))
sns.scatterplot(df, x = 'gritDarea', y = 'bias', ax = ax)
ax.set_yscale('symlog')
fig.savefig('../picture/test111.png', dpi = 600)