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

# read new forcings
df_P = pd.concat([pd.read_csv(fname).assign(year=int(os.path.basename(fname).split('_')[-1][:4])) for fname in glob.glob('../data_mswx/GRIT*temporal*P_[0-9]*csv')])
df_Tmax = pd.concat([pd.read_csv(fname).assign(year=int(os.path.basename(fname).split('_')[-1][:4])) for fname in glob.glob('../data_mswx/GRIT*temporal*Tmax_[0-9]*csv')])
df_Tmin = pd.concat([pd.read_csv(fname).assign(year=int(os.path.basename(fname).split('_')[-1][:4])) for fname in glob.glob('../data_mswx/GRIT*temporal*Tmin_[0-9]*csv')])
df_Snowfall = pd.concat([pd.read_csv(fname).assign(year=int(os.path.basename(fname).split('_')[-1][:4])) for fname in glob.glob('../data_mswx/GRIT*temporal*snowfall_[0-9]*csv')])
df_Snowmelt = pd.concat([pd.read_csv(fname).assign(year=int(os.path.basename(fname).split('_')[-1][:4])) for fname in glob.glob('../data_mswx/GRIT*temporal*snowmelt_[0-9]*csv')])

for name in ['P','Tmax','Tmin','Snowfall','Snowmelt']:
    df = df.merge(
        eval('df_'+name).rename(columns={'annMax':'annMax'+name,'annSum':'annSum'+name,'annAve':'annAve'+name,'annMin':'annMin'+name}),
        on = ['ohdb_id','year']
    )

# tmp = pd.read_csv(glob.glob(f'../data/{target}*seasonal4*rp.csv')[0])
# df = df.merge(tmp[['ohdb_id',f'{target}date','rp']], on = ['ohdb_id',f'{target}date'])
# df = df.loc[df.rp>=1,:].reset_index(drop=True)

# limit to those catchment area less than 100,000 km2
df = df.loc[(df.gritDarea<=100000)&(df.gritDarea>=500),:]
print(df.ohdb_id.unique().shape)

# limit to those catchments with at least 0.1% changes in the urban area over the study period (1982-2023)
tmp = df.groupby('ohdb_id').ImperviousSurface.apply(lambda x:x.max()-x.min()).reset_index()
df = df.loc[df.ohdb_id.isin(tmp.loc[tmp.ImperviousSurface>=0.1,'ohdb_id'].values),:]
print(df.ohdb_id.unique().shape)

# define outName to better name the ouput files
outName = re.sub('final_dataset_', '', os.path.basename(fname).split('.')[0])

# limit gauges to those with minimal influences of dams:
    # 2. percentage of reservoir area to catchment area less than 10
connect = pd.read_csv('../data/basin_reservoir_darea_ratio.csv')
connect = connect.loc[(connect.ratio>=10),:]
df = df.loc[~df.ohdb_id.isin(connect.ohdb_id),:].reset_index(drop=True)

# limit to upstream gauges
gdf = pd.read_csv('../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset_onlyUpstream.csv')
df = df.loc[df.ohdb_id.isin(gdf.ohdb_id.values),:].reset_index(drop=True)

print(df.shape, df.ohdb_id.unique().shape[0])

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 'aridity',
                'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                'clay', 'sand', 'silt',
                'Porosity_x', 'logK_Ice_x',
                'ohdb_latitude', 'ohdb_longitude', 
                'year', 
                'climate', 
                # 'basin_id', 
                'p_3', 'tmax_3', 'tmin_3', 'swd_3', 'snowmelt_3', 'snowfall_3',
                'p_7', 'tmax_7', 'tmin_7', 'swd_7', 'snowmelt_7', 'snowfall_7', 
                'p_15', 'tmax_15', 'tmin_15', 'swd_15', 'snowmelt_15', 'snowfall_15', 
                'p_30', 'tmax_30', 'tmin_30', 'swd_30', 'snowmelt_30', 'snowfall_30',
#               'annSumP','annMaxP','annAveTmax','annAveTmin','annSumSnowfall','annMaxSnowfall','annSumSnowmelt','annMaxSnowmelt',
                'slope_fdc', 
                'Q10_50', 
                'high_q_freq', 'low_q_freq', 'zero_q_freq', 'high_q_dur', 'low_q_dur', 
                'runoff_ratio', 'stream_elas', 'p_mean',
                'cv', 'BFI', 
                # 'noResRatio', 'lagT', 
                'FI', 'hfd_mean',
                'tmax_ave', 'tmax_std',
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
y = y + 0.1
y = y / df['gritDarea'].values * 86.4
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

ml=xgb.XGBRegressor(
    eval_metric='rmsle', 
    tree_method="hist", 
    device=device, 
    enable_categorical = True,
    feature_types = feature_types,
)

random_search = RandomizedSearchCV(
    ml, 
    param_distributions = param_dist, 
    n_iter = 100, 
    cv = 3, 
    # scoring='neg_root_mean_squared_error',
    # scoring = kge_scorer,
    verbose = 0, 
    random_state = 42, 
    n_jobs = 8
)

# tune outcome model
random_search.fit(X0, y)
outcome_model = random_search.best_estimator_

# tune treatment model
random_search.fit(X0, T0)
treatment_model = random_search.best_estimator_

pred_outcome = cross_val_predict(outcome_model, X0, y, cv = 10)
pred_outcome = np.exp(pred_outcome)
pred_outcome = pred_outcome / 86.4 * df['gritDarea'].values
pred_outcome = pred_outcome - 0.1
df['pred_outcome'] = pred_outcome

pred_treatment = cross_val_predict(treatment_model, X0, T0, cv = 10)
df['pred_treatment'] = pred_treatment

print('Finish cross-validation for outcome and treatment models')

cf_dml = CausalForestDML(
    model_y=outcome_model, 
    model_t=treatment_model, 
    cv=3, 
    n_estimators=500, 
    min_samples_leaf=10,
    random_state=1234,
    verbose=0,
    n_jobs = 4,
)

cf_dml.fit(y, T0, X = X0)

# Estimate treatment effects
treatment_effects = cf_dml.effect(X0)

# Calculate default (95%) confidence intervals for the test data
te_lower, te_upper = cf_dml.effect_interval(X0)

# transform to percentage change since the target variable is log-transformed
treatment_effects = (np.exp(treatment_effects) - 1) * 100
te_lower = (np.exp(te_lower) - 1) * 100
te_upper = (np.exp(te_upper) - 1) * 100

df['diff0'] = treatment_effects
df['te_lower'] = te_lower
df['te_upper']= te_upper

print(df.groupby('climate_label').apply(
    lambda x:pd.Series([x.loc[(x['diff0']>0)&(x.te_lower>0),:].shape[0]/x.shape[0]*100, x['diff0'].mean()],index=['frac','mean'])))

df0 = df.groupby(['ohdb_longitude','ohdb_latitude','climate_label','aridity','gritDarea'])[['diff0','te_lower','te_upper']].mean().reset_index()

# plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from src.plot_utils import plot_scatter,plot_map
import seaborn as sns
fig, ax = plt.subplots(figsize = (7, 4), subplot_kw = {'projection':ccrs.EqualEarth()})
ax.set_facecolor('none')

lons = df0.ohdb_longitude.values
lats = df0.ohdb_latitude.values
vals = df0['diff0'].values
name = target
vmin, vmax, vind = -5, 5, 0.5
cmap = plt.cm.RdBu_r
norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
title = 'Treatment effects'
label = f'{name} in $\Delta$%'
_, ras = plot_map(ax, lons, lats, vals, vmin, vmax, vind, cmap, title, label, norm = norm, fontSize = 11, size = 3)
# add colorbar
cax = ax.inset_axes([.35, .02, 0.25, .03])
cbar = plt.colorbar(ras, cax = cax, orientation = 'horizontal', extend = 'both')
cax.tick_params(labelsize = 10)
cax.set_title(label, size = 10, pad = 5)
# add boxplot to show the impact for dry (AI<1) and wet (AI>1) catchments
df0['tmp'] = np.where(df0['aridity']>0.65, 'wet', 'dry')

print(df0.loc[df0.tmp=='wet','diff0'].mean(), df0.loc[df0.tmp=='wet','diff0'].std())
print(df0.loc[df0.tmp=='dry','diff0'].mean(), df0.loc[df0.tmp=='dry','diff0'].std())

axin = ax.inset_axes([0.08, .05, .1, .3])
sns.boxplot(df0, 
            x = 'tmp', y = 'diff0', ax = axin, 
            showfliers = False, width = .7, 
            whis = [2.5, 97.5],
            color = '#c2dcea',
            showmeans = True,
            capprops = {'linewidth': 0},
            boxprops={'edgecolor': 'none'},  # No edge line
            meanprops={'marker': 'o',
                   'markerfacecolor': 'white',
                   'markeredgecolor': 'black',
                   'markersize': '4'},
            medianprops={'color': 'black', 'linewidth': 1},  # Black median line
            whiskerprops={'color': 'black', 'linewidth': 1},  # Black whiskers
           )
axin.set_facecolor('none')
axin.set_yticks([0,9])
axin.set_xlabel(None)
axin.set_ylabel(f'{name} in $\Delta$%', fontsize = 8)
axin.tick_params(labelsize = 8,  which='both', top = False, right = False)
axin.spines["top"].set_visible(False) 
axin.spines["right"].set_visible(False) 
    
ax2 = ax.inset_axes([1.1, .2, .5, .8])
sns.scatterplot(df0, x = 'gritDarea', y = 'diff0', ax = ax2)

# add ax3 and ax4 to evaluate the treatment and outcome models
ax3 = ax.inset_axes([.1, -1.2, .6, 1])
ax4 = ax3.inset_axes([1.3, 0, 1, 1])
sns.scatterplot(df, x = 'ImperviousSurface', y = 'pred_treatment', ax = ax3)
sns.scatterplot(df, x = 'Q', y = 'pred_outcome', ax = ax4)
ax3.axline((1, 1), slope=1, color='red', linestyle='--', label='1:1 Line')
ax4.axline((1, 1), slope=1, color='red', linestyle='--', label='1:1 Line')
ax3.set_title('Treatment model', fontsize = 11)
ax4.set_title('Outcome model', fontsize = 11)
fig.savefig('../picture/eval.png', dpi = 600)

# print importance of counfounders
predictors.remove('ImperviousSurface')
a = pd.Series(cf_dml.feature_importances(), index = predictors)
print(a.sort_values())