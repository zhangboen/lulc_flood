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

# check GPU
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    GPU = True
    if result.returncode != 0:
        GPU = False
except FileNotFoundError:
    GPU = False

if GPU:
    import cupy as cp

try:
    fname = sys.argv[1]
except:
    raise Exception('must specify the input csv name')

target = os.path.basename(fname).split('_')[0]
model = 'xgb'

print(sys.argv)

# read dataset
df = pd.read_csv(fname)

# get annual dataset
if target == 'Qmax7':
    df = df.groupby(['ohdb_id','year']).apply(lambda x:x.iloc[np.argmax(x.Q),:]).reset_index(drop = True)
else:
    df = df.groupby(['ohdb_id','year']).apply(lambda x:x.iloc[np.argmin(x.Q),:]).reset_index(drop = True)

# # limit to those events with at least 1-year return period
# df = df.loc[df.rp>=1,:]

# limit to those catchment area less than 100,000 km2 and greater than 500 km2
df = df.loc[(df.gritDarea<=100000)&(df.gritDarea>=500),:]
print(df.ohdb_id.unique().shape)

# limit to those catchments with at least 0.1% changes in the urban area over the study period (1982-2023)
tmp = df.groupby('ohdb_id').ImperviousSurface.apply(lambda x:x.max()-x.min()).reset_index()
df = df.loc[df.ohdb_id.isin(tmp.loc[tmp.ImperviousSurface>=0.1,'ohdb_id'].values),:]
print(df.ohdb_id.unique().shape)

# define outName to better name the ouput files
outName = re.sub('final_dataset_', '', os.path.basename(fname).split('.')[0])

# limit to upstream gauges
gdf = pd.read_csv('../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset_onlyUpstream.csv')
df = df.loc[df.ohdb_id.isin(gdf.ohdb_id.values),:].reset_index(drop=True)

# limit to 1982-2021 to make a 40-year balanced panel data for DynamicDML
df = df.loc[(df.year>=1982)&(df.year<=2021),:]
df1 = df.groupby('ohdb_id')['year'].count().reset_index()
df = df.loc[df.ohdb_id.isin(df1.loc[df1.year==40,'ohdb_id'].values),:]

# only use random 200 catchments
ohdb_ids = np.random.choice(df.ohdb_id.unique(), 200, replace = False)
df = df.loc[df.ohdb_id.isin(ohdb_ids),:]

print(df.shape, df.ohdb_id.unique().shape[0])

# sort df as time within each gauge
df = df.sort_values(['gauge_id','year']).reset_index(drop=True)

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 'aridity', 
                # 'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                # 'clay', 'sand', 'silt',
                # 'Porosity_x', 'logK_Ice_x',
                'p_7', 'tmax_7', 'tmin_7', 'snowmelt_7', 'snowfall_7', 
                'cv', 'BFI', 
                # 'FI', 'stream_elas', 
                'ImperviousSurface', 
                # 'forest', 'crop', 'grass', 'water', 'wetland',
                # 'GDP', 'population'
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
y = y + 0.01
y = np.log(y)

# get feature dtype
feature_types = X.agg(lambda x: 'q').to_dict()
for a in ['climate', 'season_id', 'Main_Purpose_id', 'basin_id', 'gauge_id']:
    if a in predictors:
        feature_types[a] = 'c'
        X[a] = X[a].astype(np.int16)
# feature_types['freeze'] = 'i'
feature_types = list(feature_types.values())

# transfrom from numpy array to cupy array for GPU modeling
if GPU:
    X = cp.array(X)

if GPU:
    device = 'cuda'
else:
    device = 'cpu'


#################################################################################################
#         Machine learning causal inference
#################################################################################################
from econml.panel.dml import DynamicDML
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.metrics import make_scorer
from econml.score import RScorer

# change from GPU to DataFrame to run
if GPU:
    X = cp.asnumpy(X)

X = pd.DataFrame(data = X, columns = predictors)

# define covariates  (X), treatment (T)
T0 = X['ImperviousSurface'].values
X0 = X.drop(columns = ['ImperviousSurface']).values

# change to GPU
if GPU:
    X0 = cp.array(X0)

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
    n_iter = 50, 
    cv = 3, 
    verbose = 5, 
    random_state = 42, 
)

# # tune outcome model
# random_search.fit(X0, y)
# outcome_model = random_search.best_estimator_

# # tune treatment model
# random_search.fit(X0, T0)
# treatment_model = random_search.best_estimator_

# Fit the causal forest
if GPU:
    X0 = cp.asnumpy(X0)

cf_dml = DynamicDML(
    # model_y=outcome_model, 
    # model_t=treatment_model, 
    cv=2, 
    random_state=1234
)

cf_dml.fit(y, T0, X = X0, groups = df.gauge_id.values, inference = 'bootstrap')

out = {'mdl':cf_dml,'predictor':predictors}

pickle.dump(out, open(f'../results/dynamic_causal_model_{model}_{outName}.pkl', 'wb'))

# Estimate treatment effects
treatment_effects = cf_dml.effect(X0)

# Calculate default (95%) confidence intervals for the test data
te_lower, te_upper = cf_dml.effect_interval(X0)

# transform to percentage change since the target variable is log-transformed
treatment_effects = (np.exp(treatment_effects) - 1) * 100
te_lower = (np.exp(te_lower) - 1) * 100
te_upper = (np.exp(te_upper) - 1) * 100

df['treatment_effects'] = treatment_effects
df['te_lower'] = te_lower
df['te_upper']= te_upper
df0 = df.groupby(['ohdb_id','climate_label'])[['treatment_effects','te_lower','te_upper']].mean().reset_index()

print(df0.groupby('climate_label').treatment_effects.agg(
    low = lambda x:x.quantile(.25),
    ave = lambda x:x.mean(),
    upp = lambda x:x.quantile(.75),
))

for climate in ['dry','temperate','tropical','cold']:
    df11 = df.loc[df.climate_label==climate,:]
    ratio = df11.loc[(df11.treatment_effects>0)&(df11.te_lower>0),:].shape[0] / df11.shape[0]
    ratio2 = df11.loc[(df11.treatment_effects<0)&(df11.te_upper<0),:].shape[0] / df11.shape[0]
    print(climate, '%.2f'%ratio, ' gauges positive ATE', '%.2f'%ratio2, ' gauges negative ATE')

# save 
df.to_csv(f'../results/dynamic_causal_{model}_{outName}.csv', index = False)