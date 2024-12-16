import os,glob,sys,re
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import hydroeval as he
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer
import pickle
import shap
import cupy as cp
import tqdm
import warnings

################# define a custom loss function for lightGBM #########################
def mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / np.mean(y_true)

def mape_loss(y_true, y_pred):
    grad = np.where(y_true == 0, y_pred/0.001, (y_pred - y_true) / np.abs(y_true))
    hess = np.where(y_true == 0, 1000, 1 / np.abs(y_true))
    return grad, hess

def kge_score(y_true, y_pred):
    kge0 = he.kge(y_pred, y_true).squeeze()[0]
    return float(kge0)

################# define a custom loss function for lightGBM #########################

try:
    fname = sys.argv[1]
except:
    raise Exception('must specify the input csv name')

try:
    purpose = sys.argv[2]
except:
    warnings.warn('the purpose is not specified, so the default is to conduct cross validation')
    purpose = 'cv'

target = os.path.basename(fname).split('_')[0]
model = 'xgb'

print(sys.argv)

# read dataset
df = pd.read_csv(fname)

if 'HYBAS_ID' not in df.columns:
    s = pd.read_csv('../data/basin_attributes_new.csv')
    df = df.merge(s[['ohdb_id','HYBAS_ID']], on = 'ohdb_id')

# define outName to better name the ouput files
outName = re.sub('final_dataset_', '', os.path.basename(fname).split('.')[0])

# create label-encoding variable for gauge id
x = pd.DataFrame({'ohdb_id':df.ohdb_id.unique(),'gauge_id':np.arange(df.ohdb_id.unique().shape[0])})
df = df.merge(x, on = 'ohdb_id')

# create label-encoding variable for basin id
x = pd.DataFrame({'HYBAS_ID':df.HYBAS_ID.unique(),'basin_id':np.arange(df.HYBAS_ID.unique().shape[0])})
df = df.merge(x, on = 'HYBAS_ID')

# Create a binary feature to indicate whether the temperature is below freezing
df['freeze'] = np.where(df.tmax_3 < 0, 1, 0)

# create label-encoding variable for dam purpose
x = pd.DataFrame({'Main_Purpose':df.Main_Purpose.unique(),'Main_Purpose_id':np.arange(df.Main_Purpose.unique().shape[0])})
df = df.merge(x, on = 'Main_Purpose')

# create label-encoding variable for season 
x = pd.DataFrame({'season':df.season.unique(),'season_id':np.arange(df.season.unique().shape[0])})
df = df.merge(x, on = 'season')
df.year = df.year.astype(np.float32)

# # assign weights to the inverse of length of records
# num_records = df.groupby('ohdb_id')['Q'].count().reset_index().rename(columns={'Q':'weight'})
# num_records['weight'] = 1 / num_records['weight']
# df = df.merge(num_records, on = 'ohdb_id')

df = df.iloc[np.random.choice(1500000, 1000, replace = False),:]
print(df.shape)

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 'aridity', 

                'sedimentary', 'plutonic', 'volcanic', 'metamorphic',

                # 'clay_layer1', 'clay_layer6', 'clay_layer3', 'clay_layer4', 'clay_layer2', 'clay_layer5',
                # 'sand_layer1', 'sand_layer6', 'sand_layer3', 'sand_layer4', 'sand_layer2', 'sand_layer5',
                # 'silt_layer1', 'silt_layer6', 'silt_layer3', 'silt_layer4', 'silt_layer2', 'silt_layer5',
                'clay', 'sand', 'silt',

                'Porosity_x', 'logK_Ice_x',

                'ohdb_latitude', 'ohdb_longitude', 
                
                'year', 'season_id',
                
                'freeze', 'climate', 'basin_id',

                # 'swe_3', 'swmelt_3', 'srad_3', 't2max_3', 't2min_3', 'evap_3', 'pr_3',
                # 'swe_7', 'swmelt_7', 'srad_7', 't2max_7', 't2min_7', 'evap_7', 'pr_7',
                # 'swe_15', 'swmelt_15', 'srad_15', 't2max_15', 't2min_15', 'evap_15', 'pr_15',
                # 'swe_30', 'swmelt_30', 'srad_30', 't2max_30', 't2min_30', 'evap_30', 'pr_30',

                'lwd_3', 'p_3', 'pres_3', 'relhum_3', 'swd_3', 'spechum_3', 'tmax_3', 'tmin_3', 'wind_3', 
                'lwd_7', 'p_7', 'pres_7', 'relhum_7', 'swd_7', 'spechum_7', 'tmax_7', 'tmin_7', 'wind_7', 
                'lwd_15', 'p_15', 'pres_15', 'relhum_15', 'swd_15', 'spechum_15', 'tmax_15', 'tmin_15', 'wind_15', 
                'lwd_30', 'p_30', 'pres_30', 'relhum_30', 'swd_30', 'spechum_30', 'tmax_30', 'tmin_30', 'wind_30', 
                'p_365',

                'runoff_ratio', 'slope_fdc', 'Q10_50', 'high_q_freq', 'low_q_freq', 
                'zero_q_freq', 'cv', 'high_q_dur', 'low_q_dur', 'BFI', 'lagT', 'noResRatio', 'FI', 'p_mean', 
                'stream_elas', 'hfd_mean',

                'tmax_ave', 'tmax_std',

                'ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland',

                'res_darea_normalize', 'Year_ave', 'Main_Purpose_id',
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
X = df[predictors].round(3).astype(np.float32)
y = df['Q'].values
y = y / df['gritDarea'].values * 86.4
if (df.Q==0).any():
    y = y + 0.1
y = np.log(y)
y = np.round(y, 3)

# get feature dtype
feature_types = X.agg(lambda x: 'q').to_dict()
for a in ['climate', 'season_id', 'Main_Purpose_id', 'basin_id']:
    feature_types[a] = 'c'
    X[a] = X[a].astype(np.int16)
feature_types['freeze'] = 'i'
feature_types = list(feature_types.values())

X = X.values

# transfrom from numpy array to cupy array for GPU modeling
X = cp.array(X)

#################################################################################################
#         Model tuning and cross-validation test 
#################################################################################################
# exhaustively search for the optimal hyperparameters
params = {
    'eval_metric':'rmsle', 
    'tree_method':"hist", 
    'device':"cuda", 
    'enable_categorical' : True,
    'feature_types' : feature_types,
    'max_depth':7,
    'min_child_weight':3,
    'gamma' : 0.05,
    'colsample_bytree' : 0.8,
    'n_estimators' : 800,
    'subsample' : 0.8,
    'learning_rate' : 0.01
}


d = xgb.DMatrix(X, label=y, enable_categorical=True, feature_types = feature_types)
model = xgb.train(params, d)

# Get shap values
shap_and_base_values = model.predict(xgb.DMatrix(X, enable_categorical=True, feature_types = feature_types), pred_contribs=True)

# Compute shap interaction values using GPU
shap_interaction_values = model.predict(xgb.DMatrix(X, enable_categorical=True, feature_types = feature_types), pred_interactions=True)