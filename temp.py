import os,glob,sys,re
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import hydroeval as he
from sklearn.model_selection import KFold
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

df['climate_label'] = df.climate.map({1:'tropical',2:'dry',3:'temperate',4:'cold',5:'polar'})

if 'HYBAS_ID' not in df.columns:
    s = pd.read_csv('../data/basin_attributes_new.csv')
    df = df.merge(s[['ohdb_id','HYBAS_ID']], on = 'ohdb_id')

# create label-encoding variable for gauge id
x = pd.DataFrame({'ohdb_id':df.ohdb_id.unique(),'gauge_id':np.arange(1, df.ohdb_id.unique().shape[0]+1)})
df = df.merge(x, on = 'ohdb_id')

# create label-encoding variable for basin id
x = pd.DataFrame({'HYBAS_ID':df.HYBAS_ID.unique(),'basin_id':np.arange(df.HYBAS_ID.unique().shape[0])})
df = df.merge(x, on = 'HYBAS_ID')

# create label-encoding variable for dam purpose
x = pd.DataFrame({'Main_Purpose':df.Main_Purpose.unique(),'Main_Purpose_id':np.arange(df.Main_Purpose.unique().shape[0])})
df = df.merge(x, on = 'Main_Purpose')

# create label-encoding variable for season 
x = pd.DataFrame({'season':df.season.unique(),'season_id':np.arange(df.season.unique().shape[0])})
df = df.merge(x, on = 'season').reset_index(drop=True)
df.year = df.year.astype(np.float32)

# limit gauges to those with minimal influences of dams:
    # 2. percentage of reservoir area to catchment area less than 10
connect = pd.read_csv('../data/basin_reservoir_darea_ratio.csv')
connect = connect.loc[(connect.ratio>=10),:]
df = df.loc[~df.ohdb_id.isin(connect.ohdb_id),:].reset_index(drop=True)

# limit gauges to those with positive STD of urban area
df1 = df.groupby(['ohdb_id','year']).ImperviousSurface.apply(lambda x:x.iloc[0]).reset_index()
df1 = df1.groupby('ohdb_id').ImperviousSurface.std()
df1 = df1.loc[df1>0]
df = df.loc[df.ohdb_id.isin(df1.index.values),:].reset_index(drop=True)

# calculate 1/STD for each gauge as sample weight
STD = df.groupby('ohdb_id').Q.std() + 0.1
STD.name = 'std_inv'
STD_inv = 1/STD
STD_inv = STD_inv.reset_index()
df = df.merge(STD_inv, on = 'ohdb_id')

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 
                'aridity', 'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                'clay', 'sand', 'silt', 'Porosity_x', 'logK_Ice_x',
                'ohdb_latitude', 'ohdb_longitude', 
                'year', 'climate', 'season_id', 'basin_id', 
                'p_3', 'tmax_3', 'tmin_3', 'swd_3', 'relhum_3', 'wind_3',
                'p_7', 'tmax_7', 'tmin_7', 'swd_7', 'relhum_7', 'wind_7',
                'p_15', 'tmax_15', 'tmin_15', 'swd_15', 'relhum_15', 'wind_15',
                'p_30', 'tmax_30', 'tmin_30', 'swd_30', 'relhum_30', 'wind_30',
                'p_365', 'runoff_ratio', 'slope_fdc', 'Q10_50', 
                'high_q_freq', 'low_q_freq', 'zero_q_freq', 'high_q_dur', 'low_q_dur', 
                'cv', 'BFI', 'noResRatio', 
                'FI', 'lagT', 'stream_elas', 'hfd_mean',
                'p_mean', 'tmax_ave', 'tmax_std',
                'ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland',
                'res_darea_normalize', 'Year_ave', 'Main_Purpose_id',
            ]

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

# Custom Objective Function for MSPE
def msp_objective(y_true, y_pred, sample_weight):
    epsilon = 1e-10  # Small constant to avoid division by zero
    gradient = -2 * (y_true - y_pred) / (y_true**2 + epsilon)
    hessian = 2 / (y_true**2 + epsilon)
    return gradient, hessian

# exhaustively search for the optimal hyperparameters
xgb_regressor=xgb.XGBRegressor(
    eval_metric='rmsle', 
    tree_method="hist", 
    objective=msp_objective,
    device='cpu', 
    enable_categorical = True,
    feature_types = feature_types,
    max_depth = 9,
    learning_rate = 0.1,
    n_estimators = 500,
    subsample = 0.9,
    colsample_bytree = 0.9,
)

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to manually handle cross-validation and sample weights
def cross_val_predict_with_weights(model, X, y, sample_weight, cv):
    predictions = np.zeros(len(y))  # Store the predictions
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        sample_weight_train = sample_weight[train_idx]
        
        # Train model with sample weights
        model.fit(X_train, y_train, sample_weight=sample_weight_train)
        
        # Make predictions on the test fold
        predictions[test_idx] = model.predict(X_test)
    return predictions

# Get cross-validated predictions with sample weights
sample_weights = df.std_inv.values
y_pred = cross_val_predict_with_weights(xgb_regressor, X, y, sample_weights, kf)
y_pred = np.exp(y_pred)
y_pred = y_pred - 0.1
y_pred = y_pred * df['gritDarea'].values / 86.4

# prediction cannot be negative
y_pred = np.where(y_pred < 0, 0, y_pred)  

df['pred'] = y_pred
print(df)

# calculate station-based KGE and Pearson correlation
df_sta = df.groupby(['ohdb_id','climate_label']).apply(
    lambda x: pd.Series(
        he.kge(x.pred.values, x.Q.values).squeeze().tolist() + [
            np.mean(x.pred.values-x.Q.values) / np.mean(x.Q.values) * 100,
        ], index = ['KGE','r','alpha','beta','MAPE'])
).reset_index().groupby('climate_label')[['KGE','r','alpha','beta','MAPE']].mean().reset_index()

print(df_sta)