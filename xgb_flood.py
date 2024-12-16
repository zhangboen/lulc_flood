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
X = df[predictors].astype(np.float32)
y = df['Q'].astype(np.float32).values
y = y / df['gritDarea'].values * 86.4
if (df.Q==0).any():
    y = y + 0.1
y = np.log(y)

# get feature dtype
feature_types = X.agg(lambda x: 'q').to_dict()
for a in ['climate', 'season_id', 'Main_Purpose_id', 'basin_id']:
    feature_types[a] = 'c'
    X[a] = X[a].astype(np.int16)
feature_types['freeze'] = 'i'
feature_types = list(feature_types.values())

# transfrom from numpy array to cupy array for GPU modeling
X = cp.array(X)

if purpose == 'cv':
    #################################################################################################
    #         Model tuning and cross-validation test 
    #################################################################################################
    # exhaustively search for the optimal hyperparameters
    ml=xgb.XGBRegressor(
        eval_metric='rmsle', 
        tree_method="hist", 
        device="cuda", 
        enable_categorical = True,
        feature_types = feature_types,
    )

    # set up our search grid
    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.05, 0.1, 0.2, 0.3],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # try out every combination of the above values
    random_search = RandomizedSearchCV(
        ml, 
        param_distributions = param_dist, 
        n_iter = 100, 
        cv = 3, 
        verbose = 5, 
        random_state = 42, 
    )
    random_search.fit(
        X, 
        y, 
        # sample_weight = df.weight.values,
    )
    best_model = random_search.best_estimator_
    ml = best_model
    try:
        pickle.dump(ml, open(f'../results/{model}_{outName}.pkl', 'wb'))
    except:
        print('fail to save the model')

    #=========================================================================
    # change from GPU to CPU to run
    X = cp.asnumpy(X)

    y_pred = cross_val_predict(ml, X, y, cv = 10)
    y_pred = np.exp(y_pred)
    if (df.Q==0).any():
        y_pred = y_pred - 0.1
    y_pred = y_pred * df['gritDarea'].values / 86.4

    # prediction cannot be negative
    y_pred = np.where(y_pred < 0, 0, y_pred)  

    df['pred'] = y_pred
    df.to_csv(f'../results/{model}_cv10_{outName}_raw_result.csv', index = False)

    # calculate station-based KGE and Pearson correlation
    df['climate_label'] = df.climate.map({1:'tropical',2:'dry',3:'temperate',4:'cold',5:'polar'})
    df_sta = df.groupby(['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label']).apply(
        lambda x: pd.Series(
            he.kge(x.pred.values, x.Q.values).squeeze().tolist() + [
                np.sum((x.pred.values-x.Q.values)**2) / np.sum(x.Q.values**2) * 100
            ], index = ['KGE','r','alpha','beta','nRMSE'])
    ).reset_index()
    df_sta.to_csv(f'../results/{model}_cv10_{outName}_station_based_result.csv', index = False)

    num = df_sta.loc[df_sta.KGE>0,:].shape[0]
    print(f'{model} {num/df_sta.shape[0]*100//1}% stations have KGE > 0')
    num = df_sta.loc[df_sta.KGE>0.3,:].shape[0]
    print(f'{model} {num/df_sta.shape[0]*100//1}% stations have KGE > 0.3')
    num = df_sta.loc[df_sta.r**2>=0.3,:].shape[0]
    print(f'{model} {num/df_sta.shape[0]*100//1}% stations have R2 >= 0.3')
    print(f'{model} Median KGE   is {df_sta.KGE.median()*100//1/100}   Ave KGE   is {df_sta.KGE.mean()*100//1/100}')
    print(f'{model} Median r     is {df_sta.r.median()*100//1/100}   Ave r     is {df_sta.r.mean()*100//1/100}')
    print(f'{model} Median nRMSE is {df_sta.nRMSE.median()*100//1/100}   Ave nRMSE is {df_sta.nRMSE.mean()*100//1/100}')

elif purpose == 'shap':
    #################################################################################################
    #         SHAP analysis
    #################################################################################################
    
    # load saved model
    ml = pickle.load(open(f'../results/{model}_{outName}.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = 'cuda'
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model
    ml.fit(X, y)

    # calculate shap values
    explainer = shap.TreeExplainer(ml)
    shap_values = explainer.shap_values(X, check_additivity = False)
    pickle.dump(shap_values, open(f'../results/shap_values_{model}_{outName}.pkl','wb'))
    print(type(shap_values))
    del shap_values

    # Compute shap interaction values using GPU by splitting samples into five parts
    index = np.array_split(np.arange(X.shape[0]), 10)
    for i in tqdm.tqdm(np.arange(10)):
        if os.path.exists(f'../results/shap_interaction_values_{model}_{outName}_chunk{i}.pkl'):
            continue
        X0 = X[index[i],:]
        shap_interaction_values0 = explainer.shap_interaction_values(X0)
        pickle.dump(shap_interaction_values0, open(f'../results/shap_interaction_values_{model}_{outName}_chunk{i}.pkl','wb'))

elif purpose == 'pdp':
    #################################################################################################
    #         PDP and ALE
    #################################################################################################
    from sklearn.inspection import partial_dependence
    # load saved model
    ml = pickle.load(open(f'../results/{model}_{outName}.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = 'cuda'
    ml = xgb.XGBRegressor(**param_dict)

    # refit model
    ml.fit(X, y)

    # Generate partial dependence plot (PDP) for a specific feature
    for i in ['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland']:
        PDP0 = partial_dependence(ml, X, i, kind = 'individual')
        PDP0 = PDP0['individual']
        print(PDP0.shape)