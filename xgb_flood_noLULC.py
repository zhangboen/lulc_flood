import os,glob,sys,re
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import hydroeval as he
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV, KFold
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

# # connect new meteorological forcings
# df_1 = pd.read_csv(f'../data/{target}_seasonal4_multi_MSWX_rain_onlyGridGreaterThan0.1_AveStd.csv')
# df = df.merge(df_1, on = ['ohdb_id',target+'date'])
# del df_1

# exclude reservoir gauges by identifying the name that contains the substring of 'RESERVOIR'
df_attr = pd.read_csv('../../data/OHDB/OHDB_v0.2.3/OHDB_metadata/OHDB_metadata.csv')
df_attr = df_attr.loc[~df_attr.ohdb_station_name.isna(),:]
ohdb_ids = df_attr.loc[df_attr.ohdb_station_name.str.contains('RESERVOIR'),'ohdb_id'].values
df = df.loc[~df.ohdb_id.isin(ohdb_ids),:]

if target == 'Qmax7':
    df = df.loc[df.Q>0,:]

# # exclude those reservoir gauges with the minimum distance less than 1 km using GLAKES data from 10.1038/s41467-022-33239-3
# df_res = pd.read_csv('../data/basin_connect_GLAKES.csv')
# df_res = df_res.loc[(df_res.min_dis_km<=1),:]
# print(df_res.shape)
# df = df.loc[~df.ohdb_id.isin(df_res.ohdb_id.values),:]

# # get annual dataset
# if target == 'Qmax7':
#     df = df.groupby(['ohdb_id','year']).apply(lambda x:x.iloc[np.argmax(x.Q),:]).reset_index(drop = True)
# else:
#     df = df.groupby(['ohdb_id','year']).apply(lambda x:x.iloc[np.argmin(x.Q),:]).reset_index(drop = True)

# # limit to those events with at least 1-year return period
# df = df.loc[df.rp>=1,:]

# limit to those catchment area less than 100,000 km2
df = df.loc[(df.gritDarea<=100000),:]
print(df.ohdb_id.unique().shape)

# limit to catchments with at least 80 seasonal samples
tmp = df.groupby('ohdb_id').Q.count()
df = df.loc[df.ohdb_id.isin(tmp.loc[tmp>=80].index),:]

# # limit to those catchments with at least 0.1% changes in the urban area over the study period (1982-2023)
# tmp = df.groupby('ohdb_id').ImperviousSurface.apply(lambda x:x.max()-x.min()).reset_index()
# df = df.loc[df.ohdb_id.isin(tmp.loc[tmp.ImperviousSurface>=0.1,'ohdb_id'].values),:]
# print(df.ohdb_id.unique().shape)

# define outName to better name the ouput files
outName = re.sub('final_dataset_', '', os.path.basename(fname).split('.')[0]) + '_simple_noLULC'

# # assign weights to the inverse of length of records
# num_records = df.groupby('ohdb_id')['Q'].count().reset_index().rename(columns={'Q':'weight'})
# num_records['weight'] = 1 / num_records['weight']
# df = df.merge(num_records, on = 'ohdb_id')

# limit gauges to those with minimal influences of dams:
    # 2. percentage of reservoir area to catchment area less than 10
# from pyogrio import read_dataframe
# import geopandas as gpd
# df_attr = pd.read_csv('../data/basin_attributes_new.csv')
# gdf_dam = read_dataframe('../../data/geography/GDAT_data_v1/data/GDAT_v1_dams.shp').to_crs('epsg:8857')
# gdf_res = read_dataframe('../../data/geography/GDAT_data_v1/data/GDAT_v1_catchments.shp').to_crs('epsg:8857')
# gdf_res['RESarea'] = gdf_res.area / 1000000
# gdf_dam = gdf_dam.merge(gdf_res[['Feature_ID','RESarea']], on = 'Feature_ID')
# gdf_basin = read_dataframe('../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg')
# connect = gpd.sjoin(gdf_dam[['Feature_ID','RESarea','geometry']], gdf_basin, how = 'right', predicate = 'within').dropna()
# connect = connect.groupby('ohdb_id').apply(lambda x: pd.Series([
#         x.Feature_ID.count(),
#         x.RESarea.sum(),
#         x.gritDarea.sum()
#     ], index = ['countD','RESarea','gritDarea']),
#     # include_groups=False,
# ).fillna(0).reset_index()
# connect['ratio'] = connect.RESarea / connect.gritDarea * 100
# connect.to_csv('../data/basin_reservoir_darea_ratio.csv', index = False)
# connect = pd.read_csv('../data/basin_reservoir_darea_ratio.csv')
# connect = connect.loc[(connect.ratio>=10),:]
# df = df.loc[~df.ohdb_id.isin(connect.ohdb_id),:].reset_index(drop=True)

# # limit gauges to those with positive STD of urban area
# df1 = df.groupby(['ohdb_id','year']).ImperviousSurface.apply(lambda x:x.iloc[0]).reset_index()
# df1 = df1.groupby('ohdb_id').ImperviousSurface.std()
# df1 = df1.loc[df1>0]
# df = df.loc[df.ohdb_id.isin(df1.index.values),:].reset_index(drop=True)

# # limit to upstream gauges
# gdf = pd.read_csv('../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset_onlyUpstream.csv')
# df = df.loc[df.ohdb_id.isin(gdf.ohdb_id.values),:].reset_index(drop=True)

# # limit to 1993-2012 to make a 20-year balanced panel data for DynamicDML
# df = df.loc[(df.year>=1993)&(df.year<=2012),:]
# df1 = df.groupby('ohdb_id')['year'].count().reset_index()
# df = df.loc[df.ohdb_id.isin(df1.loc[df1.year==20,'ohdb_id'].values),:]

print(df.shape, df.ohdb_id.unique().shape[0])

# sort df as time within each gauge
df = df.sort_values(['gauge_id','year']).reset_index(drop=True)

if target == 'Qmax7':
    t = 7 # 7-day average for antecedent conditions if high river flows are investigated
else:
    t = 30 # 30-day average for antecedent conditions if low river flows are investigated

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 
                'aridity', 
                'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                'clay', 'sand', 'silt',
                'Porosity_x', 'logK_Ice_x',
                'ohdb_latitude', 'ohdb_longitude', 
                'year', 
                'climate', 
                'season_id',
                'basin_id',
                'p_'+str(t), 'tmax_'+str(t), 'tmin_'+str(t), 'swd_'+str(t), 'snowmelt_'+str(t), 'snowfall_'+str(t), 'evap_'+str(t), 'smrz_'+str(t),
                # 'ImperviousSurface', 
                # 'forest', 'crop', 'grass', 'water', 'wetland',
                'res_darea_normalize', 'Year_ave', 'Main_Purpose_id',
                'form_factor', 'LAI', 'FAPAR',

                # 'p_3', 'tmax_3', 'tmin_3', 'swd_3', 'snowmelt_3', 'snowfall_3', 'evap_3', 'smrz_3',
                # 'p_7', 'tmax_7', 'tmin_7', 'swd_7', 'snowmelt_7', 'snowfall_7', 'evap_7', 'smrz_7',
                # 'p_15', 'tmax_15', 'tmin_15', 'swd_15', 'snowmelt_15', 'snowfall_15', 'evap_15', 'smrz_15',
                # 'p_30', 'tmax_30', 'tmin_30', 'swd_30', 'snowmelt_30', 'snowfall_30', 'evap_30', 'smrz_30',
                # 'runoff_ratio', 
                # 'slope_fdc', 
                # 'Q10_50', 
                # 'high_q_freq', 'low_q_freq', 'zero_q_freq', 'high_q_dur', 'low_q_dur', 
                # 'cv', 'BFI', 
                # 'FI', 'stream_elas', 'hfd_mean',
                # 'p_mean', 'tmax_ave', 'tmax_std',
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
def LogTrans(y, darea = None):
    y = y + 0.1
    y = y / darea * 86.4
    y = np.log(y)
    return y
def InvLogTrans(y, darea = None):
    y = np.exp(y)
    y = y * darea / 86.4
    y = y - 0.1
    return y

# transform predictand
y = df['Q'].astype(np.float32).values
y = LogTrans(y, darea = df['gritDarea'].values)

X = df[predictors].astype(np.float32)

# get feature dtype
feature_types = X.agg(lambda x: 'q').to_dict()
for a in ['climate', 'season_id', 'Main_Purpose_id', 'basin_id', 'gauge_id', 'koppen_id', 'year']:
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

if purpose == 'cv':
    #################################################################################################
    #         Model tuning and cross-validation test 
    #################################################################################################
    # # custom function to avoid overestimate small values
    # def my_assymetric_error_wrapper(tau, delta):
    #     def my_assymetric_error(y_pred, y_true):
    #         error = (y_pred - y_true)
    #         grad = np.where(((y_true<tau)&(error>0)), delta*2*error, 2*error)
    #         hess = np.where(((y_true<tau)&(error>0)), delta*2, 2)
    #         return grad, hess
    #     return my_assymetric_error

    # exhaustively search for the optimal hyperparameters
    ml=xgb.XGBRegressor(
        eval_metric='rmsle', 
        tree_method="hist", 
        device=device, 
        # objective = my_assymetric_error_wrapper(tau = 0, delta = 9),
        enable_categorical = True,
        feature_types = feature_types,
    )

    # set up our search grid
    param_dist = {
        'max_depth': [3, 5, 7, 9],
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
        n_jobs = 8
    )
    random_search.fit(
        X, 
        y, 
        # sample_weight=df['std_inv'].values, # use 1/(STD+0.1) as sample weight
    )

    best_model = random_search.best_estimator_
    ml = best_model
    try:
        pickle.dump(ml, open(f'../results/{model}_{outName}.pkl', 'wb'))
    except:
        print('fail to save the model')

    params = ml.get_params()
    for name in param_dist.keys():
        print(f'{name:>20}:{params[name]:>10}')

    #=========================================================================
    # # change from GPU to CPU to run
    # X = cp.asnumpy(X)

    # Set a fixed cross-validator with seed
    cv = KFold(n_splits = 10, shuffle = True, random_state = 42)

    y_pred = cross_val_predict(ml, X, y, cv = cv)
    y_pred = InvLogTrans(y_pred, darea = df['gritDarea'].values)

    # prediction cannot be negative
    y_pred = np.where(y_pred < 0, 0, y_pred)  

    df['pred'] = y_pred
    df.to_csv(f'../results/{model}_cv10_{outName}_raw_result.csv', index = False)

    # calculate station-based KGE and Pearson correlation
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
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model
    ml.fit(X, y)

    # calculate shap values
    explainer = shap.TreeExplainer(ml)
    shap_values = explainer.shap_values(X, check_additivity = False)
    pickle.dump(shap_values, open(f'../results/shap_values_{model}_{outName}.pkl','wb'))

    del shap_values

    # Compute shap interaction values by using the annual maximum Qmax7 or annual minimum Qmin7
    if GPU:
        X = cp.asnumpy(X)
    df_annual = pd.DataFrame(X, columns = predictors)
    df_annual[['ohdb_id','year']] = df[['ohdb_id','year']]
    df_annual['Q'] = y
    if target == 'Qmax7':
        df_annual = df_annual.groupby(['ohdb_id','year']).apply(lambda x:x.iloc[np.argmax(x.Q),:]).reset_index(drop = True)
    else:
        df_annual = df_annual.groupby(['ohdb_id','year']).apply(lambda x:x.iloc[np.argmin(x.Q),:]).reset_index(drop = True)
    del X
    print(df_annual.shape)
    
    X0 = df_annual[predictors]
    if GPU:
        X0 = cp.array(X0)
    shap_interaction_values0 = explainer.shap_interaction_values(X0)
    out = { 
        'predictor': predictors,
        'data': df_annual,
        'shap_interaction':shap_interaction_values0
        }
    pickle.dump(out, open(f'../results/shap_interaction_values_{model}_{outName}.pkl','wb'))

elif purpose == 'ale':
    #################################################################################################
    #         ALE
    #################################################################################################
    from src.ale import ale
    
    # load saved model
    ml = pickle.load(open(f'../results/{model}_{outName}.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model
    ml.fit(X, y)
    print('Finish model fitting')

    # Generate Accumulated Local Effects (ALE) for a specific feature
    # for i in ['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland', 
    #             'aridity', 'runoff_ratio', 'cv', 'BFI', 'p_365', 'logK_Ice_x']:
    var = 'ImperviousSurface'
    if var not in predictors:
        sys.exit(0)
    mc_ale = {}
    mc_quantiles = {}
    for climate in df.climate_label.unique():
        train_set_rep = df.loc[df.climate_label==climate,predictors]
        mc_ale0, mc_quantiles0 = ale(
            ml,
            train_set_rep,
            var,
            bins = 100,
            monte_carlo_rep = 100,
            monte_carlo = True,
            monte_carlo_ratio = 0.1,
            log = True,
        )
        mc_ale[climate] = mc_ale0
        mc_quantiles[climate] = mc_quantiles0
    pickle.dump(mc_ale, open(f'../results/mc_ale_{var}_{model}_{outName}.pkl','wb'))
    pickle.dump(mc_quantiles, open(f'../results/mc_quantiles_{var}_{model}_{outName}.pkl','wb'))

elif purpose == 'ale_time':
    #################################################################################################
    #         ALE
    #################################################################################################
    from src.ale import ale
    
    # load saved model
    ml = pickle.load(open(f'../results/{model}_{outName}.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model
    ml.fit(X, y)
    print('Finish model fitting')

    # Generate Accumulated Local Effects (ALE) for a specific feature
    # for i in ['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland', 
    #             'aridity', 'runoff_ratio', 'cv', 'BFI', 'p_365', 'logK_Ice_x']:
    feature = 'ImperviousSurface'
    df['tmp'] = np.where(df.aridity <= 0.65, 'dry', 'wet')
    for year in range(1982, 2021):
        mc_ale = {}
        mc_quantiles = {}
        train_set_rep0 = df.loc[(df.year==year), :]
        if train_set_rep0.shape[0] == 0:
            continue
        for climate in ['dry','wet']:
            train_set_rep = train_set_rep0.loc[(train_set_rep0.tmp==climate), predictors]
            mc_ale0, mc_quantiles0 = ale(
                ml,
                train_set_rep,
                feature,
                bins = 100,
                monte_carlo_rep = 100,
                monte_carlo = True,
                monte_carlo_ratio = 0.1,
            )
            mc_ale[climate] = mc_ale0
            mc_quantiles[climate] = mc_quantiles0
        pickle.dump(mc_ale, open(f'../results/mc_ale_{feature}_dry_wet_{year}_{model}_{outName}.pkl','wb'))
        pickle.dump(mc_quantiles, open(f'../results/mc_quantiles_{feature}_dry_wet_{year}_{model}_{outName}.pkl','wb'))

elif purpose == 'ale2':
    #################################################################################################
    #         2D ALE
    #################################################################################################
    from src.ale import ale
    
    # load saved model
    ml = pickle.load(open(f'../results/{model}_{outName}.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model
    ml.fit(X, y)
    print('Finish model fitting')

    # Generate Accumulated Local Effects (ALE) for 
        # ImperviousSurface
        # aridity
    x1 = 'ImperviousSurface' 
    x2 = 'p_7'
    ale_eff = ale(
        train_set = df[predictors],
        model = ml,
        features = [x1, x2],
        bins = 100,
        monte_carlo_rep = 100,
        monte_carlo = True,
        monte_carlo_ratio = 0.1,
    )
    pickle.dump(ale_eff, open(f'../results/ale2D_{x1}&{x2}_{model}_{outName}.pkl','wb'))

elif purpose == 'pdp':
    #################################################################################################
    #         PDP
    #################################################################################################
    from sklearn.inspection import partial_dependence
    
    # load saved model
    ml = pickle.load(open(f'../results/{model}_{outName}.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model
    ml.fit(X, y)
    print('Finish model fitting')

    # Generate Accumulated Local Effects (ALE) for a specific feature
    for i in ['ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland', 
                'aridity', 'runoff_ratio', 'cv', 'BFI', 'p_365', 'logK_Ice_x']:
        result = {}
        idx = predictors.index(i)
        for climate in df.climate_label.unique():
            train_set_rep = df.loc[df.climate_label==climate,predictors]
            result0 = partial_dependence(ml, train_set_rep, [idx], kind = 'both')
            result[climate] = result0
        pickle.dump(result, open(f'../results/pdp_{i}_{model}_{outName}.pkl','wb'))

elif purpose == 'sensitivity':
    #################################################################################################
    #         Sensitivity analysis: Add impervious surface by 1% 
    #################################################################################################
    # load saved model
    ml = pickle.load(open(f'../results/{model}_{outName}.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model and make predictions
    ml.fit(X, y)
    print('Finish model fitting')
    base = ml.predict(X)

    # add urban area by 1% for each location
    if GPU:
        X = cp.asnumpy(X)
    idx = predictors.index('ImperviousSurface')
    # idx_rainfall = predictors.index('pr_mswep_3')
    if GPU:
        X[:,idx] = X[:,idx] + 10 # the unit is % already
        # X[:,idx_rainfall] = X[:,idx_rainfall] * 1.1 # 3-day rainfall increase by 10%
    else:
        X.iloc[:,idx] = X.iloc[:,idx] + 10 # the unit is % already
        # X.iloc[:,idx_rainfall] = X.iloc[:,idx_rainfall] * 1.1 # 3-day rainfall increase by 10%
    if GPU:
        X = cp.array(X)
    comp = ml.predict(X)
    diff = (np.exp(comp - base) - 1) * 100
    df['diff+010urban'] = diff

    df.to_csv(f'../results/sensitivity_+010urban_diff_in_percentage_{model}_{outName}.csv', index = False)

elif purpose == 'causal':
    #################################################################################################
    #         Machine learning causal inference
    #################################################################################################
    from econml.dml import CausalForestDML, NonParamDML, SparseLinearDML, KernelDML, LinearDML
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
    # try out every combination of the above values
    def score_kge(y, y_pred):
        return he.kge(y_pred, y).ravel()[0]
    kge_scorer = make_scorer(score_kge)

    random_search = RandomizedSearchCV(
        ml, 
        param_distributions = param_dist, 
        n_iter = 50, 
        cv = 3, 
        # scoring='neg_root_mean_squared_error',
        # scoring = kge_scorer,
        verbose = 5, 
        random_state = 42, 
        n_jobs = 8
    )

    # tune outcome model
    random_search.fit(X0, y)
    outcome_model = random_search.best_estimator_

    # tune treatment model
    random_search.fit(X0, T0)
    treatment_model = random_search.best_estimator_

    # cross-validate outcome_model and treatment_model
    y_pred = cross_val_predict(outcome_model, X0, y, cv = 10, n_jobs = 3)
    df['obs_outcome'] = y
    df['sim_outcome'] = y_pred
    T_pred = cross_val_predict(treatment_model, X0, T0, cv = 10, n_jobs = 3)
    df['obs_treatment'] = T0
    df['sim_treatment'] = T_pred
    
    # Evaluate the outcome model anD treatment model
    for i,model1 in enumerate(['outcome','treatment']):
        if i == 0:
            df['obs'] = df['obs_outcome']; df['pred'] = df['sim_outcome']
        else:
            df['obs'] = df['obs_treatment']; df['pred'] = df['sim_treatment']
        df_sta = df.groupby('ohdb_id').apply(
                lambda x: pd.Series(
                    he.kge(x.pred.values, x.obs.values).squeeze().tolist() + [
                        np.sum((x.pred.values-x.obs.values)**2) / np.sum(x.obs.values**2) * 100
                    ], index = ['KGE','r','alpha','beta','nRMSE']),
                # include_groups=False
            ).reset_index()
        num = df_sta.loc[df_sta.KGE>0,:].shape[0]
        print(f'{model1} {num/df_sta.shape[0]*100//1}% stations have KGE > 0')
        num = df_sta.loc[df_sta.KGE>0.3,:].shape[0]
        print(f'{model1} {num/df_sta.shape[0]*100//1}% stations have KGE > 0.3')
        num = df_sta.loc[df_sta.r**2>=0.3,:].shape[0]
        print(f'{model1} {num/df_sta.shape[0]*100//1}% stations have R2 >= 0.3')
        print(f'{model1} Median KGE   is {df_sta.KGE.median()*100//1/100}     Ave KGE   is {df_sta.KGE.mean()*100//1/100}')
        print(f'{model1} Median r     is {df_sta.r.median()*100//1/100}       Ave r     is {df_sta.r.mean()*100//1/100}')
        print(f'{model1} Median alpha is {df_sta.alpha.median()*100//1/100}   Ave alpha is {df_sta.alpha.mean()*100//1/100}')
        print(f'{model1} Median beta  is {df_sta.beta.median()*100//1/100}    Ave beta  is {df_sta.beta.mean()*100//1/100}')
        print(f'{model1} Median nRMSE is {df_sta.nRMSE.median()*100//1/100}   Ave nRMSE is {df_sta.nRMSE.mean()*100//1/100}')

    # Fit the causal forest
    if GPU:
        X0 = cp.asnumpy(X0)

    cf_dml = CausalForestDML(
        model_y=outcome_model, 
        model_t=treatment_model, 
        cv=3, 
        n_estimators=500, 
        min_samples_leaf=10,
        random_state=1234,
        verbose=5,
        n_jobs = 8,
    )

    cf_dml.fit(y, T0, X = X0, inference = 'bootstrap')

    out = {'mdl':cf_dml,'predictor':predictors}

    pickle.dump(out, open(f'../results/causal_model_{model}_{outName}.pkl', 'wb'))

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
    df.to_csv(f'../results/causal_{model}_{outName}.csv', index = False)

    # print importance of counfounders
    predictors.remove('ImperviousSurface')
    a = pd.Series(cf_dml.feature_importances(), index = predictors)
    print(a.sort_values())

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (4,4))
    sns.boxplot(data = df, x = 'climate_label', y = 'treatment_effects', whis = [2.5, 97.5], ax = ax)
    fig.savefig('../picture/temp.png', dpi = 600)