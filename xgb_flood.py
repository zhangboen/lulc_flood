import os,glob,sys,re,json,joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import hydroeval as he
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer
import pickle
import shap
from datetime import datetime
from tqdm import tqdm
import argparse
from src.utils import check_GPU, LogTrans, InvLogTrans, aleplot_1D_continuous, pdpplot_1D_continuous, undersample
import multiprocessing as mp
import statsmodels.api as sm
# import lmoments3 as lm
# from lmoments3 import distr

# check if GPU is available
GPU = check_GPU()
if GPU:
    import cupy as cp

def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, choices=['Qmax7', 'Qmin7'])
    parser.add_argument('--fname', type=str, help='Input filename for modelling')
    parser.add_argument('--purpose', type=str, required=True, choices=["cv", "shap", "ale", "ale2", "pdp", "importance", "sensitivity", "real_sensitivity"])
    parser.add_argument('--mode', type=str, default='onlyUrban', choices=["noLULC", "onlyUrban", "onlyForest", "all"])
    parser.add_argument('--seed', type=int, help="Random seed")
    parser.add_argument('--model', type=str, default='xgb')
    parser.add_argument('--gpu', type=bool, default=GPU)

    # args related to model training
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iteration in RandomizedSearchCV')

    # args related to ALE/PDP/SHAP
    parser.add_argument('--n_explain', type=int, default=100, help='Number of ML interpretation based on ALE/PDP/SHAP')
    parser.add_argument('--min_interval', type=float, default=0, help='Minimum interval of bins for calculating ALE/PDP')
    parser.add_argument('--even', action=argparse.BooleanOptionalAction, default=False, help='Whether use an dataframe with an evenly distributed feature value to calculate ALE/PDP')
    
    # args related to sensitivity analysis
    parser.add_argument('--delta_feature', type=int, default=10, help='Response of target variable to increasing feature by the value of delta_feature')
    
    parser.add_argument('--run_dir', type=str, required=False, help="Path to run directory.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel threads for processing")
    parser.add_argument('--m3s', action=argparse.BooleanOptionalAction, default=False, help='Whether use the unit of m3s for the target variable')
    parser.add_argument('--log', action=argparse.BooleanOptionalAction, default=True, help='Whether log-transform the target variable')
    parser.add_argument('--meteo_name', type=list,
                        default=[   'p', 'tmax', 'tmin', 'swd', 'snowmelt', 
                        # 'smrz',
                        ], help="Meteorological variable name")
    parser.add_argument('--attr_name', type=list,
                        default=[   'BDTICM', 'elevation', 
                        'slope', 
                        'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                                    'clay', 
                                    'sand', 'silt', 'Porosity_x', 'logK_Ice_x', 
                                    # 'year', 'climate', 'basin_id', 
                                    # 'aridity', 'season_id', 
                                    # 'ohdb_latitude', 'ohdb_longitude', 'LAI',
                                    # 'res_darea_normalize', 'Year_ave', 'Main_Purpose_id', 
                                    'form_factor', 
                        ], help='Static attribute name')
    parser.add_argument('--lulc_name', type=list,
                        default = ['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland'],
                        help='Land-cover name')
    cfg = vars(parser.parse_args())

    # get file path
    if cfg['target'] is None:
        cfg['target'] = os.path.basename(cfg['fname']).split('_')[0]

    target = cfg['target']
    if cfg['purpose'] == 'cv' and cfg['fname'] is None:
        raise Exception('input file names must be provided for cross-validation')

    # get device name: cuda or cpu
    if cfg['gpu']:
        device = 'cuda'
    else:
        device = 'cpu'
    cfg['device'] = device

    # update land-cover
    if cfg['mode'] == 'noLULC':
        cfg['lulc_name'] = []
    if cfg['mode'] == 'onlyUrban':
        cfg['lulc_name'] = ['ImperviousSurface']
        cfg['feature'] = 'ImperviousSurface'
    if cfg['mode'] == 'onlyForest':
        cfg['lulc_name'] = ['forest']
        cfg['feature'] = 'forest'

    # Validation checks
    if cfg["seed"] is None:
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))
    if cfg['run_dir'] is not None:
        cfg['seed'] = int(Path(cfg['run_dir']).name.split('_')[-1][4:])

    if (cfg["purpose"] != 'cv') and (cfg["run_dir"] is None):
            raise ValueError("In non-cross-validation purpose a run directory (--run_dir) has to be specified")

    # convert path to PosixPath object
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])

    return cfg

def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    target = cfg['target']
    mode = cfg['mode']
    seed = cfg["seed"]
    run_name = f'run_{target}_{mode}_{day}{month}_{hour}{minute}_seed{seed}'

    if cfg['run_dir'] is None:
        cfg['run_dir'] = Path(__file__).absolute().parent.parent / "results" / run_name
        if not cfg["run_dir"].is_dir():
            cfg["run_dir"].mkdir(parents=True)

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg

def load_data(cfg,  resample = False):
    # read dataset
    df = pd.read_pickle(cfg['fname'])

    # shuffle data 
    if resample:
        df = df.sample(frac=1, random_state = cfg['seed']).reset_index(drop=True)

    # # limit to those catchments with at least 0.1% changes in the urban area over the study period (1982-2023)
    # tmp = df.groupby('ohdb_id').ImperviousSurface.apply(lambda x:x.max()-x.min()).reset_index()
    # df = df.loc[df.ohdb_id.isin(tmp.loc[tmp.ImperviousSurface>=0.1,'ohdb_id'].values),:]
    # print(df.ohdb_id.unique().shape)

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

    # predictand
    y = df['Q'].astype(np.float32).values
    if cfg['log'] == True and cfg['m3s'] == True:
        raise Exception('log and m3s cannot be both True')
    elif cfg['log'] == True and cfg['m3s'] == False:
        y = LogTrans(y, darea = df['gritDarea'].values, addition = 0.1, log = True)
    elif cfg['log'] == False and cfg['m3s'] == False:
        y = y

    # predictors
    predictors = cfg['meteo_name'] + cfg['lulc_name'] + cfg['attr_name']
    cfg['predictors'] = predictors
    X = df[predictors].astype(np.float32)

    # get feature dtype
    cate_attr = []
    feature_types = X.agg(lambda x: 'q').to_dict()
    for a in ['climate', 'season_id', 'Main_Purpose_id', 'basin_id', 'gauge_id', 'koppen_id', 'year']:
        if a in predictors:
            cate_attr.append(a)
            feature_types[a] = 'c'
            X[a] = X[a].astype(np.int16)
    feature_types = list(feature_types.values())
    cfg['feature_types'] = feature_types
    cfg['categorical_attr'] = cate_attr

    # transfrom from numpy array to cupy array for GPU modeling
    if cfg['gpu']:
        X = cp.array(X)

    return df, X, y, cfg

def cv_func(cfg):
    #################################################################################################
    #         Model tuning and cross-validation test 
    #################################################################################################
    # load data
    df, X, y, cfg = load_data(cfg, resample = True)

    # exhaustively search for the optimal hyperparameters
    ml=xgb.XGBRegressor(
        eval_metric='rmsle', 
        tree_method="hist", 
        device=cfg['device'], 
        # objective = my_assymetric_error_wrapper(tau = 0, delta = 9),
        enable_categorical = True,
        feature_types = cfg['feature_types'],
    )

    # set up our search grid
    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    model = cfg["model"]
    mode = cfg['mode']

    if not os.path.exists(cfg["run_dir"] / f'{model}_{mode}_model.pkl'):
        # try out every combination of the above values
        cv = KFold(n_splits = 3, shuffle = True)
        random_search = RandomizedSearchCV(
            ml, 
            param_distributions = param_dist, 
            n_iter = cfg['n_iter'], 
            cv = cv, 
            verbose = 5, 
            random_state = cfg['seed'], 
            n_jobs = cfg['num_workers']
        )
        random_search.fit(X, y)

        ml = random_search.best_estimator_
        try:
            pickle.dump(ml, (cfg["run_dir"] / f'{model}_{mode}_model.pkl').open('wb'))
        except:
            print('fail to save the model')
    else:
        # load saved model
        ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
        param_dict = ml.get_params()
        param_dict['device'] = cfg['device']
        ml = xgb.XGBRegressor(**param_dict)

    #=========================================================================
    # Set a fixed cross-validator with seed
    cv = KFold(n_splits = 10, shuffle = True, random_state = cfg['seed'])

    y_pred = cross_val_predict(ml, X, y, cv = cv)
    if cfg['m3s'] is False:
        if cfg['log']:
            y_pred = InvLogTrans(y_pred, darea = df['gritDarea'].values, addition = 0.1, log = True)
        else:
            y_pred = InvLogTrans(y_pred, darea = df['gritDarea'].values, addition = 0.1, log = False)

    # prediction cannot be negative
    y_pred = np.where(y_pred < 0, 0, y_pred)  

    df['pred'] = y_pred
    df_out = df[['ohdb_id',cfg['target']+'date','pred']]
    df_out.to_csv((cfg["run_dir"] / f'{model}_{mode}_cv10_raw_result.csv'), index = False)

    # calculate station-based KGE and Pearson correlation
    from parallel_pandas import ParallelPandas
    ParallelPandas.initialize(n_cpu=24, split_factor=24)
    df_sta = df.groupby(['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label']).p_apply(
        lambda x: pd.Series(
            he.kge(x.pred.values, x.Q.values).squeeze().tolist() + [
                np.sqrt(np.mean((x.pred.values - x.Q.values)**2)) / (np.max(x.Q.values) - np.min(x.Q.values)) * 100,
                np.sqrt(np.mean((x.pred.values - x.Q.values)**2)) / np.mean(x.Q.values) * 100,
            ], index = ['KGE','r','alpha','beta','nRMSEminmax', 'nRMSEmean'])
    ).reset_index()
    df_sta.to_csv(cfg["run_dir"] / f'{model}_{mode}_cv10_station_based_result.csv', index = False)

def importance_func(cfg):
    #################################################################################################
    #         Evaluate model performance with and without urban area in the ML
    #         Run 100 times to get the average nRMSE
    #################################################################################################
    mode = cfg['mode']
    predictors = cfg['predictors']
    GPU = cfg['gpu']
    run_dir = cfg['run_dir']
    model = cfg['model']
    log = cfg['log']
    m3s = cfg['m3s']

    # load data
    df, X, y, cfg = load_data(cfg)

    if mode == 'noLULC' or mode == 'onlyUrban':
        raise Exception('When purpose is importance, mode should be nothing but {mode} is given!')
    df_out = []
    nn = 10
    for mode in ['noLULC','onlyUrban']:
        if mode == 'noLULC':
            s = [item for item in predictors if item not in ['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland']]
        elif mode == 'onlyUrban':
            s = [item for item in predictors if item not in ['forest', 'crop', 'grass', 'water', 'wetland']]
        X = df[s].astype(np.float32)
        if GPU:
            X = cp.array(X)
        ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl','rb'))
        param_dict = ml.get_params()
        param_dict['device'] = cfg['device']
        ml = xgb.XGBRegressor(**param_dict)
        for i in range(nn):
            cv = KFold(n_splits = 10, shuffle = True, random_state = int(12 * (i+1)))
            y_pred = cross_val_predict(ml, X, y, cv = cv)
            y_pred = InvLogTrans(y_pred, darea = df['gritDarea'].values, log = log, m3s = m3s)
            y_pred = np.where(y_pred < 0, 0, y_pred)  
            df['pred'+str(i)] = y_pred
            print(mode, i)
        df_out = df[['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','Q']+['pred'+str(i) for i in range(nn)]]
        df_out.to_csv(cfg["run_dir"] / f'{model}_{mode}_importance_run{nn}_{feature}.csv', index = False)

def effect_raw_func(df, cfg, ml, group, n_explain0, fold = 5):
    # split dataframe into 5 splits for each climate region 
    df['split'] = 1
    kf = KFold(n_splits = fold, shuffle = True)
    for climate in df.climate_label.unique():
        df0 = df.loc[df.climate_label==climate,:]
        for i,(train_idx, test_idx) in enumerate(kf.split(df0)):
            idx = df0.iloc[test_idx,:].index.values
            df.loc[df.index.isin(idx),'split'] = i+1

    y = df['Q'].astype(np.float32).values
    if cfg['log'] == True and cfg['m3s'] == True:
        raise Exception('log and m3s cannot be both True')
    elif cfg['log'] == True and cfg['m3s'] == False:
        y = LogTrans(y, darea = df['gritDarea'].values, addition = 0.1, log = True)
    elif cfg['log'] == False and cfg['m3s'] == False:
        y = y

    # index of ImperviousSurface for PDP calculation
    predictors = cfg['predictors']
    index_pdp = predictors.index(cfg['feature'])

    # cross-validated ALE/PDP
    res_df_all = []
    for split_id in range(1, fold+1):
        X_train = df.loc[df.split!=split_id,cfg['predictors']]
        idx = X_train.index.values
        y_train = y[idx]

        # refit model
        if cfg['gpu']:
            X_train = cp.array(X_train.values)
        ml.fit(X_train, y_train)

        train_set_rep = df.loc[(df.split==split_id),:].reset_index()
        res_df = aleplot_1D_continuous(train_set_rep, cfg['predictors'], ml, cfg['feature'], grid_size = 100, group=group, 
                                        log = cfg['log'], m3s = cfg['m3s'], min_interval = cfg['min_interval'])
        res_df_all.append(res_df)
    res_df_all = pd.concat(res_df_all)
    res_df_all['n_explain'] = n_explain0
    return res_df_all

def effect_func(cfg):
    #################################################################################################
    #         Cross-validate ALE or PDP: e.g., Use the 80% gauges for each climate to train model
    #         Use the remaining 20% gauges to calculate ALE; repeat this five times
    #################################################################################################
    mode = cfg['mode']
    predictors = cfg['meteo_name'] + cfg['lulc_name'] + cfg['attr_name']
    run_dir = cfg['run_dir']
    model = cfg['model']
    device = cfg['device']
    feature = cfg['feature']
    purpose = cfg['purpose']
    min_interval = float(cfg['min_interval'])
    n_explain = int(cfg['n_explain'])
    
    if feature not in predictors:
        raise Exception('feature of interest is not included in the predictors')
    
    df, _, _, cfg = load_data(cfg)

    ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = cfg['device']
    ml = xgb.XGBRegressor(**param_dict)

    # repeat cross-validated ALE/PDP for n_explain times
    df_out = []
    group = 'climate_label'
    for n_explain0 in tqdm(range(n_explain)):
        df0 = effect_raw_func(df, cfg, ml, group, n_explain0, fold = 5)
        df0['n_explain'] = n_explain0
        df_out.append(df0)
    df_out = pd.concat(df_out)
    df_out.to_csv(cfg["run_dir"] / f'{model}_{mode}_{purpose}_{feature}_min_interval_{min_interval}.csv', index = False)

def lowess_func(par, frac = 0.666):
    sample_df, xvals, i = par
    lowess1 = sm.nonparametric.lowess(sample_df['shap'], sample_df['feature'], frac=frac, xvals=xvals, return_sorted = True)
    out = pd.DataFrame({'feature':xvals,'shap'+str(i):lowess1}).set_index('feature')
    return out

def shap_func(cfg, lowess = False, fold = 5):
    #################################################################################################
    #         SHAP analysis
    #################################################################################################
    mode = cfg['mode']
    predictors = cfg['meteo_name'] + cfg['lulc_name'] + cfg['attr_name']
    GPU = cfg['gpu']
    run_dir = cfg['run_dir']
    model = cfg['model']
    log = cfg['log']
    m3s = cfg['m3s']
    device = cfg['device']
    feature = cfg['feature']

    # load data
    df, X, y, cfg = load_data(cfg)

    # load saved model
    ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    shap_values = np.zeros(X.shape, dtype=np.float32)
    shap_interaction_values = np.zeros((X.shape[0], X.shape[1], X.shape[1]), dtype=np.float32)

    seed = int(np.random.uniform(low=0, high=1e6))
    kf = KFold(n_splits = fold, shuffle = True, random_state = seed)
    for j,(train_idx, test_idx) in enumerate(kf.split(df)):
        X_train = cp.array(df.iloc[train_idx,:][predictors])
        y_train = y[train_idx]
        X_test = cp.array(df.iloc[test_idx,:][predictors])

        # refit model
        ml.fit(X_train, y_train)

        # calculate shap values
        explainer = shap.TreeExplainer(ml)
        shap_values0 = explainer.shap_values(X_test, check_additivity = False)
        shap_interaction_values0 = explainer.shap_interaction_values(X_test)

        shap_values[test_idx,:] = shap_values0
        shap_interaction_values[test_idx,:,:] = shap_interaction_values0
    
    pickle.dump(shap_values, open(cfg["run_dir"] / f'{model}_{mode}_shap_values_explain_{seed}.pkl','wb'))
    pickle.dump(shap_interaction_values, open(cfg["run_dir"] / f'{model}_{mode}_shap_interaction_values_explain_{seed}.pkl','wb'))

def sensitivity_func(cfg):
    #################################################################################################
    #         Sensitivity analysis: Add impervious surface by 1% 
    #################################################################################################
    mode = cfg['mode']
    predictors = cfg['meteo_name'] + cfg['lulc_name'] + cfg['attr_name']
    GPU = cfg['gpu']
    run_dir = cfg['run_dir']
    model = cfg['model']
    log = cfg['log']
    m3s = cfg['m3s']
    device = cfg['device']
    feature = cfg['feature']
    target = cfg['target']
    delta_feature = int(cfg['delta_feature'])

    # load data
    df, X, y, cfg = load_data(cfg)

    # load saved model
    ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model and make predictions
    ml.fit(X, y)
    print('Finish model fitting')
    base = ml.predict(X)
    if log:
        base = np.exp(base)

    # add urban area by 10% for each location
    if GPU:
        X = cp.asnumpy(X)
    
    idx = predictors.index(feature)
    if GPU:
        X[:,idx] = X[:,idx] + delta_feature # the unit is % already
    else:
        X.iloc[:,idx] = X.iloc[:,idx] + delta_feature # the unit is % already
    if GPU:
        X = cp.array(X)
    comp = ml.predict(X)
    if log:
        comp = np.exp(comp)
    diff = (comp - base) / base * 100
    df['diff'] = diff
    df['base'] = base
    df['future'] = comp
    df = df[['ohdb_id', target+'date', 'base', 'future', 'diff']]
    df.to_csv(cfg["run_dir"] / f'{model}_{mode}_{purpose}_+0{delta_feature}{feature}_diff_in_percentage.csv', index = False)

def fit_gev_lmoments(x):
    # Calculate the first few L-moments
    # By default, lmoments() calculates L1, L2, L3, L4 (or upto 4 if not specified)
    paras = distr.gev.lmom_fit(x)
    fitted_gev = distr.gev(**paras)
    return lambda xx:fitted_gev.cdf(xx)

def real_sensitivity_func(cfg):
    #####################################################################################################
    #         Realistic-scenario sensitivity analysis: Add impervious surface by different SSP scenarios
    #         We used the urban projection datasets from Gao et al., 2021 Nature communications paper
    #         We calculated the difference between 2010 (base year) and 2100, and added this difference to
    #         the dataframe
    #####################################################################################################
    mode = cfg['mode']
    predictors = cfg['meteo_name'] + cfg['lulc_name'] + cfg['attr_name']
    GPU = cfg['gpu']
    run_dir = cfg['run_dir']
    model = cfg['model']
    log = cfg['log']
    m3s = cfg['m3s']
    device = cfg['device']
    feature = cfg['feature']
    target = cfg['target']

    # load data
    df, X, y, cfg = load_data(cfg)

    # load saved model
    ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    # refit model and make predictions
    ml.fit(X, y)
    print('Finish model fitting')
    base = ml.predict(X)
    if log:
        base = np.exp(base)
    df['base'] = base

    # add urban area by 10% for each location
    if GPU:
        X = cp.asnumpy(X)
    
    idx = predictors.index(feature)

    # read urban projections based on different SSPs
    for ssp in ['ssp1','ssp2','ssp3','ssp4','ssp5']:
        df_2010 = pd.read_csv(f'../data_urban_projection/{ssp}_2010.csv')
        df_2100 = pd.read_csv(f'../data_urban_projection/{ssp}_2100.csv')

        df_2010 = df_2010[df_2010.columns[df_2010.columns.str.contains('OHDB')].tolist()+['stat']].set_index('stat').T.loc[:,['mean']]
        df_2100 = df_2100[df_2100.columns[df_2100.columns.str.contains('OHDB')].tolist()+['stat']].set_index('stat').T.loc[:,['mean']]

        df_diff = pd.merge(df_2010.reset_index(), df_2100.reset_index(), on = 'index', suffixes = ('_2010', '_2100'))
        df_diff = df_diff.rename(columns={'index':'ohdb_id'})

        df_diff['diff'] = df_diff['mean_2100'] * 100 - df_diff['mean_2010'] * 100
        df_diff = df_diff.dropna()
        df_diff = df_diff.rename(columns={'diff':'urban_diff','mean_2010':'urban_2010','mean_2100':'urban_2100'})

        X0 = X.copy()
        if GPU:
            X0 = pd.DataFrame(data = X0, columns = predictors)
        X0 = pd.concat([X0,df[['ohdb_id']]], axis = 1).merge(df_diff[['ohdb_id','urban_diff']], on = 'ohdb_id')
        X0[feature] = X0[feature] + X0['urban_diff']
        X1 = X0[predictors].copy()

        if GPU:
            X1 = cp.array(X1)
        comp = ml.predict(X1)
        if log:
            comp = np.exp(comp)
        df[f'{target}_'+ssp] = comp
        df['urban_diff_'+ssp] = X0['urban_diff'].values
        print('Finish calculating ', ssp)
    df = df.rename(columns={'base':f'{target}_base'})
    cols = ['ohdb_id', target+'date', f'{target}_base'] + [f'{target}_{b}' for b in ['ssp1','ssp2','ssp3','ssp4','ssp5']]
    cols = cols + [f'urban_diff_{b}' for b in ['ssp1','ssp2','ssp3','ssp4','ssp5']]
    df = df[cols]
    df.to_csv(cfg["run_dir"] / f'{model}_{mode}_{purpose}_{feature}_diff_in_percentage.csv', index = False)

def ale_time_func(cfg):
    #################################################################################################
    #         Time-varying ALE (for each year)
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

def ale2_func(cfg):
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
    x2 = 'smrz'
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

def causal_func(cfg):
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
        random_state = cfg['seed'], 
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
                        np.sqrt(np.mean((x.pred.values-x.obs.values)**2)) / (np.max(x.obs.values) - np.min(x.obs.values)) * 100
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
        random_state=cfg['seed'],
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
    
    for climate in df.climate_label.unique():
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

if __name__ == "__main__":
    # get new args
    config = get_args()

    # get old args
    if config['run_dir'] is not None:
        with open(config["run_dir"] / 'cfg.json', 'r') as fp:
            user_cfg = json.load(fp)
    
        # update old args using new args
        for k,v in user_cfg.items():
            if k in config.keys():
                if v != config[k] and k not in ['attr_name','meteo_name', 'fname']:
                    user_cfg[k] = config[k]

        # add new args
        for k,v in config.items():
            if k not in user_cfg:
                user_cfg[k] = v
        config = user_cfg

    else:
        # setup modelling folder
        config = _setup_run(config)

    # print config to terminal
    for key, val in config.items():
        print(f"{key}: {val}")

    purpose = config['purpose']
    # processing
    if purpose == 'pdp' or purpose == 'ale':
        purpose = 'effect_func'
    else:
        purpose = purpose + '_func'
    globals()[purpose](config)