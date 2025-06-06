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
from pathlib import Path
from tqdm import tqdm
import argparse
from src.utils import check_GPU, LogTrans, InvLogTrans, aleplot_1D_continuous, pdpplot_1D_continuous, undersample
import multiprocessing as mp
import statsmodels.api as sm

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
    parser.add_argument('--fname', type=str, required=True, help='Input filename for modelling')
    parser.add_argument('--purpose', type=str, required=True, choices=["cv", "shap", "ale", "pdp", "cv_pdp", "cv_ale", "importance", "sensitivity"])
    parser.add_argument('--mode', type=str, default='onlyUrban', choices=["noLULC", "onlyUrban", "all"])
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--feature', type=str, 
                        default='ImperviousSurface', 
                        choices=['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland'], 
                        help='Feature of interest')
    parser.add_argument('--model', type=str, default='xgb')
    parser.add_argument('--gpu', type=bool, default=GPU)

    # args related to model training
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iteration in RandomizedSearchCV')

    # args related to ALE/PDP/SHAP
    parser.add_argument('--n_inter', type=int, default=100, help='Number of ML interpretation based on ALE/PDP/SHAP')
    parser.add_argument('--min_interval', type=float, default=0, help='Minimum interval of bins for calculating ALE/PDP')
    parser.add_argument('--even', action=argparse.BooleanOptionalAction, default=False, help='Whether use an dataframe with an evenly distributed feature value to calculate ALE/PDP')
    
    # args related to sensitivity analysis
    parser.add_argument('--delta_feature', type=int, default=10, help='Response of target variable to increasing feature by the value of delta_feature')
    
    parser.add_argument('--run_dir', type=str, required=False, help="Path to run directory.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel threads for processing")
    parser.add_argument('--m3s', action=argparse.BooleanOptionalAction, default=False, help='Whether use the unit of m3s for the target variable')
    parser.add_argument('--log', action=argparse.BooleanOptionalAction, default=True, help='Whether log-transform the target variable')
    parser.add_argument('--meteo_name', type=list,
                        default=[   'p', 'tmax', 'tmin', 'swd', 'snowmelt', 'smrz',
                        ], help="Meteorological variable name")
    parser.add_argument('--attr_name', type=list,
                        default=[   'BDTICM', 'elevation', 'slope', 'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                                    'clay', 'sand', 'silt', 'Porosity_x', 'logK_Ice_x', 
                                    'year', 'climate', 'season_id', 'basin_id', 
                                    # 'aridity',
                                    # 'ohdb_latitude', 'ohdb_longitude', 
                                    'res_darea_normalize', 'Year_ave', 'Main_Purpose_id', 'form_factor', 'LAI',
                        ], help='Static attribute name')
    parser.add_argument('--lulc_name', type=list,
                        default = ['ImperviousSurface', 'forest', 'crop', 'grass', 'water', 'wetland'],
                        help='Land-cover name')
    cfg = vars(parser.parse_args())

    # get target of this model
    target = os.path.basename(cfg['fname']).split('_')[0]
    cfg['target'] = target

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

def effect_func(cfg):
    #################################################################################################
    #         ALE or PDP estimation
    #################################################################################################
    from sklearn.inspection import partial_dependence

    model = cfg["model"]
    purpose = cfg['purpose']
    feature = cfg['feature']
    mode = cfg['mode']
    min_interval = float(cfg['min_interval'])
    even = cfg['even']

    # load saved model
    ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = cfg['device']
    ml = xgb.XGBRegressor(**param_dict)
    
    # load data
    df, X, y, cfg = load_data(cfg)

    # refit model
    ml.fit(X, y)

    # downsample to make a even dataset
    if even:
        df = undersample(df, feature)

    predictors = cfg['predictors']
    if feature not in predictors:
        raise Exception('feature of interest is not included in the predictors')

    res_df_all = []
    for climate in df.climate_label.unique():
        train_set_rep = df.loc[df.climate_label==climate,:]
        if cfg["purpose"] == 'ale':
            res_df = aleplot_1D_continuous(
                train_set_rep, predictors, ml, feature, grid_size = 100, 
                log = cfg['log'], m3s = cfg['m3s'], monte_carlo = 100, monte_ratio = 0.1, min_interval = min_interval,
            )
        elif cfg["purpose"] == 'pdp':
            train_set_rep = df.loc[df.climate_label==climate,predictors]
            idx = predictors.index(feature)
            # res_df = partial_dependence(ml, train_set_rep, [idx], kind = 'average')
            res_df = pdpplot_1D_continuous(
                train_set_rep, predictors, ml, feature, grid_size = 100,
                log = cfg['log'], m3s = cfg['m3s'], monte_carlo = 100, monte_ratio = 0.1, min_interval = min_interval,
            )
            
            # # Extract values
            # try:
            #     feature_values = res_df['values'][0]
            # except:
            #     feature_values = res_df['grid_values'][0]
            # avg_predictions = res_df['average'][0]

            # # back-transform and centeralize
            # if cfg['log']:
            #     avg_predictions = np.exp(avg_predictions)
            # avg_predictions = avg_predictions - avg_predictions[0]

            # # Convert to DataFrame for inspection
            # res_df = pd.DataFrame({
            #     feature: feature_values,
            #     'eff': avg_predictions
            # })
            # # divide by average prediction to interpret PDP in percentage terms
            # if cfg['log']:
            #     baseline = np.mean(np.exp(ml.predict(train_set_rep[predictors])))
            # else:
            #     baseline = np.mean(ml.predict(train_set_rep[predictors]))
            # res_df['eff'] = res_df['eff'] / baseline * 100
        res_df['climate'] = climate
        res_df_all.append(res_df)
    res_df_all = pd.concat(res_df_all).reset_index()

    outName = cfg["run_dir"] / f'{model}_{mode}_{purpose}_{feature}_min_interval_{min_interval}.csv'
    if even:
        outName = str(outName)[:-4] + '_even.csv'
    res_df_all.to_csv(outName, index = False)

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

def cv_effect_raw_func(cfg, df, y, ml):
    # five-fold cross-validated ALE/PDP
    fold = 5  

    # split dataframe into 5 splits for each climate region 
    df['split'] = 1
    kf = KFold(n_splits = fold, shuffle = True)
    for climate in df.climate_label.unique():
        df0 = df.loc[df.climate_label==climate,:]
        for i,(train_idx, test_idx) in enumerate(kf.split(df0)):
            idx = df0.iloc[test_idx,:].index.values
            df.loc[df.index.isin(idx),'split'] = i+1

    # index of ImperviousSurface for PDP calculation
    predictors = cfg['predictors']
    index_pdp = predictors.index(cfg['feature'])

    # cross-validated ALE/PDP
    res_df_all = []
    for i in range(fold):
        X_train = df.loc[df.split!=i+1,predictors]
        idx = X_train.index.values
        y_train = y[idx]

        # refit model
        ml.fit(X_train, y_train)
        print(f'Finish model fitting for fold {i}')

        for climate in df.climate_label.unique():
            train_set_rep = df.loc[(df.climate_label==climate)&(df.split==i+1),:]
            if purpose == 'ale':
                res_df = aleplot_1D_continuous(train_set_rep, predictors, ml, feature, grid_size = 100, log = cfg['log'], m3s = cfg['m3s'], min_interval = cfg['min_interval'])
            elif purpose == 'pdp':
                res_df = pdpplot_1D_continuous(train_set_rep, predictors, ml, feature, grid_size = 100, log = cfg['log'], m3s = cfg['m3s'], min_interval = cfg['min_interval'])
                
                # Extract values
                try:
                    feature_values = res_df['values'][0]
                except:
                    feature_values = res_df['grid_values'][0]
                avg_predictions = res_df['average'][0]
                
                # back-transform and centeralize
                if log:
                    avg_predictions = np.exp(avg_predictions)
                avg_predictions = avg_predictions - avg_predictions[0]

                # Convert to DataFrame for inspection
                res_df = pd.DataFrame({
                    feature: feature_values,
                    'eff': avg_predictions
                })
                # divide by average prediction to interpret PDP in percentage terms
                if log:
                    baseline = np.mean(np.exp(ml.predict(train_set_rep[predictors])))
                else:
                    baseline = np.mean(ml.predict(train_set_rep[predictors]))
                res_df['eff'] = res_df['eff'] / baseline * 100
            res_df['climate'] = climate
            res_df['fold'] = i
            res_df_all.append(res_df)
            print(f'Finish calculating {purpose} for fold {i} in {climate}')
    res_df_all = pd.concat(res_df_all).reset_index()

def cv_effect_func(cfg):
    #################################################################################################
    #         Cross-validate ALE or PDP: e.g., Use the 80% gauges for each climate to train model
    #         Use the remaining 20% gauges to calculate ALE; repeat this five times
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
    purpose = cfg['purpose']
    min_interval = float(cfg['min_interval'])
    n_inter = int(cfg['n_inter'])

    # load data
    df, X, y, cfg = load_data(cfg)

    # load saved model
    ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = device
    ml = xgb.XGBRegressor(**param_dict)
    
    if feature not in predictors:
        raise Exception('feature of interest is not included in the predictors')
    
    fold = 5  # five-fold cross-validated ALE/PDP

    for n_inter0 in range(n_inter):
        # split dataframe into 5 splits for each climate region 
        df['split'] = 1
        kf = KFold(n_splits = fold, shuffle = True, random_state = cfg['seed'])
        for climate in df.climate_label.unique():
            df0 = df.loc[df.climate_label==climate,:]
            for i,(train_idx, test_idx) in enumerate(kf.split(df0)):
                idx = df0.iloc[test_idx,:].index.values
                df.loc[df.index.isin(idx),'split'] = i+1

        # index of ImperviousSurface for PDP calculation
        index_pdp = predictors.index(feature)

        # cross-validated ALE
        res_df_all = []
        for i in range(fold):
            X_train = df.loc[df.split!=i+1,predictors]
            idx = X_train.index.values
            y_train = y[idx]

            # refit model
            ml.fit(X_train, y_train)
            print(f'Finish model fitting for fold {i}')

            for climate in df.climate_label.unique():
                train_set_rep = df.loc[(df.climate_label==climate)&(df.split==i+1),:]
                if purpose == 'cv_ale':
                    res_df = aleplot_1D_continuous(train_set_rep, predictors, ml, feature, grid_size = 100, log = log, m3s = m3s, min_interval = min_interval)
                elif purpose == 'cv_pdp':
                    res_df = pdpplot_1D_continuous(train_set_rep, predictors, ml, feature, grid_size = 100, log = log, m3s = m3s, min_interval = min_interval)
                    
                    # Extract values
                    try:
                        feature_values = res_df['values'][0]
                    except:
                        feature_values = res_df['grid_values'][0]
                    avg_predictions = res_df['average'][0]
                    
                    # back-transform and centeralize
                    if log:
                        avg_predictions = np.exp(avg_predictions)
                    avg_predictions = avg_predictions - avg_predictions[0]

                    # Convert to DataFrame for inspection
                    res_df = pd.DataFrame({
                        feature: feature_values,
                        'eff': avg_predictions
                    })
                    # divide by average prediction to interpret PDP in percentage terms
                    if log:
                        baseline = np.mean(np.exp(ml.predict(train_set_rep[predictors])))
                    else:
                        baseline = np.mean(ml.predict(train_set_rep[predictors]))
                    res_df['eff'] = res_df['eff'] / baseline * 100
                res_df['climate'] = climate
                res_df['fold'] = i
                res_df_all.append(res_df)
                print(f'Finish calculating {purpose} for fold {i} in {climate}')
        res_df_all = pd.concat(res_df_all).reset_index()
        res_df_all.to_csv(cfg["run_dir"] / f'{model}_{mode}_{purpose}_{feature}_min_interval_{min_interval}.csv', index = False)

def lowess_func(par, frac = 0.666):
    sample_df, xvals, i = par
    lowess1 = sm.nonparametric.lowess(sample_df['shap'], sample_df['feature'], frac=frac, xvals=xvals, return_sorted = True)
    out = pd.DataFrame({'feature':xvals,'shap'+str(i):lowess1}).set_index('feature')
    return out

def shap_func(cfg, lowess = False):
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
    
    # refit model
    ml.fit(X, y)

    # calculate shap values
    if not os.path.exists(cfg["run_dir"] / f'{model}_{mode}_shap_explainer.pkl') or not os.path.exists(cfg["run_dir"] / f'{model}_{mode}_shap_values.pkl'):
        explainer = shap.TreeExplainer(ml)
        shap_values = explainer.shap_values(X, check_additivity = False)

        pickle.dump(explainer, open(cfg["run_dir"] / f'{model}_{mode}_shap_explainer.pkl','wb'))
        pickle.dump(shap_values, open(cfg["run_dir"] / f'{model}_{mode}_shap_values.pkl','wb'))
    else:
        explainer = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_shap_explainer.pkl','rb'))

    # if not os.path.exists(cfg["run_dir"] / f'{model}_{mode}_shap_interaction_values.pkl'):
    shap_interaction_values0 = explainer.shap_interaction_values(X)
    pickle.dump(shap_interaction_values0, open(cfg["run_dir"] / f'{model}_{mode}_shap_interaction_values.pkl','wb'))

    # estimate LOWESS line in SHAP dependence plots for dry and wet catchments, respectively
    if lowess:
        shap_values = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_shap_values.pkl','rb'))
        
        if not os.path.exists(cfg["run_dir"] / f'{model}_{mode}_shap_lowess_{feature}_dry_wet.csv'):
            idx = predictors.index(feature)

            if GPU:
                X0 = cp.asnumpy(X)
            else:
                X0 = X

            tmp = pd.DataFrame({
                'aridity':df.aridity.values,
                'feature':X0[:,idx],
                'shap':shap_values[:,idx]
            })
            tmp['catch'] = np.where(tmp.aridity.values <= 0.65, 'dry', 'wet')
            
            n_boot = 50

            df_out = []
            for catch in ['dry','wet']:
                df1 = tmp.loc[tmp.catch == catch,:].drop(columns=['aridity'])
                xvals = np.linspace(df1.feature.min(), df1.feature.max(), 100)
                lowess0 = sm.nonparametric.lowess(df1['shap'].values, df1['feature'].values, xvals=xvals, frac=0.5, return_sorted = True)   
                lowess0 = pd.DataFrame({'feature':xvals,'shap':lowess0}).set_index('feature')

                # bootstrap
                pool = mp.Pool(cfg['num_workers'])
                pars = [[df1.sample(frac=0.1, replace=True), ss, xvals] for ss in range(n_boot)]
                boot_lowess = list(tqdm(pool.imap(lowess_func, pars), total=n_boot))
                boot_lowess = pd.concat(boot_lowess, axis = 1)

                lowess0 = pd.concat([lowess0, boot_lowess], axis = 1).reset_index()
                lowess0['catch'] = catch
                df_out.append(lowess0)
            df_out = pd.concat(df_out)
            df_out.to_csv(cfg["run_dir"] / f'{model}_{mode}_shap_lowess_{feature}_dry_wet.csv', index = False)

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
    df = df[['ohdb_id',target+'date','diff']]
    df.to_csv(cfg["run_dir"] / f'{model}_{mode}_{purpose}_+0{delta_feature}{feature}_diff_in_percentage.csv', index = False)

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
            if v != config[k] and k not in ['lulc_name','meteo_name','attr_name']: # update only attributes that are different and no these three attributes
                user_cfg[k] = config[k]
        config = user_cfg
    else:
        # setup modelling folder
        config = _setup_run(config)

    # print config to terminal
    for key, val in config.items():
        print(f"{key}: {val}")

    purpose = config['purpose']
    # processing
    if purpose == 'cv_pdp' or purpose == 'cv_ale':
        purpose = 'cv_effect_func'
    elif purpose == 'pdp' or purpose == 'ale':
        purpose = 'effect_func'
    else:
        purpose = purpose + '_func'
    globals()[purpose](config)