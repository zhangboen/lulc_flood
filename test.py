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
    print('GPU is availabel')
    import cupy as cp

def load_data(cfg,  resample = False):
    # read dataset
    df = pd.read_pickle(cfg['fname'])

    # shuffle data 
    if resample:
        df = df.sample(frac=1, random_state = cfg['seed']).reset_index(drop=True)

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

def effect_wrapper(par, grid_size = 100):
    df, split_id, ml, group, cfg = par
    feature = cfg['feature']

    y = df['Q'].astype(np.float32).values
    if cfg['log'] == True and cfg['m3s'] == True:
        raise Exception('log and m3s cannot be both True')
    elif cfg['log'] == True and cfg['m3s'] == False:
        y = LogTrans(y, darea = df['gritDarea'].values, addition = 0.1, log = True)
    elif cfg['log'] == False and cfg['m3s'] == False:
        y = y

    X_train = df.loc[df.split!=split_id,cfg['predictors']]
    idx = X_train.index.values
    y_train = y[idx]

    # refit model
    if cfg['gpu']:
        X_train = cp.array(X_train.values)
    ml.fit(X_train, y_train)

    train_set_rep = df.loc[(df.split==split_id),:].reset_index()
    res_df = aleplot_1D_continuous(train_set_rep, cfg['predictors'], ml, feature, grid_size = grid_size, group=group, 
                                    log = cfg['log'], m3s = cfg['m3s'], min_interval = cfg['min_interval'])
    return res_df

def effect_raw_func(df, cfg, ml, group, n_explain0, fold = 5):
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
    purpose = cfg['purpose']
    feature = cfg['feature']

    # cross-validated ALE/PDP
    # pool = mp.Pool(fold)
    pars = [(df, split_id, ml, group, cfg) for split_id in range(1,fold+1)]
    res_df_all = [effect_wrapper(par) for par in pars]
    res_df_all = pd.concat(res_df_all)
    res_df_all['n_explain'] = n_explain0
    return res_df_all

if __name__ == '__main__':
    with open('../results/run_Qmin7_onlyUrban_0506_1631_seed152387/cfg.json', 'r') as fp:
        cfg = json.load(fp)

    cfg['device'] = 'cuda'
    cfg['gpu'] = True
    cfg['purpose'] = 'ale'
    cfg['min_interval'] = 0
    cfg['run_dir'] = Path(cfg['run_dir'])

    import time
    t1 = time.time()
    df, _, _, cfg = load_data(cfg)
    t2 = time.time()

    # load saved model
    model = cfg['model']
    mode = cfg['mode']
    ml = pickle.load(open(cfg["run_dir"] / f'{model}_{mode}_model.pkl', 'rb'))
    param_dict = ml.get_params()
    param_dict['device'] = cfg['device']
    ml = xgb.XGBRegressor(**param_dict)
    t3 = time.time()

    tmp = effect_raw_func(df, cfg, ml, 'climate_label', 0, fold = 5)
    t4 = time.time()

    print(t2 - t1)
    print(t3 - t2)
    print(t4 - t3)