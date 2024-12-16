import os,glob,sys,re
import hydroeval as he
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
import joblib
from sklearn.metrics import make_scorer

################# define a custom loss function for lightGBM #########################
def mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / np.mean(y_true)

def mape_loss(y_true, y_pred):
    grad = np.where(y_true == 0, y_pred/0.001, (y_pred - y_true) / np.abs(y_true))
    hess = np.where(y_true == 0, 1000, 1 / np.abs(y_true))
    return grad, hess

################# define a custom loss function for lightGBM #########################


try:
    fname = sys.argv[1]
except:
    raise Exception('must specify the input csv name')


target = os.path.basename(fname).split('_')[0]

try:
    model = sys.argv[2]
except:
    print('the model is not specified, so the rf model is used')
    model = 'rf'

try:
    purpose = sys.argv[3]
except:
    print('the purpose is not specified, so the default is to conduct cross validation')
    purpose = 'cv'

if purpose not in ['cv','shap']:
    raise Exception('purpose must be cv or shap')

print(sys.argv)

# read dataset
df = pd.read_csv(fname)

# define outName to better name the ouput files
outName = re.sub('final_dataset_', '', os.path.basename(fname).split('.')[0])

# add gauge id as predictor
x = pd.DataFrame({'ohdb_id':df.ohdb_id.unique(),'gauge_id':np.arange(df.ohdb_id.unique().shape[0])})
df = df.merge(x, on = 'ohdb_id')

# add month predictor
df[target+'date'] = pd.to_datetime(df[target+'date'])
df['month'] = df[target+'date'].dt.month

# add country id as a predictor
x = pd.DataFrame({'country':df.country.unique(),'country_id':np.arange(df.country.unique().shape[0])})
df = df.merge(x, on = 'country')

# Create a binary feature to indicate whether the temperature is below freezing
df['freeze'] = np.where(df.tmax_3 < 0, 1, 0)

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 'aridity', 
                'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                'clay_layer1', 'clay_layer6', 'clay_layer3', 'clay_layer4', 'clay_layer2', 'clay_layer5',
                'sand_layer1', 'sand_layer6', 'sand_layer3', 'sand_layer4', 'sand_layer2', 'sand_layer5',
                'silt_layer1', 'silt_layer6', 'silt_layer3', 'silt_layer4', 'silt_layer2', 'silt_layer5',
                'Porosity_x', 'logK_Ice_x',

                'ohdb_latitude', 'ohdb_longitude', 'year', 'month', 'gauge_id', 'country_id', 'freeze',

                # 'swe_3', 'swmelt_3', 'srad_3', 't2max_3', 't2min_3', 'evap_3', 'pr_3',
                # 'swe_7', 'swmelt_7', 'srad_7', 't2max_7', 't2min_7', 'evap_7', 'pr_7',
                # 'swe_15', 'swmelt_15', 'srad_15', 't2max_15', 't2min_15', 'evap_15', 'pr_15',
                # 'swe_30', 'swmelt_30', 'srad_30', 't2max_30', 't2min_30', 'evap_30', 'pr_30',

                'lwd_3', 'p_3', 'pres_3', 'relhum_3', 'swd_3', 'spechum_3', 'tmax_3', 'tmin_3', 'wind_3', 
                'lwd_7', 'p_7', 'pres_7', 'relhum_7', 'swd_7', 'spechum_7', 'tmax_7', 'tmin_7', 'wind_7', 
                'lwd_15', 'p_15', 'pres_15', 'relhum_15', 'swd_15', 'spechum_15', 'tmax_15', 'tmin_15', 'wind_15', 
                'lwd_30', 'p_30', 'pres_30', 'relhum_30', 'swd_30', 'spechum_30', 'tmax_30', 'tmin_30', 'wind_30',

                'runoff_ratio', 'slope_fdc', 'Q10_50', 'high_q_freq', 'low_q_freq', 
                'zero_q_freq', 'cv', 'high_q_dur', 'low_q_dur', 'BFI', 'lagT', 'noResRatio', 'FI', 'p_mean', 
                'stream_elas', 'hfd_mean',

                'tmax_ave', 'tmax_std',

                'ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland',

                'res_darea_normalize', 'Year_ave', 

                'Main_Purpose_fisheries', 'Main_Purpose_flood control', 'Main_Purpose_hydroelectricity', 'Main_Purpose_other', 
                'Main_Purpose_irrigation', 'Main_Purpose_livestock', 'Main_Purpose_navigation', 'Main_Purpose_nores', 
                'Main_Purpose_other/not specified', 'Main_Purpose_recreation', 'Main_Purpose_water supply', 

                'climate_label_cold', 'climate_label_dry', 'climate_label_polar', 'climate_label_temperate', 'climate_label_tropical'
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

X = df[predictors]
y = df['Q'].values
y = y / df['gritDarea'].values * 86.4
if (df.Q==0).any():
    y = y + 0.1
y = np.log(y)

# create model object
if model == 'rf':
    ml = RandomForestRegressor(n_jobs = -1, n_estimators = 800)
elif model == 'svr':
    ml = SVR(kernel="poly")
elif model == 'gbm':
    ml = lgb.LGBMRegressor()
else:
    raise Exception ('model must be rf, svr, or gbm')

# hyperparamter tunning
if model in ['gbm','svr']:
    if model == 'gbm':    
        param_dist = {
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200, 300, 500, 1000],
            'max_depth': [3, 5, 7, 9],
            'min_child_samples': [5, 10, 20, 50, 100],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    if model == 'svr':
        param_dist = {
                'C': loguniform(1e-2, 1e2),
                'gamma': loguniform(1e-4, 1e-1)
        }
    random_search = RandomizedSearchCV(
        ml, 
        param_distributions = param_dist, 
        n_iter = 100, 
        cv = 3, 
        # scoring = make_scorer(mape, greater_is_better=False), 
        verbose = 2, 
        random_state = 42, 
        n_jobs = -1
    )
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    ml = best_model

if purpose == 'cv':
    # 10-fold spatial temporal cross validation
    y_pred = cross_val_predict(ml, X, y, cv = 10, n_jobs = 4)
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
    # trained ML model
    ml.fit(X, y)

    # save trained ML model
    joblib.dump(ml, f'../results/{model}_{outName}.pkl')

    # shap analysis
    import shap
    import fasttreeshap
    from src.myshap import createShapExplanation

    explainer = fasttreeshap.TreeExplainer(ml, algorithm = 'auto', n_jobs = -1)
    shap_values = explainer(X)
    shap_values = createShapExplanation(shap_values)
    # save output
    with open(f"../results/shap_object_{outName}.pkl", 'wb') as file:  
        pickle.dump(shap_values, file)
    with open(f"../results/shap_explainer_object_{outName}.pkl", 'wb') as file:  
        pickle.dump(explainer, file)