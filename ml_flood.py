import os,glob,sys
import hydroeval as he
import numpy as np
import pandas as pd
import pickle
import shap
import fasttreeshap
from src.myshap import createShapExplanation
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

target = sys.argv[1]
if target not in ['Qmax7','Qmin7']:
    raise Exception('target must be Qmax7 or Qmin7')
try:
    SHAP = sys.argv[2]
except:
    SHAP = 'no'

print(sys.argv)

# read dataset
df = pd.read_csv(f'../data/{target}_final_dataset_filter.csv')

df = df.iloc[:50000,:]

# define predictors and predictand
predictors = [  'BDTICM', 'elevation', 'slope', 'aridity', 
                'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                'clay_layer1', 'clay_layer6', 'clay_layer3', 'clay_layer4', 'clay_layer2', 'clay_layer5',
                'sand_layer1', 'sand_layer6', 'sand_layer3', 'sand_layer4', 'sand_layer2', 'sand_layer5',
                'silt_layer1', 'silt_layer6', 'silt_layer3', 'silt_layer4', 'silt_layer2', 'silt_layer5',
                'Porosity_x', 'logK_Ice_x', 'dam', 'climate',

                'ohdb_latitude', 'ohdb_longitude', 'year',

                'snow_depth_water_equivalent', 'snowmelt_sum',
                'surface_net_solar_radiation_sum', 'temperature_2m_max',
                'temperature_2m_min', 'total_precipitation_sum', 'total_evaporation_sum',

                'ImperviousSurface', 'crop', 'forest', 'grass', 'water', 'wetland'
            ]

X = df[predictors]
y = df['Q'].values
y = np.log(y / df['gritDarea'] * 86.4)

# create random forest object
rf = RandomForestRegressor(n_jobs = 32, n_estimators = 500)

# 10-fold spatial temporal cross validation
y_pred = cross_val_predict(rf, X, y, cv = 10)
y_pred = np.exp(y_pred)  * df['gritDarea'].values / 86.4
df['pred'] = y_pred

# calculate station-based KGE and Pearson correlation
df_sta = df.groupby(['ohdb_id','ohdb_latitude', 'ohdb_longitude','climate_label']).apply(
    lambda x: pd.Series(
        he.kge(x.pred.values, x.Q.values).squeeze().tolist() + [
            np.sum((x.pred.values-x.Q.values)**2) / np.sum(x.Q.values**2) * 100
        ], index = ['KGE','r','alpha','beta','nRMSE'])
).reset_index()

print(df_sta.loc[df_sta.KGE>0,:].shape, df_sta.loc[df_sta.KGE>0.3,:].shape, df_sta.KGE.median(), df_sta.r.median(), df_sta.nRMSE.median())

# shap analysis
if SHAP in ['yes','y','YES','Yes']:
    print('shap test')
    rf.fit(X, y)
    explainer = fasttreeshap.TreeExplainer(rf, algorithm = 'auto', n_jobs = -1)
    shap_values = explainer(X)
    shap_values = createShapExplanation(shap_values)
    # save output
    with open(f"../results/shap_object_{target}_{model}.pkl", 'wb') as file:  
        pickle.dump(shap_values, file)
    with open(f"../results/shap_explainer_object_{target}_{model}.pkl", 'wb') as file:  
        pickle.dump(explainer, file)