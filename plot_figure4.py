from src.plot_utils import *
from pathlib import Path
import json,string
from scipy import stats
from scipy.stats import linregress
from tqdm import tqdm
import pickle

dir_Qmax7 = Path('../results/run_Qmax7_onlyUrban_0506_1359_seed824956/')
dir_Qmin7 = Path('../results/run_Qmin7_onlyUrban_0506_1357_seed220973/')

with open(dir_Qmax7 / 'cfg.json', 'r') as fp:
    cfg_Qmax7 = json.load(fp)
with open(dir_Qmin7 / 'cfg.json', 'r') as fp:
    cfg_Qmin7 = json.load(fp)

predictors = cfg_Qmax7['meteo_name'] + cfg_Qmax7['lulc_name'] + cfg_Qmax7['attr_name']

idx = predictors.index('ImperviousSurface')
idx_smrz = predictors.index('smrz')

def read(par):
    name, seed = par
    try:
        inter0 = pickle.load(open(eval('dir_'+name) / f'xgb_onlyUrban_shap_interaction_values_explain_{seed}.pkl', 'rb'))
    except:
        return

    tmp = pd.DataFrame(data = np.abs(inter0[:,idx,:]), columns = predictors)
    tmp['max_inter_feature'] = tmp.idxmax(axis=1)
    tmp['max_inter_value'] = tmp.iloc[:,:-1].max(axis=1)

    # interaction bewteen urban area and soil moisture
    tmp['inter_urban_smrz'] = inter0[:, idx, idx_smrz]
    tmp['rank_urban_smrz'] = tmp.iloc[:,:-3].rank(1)['smrz'].values

    tmp = tmp[[
        'rank_urban_smrz',
        'inter_urban_smrz',
        'max_inter_value',
        # 'max_inter_feature'
    ]]
    return tmp

if __name__ == '__main__':
    name = 'Qmax7'
    seeds = [str(s).split('_')[-1].split('.')[0] for s in dir_Qmax7.glob('*shap_interaction_values_explain_*pkl')]
    if not os.path.exists(dir_Qmax7 / 'xgb_onlyUrban_shap_interaction_values_explain_ensemble_mean.pkl'):
        with mp.Pool(12) as p:
            df_Qmax7 = list(tqdm(p.imap(read, [(name,seed) for seed in seeds])))
        df_Qmax7 = [a for a in df_Qmax7 if a is not None]
        df_Qmax7 = sum(df_Qmax7) / len(df_Qmax7)
        tmp = pd.read_pickle(cfg_Qmax7['fname'])
        df_Qmax7 = pd.concat([df_Qmax7, tmp[['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']]], axis = 1)
        df_Qmax7 = df_Qmax7.groupby(['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']).mean().reset_index()
        df_Qmax7.to_pickle(dir_Qmax7 / 'xgb_onlyUrban_shap_interaction_values_explain_ensemble_mean.pkl')
    else:
        df_Qmax7 = pd.read_pickle(dir_Qmax7 / 'xgb_onlyUrban_shap_interaction_values_explain_ensemble_mean.pkl')

    name = 'Qmin7'
    seeds = [str(s).split('_')[-1].split('.')[0] for s in dir_Qmin7.glob('*shap_interaction_values_explain_*pkl')]
    if not os.path.exists(dir_Qmin7 / 'xgb_onlyUrban_shap_interaction_values_explain_ensemble_mean.pkl'):
        with mp.Pool(12) as p:
            df_Qmin7 = list(tqdm(p.imap(read, [(name,seed) for seed in seeds])))
        df_Qmin7 = [a for a in df_Qmin7 if a is not None]
        df_Qmin7 = sum(df_Qmin7) / len(df_Qmin7)
        tmp = pd.read_pickle(cfg_Qmin7['fname'])
        df_Qmin7 = pd.concat([df_Qmin7, tmp[['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']]], axis = 1)
        df_Qmin7 = df_Qmin7.groupby(['ohdb_id','ohdb_longitude','ohdb_latitude','climate_label','aridity']).mean().reset_index()
        df_Qmin7.to_pickle(dir_Qmin7 / 'xgb_onlyUrban_shap_interaction_values_explain_ensemble_mean.pkl')
    else:
        df_Qmin7 = pd.read_pickle(dir_Qmin7 / 'xgb_onlyUrban_shap_interaction_values_explain_ensemble_mean.pkl')

    fig, axes = plt.subplots(2, 2, figsize = (8, 7))
    for i,name in enumerate(['Qmin7', 'Qmax7']):
        df1 = eval('df_'+name)
        df1['rank_urban_smrz'] = len(predictors) - df1['rank_urban_smrz']
        ax = axes[i,0]
        sns.ecdfplot(df1, 
                    x = 'rank_urban_smrz', 
                    hue = 'climate_label', 
                    lw = 2,
                    ax = ax, 
                    palette = palette)
        ax.set_xlabel('Rank of SHAP interaction values between\nurban area and soil moisture', fontsize = 11)
        sns.move_legend(ax, 'upper left', title = None, fontsize = 11)
        ax.text(-.15, 1, string.ascii_letters[(i+1)*2-2], 
                weight = 'bold', transform = ax.transAxes, fontsize = 11)
        ax.tick_params(axis = 'both', labelsize = 11)
        ax.set_ylabel('Proportion', fontsize = 11)
        
        ax2 = axes[i,1]
        sns.regplot(df1, x = 'aridity', y = 'rank_urban_smrz', ax = ax2, line_kws={'color': 'red'}, robust = True)
        ax2.set_xlabel('Catchment aridity', fontsize = 11)
        ax2.set_ylabel('Rank of SHAP interaction values\nbetweenurban area and soil moisture', fontsize = 11)
        ax2.tick_params(axis = 'both', labelsize = 11)
        slope, intercept, r_value, p_value, std_err = linregress(df1.aridity.values, df1.rank_urban_smrz.values)
        label = f'r = {r_value:.2f} p = {p_value:.3f}' if p_value > 0.01 else f'r = {r_value:.2f} p < 0.01'
        ax2.text(.95, .05,
                label, 
                transform = ax2.transAxes, ha = 'right')
        ax2.text(-.3, 1, string.ascii_letters[(i+1)*2-1], 
                weight = 'bold', transform = ax2.transAxes, fontsize = 11)

    fig.tight_layout()
    fig.savefig(dir_Qmax7 / 'fig4.png', dpi = 600)