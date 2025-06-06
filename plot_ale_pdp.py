from src.plot_utils import *
from pathlib import Path
import sys,json
from scipy.interpolate import interp1d

feature = 'ImperviousSurface'
featureName = 'urban area'

Qmin7Fname = '../data/Qmin7_final_dataset_seasonal4.pkl'
Qmax7Fname = '../data/Qmax7_final_dataset_seasonal4.pkl'

par_map = pd.read_csv('../data/predictors.csv')

dir_Qmax7 = '../results/run_Qmax7_onlyUrban_2805_2303_seed107692'
dir_Qmin7 = '../results/run_Qmin7_onlyUrban_2805_2303_seed387407'

with open(Path(dir_Qmin7) / 'cfg.json', 'r') as fp:
    cfg = json.load(fp)
model = cfg['model']
mode = cfg['mode']
feature = cfg['feature']

type0 = sys.argv[1]

Qmin7Fname = Path(dir_Qmin7) / f'{model}_{mode}_{type0}_{feature}.csv'
Qmax7Fname = Path(dir_Qmax7) / f'{model}_{mode}_{type0}_{feature}.csv'

df_eff_Qmin7 = pd.read_csv(Qmin7Fname)
df_eff_Qmax7 = pd.read_csv(Qmax7Fname)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4))
for i,name in enumerate(['Qmin7','Qmax7']):
    df = eval('df_eff_'+name)
    ax = eval('ax'+str(i+1))
    if type0 in ['cv_pdp', 'cv_ale','ale']:
        for climate in ['tropical','dry','temperate','cold']:
            tmp = df.loc[df.climate==climate,:]
            # Find common x-axis range
            min_x = tmp[feature].min()
            max_x = tmp[feature].max()
            
            groupName = 'fold' if type0.startswith('cv') else 'monte'
            n_sample = tmp.groupby([groupName])['eff'].count().min()
            x_range = np.linspace(min_x, max_x, n_sample)  # Adjust number of points as needed

            # Interpolate each group to common x-axis
            interpolated_data = {}
            for group in tmp[groupName].unique():
                group_df =tmp[tmp[groupName] == group]
                f = interp1d(group_df[feature], group_df['eff'], kind='linear', fill_value='extrapolate')
                interpolated_data[group] = pd.DataFrame({'x': x_range, 'y': f(x_range), 'group': group})

            # Concatenate interpolated data
            interpolated_df = pd.concat(interpolated_data.values())

            # Calculate average y values for each x
            df_avg = interpolated_df.groupby('x').apply(
                lambda x: pd.Series([x.y.mean(), x.y.quantile(.025), x.y.quantile(.975)], index = ['ave','low','upp'])
            ).reset_index()
            
            # Create the lineplot with individual lines
            ax.plot(df_avg.x.values, df_avg.ave.values, color = palette[climate], lw = 2, label = climate, zorder = 3)
            ax.fill_between(
                df_avg.x.values, 
                df_avg.low.values, 
                df_avg.upp.values, 
                color = palette[climate], 
                ec = 'none',
                alpha = .3)
            ax.legend(fontsize = 11)
    elif type0 == 'pdp':
        sns.lineplot(data = df, x = feature, y = 'eff', hue = 'climate', ax = ax, palette = palette, lw = 2)
        sns.move_legend(ax, 'lower right', title = None, fontsize = 10)
    ax.set_xlabel(f'{featureName} (%)', fontsize = 10)
    ax.set_ylabel(f'{name} in $\Delta$%', fontsize = 10)
    if i == 0:
        ax.set_title(f'Effects of {featureName.lower()} on low flow', fontsize = 11)
    else:
        ax.set_title(f'Effects of {featureName.lower()} on high flow', fontsize = 11)
    ax.tick_params(labelsize=10)
    ax.text(-.1, 1.1, ['a','b'][i], weight = 'bold', ha = 'center', va = 'center', transform = ax.transAxes, fontsize = 12)
fig.tight_layout()
fig.savefig(Path(dir_Qmax7) / f'{type0}_{feature}.png', dpi = 600)
