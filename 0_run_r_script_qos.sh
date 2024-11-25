#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=extract
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-evoflood
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --qos=priority
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000

module purge
module load UDUNITS; module load GDAL; module load R/4.3.2-gfbf-2023a

for i in ../gleam_data/*GLEAM*nc; do
    fname=../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg
    out=${i::-3}.csv
    Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r $i $fname mean $out
done

for i in ../../data/geography/soilgrids/*tif; do
    fname=../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg
    out=../geography/$(basename ${i::-4})_median.csv
    Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r $i $fname median $out
done



df_all = []
for i in ['lulc','text','depth','elevation','slope','aridity_index']:
    if i == 'lulc':
        df = pd.concat([pd.read_csv(s).set_index('ohdb_id') for s in glob.glob(f'../ee_lulc/*csv')], axis = 1)
        df = df.loc[:,df.columns.str.contains('2015')]
    elif i == 'text':
        df1 = pd.read_csv('../geography/clay_0-5cm_mean_1000_median.csv').T
        df2 = pd.read_csv('../geography/sand_0-5cm_mean_1000_median.csv').T
        df3 = pd.read_csv('../geography/silt_0-5cm_mean_1000_median.csv').T
        df1.columns = ['clay']
        df2.columns = ['sand']
        df3.columns = ['silt']
        df = pd.concat([df1,df2,df3], axis = 1)
    elif i == 'depth':
        df = pd.read_csv('../geography/BDTICM_M_1km_ll_median.csv').T
        df.columns = ['soilDep']
    else:
        df = pd.read_csv(f'../geography/{i}_median.csv').T
        df.columns = [i]
    df_all.append(df)
df_all0 = pd.concat(df_all, axis = 1).reset_index().rename(columns={'index':'ohdb_id'})
name = 'australia'
df_meta = pd.read_csv(f'../tutorial/dataset/{name}/station_metadata.csv')
df_all = df_all0.loc[df_all0.ohdb_id.isin(df_meta.ohdb_id.values),:].rename(columns={'index':'ohdb_id'})
df_all.to_csv(f'../tutorial/dataset/{name}/station_attr.txt', index = False)

        