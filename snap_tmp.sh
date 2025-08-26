#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=calcarea
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-drift
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000
#SBATCH --qos=priority

domain=$1  
if [ ! -r /data/ouce-drift/cenv1021/data/GRIT/fdr/${domain} ]; then
    mkdir /data/ouce-drift/cenv1021/data/GRIT/fdr/$domain
fi

ohdb_name=/data/ouce-drift/cenv1021/data/OHDB/OHDB_v0.2.3/OHDB_metadata/OHDB_metadata_correct_coord.gpkg
domain_name=/data/ouce-drift/cenv1021/data/GRIT/fdr/domains60.gpkg
gauge_name=/data/ouce-drift/cenv1021/data/GRIT/fdr/${domain}/gauge.gpkg

# get target gauges
module purge
module load Anaconda3
python - <<PY
from pyogrio import read_dataframe,write_dataframe
import pandas as pd
import geopandas as gpd
import glob,os,re
# clip gauges within domain
df = read_dataframe("$ohdb_name")
gdf0 = read_dataframe("$domain_name")
gdf0 = gdf0.loc[gdf0.domain=="$domain",:]
df = gpd.sjoin(df, gdf0).reset_index(drop=True).reset_index(drop=True)
# limit to selected gagues
df_tmp = pd.read_csv('/data/ouce-drift/cenv1021/attribution_test/data/OHDB_metadata_at_least_80_complete_seasonal_records_during_1982_2023.csv')
df = df.loc[df.ohdb_id.isin(df_tmp.ohdb_id.values),:].reset_index(drop=True)
# remove those gauges already processed
gdf_GRIT1 = read_dataframe('../basin_boundary/MIX_good.gpkg')
gdf_GRIT1 = gdf_GRIT1.loc[~gdf_GRIT1.reach_id.isna(),:]
df = df.loc[~df.ohdb_id.isin(gdf_GRIT1.ohdb_id.values),:]
tmp = [os.path.basename(s) for s in glob.glob('../../data/GRIT/full_catchment/raw/GRITv06*nearest.gpkg')]
tmp = [re.search(r'OHDB_\d+',a).group(0) for a in tmp]
df = df.loc[~df.ohdb_id.isin(tmp),:]
print(df.shape)
write_dataframe(df, '../../data/GRIT/fdr/${domain}/gauge.gpkg')
dir0 = '/data/ouce-drift/cenv1021/data/GRIT/fdr/'
gdf1 = read_dataframe(dir0 + 'GRITv06_drainage_area_tile_index_EPSG8857.gpkg')
gdf1 = gdf1.loc[gdf1.intersects(gdf0.geometry.values[0]),:]
gdf1.location.to_csv(dir0 + '${domain}/locations_tmp.txt', index = False, header = False)
PY

cd /data/ouce-drift/cenv1021/data/GRIT/fdr/$domain
module purge
module load GRASS
# Grab "Feature Count: N" from ogrinfo summary
COUNT=$(ogrinfo -ro -so "gauge.gpkg" "gauge" 2>/dev/null \
        | awk -F': ' '/^Feature Count:/ {print $2; exit}')

COUNT=${COUNT:-0}
if [[ "$COUNT" -eq 0 ]]; then
  echo "No features in gauge.gpkg → exiting."
  exit 1   # use 0 if you want a graceful stop
fi

# merge intersected fcc raster files
if [ ! -f fcc.tif ]; then
    mapfile -t files < locations_tmp.txt
    cd /data/ouce-evoflood/results/global_river_topology/
    gdal_merge.py -o /data/ouce-drift/cenv1021/data/GRIT/fdr/${domain}/fcc.tif "${files[@]}"
fi
rm *tmp*

# save flow direction raster from GRASS evoflood directory
dir0=/data/ouce-drift/cenv1021/data/GRIT/fdr/$domain
cp /data/ouce-evoflood/results/global_river_topology/GRASS/${domain}/hand/ . -r
cp /data/ouce-evoflood/results/global_river_topology/GRASS/${domain}/PERMANENT/ . -r
cp /data/ouce-evoflood/results/global_river_topology/GRASS/${domain}/flow_accumulation/ . -r
mv flow_accumulation/* PERMANENT/

export GRASS_ADDON_BASE="$HOME/src/grass_addons"
grass hand/ --exec bash -euo pipefail -c '
  g.region raster=drainage -a
  v.import -o input=gauge.gpkg layer=gauge output=gauges --overwrite
  r.import -o input=fcc.tif output=accum --overwrite
  r.stream.snap input=gauges accumulation=accum output=gauges_snap radius=2 --overwrite
  r.stream.basins direction=drainage points=gauges_snap basins=basins_rast --overwrite
  r.to.vect input=basins_rast output=basins_poly type=area --overwrite
  v.out.ogr input=basins_poly output=basins_poly.gpkg \
          format=GPKG layer=basins_poly --overwrite
'

# add ohdb_id to basins
module purge
module load Anaconda3
python - <<PY
from pyogrio import read_dataframe,write_dataframe
import pandas as pd
import geopandas as gpd
import glob,os,re
gdf_basin = read_dataframe('basins_poly.gpkg')
gdf_basin = gdf_basin.dissolve(by='value')
gdf_gauge = read_dataframe('gauge.gpkg')
gdf_gauge['fid'] = gdf_gauge.index.values + 1
gdf_basin = gdf_basin.merge(gdf_gauge[['ohdb_id','fid']], left_on = 'value', right_on = 'fid')
write_dataframe(gdf_basin, '../${domain}_basins_poly.gpkg')
PY

# del intermediate folders
cd ..
rm ${domain}/ -r