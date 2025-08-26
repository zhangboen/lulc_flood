#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=calcarea
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-drift
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000
#SBATCH --qos=priority

domain=$1  
if [ ! -r /data/ouce-evoflood/results/global_river_topology/GRASS/${domain} ]; then
    echo "no such domain ${domain}"
    exit
fi

dir_raw=$(pwd)

# save flow direction raster from GRASS evoflood directory
module purge
module load GRASS
dir0=/data/ouce-drift/cenv1021/data/GRIT/fdr
cd $dir0
mkdir $domain
cd $domain
cp /data/ouce-evoflood/results/global_river_topology/GRASS/${domain}/hand/ . -r
cp /data/ouce-evoflood/results/global_river_topology/GRASS/${domain}/PERMANENT/ . -r
grass hand/ \
  --exec r.out.gdal input=drainage output=../${domain}_fdr_tmp.tif format=GTiff
cd ..
rm ${domain}/ -r

# change flow direction values
module purge
module load PROJ/9.2.0-GCCcore-12.3.0; module load GDAL/3.7.1-foss-2023a-spatialite
gdal_calc.py -A ${domain}_fdr_tmp.tif --outfile="${domain}_fdr.tif" \
  --calc="numpy.where(A==1, 1, 
  numpy.where(A==2, 128, 
  numpy.where(A==3, 64, 
  numpy.where(A==4, 32, 
  numpy.where(A==5, 16, 
  numpy.where(A==6, 8, 
  numpy.where(A==7, 4, 
  numpy.where(A==8, 2, A))))))))"
rm ${domain}_fdr_tmp.tif

# polygonize flow direction raster to get extent
if [ ! -f ${domain}_poly.gpkg ]; then
  gdal_calc.py -A ${domain}_fdr.tif --outfile="${domain}_tmp.tif" --calc="1*(A>=0)"
  gdal_polygonize.py ${domain}_tmp.tif -f "GPKG" ${domain}_poly.gpkg OUTPUT DN -b 1
  rm ${domain}_tmp.tif
fi

# dissolve
ogr2ogr -f GPKG ${domain}_dissolve.gpkg ${domain}_poly.gpkg \
    -dialect sqlite \
    -sql "SELECT ST_Union(geom) AS geom FROM OUTPUT" \
    -nln ${domain}_dissolve

# intersect flow direction with fcc tile index GPKG to intersected file names
if [ ! -f ${domain}_intersect_tmp.gpkg ]; then
  ogr2ogr -f GPKG ${domain}_intersect_tmp.gpkg GRITv06_drainage_area_tile_index_EPSG8857.gpkg GRITv06_drainage_area_tile_index_EPSG8857 \
    -clipsrc ${domain}_dissolve.gpkg -clipsrclayer ${domain}_dissolve \
    -nln a_intersect_b -nlt PROMOTE_TO_MULTI
fi

# write intersected file names to txt
ogr2ogr -f CSV /vsistdout/ ${domain}_intersect_tmp.gpkg \
  -dialect SQLite \
  -sql "SELECT DISTINCT location FROM a_intersect_b WHERE location IS NOT NULL" |
  tail -n +2 > ${domain}_locations_tmp.txt

# merge intersected fcc raster files
if [ ! -f ${dir0}/${domain}_fcc.tif ]; then
    mapfile -t files < ${domain}_locations_tmp.txt
    cd /data/ouce-evoflood/results/global_river_topology/
    gdal_merge.py -o ${dir0}/${domain}_fcc.tif "${files[@]}"
fi
cd $dir0
rm ${domain}_*tmp*

# use python script identify_watershed_boundary.py for each OHDB gauge
upa_name=${dir0}/${domain}_fcc.tif
fdr_name=${dir0}/${domain}_fdr.tif
module purge
module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whitebox
cd $dir_raw
python get_basin_boundary_GRIT_raster.py $fdr_name $upa_name $domain