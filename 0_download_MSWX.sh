#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=python
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-evoflood
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --qos=priority
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000

module purge
module load rclone
cd $DATA/data/MSWX

if [ ! -d "$1" ]; then
    mkdir -p "$1"
fi
cd $1

year=$2
echo $1 $2
shpName=${DATA}/attribution_test/basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg
outName=${DATA}/attribution_test/data_mswx/GRIT_catch_ave_std_${1}_${year}.csv
if [ ! -f $outName ]; then
    # rclone copy GoogleDrive:MSWX_V100/Past/${1}/ . --include "${year}???.nc" --verbose
    module purge
    module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
    cd ${DATA}/attribution_test/lulc_flood
    Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWX_new2.r $1 $shpName $year $outName
fi