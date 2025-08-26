#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=r
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-drift
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000

outDir='../../data/MSWX/Snow/'
shpName='../basin_boundary/MIX_good.gpkg'
year=$1
echo $year

module purge
module load rclone
cd $DATA/data/MSWX
for name in P Tmax Tmin; do
    cd $name
    if [ ! -f Daily/${year}365.nc ]; then
        rclone copy GoogleDrive:MSWX_V100/Past/${name}/ . --include "${year}???.nc" --verbose --drive-shared-with-me
    fi
    cd ..
done

module purge
module load ecCodes
module load Anaconda3
cd /data/ouce-drift/cenv1021/attribution_test/lulc_flood
if [ ! -f ${outDir}snowmelt_${year}.nc ]; then
    python src/calcSnow.py $year
fi
rm $DATA/data/MSWX/P/Daily/${year}???.nc
rm $DATA/data/MSWX/Tmax/Daily/${year}???.nc
rm $DATA/data/MSWX/Tmin/Daily/${year}???.nc

module purge
module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
Rscript calc_GRIT_catch_ave_snow.r ${outDir}snowmelt_${year}.nc $shpName ../data_mswx/MIX_catch_ave_snowmelt_${year}.csv
Rscript calc_GRIT_catch_ave_snow.r ${outDir}snowfall_${year}.nc $shpName ../data_mswx/MIX_catch_ave_snowfall_${year}.csv
rm ${outDir}snow*_${year}.nc
