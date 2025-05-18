#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=r
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-evoflood
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000
#SBATCH --qos=priority

outDir='../../data/MSWX/Snow/'
shpName='../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg'

module purge
module load ecCodes
module load Anaconda3

year=$1
echo $year
python calcSnow.py $year

module purge
module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
Rscript calc_GRIT_catch_ave_MSWX_new.r snowmelt $shpName $year
Rscript calc_GRIT_catch_ave_MSWX_new.r snowfall $shpName $year
rm ${outDir}snow*_${year}.nc
