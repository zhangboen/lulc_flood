#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=r
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-evoflood
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --qos=priority
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000

module purge
module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a

# for i in ../gleam_data/*GLEAM*nc; do
#     fname=../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg
#     out=${i::-3}.csv
#     Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r $i $fname mean $out
# done

# for i in ../../data/geography/soilgrids/*tif; do
#     fname=../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg
#     out=../geography/$(basename ${i::-4})_median.csv
#     Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r $i $fname median $out
# done

time Rscript "$@"



        