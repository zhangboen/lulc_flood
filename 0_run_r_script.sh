#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=r
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-drift
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
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

# for i in $(seq $1 $2); do
#     inputName=../../data/MSWEPv280/pr_MSWEP_daily_${i}_scale10_I16_all_year.nc
#     shpName=../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg
#     outputName=../data/GRIT_catch_ave_pr_MSWEP_${i}.csv
#     Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r $inputName $shpName mean $outputName
# done



        