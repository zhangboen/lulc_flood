#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=extract
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-evoflood
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --qos=priority
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000

module purge
module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
Rscript calc_GHS_POP_catch_ave.r $1