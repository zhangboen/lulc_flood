#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=xgboost
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
module load ecCodes
module load Anaconda3
# conda deactivate
# conda deactivate
# eval "$(conda shell.bash hook)"
# source activate xgb

time python "$@"