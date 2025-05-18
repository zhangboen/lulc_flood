#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=xgboost
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-drift
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=26
#SBATCH --mem-per-cpu=8000

module purge
module load ecCodes
module load Anaconda3
conda deactivate
conda deactivate
source activate xgb

time python "$@"