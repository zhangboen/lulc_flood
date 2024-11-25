#! /bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=python
#SBATCH --mail-type=NONE
#SBATCH --partition=short
#SBATCH --account=ouce-drift
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=36
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000

module purge
module load ecCodes
module load Anaconda3

python "$@"

