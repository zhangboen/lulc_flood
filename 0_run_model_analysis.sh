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
conda deactivate
conda deactivate
eval "$(conda shell.bash hook)"
source activate xgb

dir0=$1
target=$(echo $(basename $dir0) | cut -d'_' -f2)
for interval in 0 0.1; do
    for type0 in ale pdp; do
        python xgb_flood.py --fname ../data/${target}_final_dataset_seasonal4.pkl --mode onlyUrban --min_interval $interval --purpose $type0 --run_dir $dir0
        python xgb_flood.py --fname ../data/${target}_final_dataset_seasonal4.pkl --mode onlyUrban --min_interval $interval --purpose $type0 --run_dir $dir0 --even
    done
done

