#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=32G
#SBATCH --account=rrg-bangliu

module load python/3.10
module load arrow/14.0.0

export WANDB_DIR="/home/qinjerem/scratch/IFT6168/wandb/runs"

source venv/bin/activate

python code/main.py --direct "$1"