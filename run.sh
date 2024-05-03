#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=32G
#SBATCH --account=def-bangliu

module load python/3.10
module load arrow/14.0.0

source venv/bin/activate

python boundless_das.py