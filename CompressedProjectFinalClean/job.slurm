#!/bin/bash
#SBATCH --job-name=jen-train     # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # how many instances of your command are run, total, across 
#SBATCH --cpus-per-task=1              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2                   # number of gpus per node
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=12000
#SBATCH --time=100:00:00                # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=nmcneal@andrew.cmu.edu

module purge
module load anaconda3/2020.11
module load cuda-10.0
python train.py
