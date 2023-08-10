#!/bin/bash
#
#SBATCH --job-name=test_arr
#SBATCH --output=test_arr.out
#SBATCH --error=test_arr.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=2G
#SBATCH --time=0-6:00
##SBATCH --gres=gpu:1
#
#SBATCH --array=1-8
source ~/.bashrc
echo loading minconda
module load /ceph/apps/ubuntu-20/modulefiles/miniconda/4.9.2
echo loading cuda
module load /ceph/apps/ubuntu-20/modulefiles/cuda/12.0
echo initializing environment
conda activate /nfs/nhome/live/beng/.local/miniforge/envs/malis
echo start script
python3 training.py