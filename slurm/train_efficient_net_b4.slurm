#!/bin/bash
#SBATCH --job-name=train-efficient_net_b4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_efficient_net_b4/%x_%j.out
#SBATCH --error=logs/train_efficient_net_b4/%x_%j.err

# --- environment ---
module load CUDA/12.1.1
source ~/.bashrc
source /cvmfs/sling.si/modules/el7/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh && conda activate ris

# --- safety ---
set -e
set -o pipefail

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

PROJECT_ROOT=/d/hpc/home/jn16867/ris
cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# --- run ---
python scripts/train.py --config configs/efficient_net_b4.yaml
