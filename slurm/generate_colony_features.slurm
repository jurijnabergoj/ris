#!/bin/bash
#SBATCH --job-name=generate-colony-features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/generate-colony-features/%x_%j.out
#SBATCH --error=logs/generate-colony-features/%x_%j.err

# --- environment ---
module load CUDA/12.1.1
source ~/.bashrc
conda activate sam3d-objects

# --- safety ---
set -e
set -o pipefail

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

PROJECT_ROOT=/d/hpc/home/jn16867/ris
SAM_PATH=/d/hpc/home/jn16867/cso/ext/GECO2
CSO_PATH=/d/hpc/home/jn16867/cso/

cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTHONPATH=$SAM_PATH:$PYTHONPATH
export PYTHONPATH=$CSO_PATH:$PYTHONPATH

# --- run ---
# Training set

python scripts/generate_colony_features.py --dataset data/ris2026-krog1-ucni --device cuda
# python scripts/generate_colony_features.py --dataset data/ris2026-krog1-testni --device cuda

