module load CUDA/12.1.1
source ~/.bashrc && conda activate sam3d-objects
python -c "import torchvision.models as m; m.efficientnet_b0(weights='IMAGENET1K_V1')"


srun --partition=gpu --gres=gpu:1 --time=4:00:00 --mem=16G --pty bash
module load CUDA/12.1.1 && source ~/.bashrc && conda activate sam3d-objects
cd /d/hpc/home/jn16867/ris && python step1_eda.py

srun --partition=gpu --gres=gpu:1 --time=12:00:00 --mem=64G --pty bash -lc "
module load CUDA/12.1.1 &&
source ~/.bashrc &&
conda activate sam3d-objects &&
jupyter lab --ip=0.0.0.0 --no-browser --port=8890"