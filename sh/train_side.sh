#!/bin/bash
#SBATCH -J Side_cond
#SBATCH -p bme_a10080g
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH -t 120:00:00
#SBATCH -o logs/side_cond_%j.out
#SBATCH -e logs/side_cond_%j.err

export SWANLAB_MODE="local"
export SWANLAB_RESUME=must
export SWANLAB_LOGDIR="/public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection/CBCT_decouping_diffusion/swanlog"
source ~/.bashrc
source /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/.venv/bin/activate

cd /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection

python -m CBCT_decouping_diffusion.train_research \
  --config CBCT_decouping_diffusion/configs/full_to_sidecond.yaml \
  --resume /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection/CBCT_decouping_diffusion/side_cond_cosine/checkpoints/last.pt
