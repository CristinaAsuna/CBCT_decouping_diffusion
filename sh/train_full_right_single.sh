#!/bin/bash
#SBATCH -J Single_right
#SBATCH -p bme_a10080g
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH -o logs/full_right_single_%j.out
#SBATCH -e logs/full_right_single_%j.err

export SWANLAB_MODE="local"
export SWANLAB_RESUME=must
export SWANLAB_LOGDIR="/public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection/CBCT_decouping_diffusion/swanlog_single_right"
source ~/.bashrc
source /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/.venv/bin/activate

cd /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection

python -m CBCT_decouping_diffusion.train_research \
  --config CBCT_decouping_diffusion/configs/full_to_right_single.yaml \
  --resume /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection/CBCT_decouping_diffusion/output/full_right_single_cosine/checkpoints/last.pt
