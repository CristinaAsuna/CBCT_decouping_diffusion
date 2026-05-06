#!/bin/bash
#SBATCH -J Branch
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH -t 120:00:00
#SBATCH -o logs/Branch_%j.out
#SBATCH -e logs/Branch_%j.err

export SWANLAB_MODE="local"
export SWANLAB_RESUME=must
export SWANLAB_LOGDIR="/public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection/CBCT_decouping_diffusion/swanlog_branch"
source ~/.bashrc
source /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/.venv/bin/activate

cd /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection

python -m CBCT_decouping_diffusion.train_research \
  --config CBCT_decouping_diffusion/configs/full_to_brach.yaml \
  --resume /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/condition_injection/CBCT_decouping_diffusion/output/branch/checkpoints/last.pt
