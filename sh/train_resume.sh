#!/bin/bash
#SBATCH -J Branch_ddp
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:2
#SBATCH -t 120:00:00
#SBATCH -o logs/Branch_resume%j.out
#SBATCH -e logs/Branch_resume%j.err

export SWANLAB_MODE="local"
#export SWANLAB_RESUME=must
export TENSORBOARD_LOGDIR="/public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/consistent/CBCT_decouping_diffusion/resume_ts"

export SWANLAB_LOGDIR="/public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/consistent/CBCT_decouping_diffusion/swanlog_resume"
source ~/.bashrc
source /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/.venv/bin/activate

cd /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/consistent

torchrun --standalone --nproc_per_node=2 -m CBCT_decouping_diffusion.train_research \
  --config CBCT_decouping_diffusion/configs/full_to_brach_consistency_finetune.yaml\
  --resume /public_bme2/bme-cuizhm/maxquan/Projects/Replicate/Genai/palette/consistent/CBCT_decouping_diffusion/output/branch/checkpoints/step_325000.pt
