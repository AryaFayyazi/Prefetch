#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=output_qwen_dual.log
#SBATCH --time=05-00:00:00

export PYTORCH_ALLOC_CONF=expandable_segments:True

source /data/nfs/home/afayyazi/miniconda3/etc/profile.d/conda.sh
conda activate cgrpo

echo "Host         : $(hostname)"
echo "SLURM job    : $SLURM_JOB_ID"
echo "CUDA devices : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

mkdir -p /models/hf_cache/datasets /models/hf_cache/hub /datasets
export HF_HOME=/models/hf_cache
export HF_DATASETS_CACHE=/models/hf_cache/datasets
export HF_HUB_CACHE=/models/hf_cache/hub
export TRANSFORMERS_CACHE=/models/hf_cache/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /data/nfs/home/afayyazi/prefetch
python -u run_experiment_qwen_dual.py
