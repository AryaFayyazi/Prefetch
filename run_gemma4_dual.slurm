#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:2
#SBATCH --output=output_gemma4_dual.log
#SBATCH --time=05-00:00:00

export PYTORCH_ALLOC_CONF=expandable_segments:True

source /data/nfs/home/afayyazi/miniconda3/etc/profile.d/conda.sh
conda activate cgrpo

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ── Cache paths ───────────────────────────────────────────────────────────────
mkdir -p /models/hf_cache/datasets /models/hf_cache/hub
export HF_HOME=/models/hf_cache
export HF_DATASETS_CACHE=/models/hf_cache/datasets
export HF_HUB_CACHE=/models/hf_cache/hub
export TRANSFORMERS_CACHE=/models/hf_cache/hub
export TRANSFORMERS_OFFLINE=1

# Dataset cache used by datasets.load_dataset(cache_dir=...)
mkdir -p /datasets

cd /data/nfs/home/afayyazi/prefetch

python -u run_experiment_gemma4_dual.py
