#!/bin/bash
set -e # Exit immediately if a command fails

# 1. Define paths
# Use a local temp directory for the active environment (Fast & Safe)
export JOB_VENV="/tmp/uv_venv_$SLURM_JOB_ID" 
# Point the cache to your PVC so you don't download files every time
export UV_CACHE_DIR="/scratch/home/tatli/.uv_cache"

echo ">>> Creating ephemeral environment at $JOB_VENV"
uv venv "$JOB_VENV"

# 2. Activate it
source "$JOB_VENV/bin/activate"

# 3. Handle the requirements
echo ">>> Installing parent requirements..."
uv pip install -r requirements.txt

# 5. Run your training
echo ">>> Starting Training..."
# $@ passes whatever arguments you send to this script (e.g., train.py)
exec "$@"
