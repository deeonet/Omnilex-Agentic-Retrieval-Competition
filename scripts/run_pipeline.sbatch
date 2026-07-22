#!/bin/bash
# Run the pipeline test on a single A100:4 node.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/run_pipeline.sbatch
#
#SBATCH --job-name=pipeline_test
#SBATCH --account=kisski-aai-ss26
#SBATCH --partition=kisski
#SBATCH --time=06:00:00
#SBATCH -G A100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --constraint=inet
#SBATCH --output=logs/pipeline_test/pipeline_test_%j.out
#SBATCH --error=logs/pipeline_test/pipeline_test_%j.err

set -euo pipefail

cd /mnt/vast-kisski/home/felix.wolter01/u28113/Omnilex-Agentic-Retrieval-Competition

# Model is already cached in ~/.cache/huggingface; avoid hitting the network from a compute node.
export HF_HUB_OFFLINE=1

# Fail fast if the node driver can't serve the installed torch wheel.
./venv/bin/python -c "import torch; assert torch.cuda.is_available(), f'CUDA unavailable (torch {torch.__version__}, cuda {torch.version.cuda})'"

./venv/bin/python -u src/our_pipeline/pipeline_test.py