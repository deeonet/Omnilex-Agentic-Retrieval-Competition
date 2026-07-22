#!/bin/bash
# Embed a corpus with Qwen3-Embedding-4B into fp16 .npy shards (resumable).
#
# Usage:
#   mkdir -p logs
#   sbatch -J laws   -t 04:00:00              scripts/embed_corpus.sbatch laws
#   sbatch -J courts -t 06:00:00 --array=0-7  scripts/embed_corpus.sbatch courts
#
# Array tasks split the shards automatically (SLURM_ARRAY_TASK_ID/COUNT).
# Rerunning the same command after a failure resumes: completed shards are skipped.
# Afterwards assemble on CPU (no GPU job needed):
#   ./venv/bin/python -m our_pipeline.dense.build_embeddings --corpus laws --assemble
#   ./venv/bin/python -m our_pipeline.dense.build_embeddings --corpus courts --assemble
#
#SBATCH --partition=kisski    # adjust to your GPU partition
#SBATCH -G A100:4
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=logs/embed_%x_%A_%a.out

set -euo pipefail

CORPUS="${1:?usage: sbatch scripts/embed_corpus.sbatch <laws|courts>}"

cd /mnt/vast-kisski/home/felix.wolter01/u28113/Omnilex-Agentic-Retrieval-Competition

# Model is already cached in ~/.cache/huggingface; never hit the network from a compute node.
export HF_HUB_OFFLINE=1

# Fail fast if the node driver can't serve the installed cu130 torch wheel.
# If this assert trips, reinstall torch from the cu126 wheel index:
#   ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
./venv/bin/python -c "import torch; assert torch.cuda.is_available(), f'CUDA unavailable (torch {torch.__version__}, cuda {torch.version.cuda})'"

./venv/bin/python -m our_pipeline.dense.build_embeddings --corpus "$CORPUS"
