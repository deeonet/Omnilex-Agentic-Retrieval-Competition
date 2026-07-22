#!/bin/bash
# Phase 2 (background): build DQA+DInstr -> LoRA each -> linear-merge into HFM.
# Produces models/qwen-law-hfm, which serve_vllm.sh can then serve.
#
#SBATCH -A kisski-aai-ss26
#SBATCH -p kisski                # A100, 48h max
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -G A100:1
#SBATCH --job-name=omnilex-gen-lora
#SBATCH -o slurm-gen-lora-%j.out

set -e
REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python
cd "$REPO"

BASE="${BASE:-$REPO/models/qwen2.5-14b-instruct}"   # set to 7b if 14B won't fit
echo "=== generative LoRA -> HFM ==="; date; hostname; echo "base=$BASE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# LoRA/SFT stack (not in requirements by default)
"$PYTHON" -m pip install -q "trl>=0.9,<0.12" "peft>=0.11" bitsandbytes accelerate

# 0. Base model must be present (download on a login node first: download_models.py --only generator)
if [ ! -d "$BASE" ]; then
    echo "Base model $BASE missing — run: python scripts/download_models.py --only generator (on a login node with internet)"; exit 1
fi

# 1. Build DQA + DInstr
"$PYTHON" scripts/build_generative_finetune_data.py

# 2. LoRA each dataset separately
"$PYTHON" scripts/finetune_generative.py --base "$BASE" \
    --data data/processed/gen_ft/dqa.jsonl    --out models/adapters/qa    --epochs 1
"$PYTHON" scripts/finetune_generative.py --base "$BASE" \
    --data data/processed/gen_ft/dinstr.jsonl --out models/adapters/instr --epochs 1

# 3. Linear-merge into HFM
"$PYTHON" scripts/merge_hfm.py --base "$BASE" \
    --qa models/adapters/qa --instr models/adapters/instr --out models/qwen-law-hfm

echo "=== HFM ready at models/qwen-law-hfm ==="; date
echo "Serve it:  MODEL_DIR=$REPO/models/qwen-law-hfm sbatch scripts/serve_vllm.sh"
