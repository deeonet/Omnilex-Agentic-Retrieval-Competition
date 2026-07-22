#!/bin/bash
# Serve the local generator/agent/eval LLM (Qwen2.5) via vLLM's
# OpenAI-compatible server. The pipeline points API_BASE_URL at this endpoint.
#
# 48h window to maximise the allocation. After it starts, note the node
# hostname printed below and set in .env (or run the pipeline on the SAME node):
#     API_BASE_URL=http://<node>:8001/v1
#     API_MODEL=qwen2.5-14b-instruct
#
#SBATCH -A kisski-aai-ss26
#SBATCH -p kisski                # A100, 48h max (matches the account)
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -G A100:1
#SBATCH --job-name=omnilex-vllm
#SBATCH -o slurm-vllm-%j.out

set -e
REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python
cd "$REPO"

MODEL_DIR="${MODEL_DIR:-$REPO/models/qwen2.5-14b-instruct}"
PORT="${PORT:-8001}"
SERVED_NAME="$(basename "$MODEL_DIR")"

echo "=== vLLM serve ==="
echo "Host   : $(hostname)"
echo "Model  : $MODEL_DIR"
echo "Port   : $PORT"
echo "Set in .env ->  API_BASE_URL=http://$(hostname):$PORT/v1   API_MODEL=$SERVED_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""

# gpu-memory-utilization left at default 0.9 since vLLM has the GPU to itself.
# If you co-locate the embedder/reranker on this GPU, drop to ~0.55.
exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
