#!/bin/bash
# LQ-RAG fast-score path: fine-tune the embedder, then score baseline vs LQ-RAG
# on a held-out train slice (queries the embedder never saw) → presentable Macro-F1.
#
# LLM calls (decompose / eval agent) use the API in .env (deadline-safe). Retrieval
# runs fully local: fine-tuned embedder + local BGE reranker. Switch API_BASE_URL to
# the vLLM job (serve_vllm.sh) later for the fully-local story.
#
#SBATCH -A kisski-aai-ss26
#SBATCH -p kisski                # A100, 48h max (matches the account)
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -G A100:1
#SBATCH --job-name=omnilex-lqrag-score
#SBATCH -o slurm-lqrag-score-%j.out

set -e
REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python
cd "$REPO"
export PYTHONPATH="$REPO/src/our_pipeline:$REPO/src"

HELDOUT="$REPO/data/processed/embed_ft/heldout_queries.csv"
RUN_BASELINE="${RUN_BASELINE:-1}"

echo "=== LQ-RAG fine-tune + score ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Ensure CUDA torch (same guard as run_pipeline.sh)
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    "$PYTHON" -m pip install -q "torch==2.5.1+cu124" --index-url https://download.pytorch.org/whl/cu124
fi

# 0. Models (embedder base + reranker) — download if absent
[ -d "$REPO/models/multilingual-e5-large" ] || "$PYTHON" scripts/download_models.py --only embedding
[ -d "$REPO/models/bge-reranker-v2-m3" ]   || "$PYTHON" scripts/download_models.py --only reranker

# 1. Build fine-tune data (+ held-out scoring CSV)
echo "--- build embedding fine-tune data ---"
"$PYTHON" scripts/build_embedding_finetune_data.py --eval-frac 0.1

# 2. Fine-tune the embedder (MNRL)
echo "--- fine-tune embedder ---"
"$PYTHON" scripts/finetune_embedding.py --epochs 3 --batch-size 64

score_run () {
    local tag="$1"; shift
    echo ""; echo "===== RUN: $tag ====="
    env "$@" QUERY_FILE="$HELDOUT" "$PYTHON" "$REPO/src/our_pipeline/pipeline_test.py"
    cp "$REPO/output/submission.csv" "$REPO/output/submission_${tag}.csv"
    echo "saved output/submission_${tag}.csv"
}

# 3a. Baseline: off-the-shelf embedder, no eval agent.
# Force dense rebuild so the index matches THIS embedder (shared dir with 3b).
if [ "$RUN_BASELINE" = "1" ]; then
    score_run baseline \
        EMBEDDING_MODEL="$REPO/models/multilingual-e5-large" \
        RERANK_BACKEND=local ENABLE_EVAL_AGENT=false FORCE_REBUILD_DENSE=true
fi

# 3b. LQ-RAG: fine-tuned embedder + local rerank + eval agent (rebuild dense once)
score_run lqrag \
    EMBEDDING_MODEL="$REPO/models/e5-law-embed" \
    RERANK_BACKEND=local ENABLE_EVAL_AGENT=true FORCE_REBUILD_DENSE=true

echo ""; echo "=== done ==="; date
echo "Scores are printed above under each RUN's 'Scores:' block."
