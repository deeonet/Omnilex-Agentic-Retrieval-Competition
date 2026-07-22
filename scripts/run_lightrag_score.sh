#!/bin/bash
# Score the LightRAG pipeline WITHOUT the big-GPU index job.
#
# The index build (run_lightrag_index.sh) needs 2xA100 for vLLM, but query time
# does not: LightRAG's keyword-extraction LLM is pointed at the remote API from
# .env (same endpoint the agent already uses), and the e5 embedding server falls
# back to CPU. So this runs anywhere with internet:
#
#   bash scripts/run_lightrag_score.sh                     # login node, CPU
#   sbatch scripts/run_lightrag_score.sh                   # short interactive-GPU job
#   LIGHTRAG_LAWS_WORKDIR=lightrag/laws_valtest DATASET_MODE=test \
#       bash scripts/run_lightrag_score.sh                 # test predictions
#
#SBATCH -A kisski-aai-ss26
#SBATCH -p grete:interactive
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -C inet
#SBATCH --job-name=lightrag-score
#SBATCH -o slurm-lightrag-score-%j.out

set -e

REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python
EMBED_MODEL_DIR="$REPO/models/multilingual-e5-large"
EMBED_PORT="${EMBED_PORT:-8002}"

# Which LightRAG laws index to query, and which query split to run.
export LIGHTRAG_LAWS_WORKDIR="${LIGHTRAG_LAWS_WORKDIR:-$REPO/lightrag/laws_valtest}"
export DATASET_MODE="${DATASET_MODE:-val}"

cd "$REPO"

# ── Point LightRAG's query-time LLM at the remote API from .env ───────────────
# (keyword extraction only — small prompts, no GPU needed)
set -a; source "$REPO/.env"; set +a
export LIGHTRAG_LLM_BASE_URL="${API_BASE_URL:-https://chat-ai.academiccloud.de/v1}"
export LIGHTRAG_LLM_MODEL="${API_MODEL:-qwen3.5-27b}"
export LIGHTRAG_LLM_API_KEY="$API_KEY"
export LIGHTRAG_EMBED_BASE_URL="http://127.0.0.1:${EMBED_PORT}/v1"

echo "=== LightRAG pipeline scoring (GPU-free query path) ==="
echo "Host       : $(hostname)"
echo "Date       : $(date)"
echo "Workdir    : $LIGHTRAG_LAWS_WORKDIR"
echo "Dataset    : $DATASET_MODE"
echo "Query LLM  : $LIGHTRAG_LLM_MODEL @ $LIGHTRAG_LLM_BASE_URL"
echo "Embeddings : $EMBED_MODEL_DIR @ port $EMBED_PORT (CPU fallback ok)"

# ── Start the embedding server (uses GPU if present, else CPU) ────────────────
"$PYTHON" "$REPO/scripts/embedding_server.py" \
    --model "$EMBED_MODEL_DIR" \
    --port "$EMBED_PORT" \
    --host 127.0.0.1 &
EMBED_PID=$!
trap "echo 'Shutting down embedding server...'; kill $EMBED_PID 2>/dev/null; wait" EXIT

echo "Waiting for embedding server (max 10 min)..."
for i in $(seq 1 120); do
    sleep 5
    if "$PYTHON" -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:${EMBED_PORT}/health', timeout=5)
    sys.exit(0 if b'ok' in r.read().lower() else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "Embedding server ready after $((i * 5)) seconds."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: embedding server did not start."
        exit 1
    fi
done

# ── Run the pipeline (writes output/submission_lightrag.csv + submission.csv) ─
PYTHONPATH="$REPO/src/our_pipeline:$REPO/src" \
    "$PYTHON" "$REPO/src/our_pipeline/pipeline_lightrag.py"

# ── Score (val mode scores inside the pipeline too; this prints per-query) ────
if [ "$DATASET_MODE" = "val" ]; then
    echo ""
    "$PYTHON" "$REPO/scripts/evaluate_submission.py" output/submission_lightrag.csv -v
fi

echo ""
echo "=== Scoring complete: $(date) ==="
