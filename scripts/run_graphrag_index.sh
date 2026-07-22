#!/bin/bash
#SBATCH -A kisski-aai-ss26
#SBATCH -p kisski
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -G A100:4
#SBATCH -C inet
#SBATCH --job-name=graphrag-index

set -e

REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python
GRAPHRAG=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/graphrag

cd "$REPO"

echo "=== GraphRAG local-model indexing ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $($PYTHON --version)"

# ── Start LLM server (GPU 0) ───────────────────────────────────────────────────
echo "[1/3] Starting llama-cpp completion server on port 8001..."
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m llama_cpp.server \
    --model "$REPO/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
    --n_gpu_layers -1 \
    --n_ctx 8192 \
    --port 8001 \
    --host 127.0.0.1 &
LLM_PID=$!

# ── Start embedding server (GPU 1) ─────────────────────────────────────────────
echo "[2/3] Starting embedding server on port 8002..."
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$REPO/scripts/embedding_server.py" \
    --model "$REPO/models/multilingual-e5-large" \
    --port 8002 \
    --host 127.0.0.1 &
EMBED_PID=$!

# Kill both servers when this script exits (success or failure)
trap "echo 'Shutting down servers...'; kill $LLM_PID $EMBED_PID 2>/dev/null; wait" EXIT

# ── Wait for both servers to be ready ─────────────────────────────────────────
echo "Waiting for servers to be ready (max 5 min)..."
MAX_WAIT=60   # 60 x 5s = 5 minutes
for i in $(seq 1 $MAX_WAIT); do
    sleep 5

    LLM_OK=0
    "$PYTHON" -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:8001/v1/models', timeout=5)
    sys.exit(0 if b'id' in r.read() else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null && LLM_OK=1

    EMB_OK=0
    "$PYTHON" -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=5)
    sys.exit(0 if b'ok' in r.read().lower() else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null && EMB_OK=1

    echo "  [${i}/${MAX_WAIT}] LLM ready=${LLM_OK} Embed ready=${EMB_OK}"

    if [ "$LLM_OK" -eq 1 ] && [ "$EMB_OK" -eq 1 ]; then
        echo "Both servers ready after $((i * 5)) seconds."
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "ERROR: Servers did not start within timeout. Check logs above."
        exit 1
    fi
done

# ── Run GraphRAG indexing ──────────────────────────────────────────────────────
echo ""
echo "[3/3] Running graphrag index..."
"$GRAPHRAG" index --root "$REPO/graphrag" --method fast

echo ""
echo "=== Indexing complete: $(date) ==="
