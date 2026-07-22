#!/bin/bash
#SBATCH -A kisski-aai-ss26
#SBATCH -p kisski
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -G A100:2
#SBATCH -C inet
#SBATCH --job-name=lightrag-index

set -e

REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python

# Qwen weights (downloaded from HuggingFace into models/) and served name.
QWEN_MODEL_DIR="$REPO/models/Qwen2.5-14B-Instruct"
QWEN_SERVED_NAME="Qwen2.5-14B-Instruct"
EMBED_MODEL_DIR="$REPO/models/multilingual-e5-large"

# Scope + tuning for the laws index build. Override via env, e.g.:
#   WORKDIR=lightrag/laws_valtest_smoke sbatch scripts/run_lightrag_index.sh --limit 100
#
# RESUMABLE: if the job dies or hits walltime, resubmit with the same WORKDIR —
# LightRAG's doc-status store skips already-processed docs (re-inserts are logged
# as "File name already exists" failures; the originals stay processed).
WORKDIR="${WORKDIR:-$REPO/lightrag/laws_valtest}"
CODES="${CODES:-StPO,StGB,BGG,StBOG,ATSG,IVG,BV,ZGB,OR,IPRG,ZPO,SchKG}"
MAX_ASYNC="${MAX_ASYNC:-32}"
RUN_PIPELINE="${RUN_PIPELINE:-0}"   # scoring lives in run_lightrag_score.sh (no GPU needed)

# Pass smoke-test args through:  sbatch run_lightrag_index.sh --limit 50
BUILD_ARGS="$@"

cd "$REPO"

echo "=== LightRAG laws indexing ==="
echo "Host    : $(hostname)"
echo "Date    : $(date)"
echo "Python  : $($PYTHON --version)"
echo "Qwen    : $QWEN_MODEL_DIR"
echo "Embed   : $EMBED_MODEL_DIR"
echo "Workdir : $WORKDIR"
echo "Codes   : $CODES"
echo "MaxAsync: $MAX_ASYNC"
echo "Build args: ${BUILD_ARGS:-<none>}"

# ── Download local models if missing (idempotent) ─────────────────────────────
echo ""
echo "[1/5] Ensuring local models are present..."
"$PYTHON" - <<PYEOF
import os
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HOME", "$REPO/models/.cache/huggingface")

targets = [
    ("Qwen/Qwen2.5-14B-Instruct", "$QWEN_MODEL_DIR"),
    ("intfloat/multilingual-e5-large", "$EMBED_MODEL_DIR"),
]
for repo_id, local_dir in targets:
    local_dir = Path(local_dir)
    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        print(f"  {repo_id}: already present at {local_dir}, skipping")
        continue
    print(f"  {repo_id}: downloading to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.bin", "*.gguf", "*.pth", "*.msgpack", "*.h5"],
    )
    print(f"  {repo_id}: done")
PYEOF

# ── Start Qwen completion server via vLLM (GPU 0) ─────────────────────────────
echo ""
echo "[2/5] Starting vLLM Qwen server on port 8001..."
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$QWEN_MODEL_DIR" \
    --served-model-name "$QWEN_SERVED_NAME" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8001 \
    --host 127.0.0.1 &
LLM_PID=$!

# ── Start embedding server (GPU 1) ────────────────────────────────────────────
echo "[3/5] Starting embedding server on port 8002..."
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$REPO/scripts/embedding_server.py" \
    --model "$EMBED_MODEL_DIR" \
    --port 8002 \
    --host 127.0.0.1 &
EMBED_PID=$!

# Kill both servers when this script exits (success or failure).
trap "echo 'Shutting down servers...'; kill $LLM_PID $EMBED_PID 2>/dev/null; wait" EXIT

# ── Wait for both servers to be ready (max 15 min — vLLM load is slow) ─────────
echo "Waiting for servers to be ready (max 15 min)..."
MAX_WAIT=180   # 180 x 5s = 15 minutes
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

# ── Build the LightRAG laws index (scoped to $CODES) ──────────────────────────
echo ""
echo "[4/5] Building LightRAG laws index..."
"$PYTHON" "$REPO/scripts/build_lightrag_index.py" --laws \
    --codes "$CODES" \
    --max-async "$MAX_ASYNC" \
    --workdir "$WORKDIR" \
    $BUILD_ARGS

# ── Run the val-set pipeline against the freshly built index ──────────────────
if [ "$RUN_PIPELINE" -eq 1 ]; then
    echo ""
    echo "[5/5] Running pipeline_lightrag.py against $WORKDIR (val set)..."
    LIGHTRAG_LAWS_WORKDIR="$WORKDIR" \
    PYTHONPATH="$REPO/src/our_pipeline:$REPO/src" \
    "$PYTHON" "$REPO/src/our_pipeline/pipeline_lightrag.py"
else
    echo ""
    echo "[5/5] Skipping pipeline run (RUN_PIPELINE=0)."
fi

echo ""
echo "=== Indexing complete: $(date) ==="
