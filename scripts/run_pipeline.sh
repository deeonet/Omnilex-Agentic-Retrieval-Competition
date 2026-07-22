#!/bin/bash
#SBATCH -A kisski-aai-ss26
#SBATCH -p grete:interactive
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -C inet
#SBATCH --job-name=omnilex-pipeline

set -e

REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python

cd "$REPO"

echo "=== Omnilex retrieval pipeline (deterministic decompose-loop) ==="
echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "Python: $($PYTHON --version)"
echo ""

# ── GPU diagnostics ─────────────────────────────────────────────────────────
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""

# ── Ensure torch sees CUDA (same fix as run_hybrid_index.sh) ─────────────────
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA unavailable — downgrading torch to 2.5.1+cu124 ..."
    "$PYTHON" -m pip install -q "torch==2.5.1+cu124" \
        --index-url https://download.pytorch.org/whl/cu124
fi
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo ""

# ── Run the pipeline (writes output/submission.csv + output/run_logs.jsonl) ──
# pipeline_test.py imports its siblings as top-level modules (from constants import ...),
# so src/our_pipeline must be on PYTHONPATH and cwd must be the repo root.
PYTHONPATH="$REPO/src/our_pipeline" "$PYTHON" "$REPO/src/our_pipeline/pipeline_test.py"

echo ""
echo "=== Pipeline complete: $(date) ==="
