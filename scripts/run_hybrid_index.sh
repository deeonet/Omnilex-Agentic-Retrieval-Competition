#!/bin/bash
#SBATCH -A kisski-aai-ss26
#SBATCH -p kisski
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -G A100:4
#SBATCH -C inet
#SBATCH --job-name=hybrid-index

set -e

REPO=/mnt/vast-kisski/home/oludayo.fayemi/u28157/scratch/Omnilex-Agentic-Retrieval-Competition
PYTHON=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/python
PIP=/mnt/vast-kisski/home/oludayo.fayemi/u28157/miniconda3/envs/mypython/bin/pip

cd "$REPO"

echo "=== Hybrid pipeline index build ==="
echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "Python: $($PYTHON --version)"
echo ""

# ── GPU diagnostics ────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
           --format=csv,noheader || true
echo ""

# ── Fix CUDA/torch mismatch ────────────────────────────────────────────────────
# Cluster driver 570.211.01 supports CUDA ≤ 12.8 (cuDriverGetVersion = 12080).
# The conda env has torch 2.12.0+cu130 which requires CUDA 13.0 — incompatible.
# torch 2.5.1+cu124 only needs driver ≥ 550.54.15, so it works here.
# pip keeps the existing version unless explicitly pinned; use == to force downgrade.
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA unavailable — downgrading torch from cu130 to 2.5.1+cu124 ..."
    "$PIP" install -q \
        "torch==2.5.1+cu124" \
        --index-url https://download.pytorch.org/whl/cu124
    echo "torch reinstalled: $("$PYTHON" -c 'import torch; print(torch.__version__)')"
    echo ""
fi

# ── Verify CUDA ────────────────────────────────────────────────────────────────
"$PYTHON" -c "
import torch, sys
avail = torch.cuda.is_available()
n     = torch.cuda.device_count()
print(f'torch         : {torch.__version__}')
print(f'CUDA available: {avail}   GPU count: {n}')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB')
if not avail:
    print('ERROR: CUDA still unavailable after torch reinstall. Aborting.')
    sys.exit(1)
"

echo ""

# ── Build indices ──────────────────────────────────────────────────────────────
"$PYTHON" "$REPO/scripts/build_hybrid_indices.py"

echo ""
echo "=== Build complete: $(date) ==="
