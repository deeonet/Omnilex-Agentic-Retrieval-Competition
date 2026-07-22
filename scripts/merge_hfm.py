"""Linear-merge the DQA + DInstr LoRA adapters into the HFM (LQ-RAG Algorithm 2).

Loads the base model, adds both adapters, linearly combines them (equal weight),
bakes the result into full weights, and saves a standalone model dir that vLLM
can serve directly (point serve_vllm.sh MODEL_DIR at it).

Runs on CPU by default (bf16, needs ~2x model RAM) so it fits any GPU size.

Usage:
    python scripts/merge_hfm.py \
        --base models/qwen2.5-14b-instruct \
        --qa   models/adapters/qa \
        --instr models/adapters/instr \
        --out  models/qwen-law-hfm
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "models" / "qwen2.5-14b-instruct"))
    ap.add_argument("--qa", default=str(REPO / "models" / "adapters" / "qa"))
    ap.add_argument("--instr", default=str(REPO / "models" / "adapters" / "instr"))
    ap.add_argument("--out", default=str(REPO / "models" / "qwen-law-hfm"))
    ap.add_argument("--weights", type=float, nargs=2, default=[0.5, 0.5])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading base (CPU, bf16)…")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, args.qa, adapter_name="qa")
    model.load_adapter(args.instr, adapter_name="instr")

    print(f"Linear-merging adapters with weights {args.weights}…")
    model.add_weighted_adapter(
        adapters=["qa", "instr"],
        weights=args.weights,
        adapter_name="hfm",
        combination_type="linear",
    )
    model.set_adapter("hfm")
    merged = model.merge_and_unload()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"Saved merged HFM to {args.out}")


if __name__ == "__main__":
    main()
