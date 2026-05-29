"""Prepare competition data for GraphRAG indexing.

Converts laws_de.csv and court_considerations.csv to text files grouped
by statute (for laws) and by BGE volume+division (for court decisions).
Grouping reduces file count while keeping related provisions together,
which improves GraphRAG entity extraction quality.

Usage:
    python scripts/prepare_graphrag_data.py [--laws-only] [--courts-only]
    python scripts/prepare_graphrag_data.py --max-court-volumes 50

Output:
    graphrag/input/laws/       one .txt file per Swiss statute (e.g. OR.txt)
    graphrag/input/courts/     one .txt file per BGE volume+division
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "llm-agentic-legal-information-retrieval"
OUTPUT_DIR = PROJECT_ROOT / "graphrag" / "input"

LAWS_CSV = DATA_DIR / "laws_de.csv"
COURTS_CSV = DATA_DIR / "court_considerations.csv"


def extract_statute(citation: str) -> str:
    """Extract statute abbreviation from a law citation.

    'Art. 1 OR'           -> 'OR'
    'Art. 104 ZGB'        -> 'ZGB'
    'Art. 10a Abs. 1 USG' -> 'USG'
    'Art. 264m StGB'      -> 'StGB'
    Returns 'UNKNOWN' if no abbreviation can be found.
    """
    # Match statute abbreviation after the article+abs number.
    # Swiss statute abbreviations are 2–6 chars, starting uppercase,
    # possibly with lowercase (e.g. StGB, StPO, SchKG).
    match = re.search(
        r"Art\.?\s+\d+\w*(?:\s+Abs\.?\s*\d+\w*)*\s+([A-Z][A-Za-z]{1,5})\b",
        citation,
    )
    if match:
        return match.group(1)
    # Fallback: last token that looks like a statute abbreviation
    tokens = citation.split()
    for tok in reversed(tokens):
        if re.fullmatch(r"[A-Z][A-Za-z0-9]{1,5}", tok) and tok not in {"E.", "Abs."}:
            return tok
    return "UNKNOWN"


def extract_bge_volume_division(citation: str) -> str:
    """Extract BGE volume and division from a court citation.

    'BGE 139 I 2 E. 1' -> 'BGE_139_I'
    'BGE 121 III 38 E. 2b' -> 'BGE_121_III'
    Returns 'BGE_OTHER' if no match.
    """
    match = re.match(r"BGE\s+(\d+)\s+(I{1,3}V?|VI?|VII|VIII|IX|X{0,3}(?:IX|IV|V?I{0,3}))", citation)
    if match:
        vol = match.group(1)
        div = match.group(2)
        return f"BGE_{vol}_{div}"
    # BGer docket-style: BGer 4A_123/2020
    match2 = re.match(r"BGer\s+(\w+)_(\d+)/(\d+)", citation)
    if match2:
        chamber = match2.group(1)
        year = match2.group(3)
        return f"BGer_{chamber}_{year}"
    return "BGE_OTHER"


def prepare_laws(output_dir: Path) -> None:
    print("Loading laws corpus...")
    df = pd.read_csv(LAWS_CSV)
    print(f"  {len(df):,} law articles")

    groups: dict[str, list[str]] = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping laws by statute"):
        citation = str(row.get("citation", ""))
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        statute = extract_statute(citation)

        block = f"## {citation}\n"
        if title and title != "nan":
            block += f"### {title}\n"
        block += f"{text}\n"
        groups[statute].append(block)

    laws_dir = output_dir / "laws"
    laws_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Writing {len(groups)} statute files...")
    for statute, blocks in tqdm(groups.items(), desc="Writing statute files"):
        out_path = laws_dir / f"{statute}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"# Swiss Federal Law: {statute}\n\n")
            f.write("\n---\n\n".join(blocks))

    total_files = len(groups)
    total_articles = sum(len(b) for b in groups.values())
    print(f"  Done: {total_files} statute files, {total_articles:,} articles")


def prepare_courts(
    output_dir: Path,
    max_volumes: int | None = None,
    max_excerpts: int | None = None,
) -> None:
    """Stream court decisions CSV to avoid loading the full 2.4 GB file into RAM."""
    # GraphRAG/Arrow hard limit is 2 GB per array. Keep well under that.
    SIZE_WARN_BYTES = 1_200_000_000  # 1.2 GB warning threshold

    print("Streaming court decisions corpus (2.4 M rows — not loaded all at once)...")

    groups: dict[str, list[str]] = defaultdict(list)
    total_rows = 0
    chunk_size = 50_000

    with tqdm(desc="Reading court excerpts", unit=" rows") as pbar:
        for chunk in pd.read_csv(COURTS_CSV, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                citation = str(row.get("citation", ""))
                text = str(row.get("text", ""))
                volume_key = extract_bge_volume_division(citation)
                groups[volume_key].append(f"## {citation}\n{text}\n")
                total_rows += 1
                pbar.update(1)

                if max_excerpts is not None and total_rows >= max_excerpts:
                    print(f"  Reached --max-court-excerpts={max_excerpts}, stopping early.")
                    break
            else:
                continue  # inner loop didn't break — keep reading chunks
            break          # inner loop broke — stop outer loop too

    print(f"  Read {total_rows:,} excerpts across {len(groups)} BGE volumes")

    courts_dir = output_dir / "courts"
    courts_dir.mkdir(parents=True, exist_ok=True)

    volume_keys = sorted(groups.keys())
    if max_volumes is not None:
        print(f"  Limiting to {max_volumes} BGE volumes (out of {len(volume_keys)})")
        volume_keys = volume_keys[:max_volumes]

    print(f"  Writing {len(volume_keys)} BGE volume files...")
    total_bytes = 0
    for key in tqdm(volume_keys, desc="Writing BGE volume files"):
        blocks = groups[key]
        out_path = courts_dir / f"{key}.txt"
        content = f"# Swiss Federal Court Decisions: {key.replace('_', ' ')}\n\n" + "\n---\n\n".join(blocks)
        out_path.write_text(content, encoding="utf-8")
        total_bytes += len(content.encode())

    written = sum(len(groups[k]) for k in volume_keys)
    size_mb = total_bytes / 1_000_000
    print(f"  Done: {len(volume_keys)} BGE volume files, {written:,} excerpts, {size_mb:.0f} MB")

    if total_bytes > SIZE_WARN_BYTES:
        print()
        print(f"  WARNING: court output is {size_mb:.0f} MB — close to GraphRAG's 2 GB limit.")
        print("  Consider using --max-court-excerpts 200000 to stay safely under it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laws-only", action="store_true", help="Only prepare law articles")
    parser.add_argument("--courts-only", action="store_true", help="Only prepare court decisions")
    parser.add_argument(
        "--max-court-volumes",
        type=int,
        default=None,
        metavar="N",
        help="Limit court output to N BGE volume files (useful for a quick test run)",
    )
    parser.add_argument(
        "--max-court-excerpts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop reading courts after N excerpts total. "
            "Avoids the 2 GB GraphRAG/Arrow array limit. "
            "Recommended: 200000 (≈ 200 MB of court text)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Error: data directory not found at {DATA_DIR}", file=sys.stderr)
        print("Run download_kaggle_data.py first.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.courts_only:
        if not LAWS_CSV.exists():
            print(f"Warning: {LAWS_CSV} not found, skipping laws", file=sys.stderr)
        else:
            prepare_laws(args.output_dir)

    if not args.laws_only:
        if not COURTS_CSV.exists():
            print(f"Warning: {COURTS_CSV} not found, skipping courts", file=sys.stderr)
        else:
            prepare_courts(
                args.output_dir,
                max_volumes=args.max_court_volumes,
                max_excerpts=args.max_court_excerpts,
            )

    # Final size check — warn before GraphRAG hits the 2 GB Arrow limit
    total_input_bytes = sum(
        f.stat().st_size for f in args.output_dir.rglob("*.txt")
    )
    total_mb = total_input_bytes / 1_000_000
    print(f"\nTotal input size: {total_mb:.0f} MB")
    if total_input_bytes > 1_500_000_000:
        print("WARNING: Input exceeds 1.5 GB. GraphRAG will fail with the Arrow 2 GB limit.")
        print("Run with --laws-only, or add --max-court-excerpts 200000 for courts.")
    else:
        print("Size OK — within GraphRAG's 2 GB limit.")

    print(f"\nInput files ready in: {args.output_dir}")
    print("Next steps:")
    print("  1. Ensure graphrag/.env has MISTRAL_API_KEY set")
    print("  2. Run:  graphrag index --root ./graphrag")
    print("  3. Use GraphRAGSearchEngine in your retrieval pipeline")


if __name__ == "__main__":
    main()
