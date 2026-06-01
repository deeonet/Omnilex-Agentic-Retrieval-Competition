
"""Download Kaggle competition data into the repo's local data layout.

This script downloads the Omnilex competition files with kagglehub and copies
the CSV corpus/query files into ``data/raw``:

    data/raw/train.csv
    data/raw/val.csv
    data/raw/test.csv
    data/raw/laws_de.csv
    data/raw/court_considerations.csv
    data/raw/sample_submission.csv

The large files remain git-ignored by the repository's ``.gitignore``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_COMPETITION = "llm-agentic-legal-information-retrieval"
EXPECTED_FILES = (
    "train.csv",
    "val.csv",
    "test.csv",
    "laws_de.csv",
    "court_considerations.csv",
    "sample_submission.csv",
)


def find_downloaded_file(download_path: Path, filename: str) -> Path | None:
    """Find a downloaded competition file by name."""
    direct = download_path / filename
    if direct.exists():
        return direct

    matches = list(download_path.rglob(filename))
    if matches:
        return matches[0]

    return None


def copy_file(src: Path, dest: Path, overwrite: bool) -> str:
    """Copy one file, returning a short status string."""
    existed_before = dest.exists()
    if existed_before and not overwrite:
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return "overwritten" if existed_before else "copied"


def format_size(path: Path) -> str:
    """Return a compact file size string."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Kaggle Omnilex competition files into data/raw."
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug (default: {DEFAULT_COMPETITION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where CSV files should be copied (default: data/raw)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in the output directory.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run kagglehub.login() before downloading. Use this for first-time setup.",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: kagglehub. Install project requirements first:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc

    if args.login:
        kagglehub.login()

    print(f"Downloading Kaggle competition: {args.competition}")
    downloaded = Path(kagglehub.competition_download(args.competition))
    print(f"Kaggle cache path: {downloaded}")
    print(f"Copying CSV files to: {args.output_dir}")
    print()

    missing = []
    for filename in EXPECTED_FILES:
        src = find_downloaded_file(downloaded, filename)
        if src is None:
            missing.append(filename)
            print(f"missing  {filename}")
            continue

        dest = args.output_dir / filename
        existed_before = dest.exists()
        status = copy_file(src, dest, overwrite=args.overwrite)

        if status == "skipped" and existed_before:
            size = format_size(dest)
        else:
            size = format_size(src)

        print(f"{status:<8} {filename:<28} {size}")

    if missing:
        print()
        print("Warning: some expected files were not found in the Kaggle download:")
        for filename in missing:
            print(f"  - {filename}")

    print()
    print("Done. Expected local layout:")
    for filename in EXPECTED_FILES:
        print(f"  {args.output_dir / filename}")


if __name__ == "__main__":
    main()
