"""
build_corpus.py — Load transactions.json, chunk every record, write corpus.json.

This is the bridge between Step 2 (corpus prep) and Step 3 (Chroma embedding).
Step 3's embedder will read corpus.json and push each chunk's `text` field
through Voyage-3-large.

Usage:
    python scripts/build_corpus.py                 # default in/out paths
    python scripts/build_corpus.py --in data/transactions.json --out data/corpus.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src.*` imports work when this script is invoked directly from the
# repo root (e.g. `python scripts/build_corpus.py`). Pytest already handles
# this via pyproject.toml `pythonpath = ["."]`; plain scripts don't.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import build_corpus  # noqa: E402  (import-after-sys.path-fix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", type=Path,
                        default=Path("data/transactions.json"),
                        help="Input transactions JSON (default: data/transactions.json)")
    parser.add_argument("--out", dest="output_path", type=Path,
                        default=Path("data/corpus.json"),
                        help="Output chunked corpus JSON (default: data/corpus.json)")
    args = parser.parse_args()

    transactions = json.loads(args.input_path.read_text())
    corpus = build_corpus(transactions)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(corpus, indent=2))

    # Sanity print
    print(f"Built {len(corpus)} chunks from {len(transactions)} transactions")
    print(f"  Input:  {args.input_path}")
    print(f"  Output: {args.output_path}")
    print()
    print("Sample chunks (first + middle + last):")
    for chunk in (corpus[0], corpus[len(corpus) // 2], corpus[-1]):
        print(f"  [{chunk['id']}] {chunk['text']}")


if __name__ == "__main__":
    main()
