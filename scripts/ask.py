#!/usr/bin/env python3
"""
scripts/ask.py — CLI wrapper for grounded Q&A (the full RAG path).

Usage:
    python scripts/ask.py "How much did I spend on dining in Q1?"
    python scripts/ask.py "T-Mobile bills in 2025" 8

Retrieves the top-k chunks (hybrid BM25 + dense + Voyage rerank), grounds Claude
on them, and prints the answer plus the chunk ids it cited. Matches the
scripts/ = CLI entry, src/ = library convention used by build_index.py,
build_corpus.py, and retrieve.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` imports work when this script is invoked directly from the repo
# root (e.g. `python scripts/ask.py "..."`). Pytest handles this via
# pyproject.toml `pythonpath = ["."]`; plain scripts don't.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import verify_generate  # noqa: E402  (import-after-sys.path-fix)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ask.py '<question>' [k=5]")
        sys.exit(1)
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    verify_generate(query=query, k=k)


if __name__ == "__main__":
    main()
