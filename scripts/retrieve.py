#!/usr/bin/env python3
"""
scripts/retrieve.py — CLI wrapper for hybrid retrieval.

Usage:
    python scripts/retrieve.py "Vietnamese food in summer 2025"
    python scripts/retrieve.py "T-Mobile bill December 2025" 10

Loads the persisted Chroma index + corpus + Voyage client, runs three-stage
hybrid retrieval, prints top-k results. Matches scripts/build_index.py
convention from Step 3 (scripts/ = CLI entry, src/ = library).
"""
from __future__ import annotations

import sys

from src.retriever import verify_retrieve


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/retrieve.py '<query>' [k=5]")
        sys.exit(1)
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    verify_retrieve(query=query, k=k)


if __name__ == "__main__":
    main()
