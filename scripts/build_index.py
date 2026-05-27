"""
scripts/build_index.py — CLI entry to build and persist the Chroma vector index.

Reads data/corpus.json (693 chunks from Step 2), embeds each via Voyage-3-large,
persists to chroma_db/. Run once before querying.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from src.index import ...` when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import build_index  # noqa: E402


if __name__ == "__main__":
    store = build_index()
    print(f"Built index. Document count: {store._collection.count()}")
