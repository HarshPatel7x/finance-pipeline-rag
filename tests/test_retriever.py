"""
tests/test_retriever.py — smoke tests for HybridRetriever.

Loads the EXISTING persisted Chroma index (no re-embed) + the corpus JSON,
runs three cheap checks against retrieve(). Expected runtime: ~3-5 sec
(Voyage rerank API call dominates).

Pre-req:
  - `python src/index.py` has been run at least once (chroma_db/ persisted).
  - `data/corpus.json` exists (Step 2 output).
  - VOYAGE_API_KEY set in .env.

Run: `pytest tests/test_retriever.py`
"""
from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.retriever import HybridRetriever

load_dotenv()

PERSIST_DIR = "chroma_db"
CORPUS_PATH = "data/corpus.json"


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    """Load the retriever once per test module."""
    if not Path(PERSIST_DIR).exists():
        pytest.skip(f"{PERSIST_DIR}/ not found — run `python src/index.py` first")
    if not Path(CORPUS_PATH).exists():
        pytest.skip(f"{CORPUS_PATH} not found — run `python scripts/build_corpus.py` first")
    return HybridRetriever.from_corpus(chroma_dir=PERSIST_DIR, corpus_path=CORPUS_PATH)


def test_retrieve_returns_k_results(retriever: HybridRetriever) -> None:
    """A k=5 query should return exactly 5 results."""
    results = retriever.retrieve("any test query", k=5)
    assert len(results) == 5


def test_results_have_expected_shape(retriever: HybridRetriever) -> None:
    """Each result is {id, text, metadata, score} with non-empty values."""
    results = retriever.retrieve("Vietnamese food", k=3)
    assert len(results) == 3
    for r in results:
        assert set(r.keys()) == {"id", "text", "metadata", "score"}
        assert isinstance(r["id"], str) and r["id"]
        assert isinstance(r["text"], str) and r["text"]
        assert isinstance(r["metadata"], dict)
        assert isinstance(r["score"], float)


def test_semantic_carry_forward(retriever: HybridRetriever) -> None:
    """Step 3 carry-forward — Step 3's exact verify_index query should preserve Saigon hits.

    Step 3's verify_index default query was "Vietnamese food in summer 2025" and it
    returned 5/5 Saigon Noodle House on dense alone. Hybrid + rerank should preserve
    Saigon presence in top-5 (rerank may surface other Vietnamese merchants like Pho
    ahead — that's a *sharpening*, not a regression). If Saigon disappears entirely
    from top-5 for this exact query, the fused→rerank index mapping is wrong
    (most likely cause: using rerank_result.index against the wrong list).
    """
    results = retriever.retrieve("Vietnamese food in summer 2025", k=5)
    texts = [r["text"] for r in results]
    assert any("Saigon" in t for t in texts), (
        f"Expected Saigon Noodle House in top-5; got ids: {[r['id'] for r in results]}"
    )
