"""
tests/test_index.py — smoke tests for the persisted Chroma index.

These tests load the EXISTING persisted index (no re-embed) and run cheap assertions.
Run: `pytest tests/test_index.py`. Expected runtime: ~3 sec.

Pre-req: `python src/index.py` has been run at least once to persist chroma_db/.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

PERSIST_DIR = "chroma_db"


@pytest.fixture(scope="module")
def store() -> Chroma:
    """Load the persisted index once per test module."""
    if not Path(PERSIST_DIR).exists():
        pytest.skip(f"{PERSIST_DIR}/ not found — run `python src/index.py` first")
    embeddings = VoyageAIEmbeddings(model="voyage-3-large")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def test_index_has_693_chunks(store: Chroma) -> None:
    """The corpus has 693 chunks; the index should reflect that count."""
    assert store._collection.count() == 693


def test_similarity_search_returns_k_results(store: Chroma) -> None:
    """A k=5 query should return exactly 5 results."""
    results = store.similarity_search("any test query", k=5)
    assert len(results) == 5


def test_results_carry_metadata(store: Chroma) -> None:
    """Each result should carry the metadata we passed at index time."""
    results = store.similarity_search("Vietnamese food", k=1)
    assert len(results) == 1
    metadata = results[0].metadata
    assert "category" in metadata
    assert "date" in metadata
