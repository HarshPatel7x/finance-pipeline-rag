"""
  src/index.py — build the Chroma vector index from corpus.json.

  Uses Voyage-3-large (1024-dim) for embeddings. Persists the index to chroma_db/.
  """
from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # picks up VOYAGE_API_KEY from .env
  

def build_index(corpus_path: str = "data/corpus.json", persist_dir: str = "chroma_db") -> Chroma:
    """
    Build and persist a Chroma vector store from our 693-chunk corpus.

    Steps:
    1. Load corpus_path (JSON list of {id, text, metadata}) into a list.
    2. Initialize VoyageAIEmbeddings(model="voyage-3-large").
    3. Call Chroma.from_texts(...) with:
            - texts: list of chunk["text"] values
            - embedding: the Voyage embedder from step 2
            - ids: list of chunk["id"] values
            - metadatas: list of chunk["metadata"] dicts
            - persist_directory: persist_dir
    4. Return the Chroma store.

    Returns:
        A Chroma store with 693 chunks indexed.
    """
        
    with open(corpus_path) as corpus:
        corpus = json.load(corpus)

    embeddings = VoyageAIEmbeddings(model="voyage-3-large")
    store = Chroma.from_texts(
        ids = [c['id'] for c in corpus],
        embedding = embeddings,
        texts = [c['text'] for c in corpus],
        metadatas = [c['metadata'] for c in corpus],
        persist_directory = persist_dir
    )
    return store


def verify_index(
    query: str = "Vietnamese food in summer 2025",
    k: int = 5,
    persist_dir: str = "chroma_db",
) -> None:
    """
    Sanity-check the persisted index by running a sample query and printing top-k results.

    Loads the EXISTING persisted store from persist_dir — does NOT re-embed the corpus.
    Prints each result's metadata + a short text snippet so you can eyeball relevance.

    Use this after `python src/index.py` to confirm the index returns plausibly-relevant
    chunks for a known query. For "Vietnamese food in summer 2025", expect Food and Drink
    category chunks dated June/July/August 2025.
    """
    embeddings = VoyageAIEmbeddings(model="voyage-3-large")
    store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    results = store.similarity_search(query, k=k)
    print(f"\nQuery: {query!r}")
    print(f"Top {k} results:\n")
    for i, doc in enumerate(results, 1):
        snippet = doc.page_content[:120] + ("..." if len(doc.page_content) > 120 else "")
        print(f"{i}. metadata: {doc.metadata}")
        print(f"   text: {snippet}")
        print()


if __name__ == "__main__":
    store = build_index()
    print(f"Built index. Document count: {store._collection.count()}")