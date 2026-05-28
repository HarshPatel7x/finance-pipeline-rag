"""
src/retriever.py — three-stage hybrid retrieval for the RAG corpus.

Pipeline (per DECISIONS.md row 2 + Anthropic Contextual Retrieval blog):
  1. BM25 keyword retrieval (top candidate_k)        — exact-word / rare-word recall
  2. Chroma dense retrieval (top candidate_k)        — semantic recall
  3. Union + dedupe by chunk id                      — fuse the two lists
  4. Voyage rerank-2 over the fused candidates      — cross-encoder relevance
  5. Return final top k results

The bi-encoder side (Chroma + Voyage embed) does the wide cheap retrieval.
The cross-encoder side (Voyage rerank-2) does the narrow sharp ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()  # picks up VOYAGE_API_KEY from .env


def _tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenizer for BM25.

    Cheap enough for our 693-chunk corpus. If you ever swap in a smarter
    tokenizer (e.g. nltk word_tokenize), make sure to retokenize the corpus
    at index time AND every query — they must use the same tokenizer.
    """
    return text.lower().split()


class HybridRetriever:
    """Three-stage hybrid retriever: BM25 + dense → fuse → rerank.

    Wire-up:
        retriever = HybridRetriever.from_corpus()
        results = retriever.retrieve("Vietnamese food in summer 2025", k=5)
        # results = [{"id": str, "text": str, "metadata": dict, "score": float}, ...]
    """

    def __init__(
        self,
        chroma_store: Chroma,
        corpus: list[dict],
        voyage_client: voyageai.Client,
        rerank_model: str = "rerank-2",
    ):
        self.chroma_store = chroma_store
        self.corpus = corpus
        self.voyage_client = voyage_client
        self.rerank_model = rerank_model

        # BM25 index — built once, reused per query
        self._corpus_texts: list[str] = [doc["text"] for doc in corpus]
        self._tokenized_corpus: list[list[str]] = [
            _tokenize(t) for t in self._corpus_texts
        ]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

        # text → (id, metadata) lookup
        # Chroma does NOT round-trip chunk ids in similarity_search() results;
        # we recover the id by matching page_content against the original corpus text.
        # Safe here because our 693 corpus texts are unique per chunk.
        self._by_text: dict[str, dict] = {
            doc["text"]: {"id": doc["id"], "metadata": doc["metadata"]}
            for doc in corpus
        }

    @classmethod
    def from_corpus(
        cls,
        chroma_dir: str = "chroma_db",
        corpus_path: str = "data/corpus.json",
        rerank_model: str = "rerank-2",
    ) -> "HybridRetriever":
        """Convenience constructor — load Chroma + corpus + Voyage client from disk."""
        embedder = VoyageAIEmbeddings(model="voyage-3-large")
        chroma_store = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embedder,
        )
        with open(corpus_path) as f:
            corpus = json.load(f)
        voyage_client = voyageai.Client()
        return cls(chroma_store, corpus, voyage_client, rerank_model)

    # --- fusion helper (scaffold) ---

    def _fuse(self, dense_docs, bm25_texts: list[str]) -> list[dict]:
        """Union the two candidate lists, dedupe by chunk id, preserve order seen.

        Args:
            dense_docs: list[langchain.schema.Document] from Chroma similarity_search.
                Each has .page_content + .metadata.
            bm25_texts: list[str] from BM25Okapi.get_top_n — raw chunk texts.

        Returns:
            list of {id, text, metadata} dicts. Dense-first ordering (dense hits
            tend to be more semantically aligned); BM25 fills any gaps.
        """
        seen_ids: set[str] = set()
        fused: list[dict] = []

        # Dense first
        for doc in dense_docs:
            text = doc.page_content
            entry = self._by_text.get(text)
            if entry is None:
                continue  # text not in our corpus (shouldn't happen — guard)
            if entry["id"] in seen_ids:
                continue
            seen_ids.add(entry["id"])
            fused.append(
                {"id": entry["id"], "text": text, "metadata": entry["metadata"]}
            )

        # BM25 next — fills anything dense missed
        for text in bm25_texts:
            entry = self._by_text.get(text)
            if entry is None:
                continue
            if entry["id"] in seen_ids:
                continue
            seen_ids.add(entry["id"])
            fused.append(
                {"id": entry["id"], "text": text, "metadata": entry["metadata"]}
            )

        return fused

    # --- the core (YOU WRITE THIS) ---

    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:
        """Three-stage hybrid retrieval.

        Step 1 — Dense: ask Chroma for candidate_k chunks whose embeddings are
                 closest (cosine) to the query embedding.
        Step 2 — BM25: ask the BM25 index for candidate_k chunks by keyword
                 overlap with the tokenized query.
        Step 3 — Fuse: union the two lists, dedupe by chunk id (use self._fuse).
        Step 4 — Rerank: hand the fused candidates + query to Voyage rerank-2;
                 it runs a cross-encoder pass per (query, chunk) pair and returns
                 top-k by relevance_score.
        Step 5 — Return list[dict] in rerank order with {id, text, metadata, score}.

        Args:
            query: natural-language question.
            k: final result count (default 5).
            candidate_k: candidates per retriever pre-fusion (default 20 → ~30 unique).

        Returns:
            list of length k: [{"id": str, "text": str, "metadata": dict, "score": float}, ...]
        """
        dense_docs = self.chroma_store.similarity_search(query, k=candidate_k)

        query_tokens = _tokenize(query)
        bm25_texts = self.bm25.get_top_n(
            query_tokens, self._corpus_texts, n=candidate_k,
        )

        fused = self._fuse(dense_docs, bm25_texts)
        
        rerank_result = self.voyage_client.rerank(
            query=query,
            documents=[c["text"] for c in fused],
            model=self.rerank_model,
            top_k=k,
        )

        return [
            {
                "id": fused[r.index]["id"],
                "text": fused[r.index]["text"],
                "metadata": fused[r.index]["metadata"],
                "score": r.relevance_score,
            }
            for r in rerank_result.results
        ]


def verify_retrieve(
    query: str = "Vietnamese food in summer 2025",
    k: int = 5,
) -> None:
    """Print top-k hybrid-retrieval results for a query.

    Loads the persisted Chroma index + corpus + Voyage client, runs hybrid
    retrieval, and prints id + score + snippet per result. No assertions —
    pure eyeball-the-output sanity check (same shape as verify_index from Step 3).
    """
    retriever = HybridRetriever.from_corpus()
    results = retriever.retrieve(query, k=k)
    print(f"\nQuery: {query!r}")
    print(f"Top {k} hybrid-rerank results:\n")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:120] + ("..." if len(r["text"]) > 120 else "")
        print(f"{i}. [{r['id']}] score={r['score']:.4f}")
        print(f"   metadata: {r['metadata']}")
        print(f"   text: {snippet}")
        print()


if __name__ == "__main__":
    verify_retrieve()
