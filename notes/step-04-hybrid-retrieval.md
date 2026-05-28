# Step 4 — Hybrid Retrieval (BM25 + dense + Voyage rerank)

*Apply slot, 2026-05-28. Cap: 2 hr from 3:15 PM. Closed under cap.*
*Beginner-language retrospective. Cross-links to [glossary](./glossary.md).*

---

## What landed

| File | Role |
|---|---|
| `src/retriever.py` | `HybridRetriever` class — wraps Chroma + BM25 + Voyage rerank into a single `retrieve(query, k)` call |
| `tests/test_retriever.py` | 3 smoke tests (k=5 count, result-shape, Step-3-carry-forward Saigon Noodle House check) |
| `scripts/retrieve.py` | CLI: `python scripts/retrieve.py "<query>" [k]` → prints top-k with id + score + snippet |

3/3 tests green. PR: `feat(retrieval): Step 4 — Hybrid retrieval (BM25 + dense + Voyage rerank)`.

---

## The 3-stage pipeline (what hybrid retrieval IS)

```
Query
  │
  ├─► BM25 retriever   ──► top-20 candidates (keyword winners)
  │
  ├─► Chroma dense ────► top-20 candidates (semantic winners)
  │
  ├─► Fuse: union the two lists, dedupe by chunk id  →  ~30 unique
  │
  └─► Voyage rerank-2 reads each (query, chunk) PAIR
         and returns the top-5 by relevance score
```

Wide + cheap retrieval → narrow + sharp ranking. The pyramid shape repeats at every retrieval scale (our 693 → web-scale search).

**Why hybrid:** BM25 is great at exact words and rare tokens (merchant names, IDs); dense is great at meaning (synonyms, related concepts). Each is blind where the other isn't. Running both removes both blind spots.

**Why rerank on top:** dense embedding compares pre-computed vectors via cosine — a coarse "are these in the same neighborhood?" signal. Rerank reads the query + chunk **together** and asks "does this exact chunk actually answer this exact question?" — sharper but expensive, so we only spend it on the ~30 candidates that survive Step 1+2.

Source: [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — measured ~49% reduction in retrieval failures vs dense-only.

---

## Bi-encoder vs cross-encoder (the names you'll see in interviews)

Two ways to use a model when comparing query + chunk:

- **Bi-encoder** (what dense embedding is): model encodes query and chunk **separately**, producing two vectors. Cosine similarity at the end is the only place they "meet." Fast, scales to millions of chunks (vectors are pre-computed once). Coarse signal.

- **Cross-encoder** (what rerank is): model reads `[query] [SEP] [chunk]` in **one pass**. Attention layers let every word in the query attend to every word in the chunk and vice versa. Outputs a single relevance score. Sharp signal, but expensive (one model call per pair) — can't be precomputed.

Quick test: a cross-encoder is **NOT a "better embedding model."** It does not produce a vector. It's a different *kind* of model with a different *kind* of output.

---

## Cost + latency reality

| What | Cost per query (30 candidates) |
|---|---|
| Dollar cost (Voyage rerank-2 ~$0.05/1M tokens) | **~$0.00009** — negligible |
| Latency (API, batched) | **~200-400 ms** |
| Quality (vs rerank-on-all-693) | **Saturates around 50-100 candidates** — going wider doesn't help |

**Latency** is the binding constraint, not dollars. Prod RAG p95 budget is usually <500 ms → caps candidate-set ~30-100.

---

## Code shape (key snippet — `src/retriever.py:retrieve`)

```python
def retrieve(self, query, k=5, candidate_k=20) -> list[dict]:
    dense_docs = self.chroma_store.similarity_search(query, k=candidate_k)
    query_tokens = _tokenize(query)
    bm25_texts = self.bm25.get_top_n(query_tokens, self._corpus_texts, n=candidate_k)
    fused = self._fuse(dense_docs, bm25_texts)
    rerank_result = self.voyage_client.rerank(
        query=query,
        documents=[c["text"] for c in fused],
        model=self.rerank_model,
        top_k=k,
    )
    return [
        {"id": fused[r.index]["id"], "text": fused[r.index]["text"],
         "metadata": fused[r.index]["metadata"], "score": r.relevance_score}
        for r in rerank_result.results
    ]
```

Fusion helper `_fuse()` is dense-first, BM25-fills-gaps, dedupe-by-id. ID round-trip from Chroma was the trickiest detail — `similarity_search()` returns `Document` with `page_content` + `metadata` but NOT the chunk id. We recover the id by matching `page_content` against the original corpus text via a dict built in `__init__` (safe because all 693 chunks are unique).

---

## Lessons logged

1. **Carry-forward query must be Step 3's exact query.** Test 3 used a different query and falsely "failed" — fixed by matching Step 3's `verify_index` default. Patterns.md anchor 2026-05-28.

2. **Scaffold docstrings should teach concepts, not paste runnable code.** First version of `retrieve()` docstring had 5 numbered code blocks identical to the implementation — effectively the answer key. Cleaned to concept-only before commit. Anchored in patterns.md as a Track C principle violation.

---

## Where this feeds next

| Step | What it adds |
|---|---|
| **Step 5 — Generation** | Take the top-5 reranked chunks + the query → Claude prompt with citation tags → cited answer. The chunks `retrieve()` returns ARE the prompt context. |
| **Step 6 — Eval harness** | DeepEval reads (question, retrieved chunks, generated answer) and computes faithfulness + contextual-recall@5 + hallucination-rate. The `retrieve()` output is the middle term of every eval. |
| **Step 8 — Latency** | Measure p95 wall-clock for `retrieve()` end-to-end. Target <200 ms. Voyage rerank dominates latency. |

---

## Glossary cross-refs

- [Hybrid retrieval](./glossary.md#hybrid-retrieval) — the 3-stage pipeline
- [BM25](./glossary.md#bm25) — "smart Ctrl+F"
- [Dense embedding](./glossary.md#dense-embedding) — bi-encoder side
- [Rerank](./glossary.md#rerank) — cross-encoder filter
- [Cosine similarity](./glossary.md#cosine-similarity) — the bi-encoder math
- [Contextual prefix](./glossary.md#contextual-prefix) — what makes the embeddings sharper before retrieval even starts

(Glossary needs `cross-encoder` + `bi-encoder` + `recall ceiling` entries added at Step 11 notes-refinement pass.)
