# Notes — Index (Head)

Navigation for the finance-pipeline-rag build notes. Each step note is a beginner-language retro: what was built, the concepts, the bugs, the decisions. Unfamiliar term? Start with the glossary.

## Concept reference
- **[glossary.md](./glossary.md)** — 19 concepts, ordered so each builds on the last: RAG → corpus → chunk → contextual prefix → embedding → vector → cosine → vector DB → dense retrieval → BM25 → hybrid retrieval → reranking → generation → eval metrics → LangChain → project hygiene → **bi/cross-encoder → recall ceiling → LLM-as-judge**.

## Build-step notes (in order)
| Step | Note | Covers |
|---|---|---|
| 1 | [step-01-repo-init.md](./step-01-repo-init.md) | Repo skeleton, requirements, env/secrets hygiene |
| 2 | [step-02-corpus-prep.md](./step-02-corpus-prep.md) | 693-record synthetic corpus + contextual prefix → JSON |
| 3–4 | [step-04-hybrid-retrieval.md](./step-04-hybrid-retrieval.md) | Chroma dense + BM25 + Voyage rerank-2; **+ Voyage retry/backoff hardening** |
| 5 | [step-05-generation.md](./step-05-generation.md) | Grounded Claude generation + CITED-line citations |
| 6 | [step-06-eval-harness.md](./step-06-eval-harness.md) | DeepEval RAG-triad + 20 golden set; **5 findings incl. judge-dependence** |

## Key findings (worth remembering)
- **Finding 1** — negatives break faithfulness → scored by exact refusal-match, excluded from the triad (4/4).
- **Finding 4** — aggregates break contextual-recall (a computed total isn't verbatim in any chunk) → excluded from recall with its own denominator (8 exact-fact, not 16).
- **Finding 5 (the big one)** — the eval score is *judge-dependent*: a weak 8B judge fails a good system (0.59/0.67/0.70); a strong 70B passes it (0.95/0.91/1.00) on the **identical** 20 cases. A quality gate needs a capable judge → CI uses a Groq-first/local-fallback judge.
- **Latency** — p95 retrieve ≈ 470 ms (two Voyage network round-trips dominate); exceeds the <200 ms target — would need a co-located/local reranker to close.

## The `eval/` harness (code)
- `run_eval.py` — the gated triad run (exits non-zero on fail, so CI can block on it).
- `judge.py` — `ResilientJudge` (Groq-first → local-Ollama fallback; instructor schema coercion).
- `golden_qa.py` — 20 verified goldens (8 exact-fact / 8 aggregate / 4 negative).
- `compare_judges.py` — the controlled weak-vs-strong judge A/B (Finding 5).
- `latency.py` — p95 retrieval latency (Step 8).
- `diagnose.py` — per-question scores + judge reasons.
