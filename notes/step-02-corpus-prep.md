# Step 2 — Corpus prep with Contextual Retrieval prefix → JSON (2026-05-27)

Compiled at the close of Step 2. ~50 min cap, finished under cap.

> **Concepts referenced in this note are defined in [`glossary.md`](./glossary.md)** — see the mermaid concept map at the top for the whole pipeline picture. Links inline below jump straight to the right section.

---

## What we built

```
finance-pipeline-rag/
├── data/
│   ├── transactions.json   # 693 synthetic transactions (12 months, 15 categories)
│   └── corpus.json         # 693 RAG chunks (prefix + record + metadata)
├── scripts/
│   ├── generate_corpus.py  # deterministic synthetic transaction generator
│   └── build_corpus.py     # chunk generator (transactions → chunks)
├── src/
│   ├── __init__.py
│   ├── prefix.py           # make_prefix() — Anthropic Contextual Retrieval prefix
│   └── chunker.py          # make_chunk() — prefix + record → chunk
├── tests/
│   ├── __init__.py
│   ├── test_prefix.py      # 14 tests
│   └── test_chunker.py     # 8 tests
└── pyproject.toml          # pytest config (pythonpath = ".")
```

22 tests, all green.

---

## Concepts taught (beginner language)

### 1. What is a "corpus"?

Latin for "body" — it's the AI/NLP word for "**the collection of text the system searches over**." For this project: all the categorized transactions. The retrieval system searches the corpus for chunks relevant to your question.

### 2. What is RAG?

**Retrieval-Augmented Generation.** Two steps:
1. **Retrieval** — find the most relevant pieces of your corpus given a question
2. **Generation** — hand those pieces to an LLM (Claude) and have it answer using them, with citations

Why it exists: LLMs don't know your private data. RAG lets them use your data without retraining the model, and at lower cost than dumping everything in the prompt.

### 3. What is Contextual Retrieval (Anthropic, Sept 2024)?

Standard RAG embeds each chunk as-is. The chunk `"Saigon Noodle House $15.87 2025-06-03"` is short and lacks context. When the user asks "Vietnamese food in summer 2025?", the embedding has to leap from short merchant name to broad concept on its own.

**Fix:** prepend a ~50-token contextual prefix to each chunk BEFORE embedding:

```
"Dining out transaction from June 2025 (summer). Restaurant spend, $15.87. | 2025-06-03 — Saigon Noodle House — $15.87 — dining out"
```

The prefix adds: semantic class ("dining out"), temporal context ("June 2025 (summer)"), and amount — all directly in the embedded text. Anthropic's benchmark: **+49% retrieval-failure-rate improvement** when combined with BM25 + reranking.

The prefix is for the search index ONLY. Claude's generation prompt sees the original record, not the prefix.

### 4. What is BM25?

A 1994 keyword-search ranking algorithm. Smart Ctrl+F. Two extra brains:
- **Rare words count more** — "the" appears in every document → BM25 ignores it; "saigon" appears in 2 docs → BM25 treats it as a strong signal
- **Long documents get a small penalty** — prevents a giant document from winning every search just by having every word

Hybrid retrieval = run BM25 + dense vector search together → combine results → catch both exact-string queries ("did I go to Saigon Noodle House?") and semantic queries ("Vietnamese food in summer 2025").

### 5. What is embedding?

Turning text into a list of numbers (a "vector") such that **similar-meaning text gets similar numbers**. Voyage-3-large produces 1024-number vectors. Closeness between vectors (cosine similarity) = closeness in meaning.

How retrieval uses it:
1. Index time: embed every chunk → store vectors in Chroma
2. Query time: embed the question → ask Chroma "give me the closest chunks to this vector"

### 6. What is LangChain?

A plumbing kit for AI apps. Provides standard interfaces + connectors between Voyage (embedding) + Chroma (vector store) + Claude (generation). Without it you'd write ~100 lines of glue per project. With it, ~20 lines.

---

## Decisions made (logged in `plans/DECISIONS.md`)

**Row 4 revised** (corpus source, 2026-05-27 Step 2.1):
- Original lock: "DynamoDB transactions" — assumed real bank data
- Revised lock: **synthetic generated corpus** — 500-1000 transactions, deterministic seed, 12-month span, 15 categories
- Reason: the predecessor `finance-pipeline` only ingested Plaid sandbox data (~100-300 records of repeated synthetic merchants). Below the 500-corpus floor needed for meaningful contextual-recall@5 metrics. Real BofA `development`-mode OAuth was a known unresolved blocker in finance-pipeline.
- README has the honest framing: "synthetic corpus seeded to exercise RAG retrieval quality."

---

## Gotchas encountered

- **`ModuleNotFoundError: No module named 'src'`** when running `scripts/build_corpus.py` directly. Root cause: `pyproject.toml`'s `pythonpath = ["."]` is only seen by pytest, not by plain `python` script invocations. Fix: prepend `sys.path.insert(0, repo_root)` at the top of any script that imports from `src/`. The `# noqa: E402` comment tells linters "yes I know this import is after a statement, that's intentional."
- **Plaid's `merchant_name` field can be `None`** (transfers, payroll). The chunker uses `transaction['merchant_name'] or transaction['name']` to fall back to `name` when `merchant_name` is null. Caught by a dedicated test case (`test_make_record_text_handles_null_merchant_name`).
- **Loose-but-useful tests** are the right pattern for Step 2.7 — instead of asserting an exact prefix string (which would lock the user to one specific wording), the tests assert that key concepts are present: month-or-season, category-in-plain-English, amount. Gives the user freedom in wording while still enforcing the Contextual Retrieval principles.

---

## Snippets worth remembering

### Deterministic seeded RNG

```python
import random
rng = random.Random(seed)  # NOT random.seed(seed) — Random() gives an isolated generator
rng.randint(1, 28)
rng.choice(merchants)
rng.gauss(mean, std)
```

### Walking month-by-month

```python
current = start.replace(day=1)
while current <= today:
    # ... use current.year, current.month ...
    if current.month == 12:
        current = current.replace(year=current.year + 1, month=1)
    else:
        current = current.replace(month=current.month + 1)
```

### Format amount with 2 decimals

```python
f"${amount:.2f}"   # 15.87 → "$15.87"
                   # 15.0  → "$15.00"
                   # -1175.0 → "$-1175.00"
```

### Longest-prefix match in a dict-of-tuples lookup

```python
for n in range(len(items), 0, -1):
    key = tuple(items[:n])
    if key in MAP:
        return MAP[key]
return fallback
```

Useful when you have a hierarchical category like `["Food", "Restaurants", "Coffee Shop"]` and want to match the most-specific entry first.

---

## What's next (Step 3 preview)

**Step 3 — Chroma vector store + Voyage embeddings persisted** (~1.5 hr).

1. Install full `requirements.txt` (LangChain + Chroma + Voyage clients) into the venv
2. Set up `.env` with VOYAGE_API_KEY and ANTHROPIC_API_KEY
3. Read `data/corpus.json`
4. Loop over chunks → send each `text` to Voyage-3-large → get a 1024-number vector back
5. Insert vector + metadata into Chroma's local persistent store under `chroma_db/`
6. Verify: run a sample query, get nearest 5 chunks back

Track C split for Step 3: I teach how a vector store is laid out + what cosine similarity means → you write the simple embed-and-store loop (it's a great `for chunk in corpus:` exercise) → I scaffold the LangChain Chroma wrapper around it.

**Heads up:** Step 3 will spend ~$1 of your Voyage credit (693 chunks × 1024-dim embedding at $0.18/1M tokens ≈ <$1 total). Same as locked in DECISIONS row 1.
