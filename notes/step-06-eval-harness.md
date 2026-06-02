# Step 6 — The Eval Harness (RAG-triad)

> Status: **machinery DONE + proven on 2 seeds; PAUSED** before the full 20-question set.
> Built 2026-06-01. Pairs with `eval/run_eval.py`, `eval/golden_qa.py`, `eval/judge.py`.

---

## What is a "harness"?

A **harness** is the code that **straps your system into a test rig, feeds it known
inputs, runs it, and reads the gauges** — so you measure quality with *one command*
instead of eyeballing answers.

- **Engine** (the thing being tested) = your RAG pipeline (`retriever` + `generator`)
- **Harness** = the `eval/` folder
- **Run button** = `python -m eval.run_eval`
- **Gauges** = faithfulness / answer-relevancy / contextual-recall → PASS/FAIL vs 0.85

Step 7 will bolt this same harness into CI, so a change that drops a score below 0.85
literally can't merge. **This is the quality gate for the whole project.**

---

## The ONE idea that trips everyone: two different LLMs

There are **two** models doing **different jobs**. Don't collapse them.

```
 question
   │
   ▼
 RETRIEVER (BM25 + dense + rerank) → top-k chunks (retrieved_ids + texts)
   │
   ▼
 GENERATOR = Claude Sonnet (EVAL_MODEL)     ← the "G" in RAG. The thing UNDER TEST.
   writes: answer + cited_ids
   │
   ▼
 LLMTestCase { input, actual_output, expected_output, retrieval_context }
   │
   ▼
 JUDGE = local llama3.1 (Ollama)            ← the GRADER. Only scores. Writes nothing.
   reads the test case → 3 metric scores
```

- **Generator writes the answer. Judge grades it.** The judge never produces `cited_ids`.
- They're **different model families on purpose**: if Claude graded Claude, it would
  score its own family too high (*self-preference bias*). A different local judge =
  honest scores + **$0 per run**.

---

## The RAG triad — 3 metrics = 3 comparisons among 4 things

The 4 things: **question / answer / expected / context**.

| Metric | Compares | Catches |
|---|---|---|
| **Faithfulness** | answer ↔ context | hallucination (did it invent facts not in the chunks?) |
| **Answer-relevancy** | question ↔ answer | did the answer actually address the question? |
| **Contextual-recall** | expected ↔ context | did *retrieval* fetch the chunks the right answer needs? |

An `LLMTestCase` is just a record with 4 named slots. Which variable fills which slot:
- `input` = the question
- `actual_output` = the model's answer
- `expected_output` = your ground-truth (golden) answer
- `retrieval_context` = the **texts** of the chunks the model was shown

---

## Bugs I hit writing `build_test_case()` (so I don't repeat them)

1. **Wrong `LLMTestCase` field names.** Used `question/answer/expected_answer/context`.
   The real fields are `input/actual_output/expected_output/retrieval_context`. → `TypeError`.
2. **The "Recover" step (ids → texts):**
   - `json.load` gives a list of **dicts** → `c.text` errors; use `c['text']`.
   - Compared a chunk's text to a **list of ids** → never matches. Match on **id**, with
     **`in`** (membership), not `==` (equality). *Reflex: matching one value against a
     collection is always `in`, never `==`.*  (Slipped on this twice in one sitting.)
   - `retrieval_context` wants a **list**, not a generator.
3. **The dangerous one — `retriever._corpus_texts`.** That's **all 693 chunks**, not what
   was retrieved. It *doesn't crash* — it silently makes **contextual-recall ≈ 1.000 every
   time** (the needed chunk is always somewhere in 693), so you'd never catch a retrieval
   miss. Lesson: a bug that runs and prints numbers is worse than one that crashes.
   **Tell: a recall of exactly 1.000 means this bug is live.**

✅ Final correct line:
```python
retrieval_context = [c['text'] for c in retriever.corpus if c['id'] in answer.retrieved_ids]
```

## `run_eval()` shape (the rollup)

Build the retriever **once**, one `get_judge()` fed to all 3 metrics. Per golden:
`build_test_case` → each `metric.measure(tc)` (returns the score float) → accumulate.
At the end, **average each metric down to one number** (N×3 scores → 3 means) and return
a `TriadReport`. The report's `.passed` does the threshold check — don't re-implement it.

---

## What the 2-seed run told us (and why low scores ≠ bug)

```
faithfulness 0.500 | answer_relevancy 0.500 | contextual_recall 0.667 → FAIL
```
Per-case diagnostic (always read the per-case detail, never trust the aggregate alone):

- **exact_fact (Costco $70.70):** answer correct, **faithfulness 1.000** → generator is fine.
- **negative (rent 2019 → refusal):** refusal correct, but **faithfulness 0.000**.

### Finding 1 — negative cases break faithfulness (FIX FIRST next session)
Retrieval **always returns top-k chunks**, even for an unanswerable question. The judge
sees 5 chunks present and reads "I don't have that" as *contradicting* them. The triad
metrics aren't built for refusals. **Plan:** score negative goldens by **exact-match on the
REFUSAL string** and **exclude them from faithfulness/recall** — else they drag every run.

### Finding 2 — relevancy dings the grounding preamble
The judge flagged "Based on the provided records…" as an irrelevant statement (0.500 on a
*correct* answer). **Option:** trim the preamble in `src/generator.py` GROUNDED_PROMPT, or
accept that a strict local judge scores relevancy conservatively.

### Finding 3 — n=2 is noise. Real signal needs the 20-case set.

---

## Final results (2026-06-01) — shipped

20-question golden set complete (8 exact_fact / 8 aggregate / 4 negative), all answers verified vs `corpus.json`.

**Run on the LOCAL `llama3.1` (8B) judge:**
```
faithfulness      0.698   (target >0.85)
answer_relevancy  0.644
contextual_recall 0.612   (over 8 exact_fact; aggregates excluded)
trick refusals    4 / 4
→ FAIL
```

### Finding 1 (FIXED) — negatives break faithfulness
Refusals scored by exact-match on the REFUSAL string, excluded from the triad. 4/4 correct.

### Finding 4 (NEW, FIXED) — aggregates break contextual_recall
`ContextualRecallMetric` attributes the *expected* answer to retrieved chunks. Aggregate answers are computed totals (`$195`, `$2,350`, `$136.26`) that never appear verbatim in any chunk → recall is structurally 0.00 even when retrieval succeeded. Same family as Finding 1. Fix: exclude `aggregate` from recall, with its OWN denominator (`recall_n` = 8 exact_fact, NOT 16) — kept in faithfulness + relevancy.

### Finding 2 (partial) — payroll sign
Payroll stored negative (`-1175`). Q5 reframed to a date question (fixed). Q6 residual: the *generator* still answers `$2,350.00` positive vs the negative chunk → minor generator-side issue, not the harness.

### Finding 5 (THE BIG ONE, PROVEN) — the score is judge-dependent; a weak judge makes the gate lie
Controlled A/B (`eval/compare_judges.py`, 2026-06-02): the SAME 20 test cases — built
**once** (one retrieve + generate pass), so question + answer + retrieved context are
identical — scored by two judges. The ONLY variable across the columns is the judge.

| metric | weak local-8B | strong OpenRouter Llama-3.3-70B |
|---|---|---|
| faithfulness | **0.593** ❌ | **0.948** ✅ |
| answer_relevancy | **0.671** ❌ | **0.906** ✅ |
| contextual_recall | **0.700** ❌ | **1.000** ✅ |
| trick refusals | 4/4 | 4/4 |

*(faith/relev over n=16 non-neg; recall over recall_n=8 exact_fact; threshold 0.85.)*

**Same system, same answers — swap only the grader and it flips FAIL → PASS on all three.**
The weak 8B isn't merely lower, it's *noisy/unreliable*: it gave faithfulness **0.0 to a
correct answer** (Q7 Foxtail) and 0.67 to the clean Costco fact (strong judge: 1.0). On
2/20 the strong judge is actually *stricter* (Q8, Q20) — so the honest framing is "the weak
judge mis-scores and on balance lowballs a good system below the gate," not "strong inflates."

**Implication: an 0.85 quality gate is only trustworthy with a capable judge.** That is the
Step-7 judge design: Groq-first / local-Ollama-fallback (`eval/judge.py`), with
OpenRouter-hosted Llama-3.3-70B as the strong cloud option — Groq's free tier caps a full
run at ~13 Q/day and its paid tier was unavailable, so OpenRouter serves the same 70B
uncapped at ~$0.10/run. Refusals (4/4) are judge-independent (string match), so they're
identical across both columns — a nice control.

> Cross-refs: `glossary.md` (bi-encoder vs cross-encoder, grounding), `step-05-generation.md`
> (the generator + CITED: line this harness grades).
