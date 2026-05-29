# Step 5 — Generation (grounded answer + citations)

*Apply slot, 2026-05-29. Cap: 1.5 hr. The "G" in RAG.*
*Beginner-language retrospective. Cross-links to [glossary](./glossary.md).*

---

## What landed

| File | Role |
|---|---|
| `src/generator.py` | `generate(query, k, model, retriever)` → `GenerationResult`. Retrieve top-k → build grounded prompt → call Claude → parse cited chunk ids. |
| `tests/test_generator.py` | 3 tests: `GenerationResult` shape, `GROUNDED_PROMPT` contract, + an integration smoke (real retrieval + Claude, asserts cited ids ⊆ retrieved ids). |

3/3 tests green (integration smoke makes a real Claude + Chroma call; retrieval returned 5 real ids `tx_0504…`).

---

## What generation IS (retrieval vs generation)

Step 4 (the retriever) gives you the **right pages** — top-k ranked chunks. It does NOT answer the question; it hands you 5 raw transaction records.

Step 5 (generation) **reads those chunks and writes the answer** — and tracks which chunk ids it used, so every claim is auditable.

```
question ─► retrieve top-k chunks (Step 4)
              │
              ├─► build GROUNDED prompt: paste chunks as context, each tagged [id]
              │
              ├─► Claude reads context + question → answer (ONLY from context)
              │
              └─► parse the CITED: line → which chunk ids were used
                     → GenerationResult(answer, cited_ids, retrieved_ids, model)
```

That's what **RAG** = **R**etrieval-**A**ugmented **G**eneration means: Step 4 = R, Step 5 = G, and "augmented" is feeding the retrieved chunks into the prompt so Claude answers from *our* records, not its training memory.

**Why grounding (the "answer ONLY from context" rule in the prompt):** "how much did I spend on dining" has no answer in Claude's head — it's private data. Without the chunks Claude would hallucinate a number. With them, it can only use what's in front of it. That rule is what drives the Step 6 targets: faithfulness > 0.85, hallucination < 5%.

---

## Two ways to call Claude: raw SDK vs LangChain (the big concept this session)

Both hit the *same* Claude API. They differ in **abstraction layer**.

| | raw `anthropic` SDK | `langchain_anthropic.ChatAnthropic` |
|---|---|---|
| import | `from anthropic import Anthropic` | `from langchain_anthropic import ChatAnthropic` |
| call | `client.messages.create(model, messages, max_tokens)` | `chat.invoke(prompt)` |
| get text | `resp.content[0].text` | `resp.content` (off an `AIMessage`) |
| best for | standalone "just call Claude" scripts | pipelines built on LangChain |

- **`claude_categorize.py` (Step 2)** uses the raw SDK — it's a one-shot categorize task, no pipeline to plug into, so the thin direct client is simplest.
- **`generator.py` (Step 5)** uses `ChatAnthropic` — it lives inside a LangChain RAG pipeline (the retriever already uses `langchain_chroma`/`langchain_voyageai`; DECISIONS locked "LangChain orchestration"; Step 6 eval is LangChain-friendly). Same ecosystem = parts compose uniformly.

**The "standard interface" ChatAnthropic gives you** = LangChain's `Runnable` / `BaseChatModel` contract, the same on every provider (Anthropic/OpenAI/Google/Ollama):
- same methods: `.invoke()` / `.batch()` / `.stream()` / `.ainvoke()`
- same input: a string **or** a list of messages (`SystemMessage`/`HumanMessage`/`AIMessage`) — **we used the string form** (passed the formatted `GROUNDED_PROMPT` straight to `.invoke()`); never built message objects
- same output: an **`AIMessage`**, text on **`.content`** — this is the *only* one of these message classes our code touched, and only as the **return** of `.invoke()`, not an input we built

Payoff: swap `ChatAnthropic` → `ChatOpenAI` and `.invoke(prompt)` + `resp.content` + the `prompt | model | parser` pipe don't change. The raw SDKs have totally different shapes (`resp.content[0].text` vs `resp.choices[0].message.content`); LangChain normalizes them.

Trade-off named honestly: this repo now uses *two* Claude clients (raw in categorize, LangChain here). Defensible (different roles), but a mild inconsistency a team might later standardize.

---

## The 3 bugs (the real lessons)

1. **Bare `except` hid a real error → silent empty answers.** A `try: … except Exception: return GenerationResult("", [], …)` swallowed the actual exception and returned a blank answer. Code "ran clean" and *looked* done, but produced garbage. **Self-attested "done" ≠ verified — only a real run caught it.** Fix: don't catch-all-and-fake-success; let the error surface (or narrow the except).

2. **`AIMessage` is not a string.** `chat.invoke(prompt)` returns an `AIMessage` object; the text is `resp.content`. Calling `.rsplit()` directly on the message threw `AttributeError` — which bug #1 then buried. Also `chat(PROMPT)` (calling the model like a function) is deprecated → use `chat.invoke(PROMPT)`. This is the LangChain output-shape from the table above.

3. **The `else` branch discarded the injected retriever.** `if retriever is None: build; else: retriever = retriever.from_corpus()` rebuilt from disk even when a caller passed one in — defeating the parameter (which exists so tests can inject a pre-built retriever and skip the rebuild). Fix: drop the `else`; if it's not None, use it as-is.

Process note: the citation-parse line was IDE-autogenerated by autocomplete, not written from blank — and it was *itself buggy* (the AIMessage/`.rsplit` issue). Autocomplete planted a hidden bug rather than saving work. Read + understood it after, but the from-blank rep didn't fire.

---

## Code shape (key idea — `src/generator.py:generate`)

```python
if retriever is None:
    retriever = HybridRetriever.from_corpus()     # call on the CLASS (always exists)
chunks = retriever.retrieve(query=query, k=k)

context = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)   # tag each chunk with its id
prompt = GROUNDED_PROMPT.format(context=context, question=query)

resp = ChatAnthropic(model=model).invoke(prompt)  # AIMessage
answer, cited_line = resp.content.rsplit("CITED:", 1)  # split off the contract line
cited_ids = [] if cited_line.strip() == "none" else [c.strip() for c in cited_line.split(",")]
```

The **prompt defines the output contract**: it tells Claude to end with `CITED: <id>, <id>`, which makes citation extraction a simple string split — no fancy tooling.

---

## Lessons logged

1. **Run it for real before saying "done."** A bare `except` that returns a fake-success value will mask a crash and ship silently-wrong output. Verify with a real run, not a vibe.
2. **Know your client's return type.** Raw SDK → `resp.content[0].text`; LangChain → `AIMessage.content`. Mixing them up is a common first-time error.
3. **Optional params exist for a reason — don't override them.** The `retriever` arg is for injection; rebuilding anyway wastes the point.
4. **Autocomplete ≠ a learning rep,** and it can plant bugs that *look* right. Read everything it produces.

---

## Where this feeds next

| Step | What it adds |
|---|---|
| **Step 6 — Eval harness** | DeepEval reads (question, retrieved chunks, generated answer) → faithfulness + contextual-recall@5 + hallucination-rate. `generate()`'s answer + `retrieved_ids` are the inputs. Switch `model=EVAL_MODEL` (Sonnet) for eval runs. |
| **Robustness TODO** | `rsplit("CITED:", 1)` will fail to unpack if the model omits the `CITED:` line; cited-id parse doesn't drop empty strings. Harden before eval. |

---

## Glossary cross-refs

- [Hybrid retrieval](./glossary.md#hybrid-retrieval) — what feeds the context here
- [Rerank](./glossary.md#rerank) — produces the top-k chunks generation grounds on

(Glossary needs new entries at Step 11 refinement pass: `RAG`, `grounding`, `faithfulness`, `Runnable / BaseChatModel interface`, `AIMessage`.)
