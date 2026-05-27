# Step 1 — Repo init + skeleton (2026-05-27)

Compiled at the close of Step 1. ~45 min from `mkdir` to first push.

> **Conceptual references for any term you see here** (RAG, corpus, embedding, LangChain, BM25, etc.) live in [`glossary.md`](./glossary.md) — read top-down for the full pipeline picture, or jump to any section.

---

## What we built

A skeleton repo ready for the RAG build:

```
finance-pipeline-rag/
├── .gitignore           # tells git "skip these files" — Python helpers + secrets
├── .env.example         # public template of API keys (no secret values)
├── README.md            # the front-door page — what / how / metrics / stack / cite
├── requirements.txt     # recipe list of Python libraries to install
└── notes/
    └── step-01-repo-init.md   # this file
```

---

## Concepts taught (beginner language)

### 1. `.gitignore` is git's "skip this" list

Git tracks files. Without `.gitignore`, it tracks EVERYTHING in the folder — including:

- **`venv/`**: Python's library box (other people's code, not yours)
- **`__pycache__/`**: auto-generated speed-up files (Python rebuilds these every run)
- **`.env`**: your API keys (Voyage + Anthropic) — **NEVER commit this**
- **`chroma_db/`**: the local vector database files (rebuilt from your CSVs)

**Rule:** write the names in `.gitignore` BEFORE you `git add -A`. If you ever accidentally commit a `.env`, you must rotate the keys — once it's in GitHub history, the keys are public forever.

### 2. `requirements.txt` is your project's recipe list

Someone (you, a recruiter cloning your repo, the GitHub CI bot) runs `pip install -r requirements.txt` → all packages install. Without this file, they'd have to guess which libraries to install.

**Loose pins vs exact pins:**
- `langchain==0.3.7` (exact) — reproducible to the byte, but you must update for every security patch
- `langchain>=0.3,<0.4` (loose) — security patches work automatically, no major-version surprises

For a skeleton: loose is fine. Tighten to exact after metrics are measured (Step 8).

### 3. README is the recruiter's 30-second impression

Four things a portfolio README has to answer instantly:

1. **What is this?** — 1 sentence at the top
2. **How does it work?** — a mermaid diagram beats prose every time
3. **What numbers?** — metrics table, honest about target vs measured
4. **What stack + why?** — shows it's not vibes-coded

**Honesty rule:** every metric in the table says **TARGET** with "Measured at Step X" until Step 6/8 lands the real numbers. Then we update the table with actual values + a one-line note if a number missed.

### 4. `.env.example` is a public template

The real `.env` is `.gitignore`-d (has your secret values). The `.env.example` version is checked in (variable names only, no secrets) so anyone cloning the repo knows what to fill in.

---

## Decisions locked (all 6 P2 decisions per `plans/DECISIONS.md §P2-decisions`)

1. **Embedding:** Voyage-3-large (paid, ~$1 for full corpus — Anthropic-stack name-drop preserved)
2. **Generation:** Claude Haiku (dev iteration) / Sonnet (eval runs) — covered by Max 5x subscription
3. **Reranker:** Voyage rerank-2 (same vendor as embed → one API integration)
4. **Corpus:** DynamoDB transactions only (from the previous finance-pipeline build)
5. **Chunks:** 512 tokens + 50-token contextual prefix (per Anthropic Contextual Retrieval pattern)
6. **Golden eval set:** 20 Q&A — Claude drafts the questions; answers computed from real transaction data via pandas/SQL (not Claude-fabricated, to keep faithfulness metric meaningful); user spot-checks 3-4 to validate
7. **Eval judge:** Ollama Llama-3.1-8B local (free, $0/run) — replaces paid DeepEval LLM judge

**Net cost target: < $5 total build.** Voyage embed + rerank are paid (~$1 each at this corpus size); everything else is free via Max 5x or local Ollama.

---

## Gotchas encountered

- **Nested git repo inside parent git repo.** `~/Desktop/Harsh/` is itself a git repo (workspace-level). Without adding `finance-pipeline-rag/` to the parent `.gitignore`, every Python helper file inside the new repo would show as untracked from the parent's `git status` — polluting checkpoint discipline. Pattern was already established (parent `.gitignore` already ignored `Bearing Fruit/`, `harsh-brain/`, `IAMBOSS/`, `system-map/`, `finance-pipeline/`, `vyasa/`). Added `finance-pipeline-rag/` alphabetical between `finance-pipeline/` and `vyasa/`.
- **GitHub Free + private repo = no server-side branch protection.** Per `~/Desktop/Harsh/harsh-brain/memories/patterns.md` 2026-05-19, the GitHub API call to set branch protection returns 403 for private repos on the Free tier ("Upgrade to GitHub Pro or make this repository public"). Workaround: install a local pre-push hook at `.git/hooks/pre-push` that blocks direct pushes to `main`. The hook is local-only — vanishes on re-clone — but it does its job for the original working copy.
- **First commit must verify source-file completeness** (per patterns.md 2026-05-19 — system-map repo previously shipped with `git init + .gitignore + CI only`, source never staged). Before `git add -A`, run `git status` and visually confirm: README, requirements.txt, .env.example, .gitignore, notes/ all expected. If `venv/` or `__pycache__/` appear in the staging area, `.gitignore` failed and we need to fix it before commit.

---

## Snippets worth remembering

### Standard Python `.gitignore` floor

```gitignore
venv/
__pycache__/
.env
*.pyc
.DS_Store
.pytest_cache/
```

These 6 lines cover ~95% of accidental commits in Python projects.

### Loose-pin syntax

```
langchain>=0.3,<0.4
```

"Any version at or above 0.3, strictly less than 0.4" — gets minor patches, blocks major-version breaks.

### Nested repo + parent `.gitignore`

```gitignore
# In the parent repo's .gitignore:
finance-pipeline-rag/
```

One line — parent git ignores the entire sub-folder. The sub-folder runs its own independent git.

### Local pre-push hook (private-repo branch-protection workaround)

```bash
# .git/hooks/pre-push
#!/bin/sh
protected_branch='main'
current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

if [ "$current_branch" = "$protected_branch" ]; then
    echo "Direct push to '$protected_branch' is blocked. Use a feature branch + PR."
    exit 1
fi
```

`chmod +x .git/hooks/pre-push` makes it active. Blocks `git push origin main`; PRs still work via the GitHub UI.

---

## What's next (Step 2 preview)

**Step 2 — Corpus prep with Contextual Retrieval prefix → JSON** (~2 hr cap). We will:

1. Load the DynamoDB-transactions CSV via pandas
2. Chunk the transactions into 512-token windows
3. Add a 50-token contextual prefix to each chunk (the Anthropic pattern: "this chunk is from {date range}, contains transactions categorized as {top categories}; …")
4. Write the chunked + prefixed corpus to a single JSON file, ready for Step 3's Voyage embedding pass

Track C approach continues: I teach the contextual-prefix concept first → you write the prefix-generation function → I scaffold the JSON I/O glue around it.
