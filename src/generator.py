"""
src/generator.py — grounded generation with citations (Step 5, the "G" in RAG).

Flow:
  retrieve top-k chunks (src.retriever.HybridRetriever)
    -> build a GROUNDED prompt (each chunk tagged with its id)
    -> call Claude
    -> return the answer PLUS the chunk ids the model said it used.

Why grounding matters: the prompt instructs "answer ONLY from context; if it's
not there, say so." That instruction is what drives the Step 6 eval targets
(faithfulness > 0.85, hallucination < 5%). Citations make every claim auditable
back to a source chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.retriever import HybridRetriever

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env

# Models per DECISIONS.md §P2-decisions: Haiku for dev, Sonnet for eval runs.
DEV_MODEL = "claude-haiku-4-5"
EVAL_MODEL = "claude-sonnet-4-6"


# --- the grounded prompt (scaffold — defines the OUTPUT CONTRACT you parse) ---
#
# Two placeholders, filled in by generate():
#   {context}  -> the retrieved chunks, each tagged with its id (you format this)
#   {question} -> the user's natural-language query
#
# The trailing CITED: line is the contract that makes citation extraction simple:
# Claude lists the ids it used; you split that line off to populate cited_ids.
GROUNDED_PROMPT = """You are a precise personal-finance records assistant.
Answer the question using ONLY the context chunks below. Each chunk is tagged \
with an id in square brackets, like [chunk_42].

Rules:
- Use only facts stated in the context. Never use outside knowledge or invent \
numbers, dates, or merchants.
- If the answer is not contained in the context, reply exactly:
  "I don't have that in the provided records."
- After your answer, on its own final line, list the ids of the chunks you \
actually relied on, in this exact format:
  CITED: <id>, <id>, ...
  If you relied on none, write: CITED: none

Context:
{context}

Question: {question}
"""


# --- citation data structure (scaffold) ---
@dataclass
class GenerationResult:
    """One grounded answer plus its provenance.

    Attributes:
        answer:        the answer text, with the trailing CITED: line stripped off.
        cited_ids:     chunk ids the model said it relied on (parsed from CITED:).
        retrieved_ids: every chunk id fed into the context (for eval / debugging —
                       lets Step 6 check cited_ids is a subset of what we showed).
        model:         the model id that produced this answer.
    """

    answer: str
    cited_ids: list[str]
    retrieved_ids: list[str] = field(default_factory=list)
    model: str = DEV_MODEL


# --- the core (YOU WRITE THIS) ---
def generate(
    query: str,
    k: int = 5,
    model: str = DEV_MODEL,
    retriever: HybridRetriever | None = None,
) -> GenerationResult:
    """Answer a question grounded in the top-k retrieved chunks, with citations.

    Your three moves (the concept; figure out the HOW yourself — that's the rep):
      - Ground     — get the top-k chunks, then shape them into context the model
                     can cite back to. (What "cite back to" requires: re-read the
                     GROUNDED_PROMPT contract and the GenerationResult fields.)
      - Ask        — turn that context + the question into the prompt, call the model.
      - Attribute  — recover which chunk ids the answer actually used, split from
                     the answer text, and return both.

    Args:
        query:     natural-language question.
        k:         number of chunks to retrieve and ground on (default 5).
        model:     model id (DEV_MODEL for dev, EVAL_MODEL for eval runs).
        retriever: optional pre-built HybridRetriever (inject one in tests to
                   avoid rebuilding the index every call).

    Returns:
        GenerationResult(answer, cited_ids, retrieved_ids, model).
    """
    if retriever is None:
        retriever = HybridRetriever.from_corpus()  

    list_of_top_k_chunks = retriever.retrieve(query=query, k=k)
    
    chat = ChatAnthropic(model=model)

    PROMPT = GROUNDED_PROMPT.format(
        context="\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in list_of_top_k_chunks),
        question=query,
    )

    retrieved_ids = [chunk['id'] for chunk in list_of_top_k_chunks]

    response = chat.invoke(PROMPT).content
    answer, cited_line = response.rsplit("CITED:", 1)
    cited_ids = [c.strip() for c in cited_line.split(",")] if cited_line.strip() != "none" else []
    
    return GenerationResult(
        answer=answer.strip(),
        cited_ids=cited_ids,
        retrieved_ids=retrieved_ids,
        model=model,
    )
    

def verify_generate(
    query: str = "How much did I spend on dining in 2026?",
    k: int = 5,
) -> None:
    """Print one grounded answer + its citations for eyeballing (like verify_retrieve).

    No assertions — pure sanity check. Confirm the answer is grounded in the
    retrieved chunks and does not invent numbers absent from the context.
    """
    result = generate(query, k=k)
    print(f"\nQuery: {query!r}  (model={result.model})")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nCited ids: {result.cited_ids}")
    print(f"Retrieved ids: {result.retrieved_ids}")
    stray = [c for c in result.cited_ids if c not in result.retrieved_ids]
    if stray:
        print(f"\n!! WARNING: cited ids not in retrieved context: {stray}")


if __name__ == "__main__":
    verify_generate()
