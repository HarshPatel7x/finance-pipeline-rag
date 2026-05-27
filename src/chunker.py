"""
chunker.py — Turn raw transactions into RAG chunks.

Each transaction becomes exactly one chunk:
    chunk.text = make_prefix(transaction) + " | " + make_record_text(transaction)

The prefix half adds semantic + temporal context for retrieval (per Anthropic
Contextual Retrieval, Sept 2024). The record-text half is the original
transaction in a single readable line — what Claude actually quotes when
it answers and cites.

Metadata (date, merchant, amount, category) is kept alongside the text so
the retriever can also filter on structured fields when needed.
"""
from __future__ import annotations

from src.prefix import make_prefix, category_label


def make_record_text(transaction: dict) -> str:
    """One-line readable representation of a single transaction.

    Format: 'YYYY-MM-DD — Merchant — $amount — plain-english-category'
    """
    cat = category_label(transaction["category"])
    return (
        f"{transaction['date']} — "
        f"{transaction['merchant_name'] or transaction['name']} — "
        f"${transaction['amount']:.2f} — {cat}"
    )


def make_chunk(transaction: dict) -> dict:
    """Build a single RAG chunk from one transaction."""
    prefix = make_prefix(transaction)
    record = make_record_text(transaction)
    return {
        "id": transaction["transaction_id"],
        "text": f"{prefix} | {record}",
        "metadata": {
            "date":          transaction["date"],
            "merchant_name": transaction["merchant_name"] or transaction["name"],
            "amount":        transaction["amount"],
            "category":      transaction["category"],
        },
    }


def build_corpus(transactions: list[dict]) -> list[dict]:
    """Chunk every transaction in the input list. Returns the chunked corpus."""
    return [make_chunk(t) for t in transactions]
