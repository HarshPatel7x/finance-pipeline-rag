"""Tests for src/chunker.py — verify chunks combine prefix + record correctly."""
import json
from pathlib import Path

import pytest

from src.chunker import build_corpus, make_chunk, make_record_text


@pytest.fixture
def sample_txn():
    return {
        "transaction_id": "tx_0001",
        "date": "2025-06-03",
        "name": "Saigon Noodle House",
        "amount": 15.87,
        "account_id": "acc_synthetic_001",
        "merchant_name": "Saigon Noodle House",
        "category": ["Food and Drink", "Restaurants"],
        "pending": False,
    }


@pytest.fixture
def income_txn():
    # Income has merchant_name == None in the original Plaid schema. Some
    # transactions in our synthetic corpus have None too — chunker must
    # handle that gracefully (fall back to `name`).
    return {
        "transaction_id": "tx_0002",
        "date": "2025-09-15",
        "name": "Direct Deposit Payroll",
        "amount": -1175.0,
        "account_id": "acc_synthetic_001",
        "merchant_name": None,
        "category": ["Transfer", "Payroll"],
        "pending": False,
    }


def test_make_record_text_shape(sample_txn):
    r = make_record_text(sample_txn)
    assert "2025-06-03" in r
    assert "Saigon Noodle House" in r
    assert "15.87" in r
    assert "dining" in r.lower()


def test_make_record_text_handles_null_merchant_name(income_txn):
    # Fallback path: merchant_name is None → use `name`
    r = make_record_text(income_txn)
    assert "Direct Deposit Payroll" in r
    assert "1175.00" in r  # formatted with .2f


def test_make_chunk_has_required_keys(sample_txn):
    c = make_chunk(sample_txn)
    assert set(c.keys()) == {"id", "text", "metadata"}


def test_make_chunk_id_matches_transaction(sample_txn):
    c = make_chunk(sample_txn)
    assert c["id"] == "tx_0001"


def test_make_chunk_text_contains_both_halves(sample_txn):
    c = make_chunk(sample_txn)
    # Should contain prefix-side signal (month, category) AND record-side signal (date, merchant)
    assert "June" in c["text"] or "summer" in c["text"].lower()  # prefix
    assert "2025-06-03" in c["text"]  # record half
    assert "Saigon Noodle House" in c["text"]
    assert "|" in c["text"]  # the separator between halves


def test_make_chunk_metadata_roundtrips(sample_txn):
    c = make_chunk(sample_txn)
    m = c["metadata"]
    assert m["date"] == "2025-06-03"
    assert m["merchant_name"] == "Saigon Noodle House"
    assert m["amount"] == 15.87
    assert m["category"] == ["Food and Drink", "Restaurants"]


def test_build_corpus_count_matches_input():
    txns = [
        {
            "transaction_id": f"tx_{i:04d}",
            "date": "2025-06-01",
            "name": "Walmart",
            "amount": 25.0,
            "account_id": "acc_synthetic_001",
            "merchant_name": "Walmart",
            "category": ["Shops", "Supermarkets and Groceries"],
            "pending": False,
        }
        for i in range(1, 6)
    ]
    corpus = build_corpus(txns)
    assert len(corpus) == 5
    # IDs in order
    assert [c["id"] for c in corpus] == [f"tx_{i:04d}" for i in range(1, 6)]


def test_build_corpus_against_real_transactions_file():
    """End-to-end: chunk the real generated corpus, verify shape + sample."""
    data_path = Path(__file__).parent.parent / "data" / "transactions.json"
    if not data_path.exists():
        pytest.skip("data/transactions.json not generated yet — run scripts/generate_corpus.py first")

    transactions = json.loads(data_path.read_text())
    corpus = build_corpus(transactions)

    assert len(corpus) == len(transactions)
    # First chunk has all required fields, non-empty text
    first = corpus[0]
    assert first["id"].startswith("tx_")
    assert len(first["text"]) > 30
    assert "|" in first["text"]
    assert "metadata" in first
