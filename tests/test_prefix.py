"""Tests for src/prefix.py — verify make_prefix produces sane contextual prefixes.

These are loose-but-useful tests: they check that key terms are present
without locking you to exact wording. Pass these and your prefix is good.
"""
import pytest

from src.prefix import (
    make_prefix,
    season_of,
    month_name,
    category_label,
)


# ---------------------------------------------------------------------------
# Helper unit tests (these should already pass — no work needed from you)
# ---------------------------------------------------------------------------


def test_season_of_winter():
    assert season_of(1) == "winter"
    assert season_of(12) == "winter"
    assert season_of(2) == "winter"


def test_season_of_summer():
    assert season_of(6) == "summer"
    assert season_of(7) == "summer"
    assert season_of(8) == "summer"


def test_month_name():
    assert month_name(1) == "January"
    assert month_name(12) == "December"
    assert month_name(6) == "June"


def test_category_label_known_paths():
    assert category_label(["Food and Drink", "Restaurants"]) == "dining out"
    assert category_label(["Food and Drink", "Restaurants", "Coffee Shop"]) == "coffee shop"
    assert category_label(["Shops", "Supermarkets and Groceries"]) == "groceries"
    assert category_label(["Service", "Subscription"]) == "subscription"
    assert category_label(["Transfer", "Payroll"]) == "income from payroll"


def test_category_label_unknown_falls_back():
    # Unknown nested category should fall back to lowercased last element
    result = category_label(["Unknown", "Mystery", "Thing"])
    assert "thing" in result.lower()


# ---------------------------------------------------------------------------
# make_prefix tests — these are the ones you need to make pass.
# ---------------------------------------------------------------------------


@pytest.fixture
def dining_transaction():
    return {
        "transaction_id": "tx_demo_001",
        "date": "2025-06-03",
        "name": "Saigon Noodle House",
        "amount": 15.87,
        "account_id": "acc_synthetic_001",
        "merchant_name": "Saigon Noodle House",
        "category": ["Food and Drink", "Restaurants"],
        "pending": False,
    }


@pytest.fixture
def subscription_transaction():
    return {
        "transaction_id": "tx_demo_002",
        "date": "2026-01-14",
        "name": "Netflix",
        "amount": 15.49,
        "account_id": "acc_synthetic_001",
        "merchant_name": "Netflix",
        "category": ["Service", "Subscription"],
        "pending": False,
    }


@pytest.fixture
def income_transaction():
    return {
        "transaction_id": "tx_demo_003",
        "date": "2025-09-15",
        "name": "Direct Deposit Payroll",
        "amount": -1175.0,
        "account_id": "acc_synthetic_001",
        "merchant_name": "Direct Deposit Payroll",
        "category": ["Transfer", "Payroll"],
        "pending": False,
    }


def test_make_prefix_returns_string(dining_transaction):
    p = make_prefix(dining_transaction)
    assert isinstance(p, str)
    assert len(p) > 0


def test_make_prefix_dining_includes_month_or_season(dining_transaction):
    p = make_prefix(dining_transaction).lower()
    # Should mention either "june" or "summer" — temporal context rule #2
    assert "june" in p or "summer" in p, f"Prefix missing temporal context. Got: {p!r}"


def test_make_prefix_dining_includes_category(dining_transaction):
    p = make_prefix(dining_transaction).lower()
    # Should mention dining or restaurant — category-in-plain-English rule #3
    assert "dining" in p or "restaurant" in p, f"Prefix missing category. Got: {p!r}"


def test_make_prefix_dining_includes_amount(dining_transaction):
    p = make_prefix(dining_transaction)
    assert "15.87" in p, f"Prefix missing amount. Got: {p!r}"


def test_make_prefix_subscription_includes_winter_or_january(subscription_transaction):
    p = make_prefix(subscription_transaction).lower()
    assert "january" in p or "winter" in p, f"Prefix missing temporal. Got: {p!r}"


def test_make_prefix_subscription_includes_subscription_word(subscription_transaction):
    p = make_prefix(subscription_transaction).lower()
    assert "subscription" in p or "recurring" in p, f"Prefix missing category. Got: {p!r}"


def test_make_prefix_income_includes_payroll_or_income(income_transaction):
    p = make_prefix(income_transaction).lower()
    assert "income" in p or "payroll" in p, f"Prefix missing category for income. Got: {p!r}"


def test_make_prefix_token_count_in_range(dining_transaction):
    p = make_prefix(dining_transaction)
    word_count = len(p.split())
    # Rough proxy for token count — words are usually close enough.
    assert 5 <= word_count <= 80, f"Prefix should be roughly 5-80 words, got {word_count}: {p!r}"


def test_make_prefix_does_not_include_original_record(dining_transaction):
    p = make_prefix(dining_transaction)
    # The prefix should NOT include the original record verbatim — that gets
    # appended by the chunker. Prefix is contextual TAG only.
    # We check by ensuring the prefix doesn't have the merchant_name + amount
    # combined in a way that duplicates the chunk text.
    # (Mentioning either alone is fine; both together as a chunk-like string is not.)
    # Soft check: prefix should be shorter than 600 chars.
    assert len(p) < 600, f"Prefix too long (likely includes the record). Got {len(p)} chars."
