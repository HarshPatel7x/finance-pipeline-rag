"""
prefix.py — Contextual Retrieval prefix generator (per Anthropic Sept 2024 pattern).

Each transaction gets a ~50-token contextual prefix that gets PREPENDED to the
chunk text BEFORE embedding. The prefix surfaces the semantic class, temporal
context, and category in plain English — so retrieval queries like "Vietnamese
food in summer 2025" can land on the right chunks even when the merchant name
doesn't contain those exact words.

The prefix does NOT go into the generation prompt — only the search index.
"""
from __future__ import annotations

from datetime import date


# ---------------------------------------------------------------------------
# Helpers (safe to use inside make_prefix). Add more here if useful.
# ---------------------------------------------------------------------------


def season_of(month: int) -> str:
    """Return a human-friendly season label for a month (1-12)."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def month_name(month: int) -> str:
    """Return the full English name of a month (1=January, 12=December)."""
    return date(2000, month, 1).strftime("%B")


# Map Plaid's nested category arrays to a single plain-English label.
# Add new mappings as the corpus grows.
CATEGORY_LABELS = {
    ("Food and Drink", "Restaurants"):                   "dining out",
    ("Food and Drink", "Restaurants", "Coffee Shop"):    "coffee shop",
    ("Food and Drink", "Restaurants", "Fast Food"):      "fast food",
    ("Shops", "Supermarkets and Groceries"):             "groceries",
    ("Shops", "Digital Purchase"):                       "online shopping",
    ("Travel", "Gas Stations"):                          "gas",
    ("Travel", "Taxi"):                                  "ride-share or taxi",
    ("Service", "Subscription"):                         "subscription",
    ("Service", "Utilities"):                            "utility bill",
    ("Service", "Telecommunication Services"):           "phone or internet bill",
    ("Service", "Financial"):                            "bank fee",
    ("Healthcare", "Pharmacies"):                        "pharmacy",
    ("Recreation", "Arts and Entertainment"):            "entertainment",
    ("Recreation", "Gyms and Fitness Centers"):          "gym or fitness",
    ("Transfer", "Payroll"):                             "income from payroll",
    ("Transfer", "Credit"):                              "refund or credit",
    ("Transfer", "Internal Account Transfer"):           "internal transfer",
    ("Payment", "Credit Card"):                          "credit card payment",
}


def category_label(plaid_category: list[str]) -> str:
    """Convert a Plaid category array to a plain-English label.

    Walks the longest-prefix match in CATEGORY_LABELS, falling back to the
    last (most specific) Plaid label lowercased.
    """
    for n in range(len(plaid_category), 0, -1):
        key = tuple(plaid_category[:n])
        if key in CATEGORY_LABELS:
            return CATEGORY_LABELS[key]
    return plaid_category[-1].lower() if plaid_category else "uncategorized"


# ---------------------------------------------------------------------------
# YOUR TURN — write the prefix template.
# ---------------------------------------------------------------------------


def make_prefix(transaction: dict) -> str:
    """Return a ~50-token contextual prefix for a single transaction.

    A good prefix surfaces (priority order):
      1. The SEMANTIC CLASS of the merchant (e.g., "Vietnamese restaurant"
         beats "Saigon Noodle House" for cuisine-type queries).
      2. TEMPORAL CONTEXT in human words ("June 2025 (early summer)" beats
         the raw ISO date for season/month queries).
      3. CATEGORY in plain English (use the `category_label(...)` helper).

    Args:
        transaction: a dict with keys: date (YYYY-MM-DD), name, amount,
            merchant_name, category (list[str]).

    Returns:
        A string of roughly 30-60 tokens, ready to be PREPENDED to the
        original chunk text by the chunker.
        
    Examples:
        >>> make_prefix({
        ...     "date": "2025-06-03",
        ...     "name": "Saigon Noodle House",
        ...     "amount": 15.87,
        ...     "merchant_name": "Saigon Noodle House",
        ...     "category": ["Food and Drink", "Restaurants"],
        ... })
        'Dining out transaction from June 2025 (summer). Restaurant spend, $15.87.'

        >>> make_prefix({
        ...     "date": "2026-01-14",
        ...     "name": "Netflix",
        ...     "amount": 15.49,
        ...     "merchant_name": "Netflix",
        ...     "category": ["Service", "Subscription"],
        ... })
        'Subscription transaction from January 2026 (winter). Recurring monthly service, $15.49.'
    """
    # ----- YOU WRITE THIS BODY -----
    # Parse the date into year + month
    # Use season_of(month) + month_name(month) for human-friendly date
    # Use category_label(transaction["category"]) for plain-English category
    # Compose a 1-2 sentence string covering: category + when + amount (+ optional class detail)
    d = date.fromisoformat(transaction['date'])

    return f"{category_label(transaction['category'])} transaction from {month_name(d.month)} {d.year} ({season_of(d.month)}). Transaction of {transaction['amount']:.2f} for {transaction['name']}"
