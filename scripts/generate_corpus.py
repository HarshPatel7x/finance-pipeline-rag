"""
generate_corpus.py — Synthesize a deterministic transaction corpus for RAG eval.

Honest framing: this is synthetic, not real bank data. The predecessor
finance-pipeline (https://github.com/HarshPatel7x/finance-pipeline) ingests
Plaid sandbox data; real BofA development-mode OAuth was never completed.
This generator produces a corpus large enough (~840 records, 12 months,
15 categories) to make the contextual-recall@5 and faithfulness metrics
meaningful.

Schema matches the predecessor's sample_transactions.json so the loaders
in src/ are corpus-agnostic — swap in real DynamoDB output without code
changes.

Usage:
    python scripts/generate_corpus.py                       # default args
    python scripts/generate_corpus.py --seed 42 --months 12 --out data/transactions.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Category catalog — one entry per Plaid-style category we want in the corpus.
# Per-month frequency is a (min, max) range; the generator draws uniformly.
# Amount range is (mean, std) for a normal distribution clamped to >$0.
# ---------------------------------------------------------------------------


@dataclass
class Category:
    name: str
    merchants: list[str]
    plaid_categories: list[list[str]]  # one or more Plaid category paths
    freq_per_month: tuple[int, int]
    amount_mean: float
    amount_std: float


CATEGORIES: list[Category] = [
    Category(
        name="groceries",
        merchants=["Walmart", "Publix", "Whole Foods Market", "Trader Joe's", "Aldi", "Costco"],
        plaid_categories=[["Shops", "Supermarkets and Groceries"]],
        freq_per_month=(4, 8),
        amount_mean=70.0, amount_std=40.0,
    ),
    Category(
        name="dining",
        merchants=["Chipotle Mexican Grill", "Olive Garden", "Pho 79", "McDonald's",
                   "Chick-fil-A", "Panera Bread", "Saigon Noodle House", "Cracker Barrel"],
        plaid_categories=[["Food and Drink", "Restaurants"]],
        freq_per_month=(8, 15),
        amount_mean=24.0, amount_std=14.0,
    ),
    Category(
        name="coffee",
        merchants=["Starbucks", "Dunkin'", "Local Roaster", "Foxtail Coffee"],
        plaid_categories=[["Food and Drink", "Restaurants", "Coffee Shop"]],
        freq_per_month=(6, 12),
        amount_mean=6.50, amount_std=1.50,
    ),
    Category(
        name="gas",
        merchants=["Shell", "Chevron", "Speedway", "Costco Gas", "BP"],
        plaid_categories=[["Travel", "Gas Stations"]],
        freq_per_month=(3, 5),
        amount_mean=40.0, amount_std=8.0,
    ),
    Category(
        name="online_shopping",
        merchants=["Amazon", "Walmart.com", "Target.com", "Best Buy", "eBay"],
        plaid_categories=[["Shops", "Digital Purchase"]],
        freq_per_month=(2, 5),
        amount_mean=55.0, amount_std=60.0,
    ),
    Category(
        name="pharmacy",
        merchants=["CVS Pharmacy", "Walgreens", "Walmart Pharmacy"],
        plaid_categories=[["Healthcare", "Pharmacies"]],
        freq_per_month=(1, 3),
        amount_mean=22.0, amount_std=12.0,
    ),
    Category(
        name="entertainment",
        merchants=["AMC Theatres", "Steam", "PlayStation Store", "Concert Tickets"],
        plaid_categories=[["Recreation", "Arts and Entertainment"]],
        freq_per_month=(2, 4),
        amount_mean=28.0, amount_std=20.0,
    ),
    Category(
        name="transport",
        merchants=["Uber", "Lyft", "Parking", "Metro"],
        plaid_categories=[["Travel", "Taxi"]],
        freq_per_month=(1, 3),
        amount_mean=18.0, amount_std=12.0,
    ),
    Category(
        name="personal_care",
        merchants=["LA Fitness", "Barbershop", "Sally Beauty"],
        plaid_categories=[["Recreation", "Gyms and Fitness Centers"]],
        freq_per_month=(1, 3),
        amount_mean=32.0, amount_std=18.0,
    ),
    Category(
        name="transfers",
        merchants=["Bank Transfer", "Zelle Payment"],
        plaid_categories=[["Transfer", "Internal Account Transfer"]],
        freq_per_month=(0, 2),
        amount_mean=250.0, amount_std=200.0,
    ),
    Category(
        name="bank_fees",
        merchants=["ATM Fee", "Wire Fee", "Foreign Transaction Fee"],
        plaid_categories=[["Service", "Financial"]],
        freq_per_month=(0, 1),
        amount_mean=15.0, amount_std=8.0,
    ),
    Category(
        name="refunds",
        merchants=["Amazon Refund", "Walmart Refund", "Returned Purchase"],
        plaid_categories=[["Transfer", "Credit"]],
        freq_per_month=(0, 1),
        amount_mean=45.0, amount_std=35.0,
    ),
]


# Fixed recurring transactions (subscriptions + utilities + income).
# These hit on a deterministic day-of-month so the corpus has clean
# "recurring spend" patterns the retriever can latch onto.
RECURRING = [
    # (merchant, day_of_month, amount, plaid_categories)
    ("Netflix",          14, 15.49, [["Service", "Subscription"]]),
    ("Spotify Premium",   5, 11.99, [["Service", "Subscription"]]),
    ("Apple iCloud",     20,  2.99, [["Service", "Subscription"]]),
    ("GitHub Pro",        7,  4.00, [["Service", "Subscription"]]),
    ("Anthropic (Claude Max 5x)", 13, 100.00, [["Service", "Subscription"]]),
    ("Electric Company", 12,  85.00, [["Service", "Utilities"]]),
    ("Internet Provider", 3,  65.00, [["Service", "Utilities"]]),
    ("T-Mobile",         18,  55.00, [["Service", "Telecommunication Services"]]),
    # Debt minimums (per harsh-context.md debt strategy: minimums only)
    ("Apple Card Payment",  25, 135.00, [["Payment", "Credit Card"]]),
    ("Bank of America Card 1 Payment",  22, 171.00, [["Payment", "Credit Card"]]),
    ("Bank of America Card 2 Payment",  28, 371.00, [["Payment", "Credit Card"]]),
    # Income (negative amount per Plaid convention: outflow = positive, inflow = negative)
    ("Direct Deposit Payroll",  1, -1175.00, [["Transfer", "Payroll"]]),
    ("Direct Deposit Payroll", 15, -1175.00, [["Transfer", "Payroll"]]),
]


# ---------------------------------------------------------------------------
# Generator core
# ---------------------------------------------------------------------------


def round_amount(amt: float) -> float:
    """Bank amounts always have 2 decimal places."""
    return round(amt, 2)


def draw_amount(rng: random.Random, mean: float, std: float) -> float:
    """Normal-distributed amount, clamped to a sensible floor of $0.50."""
    val = rng.gauss(mean, std)
    return round_amount(max(0.50, val))


def generate_records(seed: int, months_back: int) -> list[dict]:
    rng = random.Random(seed)
    today = date.today()
    start = today - timedelta(days=months_back * 30)

    # Walk month-by-month from start
    records: list[dict] = []
    txn_counter = 1

    # Iterate 12 month-buckets
    current = start.replace(day=1)
    while current <= today:
        # Variable-frequency categories
        for cat in CATEGORIES:
            n = rng.randint(*cat.freq_per_month)
            for _ in range(n):
                # Random day inside the month, clamped to today
                day = rng.randint(1, 28)
                txn_date = current.replace(day=day)
                if txn_date > today:
                    continue
                merchant = rng.choice(cat.merchants)
                plaid_cat = rng.choice(cat.plaid_categories)
                amount = draw_amount(rng, cat.amount_mean, cat.amount_std)
                # Refunds are negative (money coming back in)
                if cat.name == "refunds":
                    amount = -amount
                records.append({
                    "transaction_id": f"tx_{txn_counter:04d}",
                    "date": txn_date.isoformat(),
                    "name": merchant,
                    "amount": amount,
                    "account_id": "acc_synthetic_001",
                    "merchant_name": merchant,
                    "category": plaid_cat,
                    "pending": False,
                })
                txn_counter += 1

        # Fixed recurring transactions for this month
        for merchant, day, amount, plaid_cats in RECURRING:
            try:
                txn_date = current.replace(day=day)
            except ValueError:
                continue
            if txn_date > today:
                continue
            records.append({
                "transaction_id": f"tx_{txn_counter:04d}",
                "date": txn_date.isoformat(),
                "name": merchant,
                "amount": round_amount(amount),
                "account_id": "acc_synthetic_001",
                "merchant_name": merchant,
                "category": rng.choice(plaid_cats),
                "pending": False,
            })
            txn_counter += 1

        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    records.sort(key=lambda r: (r["date"], r["transaction_id"]))
    # Re-number transaction_ids in date order for cleanliness
    for i, r in enumerate(records, start=1):
        r["transaction_id"] = f"tx_{i:04d}"
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--months", type=int, default=12, help="Months of history (default 12)")
    parser.add_argument("--out", type=Path, default=Path("data/transactions.json"))
    args = parser.parse_args()

    records = generate_records(args.seed, args.months)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))

    # Summary print — useful as a sanity check
    by_cat: dict[str, int] = {}
    for r in records:
        key = " > ".join(r["category"])
        by_cat[key] = by_cat.get(key, 0) + 1

    print(f"Generated {len(records)} transactions")
    print(f"  Date range: {records[0]['date']} → {records[-1]['date']}")
    print(f"  Output:     {args.out}")
    print(f"  By category:")
    for cat, count in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"    {count:4d}  {cat}")


if __name__ == "__main__":
    main()
