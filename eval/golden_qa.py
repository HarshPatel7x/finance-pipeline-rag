"""
eval/golden_qa.py — the golden Q&A set the RAG harness is graded against.

A "golden" entry is a question + the ground-truth answer you already know is
correct (computed from data/corpus.json, NOT from the model). The harness runs
the real pipeline on each question and the judge compares the model's answer +
retrieved chunks against your ground truth.

Design the 20 to cover three archetypes (this spread is the whole point — each
stresses a different triad metric):
  - exact_fact  — one transaction; pins FAITHFULNESS (model must not alter the
                  number/date/merchant that is right there in the chunk).
  - aggregate   — needs several chunks summed/filtered; pins CONTEXTUAL RECALL
                  (retrieval has to pull in every relevant chunk, or the sum is
                  wrong through no fault of the generator).
  - negative    — answer is NOT in the corpus; pins the grounding guard (model
                  must reply with the refusal string, not invent an answer).

Ground-truth rule: every expected_answer must be checkable against corpus.json.
For aggregate entries, compute the number first (a one-off pandas/json sum), then
paste the result — do not eyeball it.

YOUR JOB: write 20 entries. Two seeds are filled in to show the shape + the
archetype spread. Replace the TODOs. Keep roughly 8 exact_fact / 8 aggregate /
4 negative so the eval exercises all three metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

# The exact refusal string generate() is instructed to emit when the answer is
# absent (must match src/generator.py GROUNDED_PROMPT, or negative cases fail).
REFUSAL = "I don't have that in the provided records."


@dataclass(frozen=True)
class Golden:
    """One graded question.

    question:        natural-language input fed to generate().
    expected_answer: ground truth, computed from corpus.json (NOT model output).
    archetype:       "exact_fact" | "aggregate" | "negative" — for slicing
                     scores by archetype when you read the report.
    """

    question: str
    expected_answer: str
    archetype: str


# --- two seeds (real, grounded in data/corpus.json) -------------------------
GOLDEN_SET: list[Golden] = [
    Golden(
        question="How much did I spend at Costco on June 1, 2025?",
        expected_answer="$70.70",
        archetype="exact_fact",
    ),
    Golden(
        question="How much did I spend on rent in 2019?",
        expected_answer=REFUSAL,
        archetype="negative",
    ),
    Golden(
        question="How much did I spend on the internet in 2025 summer?",
        expected_answer="$195.00",
        archetype="aggregate",
    ),
    Golden(
        question="On what did I spend 65 dollars on in June 2025?",
        expected_answer="Internet Provider",
        archetype="exact_fact",
    ),
    Golden(
        question="What was the date of the last payroll transaction in 2025?",
        expected_answer="2025-12-15",
        archetype="exact_fact",
    ),
    Golden(
        question="How much was deducted in payroll transactions in June 2025?",
        expected_answer="$-2,350.00",
        archetype="aggregate",
    ),
    Golden(
        question="When did I drink Foxtail Coffee last in 2026?",
        expected_answer="2026-05-25",
        archetype="exact_fact",
    ),
    Golden(
        question="On what did I last spent $7.36 on in 2026?",
        expected_answer="Foxtail Coffee",
        archetype="exact_fact",
    ),
    Golden(
        question="When did I go to disco or party in July 2025",
        expected_answer=REFUSAL,
        archetype="negative",
    ),
    Golden(
        question="How much did I spent on Pizzas in 2025?",
        expected_answer=REFUSAL,
        archetype="negative",
    ),
    Golden(
        question="What did Anthropic charged me in 2024?",
        expected_answer=REFUSAL,
        archetype="negative",
    ),
    Golden(
        question="Where did I spent $70.70 dollars in June 2025",
        expected_answer="Costco",
        archetype="exact_fact",
    ),
    Golden(
        question="Which supermarket did I go on 1st of June 2025?",
        expected_answer="Costco",
        archetype="exact_fact",
    ),
    Golden(
        question="How much did i spent digital purchase in June 2025?",
        expected_answer="$136.26",
        archetype="aggregate",
    ),
    Golden(
        question="How much did i spent on Spotify Premium in 5th of June 2025?",
        expected_answer="$11.99",
        archetype="exact_fact",
    ),
    Golden(
        question="How much did i spent digital on subscription in June 2025?",
        expected_answer="$134.47",
        archetype="aggregate",
    ),
    Golden(
        question="How much did I transfer between my accounts in August 2025?",
        expected_answer="$715.52",
        archetype="aggregate",
    ),
    Golden(
        question="How many times did I spend more than $5 at a coffee shop in September 2025?",
        expected_answer="4 times",
        archetype="aggregate",
    ),
    Golden(
        question="What was the maximum amount that i have spent on Trader Joe's in 2025?",
        expected_answer="$101.68",
        archetype="aggregate",
    ),
    Golden(
        question="How much did I spend on T-Mobile in summer 2025?",
        expected_answer="$165.00",
        archetype="aggregate",
    ),
]
# Set complete: 8 exact_fact / 8 aggregate / 4 negative (= 20), all verified vs corpus.json.
