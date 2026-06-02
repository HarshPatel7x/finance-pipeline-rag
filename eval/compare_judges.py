"""
eval/compare_judges.py — controlled A/B over the SAME 20 cases.

Builds the 20 LLMTestCases ONCE (one retrieve + generate pass), then scores that
*identical* set with a weak (local-8B) and a strong (OpenRouter Llama-3.3-70B) judge.
Because the question, retrieved context, and generated answer are shared, the ONLY
variable across the two columns is the judge — a clean controlled experiment.

Run: PYTHONPATH=. ./venv/bin/python -m eval.compare_judges
"""
from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

from eval.golden_qa import GOLDEN_SET
from eval.judge import ResilientJudge
from eval.run_eval import THRESHOLD, build_test_case
from src.retriever import HybridRetriever

JUDGES = {
    "weak local-8B": ["ollama/llama3.1"],
    "strong OR-70B": ["openrouter/meta-llama/llama-3.3-70b-instruct"],
}


def score_with(judge, cases) -> dict:
    """Score the pre-built cases with one judge, archetype-aware (negatives by
    refusal-match and excluded from the triad; aggregates excluded from recall)."""
    faith = FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    rel = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    rec = ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    f_sum = r_sum = c_sum = 0.0
    n = recall_n = tricks_ok = tricks_n = 0
    per_q = []
    for g, tc in cases:
        if g.archetype == "negative":
            tricks_n += 1
            ok = tc.actual_output == g.expected_answer
            tricks_ok += int(ok)
            per_q.append((g.archetype, g.question, ("REFUSAL", ok)))
            continue
        f = faith.measure(tc)
        r = rel.measure(tc)
        f_sum += f
        r_sum += r
        n += 1
        c = None
        if g.archetype != "aggregate":
            c = rec.measure(tc)
            c_sum += c
            recall_n += 1
        per_q.append((g.archetype, g.question, (round(f, 2), round(r, 2), c)))
    return {
        "faith": f_sum / (n or 1),
        "relev": r_sum / (n or 1),
        "recall": c_sum / (recall_n or 1),
        "tricks": f"{tricks_ok}/{tricks_n}",
        "n": n,
        "recall_n": recall_n,
        "per_q": per_q,
    }


def main() -> None:
    retriever = HybridRetriever.from_corpus()
    # Build the 20 test cases ONCE — identical inputs for every judge.
    cases = [(g, build_test_case(g, retriever)) for g in GOLDEN_SET]
    print(f"Built {len(cases)} test cases once (shared across judges).\n")

    results = {}
    for name, chain in JUDGES.items():
        print(f"--- scoring with {name} ({chain[0]}) ---")
        results[name] = score_with(ResilientJudge(chain), cases)

    names = list(results)
    print("\n===== 20-vs-20, IDENTICAL test cases (only the judge differs) =====")
    print(f"{'metric':<22}" + "".join(f"{n:>18}" for n in names))
    for key, label in [
        ("faith", "faithfulness"),
        ("relev", "answer_relevancy"),
        ("recall", "contextual_recall"),
        ("tricks", "trick_refusals"),
    ]:
        row = f"{label:<22}"
        for n in names:
            v = results[n][key]
            row += f"{(f'{v:.3f}' if isinstance(v, float) else v):>18}"
        print(row)
    for n in names:
        print(f"  {n}: faith/relev over n={results[n]['n']}, recall over recall_n={results[n]['recall_n']}")
    print(f"  threshold = {THRESHOLD}")

    # Per-question, both judges side by side.
    print("\n--- per-question (faith/relev/recall; · = n/a; Rok/RX = refusal pass/fail) ---")
    a, b = names

    def fmt(s):
        if s[0] == "REFUSAL":
            return "Rok" if s[1] else "RX"
        f, r, c = s
        return f"{f}/{r}/{'·' if c is None else round(c, 2)}"

    for i in range(len(cases)):
        arch, q, _ = results[a]["per_q"][i]
        sa = results[a]["per_q"][i][2]
        sb = results[b]["per_q"][i][2]
        print(f"[{i + 1:>2}] {arch:<10} weak={fmt(sa):<14} strong={fmt(sb):<14} | {q[:46]}")


if __name__ == "__main__":
    main()
