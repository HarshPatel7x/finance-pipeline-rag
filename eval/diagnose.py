"""
eval/diagnose.py — per-question diagnostic for the RAG-triad (Step 6 debugging).

NOT shipped logic and NOT the rep — a throwaway diagnostic harness (like judge.py
is plumbing). It runs the same pipeline as run_eval but prints EACH question's
three metric scores AND the judge's reason, so we can see which questions drag the
aggregate and why. Run:  PYTHONPATH=. ./venv/bin/python -m eval.diagnose
"""
from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

from eval.golden_qa import GOLDEN_SET
from eval.judge import get_judge
from eval.run_eval import THRESHOLD, build_test_case
from src.retriever import HybridRetriever


def main() -> None:
    retriever = HybridRetriever.from_corpus()
    judge = get_judge()
    faith = FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    rel = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    rec = ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True)

    for i, g in enumerate(GOLDEN_SET, 1):
        tc = build_test_case(g, retriever)
        print(f"\n[{i}] ({g.archetype}) {g.question}")
        print(f"    expected : {g.expected_answer!r}")
        print(f"    model    : {tc.actual_output!r}")
        if g.archetype == "negative":
            print(f"    REFUSAL match: {tc.actual_output == g.expected_answer}")
            continue
        f = faith.measure(tc)
        r = rel.measure(tc)
        c = rec.measure(tc)
        print(f"    faithfulness={f:.2f}  relevancy={r:.2f}  recall={c:.2f}")
        print(f"    faith reason : {faith.reason}")
        print(f"    relev reason : {rel.reason}")
        print(f"    recall reason: {rec.reason}")


if __name__ == "__main__":
    main()
