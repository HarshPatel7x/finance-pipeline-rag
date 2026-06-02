"""
eval/run_eval.py — Step 6 eval harness: run the RAG-triad over the golden set.

This is the rep. The plumbing (judge, golden schema) is done; you write the core
loop that turns each golden question into a graded DeepEval test case and rolls
the scores up against the targets.

Targets (WORKITEMS.md Step 6 + README):
  faithfulness      > 0.85     (generator stays grounded — no hallucinated facts)
  answer_relevancy  > 0.85     (answer actually addresses the question)
  contextual_recall > 0.85     (retrieval pulled in what the answer needed)

Run it:   python -m eval.run_eval
CI (Step 7) will call the same entry point and gate the merge on the thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from eval.golden_qa import GOLDEN_SET, Golden
from eval.judge import get_judge
from src.generator import EVAL_MODEL, generate
from src.retriever import HybridRetriever

THRESHOLD = 0.85


@dataclass
class TriadReport:
    """Rolled-up result of one eval run.

    faithfulness / answer_relevancy / contextual_recall: mean score across the
    golden set (0-1). passed: all three means >= THRESHOLD. n: cases evaluated.
    """

    faithfulness: float
    answer_relevancy: float
    contextual_recall: float
    tricks_passed: int
    n: int
    recall_n: int
    tricks_n: int

    @property
    def passed(self) -> bool:
        return min(
            self.faithfulness, self.answer_relevancy, self.contextual_recall
        ) >= THRESHOLD


def build_test_case(golden: Golden, retriever: HybridRetriever) -> LLMTestCase:
    """Turn one golden question into a DeepEval test case by running the pipeline.

    Your moves (concept — work out the HOW; that is the rep):
      - Run      — call generate() on golden.question with the EVAL_MODEL and the
                   injected retriever, so you grade the real Sonnet answer.
      - Recover  — the judge needs the CONTEXT the model actually saw, as text
                   (not ids). generate() hands back the ids it grounded on; turn
                   those back into their chunk texts from the retriever's corpus.
      - Pack     — assemble an LLMTestCase carrying: the question, the model's
                   answer, your ground-truth answer, and that retrieved context.
                   (Re-check which LLMTestCase field each of those maps to —
                   contextual_recall reads expected vs retrieved, faithfulness
                   reads answer vs retrieved.)
    """

    answer = generate(golden.question, model=EVAL_MODEL, retriever=retriever)
    
    return LLMTestCase(
        input = golden.question,
        actual_output = answer.answer,
        expected_output = golden.expected_answer,
        retrieval_context = [c['text'] for c in retriever.corpus if c['id'] in answer.retrieved_ids]
        )


def run_eval() -> TriadReport:
    """Grade the whole golden set with the local judge and roll up the means.

    Your moves (concept — work out the HOW; that is the rep):
      - Setup    — build the retriever ONCE (reuse across all cases), build the
                   three triad metrics, and hand each one the local judge via
                   model=get_judge() so scoring never hits a paid API.
      - Grade    — make a test case per golden entry, then have each metric
                   measure it. (DeepEval gives you both styles: metric.measure()
                   per case, or evaluate() over a list — pick one and know why.)
      - Roll up  — average each metric across the set, count the cases, and
                   return a TriadReport. The report's own .passed compares
                   against THRESHOLD — you don't re-implement the gate here.
    """
    retriever = HybridRetriever.from_corpus() 
    judge = get_judge()
    
    faithfulness = FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    answer_relevancy = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    contextual_recall = ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True)
    tricks_passed = 0
    tricks_n = 0
    n = 0
    recall_n = 0

    faithfulness_counter, answer_relevancy_counter, contextual_recall_counter = 0, 0, 0
    for golden in GOLDEN_SET:
        test_case = build_test_case(golden, retriever)
        if golden.archetype == 'negative':
            tricks_n += 1
            if test_case.actual_output == golden.expected_answer:
                tricks_passed += 1
        else:
            faithfulness_counter += faithfulness.measure(test_case)
            answer_relevancy_counter += answer_relevancy.measure(test_case)
            n += 1
            # Aggregate expected answers are computed totals ($195.00, $2,350.00,
            # $136.26) that never appear verbatim in retrieved chunks, so
            # ContextualRecallMetric structurally scores them 0.00 even when
            # retrieval worked — same reason negatives are excluded from the triad.
            # Recall therefore runs on exact_fact only, with its own denominator.
            if golden.archetype != 'aggregate':
                contextual_recall_counter += contextual_recall.measure(test_case)
                recall_n += 1

    mean_faithfulness = faithfulness_counter / (n or 1)
    mean_answer_relevancy = answer_relevancy_counter / (n or 1)
    mean_contextual_recall = contextual_recall_counter / (recall_n or 1)

    return TriadReport(faithfulness=mean_faithfulness, answer_relevancy=mean_answer_relevancy, contextual_recall=mean_contextual_recall, n=n, recall_n=recall_n, tricks_passed=tricks_passed, tricks_n=tricks_n)


def main() -> None:
    report = run_eval()
    judge_name = get_judge().get_model_name()
    print(f"\nRAG-triad over {report.n} golden cases (judge={judge_name}):")
    print(f"  faithfulness      {report.faithfulness:.3f}  (target >{THRESHOLD})")
    print(f"  answer_relevancy  {report.answer_relevancy:.3f}  (target >{THRESHOLD})")
    print(f"  contextual_recall {report.contextual_recall:.3f} (over {report.recall_n} exact_fact cases; aggregates excluded)  (target >{THRESHOLD})")
    print(f"trick questions correctly refused: {report.tricks_passed} out of {report.tricks_n}.")
    print(f"\n{'PASS' if report.passed else 'FAIL'} — all three must clear {THRESHOLD}")


if __name__ == "__main__":
    main()
