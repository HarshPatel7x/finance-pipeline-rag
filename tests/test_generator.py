"""
tests/test_generator.py — Step 5 smoke stub.

Two layers:
  - structural test always runs (no API, no index): the GenerationResult shape
    and the GROUNDED_PROMPT contract.
  - integration smoke skips until generate() is implemented (and needs the
    persisted Chroma index + ANTHROPIC_API_KEY). It checks the grounding
    invariant, not exact wording.
"""
from __future__ import annotations

import pytest

from src.generator import (
    DEV_MODEL,
    GROUNDED_PROMPT,
    GenerationResult,
    generate,
)


def test_generation_result_shape():
    """GenerationResult carries answer + provenance, with sane defaults."""
    r = GenerationResult(answer="spent $312 on dining", cited_ids=["chunk_7"])
    assert r.answer == "spent $312 on dining"
    assert r.cited_ids == ["chunk_7"]
    assert r.retrieved_ids == []        # default_factory list
    assert r.model == DEV_MODEL          # default model


def test_prompt_contract_has_placeholders_and_citation_rule():
    """The grounded prompt must expose both fill points and the CITED: contract."""
    assert "{context}" in GROUNDED_PROMPT
    assert "{question}" in GROUNDED_PROMPT
    assert "CITED:" in GROUNDED_PROMPT
    # the anti-hallucination instruction is the whole point of grounding
    assert "ONLY" in GROUNDED_PROMPT


@pytest.mark.integration
def test_generate_smoke_grounded_and_cited():
    """Smoke: real retrieval + Claude. cited_ids must be a subset of what we showed.

    Skips until generate() is implemented. Needs chroma_db/ + ANTHROPIC_API_KEY.
    """
    try:
        result = generate("How much did I spend on dining in 2026?", k=5)
    except NotImplementedError:
        pytest.skip("generate() not implemented yet — Step 5 in progress")

    assert isinstance(result, GenerationResult)
    assert result.answer.strip(), "answer should not be empty"
    assert len(result.retrieved_ids) > 0, "should have grounded on some chunks"
    # the grounding invariant: the model may only cite chunks it was shown
    stray = [c for c in result.cited_ids if c not in result.retrieved_ids]
    assert not stray, f"model cited ids it was never shown: {stray}"
