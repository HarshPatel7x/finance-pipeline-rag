"""
eval/judge.py — the LLM-as-judge for the RAG-triad metrics.

WHY not Claude as judge: our generator is Claude (Sonnet); a Claude judge would
score its own family too high (self-preference bias). We use a NON-Claude judge.

WHY this shape (custom DeepEvalBaseLLM + instructor):
  DeepEval scores via structured (Pydantic-schema) verdicts. DeepEval's stock
  LiteLLMModel asks Groq for that schema via Groq's *server-side* strict JSON
  validation — and Groq rejects partial output (e.g. a verdict missing `reason`)
  with `tool_use_failed` / `json_validate_failed`. The local Ollama judge worked
  only because Ollama doesn't strict-validate.
  Fix (DeepEval's own recommended pattern): wrap litellm with `instructor`, which
  coerces output into the schema CLIENT-SIDE and *re-asks the model* on a bad
  parse (max_retries). Groq never strict-rejects; instructor guarantees a valid
  schema instance. Groq-hosted Llama-3.3-70B → fast, free-tier, non-Claude.

This file is plumbing, not the rep — provided complete. The rep is run_eval.py.
"""
from __future__ import annotations

import os
import time

import instructor
from deepeval.models import DeepEvalBaseLLM, OllamaModel
from litellm import completion
from pydantic import BaseModel

JUDGE_MODEL = "groq/llama-3.1-8b-instant"
# Local fallback judge — no rate limits, no daily quota, just slow (~25 min/run).
LOCAL_JUDGE_MODEL = "llama3.1"
OLLAMA_URL = "http://localhost:11434"


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by Groq (via litellm) with instructor schema coercion.

    INACTIVE (get_judge returns the local Ollama judge). BEFORE activating this on
    a paid/reset Groq tier, fix 3 latent issues (audit 2026-06-01, see Task #2):
      1. call `super().__init__(model)` (base sets self.name/self.model);
      2. make `a_generate` truly async (it currently calls the sync, blocking
         `generate` with time.sleep inside the event loop) — use asyncio.to_thread;
      3. `_backoff` should match `isinstance(e, litellm.RateLimitError)` not a string,
         and raise after the loop (no silent None return).
    """

    def __init__(self, model: str = JUDGE_MODEL):
        self.model_name = model
        # JSON mode + client-side validation/retries — sidesteps Groq's strict
        # server-side schema validation that DeepEval's default path tripped on.
        self.client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)

    def load_model(self):
        return self.client

    def _backoff(self, fn, attempts: int = 12):
        # Groq free tier caps the 70B at ~12k tokens/minute; a burst of judge
        # calls trips it (`rate_limit_exceeded`). It's a throttle, not a failure —
        # wait out the per-minute window and retry.
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — narrow on the message text
                if i < attempts - 1 and "rate_limit" in str(e).lower():
                    time.sleep(5)
                    continue
                raise

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        key = os.environ["GROQ_API_KEY"]
        if schema is None:
            resp = self._backoff(
                lambda: completion(
                    model=self.model_name,
                    api_key=key,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            )
            return resp.choices[0].message.content
        return self._backoff(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                api_key=key,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                temperature=0,
                max_retries=3,
            )
        )

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


def get_judge():
    """Return the active judge for the triad metrics.

    Currently LOCAL Ollama — the Groq path (GroqJudge above) is code-complete and
    proven (instructor coercion + backoff), but Groq's FREE tier rate-limits a full
    20-question run: the 70B exhausts its ~100k tokens/day quota, and the 8B's
    6000 tokens/MINUTE throttle is blown by instructor's re-ask bursts. Flip the
    return to `GroqJudge()` once on a paid Dev tier or after the daily quota resets.
    Local has no limits — just slow. (2026-06-01)
    """
    return OllamaModel(model=LOCAL_JUDGE_MODEL, base_url=OLLAMA_URL, temperature=0)
