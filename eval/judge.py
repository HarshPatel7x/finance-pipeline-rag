"""
eval/judge.py — resilient LLM-as-judge for the RAG-triad metrics.

NON-Claude judge on purpose: the generator is Claude (Sonnet); a Claude judge would
self-prefer (grade its own family too high), inflating scores. We grade with Llama.

Strategy (user pref 2026-06-01): try the fast Groq free tier FIRST, auto-fall-back
to the local Ollama model when Groq hits a rate-limit/quota. Same model family
(llama-3) both ends → scoring stays consistent across a fallback.

Uniform path: `instructor` wraps `litellm` for BOTH backends, coercing DeepEval's
structured (Pydantic-schema) verdicts CLIENT-SIDE with re-asks. Required because
Groq strict-validates schemas server-side and rejects partial JSON
(`tool_use_failed` / `json_validate_failed`); instructor re-asks until valid, so the
return contract is identical no matter which backend served the call.

This file is plumbing, not the rep — provided complete. The rep is run_eval.py.
"""
from __future__ import annotations

import asyncio
import os
import time

import instructor
import litellm
from deepeval.models import DeepEvalBaseLLM
from litellm import completion
from pydantic import BaseModel

# Ordered backends: Groq free tier first (fast), local Ollama as the no-limit
# fallback. Override with JUDGE_CHAIN env var (comma-separated litellm model ids) —
# e.g. JUDGE_CHAIN=groq/llama-3.3-70b-versatile for a pure-Groq run.
DEFAULT_CHAIN = ["groq/llama-3.3-70b-versatile", "ollama/llama3.1"]


def _is_rate_limit(exc: BaseException) -> bool:
    """A Groq/litellm rate-limit or quota error — the typed error, or an
    instructor-wrapped one whose message carries the rate_limit code."""
    return isinstance(exc, litellm.RateLimitError) or "rate_limit" in str(exc).lower()


def _is_bad_completion(exc: BaseException) -> bool:
    """A structurally-empty / unparseable judge response — not a transport error.

    Seen in CI: OpenRouter load-balanced a request to a backend (Parasail) that
    returned finish_reason="tool_calls" with an EMPTY payload (content=None,
    tool_calls=None). instructor exhausts its re-asks against the same routed
    backend and raises InstructorRetryException wrapping a pydantic json_invalid
    ValidationError. A different backend usually succeeds, so treat this as
    advance-to-next-backend worthy rather than a hard crash."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "instructorretry" in name
        or "validationerror" in name
        or "json_invalid" in msg
        or "validation error for" in msg
    )


class ResilientJudge(DeepEvalBaseLLM):
    """Groq-first, local-Ollama-fallback judge. Sticky: once a backend hits a
    rate-limit/quota it permanently advances to the next for the rest of the run
    (no re-hammering a dead quota on every metric call)."""

    def __init__(self, chain: list[str] | None = None):
        env_chain = os.environ.get("JUDGE_CHAIN")
        self.chain = chain or (
            [m.strip() for m in env_chain.split(",")] if env_chain else DEFAULT_CHAIN
        )
        self._idx = 0
        # JSON mode + client-side validation/retries — see module docstring.
        self.client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)
        super().__init__(self.chain[self._idx])

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.chain[self._idx]

    def _kwargs(self, model: str) -> dict:
        kw = {"model": model, "temperature": 0}
        if model.startswith("groq/"):
            kw["api_key"] = os.environ["GROQ_API_KEY"]
        if model.startswith("openrouter/"):
            # OpenRouter silently load-balances across backend providers; some
            # (observed: Parasail) return an empty tool-call response instructor
            # can't parse, crashing the eval. Route only to providers that honor
            # our structured request, and exclude the known-bad one.
            kw["extra_body"] = {
                "provider": {"require_parameters": True, "ignore": ["Parasail"]}
            }
        return kw

    def _run(self, prompt: str, schema: type[BaseModel] | None):
        msgs = [{"role": "user", "content": prompt}]
        last = None
        while self._idx < len(self.chain):
            model = self.chain[self._idx]
            for attempt in range(4):
                try:
                    if schema is None:
                        r = completion(messages=msgs, **self._kwargs(model))
                        return r.choices[0].message.content
                    return self.client.chat.completions.create(
                        messages=msgs, response_model=schema, max_retries=3,
                        **self._kwargs(model),
                    )
                except Exception as e:  # noqa: BLE001 — re-raised below unless retryable
                    last = e
                    if _is_rate_limit(e):
                        msg = str(e).lower()
                        if ("per minute" in msg or "tpm" in msg) and attempt < 3:
                            time.sleep(5)  # per-minute throttle → wait the window, retry
                            continue
                        break  # daily/quota or minute-retries exhausted → advance backend
                    if _is_bad_completion(e):
                        break  # empty/unparseable provider output → advance to next backend
                    raise
            self._idx += 1  # sticky fall to the next backend
        raise RuntimeError(f"All judge backends exhausted. Last: {last}")

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return self._run(prompt, schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        # DeepEval scores via a_generate inside an event loop; run the sync path
        # off-loop so the time.sleep backoff can't stall the loop.
        return await asyncio.to_thread(self._run, prompt, schema)


def get_judge(chain: list[str] | None = None) -> ResilientJudge:
    """Return the Groq-first / local-Ollama-fallback judge for the triad metrics."""
    return ResilientJudge(chain)
