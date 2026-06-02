"""
eval/latency.py — Step 8: p95 wall-clock latency of HybridRetriever.retrieve().

Times the full 3-stage retrieve (Voyage embed → BM25 → Voyage rerank) over the
golden questions (2 passes), after a warm-up call that loads the models/clients.
Reports p50/p95/min/max/mean in ms. Voyage rerank (a network round-trip) dominates.

Run: PYTHONPATH=. ./venv/bin/python -m eval.latency
"""
from __future__ import annotations

import math
import statistics
import time

from eval.golden_qa import GOLDEN_SET
from src.retriever import HybridRetriever


def _percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def main() -> None:
    r = HybridRetriever.from_corpus()
    r.retrieve(GOLDEN_SET[0].question, k=5)  # warm-up (load models/clients)

    lat_ms: list[float] = []
    for _ in range(2):  # 2 passes over the 20 questions = 40 samples
        for g in GOLDEN_SET:
            t = time.perf_counter()
            r.retrieve(g.question, k=5)
            lat_ms.append((time.perf_counter() - t) * 1000)

    print(f"retrieve() latency over {len(lat_ms)} calls (ms):")
    print(f"  p50  {_percentile(lat_ms, 0.50):.0f}")
    print(f"  p95  {_percentile(lat_ms, 0.95):.0f}")
    print(f"  mean {statistics.mean(lat_ms):.0f}   min {min(lat_ms):.0f}   max {max(lat_ms):.0f}")


if __name__ == "__main__":
    main()
