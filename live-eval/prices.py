"""Real price table and a thread-safe spend meter with a hard cap.

Prices checked 2026-11-06 (USD per million tokens):
- Claude Haiku 5.5: $0.10 in / $0.50 out for prompts up to 100K tokens; $0.50 / $2.50 above.
- voyage-4-lite: $0.02 per million input tokens (free-tier credits ignored, so this over-counts).
- TypeSafe Jev: $0.042 per million input tokens, $0.00 output (cookbook pages).
"""
from __future__ import annotations

import threading

HAIKU = {"in": 0.10, "out": 0.50, "in_long": 0.50, "out_long": 2.50, "long_threshold": 100_000}
VOYAGE_PER_M = 0.02
JEV_IN_PER_M = 0.042
JEV_OUT_PER_M = 0.0


class SpendCapReached(RuntimeError):
    pass


def haiku_cost(in_tok: int, out_tok: int) -> float:
    if in_tok > HAIKU["long_threshold"]:
        return in_tok / 1e6 * HAIKU["in_long"] + out_tok / 1e6 * HAIKU["out_long"]
    return in_tok / 1e6 * HAIKU["in"] + out_tok / 1e6 * HAIKU["out"]


class SpendMeter:
    def __init__(self, cap_usd: float):
        self.cap = float(cap_usd)
        self.usd = 0.0
        self.counts: dict[str, float] = {}
        self._lock = threading.Lock()
        self.stopped = False

    def _bump(self, k: str, v: float):
        self.counts[k] = self.counts.get(k, 0) + v

    def check(self, projected_usd: float = 0.0):
        with self._lock:
            if self.stopped or self.usd + projected_usd > self.cap:
                self.stopped = True
                raise SpendCapReached(f"spend ${self.usd:.4f} + projected ${projected_usd:.4f} > cap ${self.cap:.2f}")

    def add_haiku(self, in_tok: int, out_tok: int):
        with self._lock:
            self.usd += haiku_cost(in_tok, out_tok)
            self._bump("haiku_in", in_tok)
            self._bump("haiku_out", out_tok)
            self._bump("haiku_calls", 1)

    def add_voyage(self, tokens: int):
        with self._lock:
            self.usd += tokens / 1e6 * VOYAGE_PER_M
            self._bump("voyage_tokens", tokens)

    def add_jev(self, in_tok: int, out_tok: int = 0, calls: int = 0, cache_hits: int = 0):
        with self._lock:
            self.usd += in_tok / 1e6 * JEV_IN_PER_M + out_tok / 1e6 * JEV_OUT_PER_M
            self._bump("jev_in", in_tok)
            self._bump("jev_out", out_tok)
            self._bump("jev_calls", calls)
            self._bump("jev_cache_hits", cache_hits)

    def snapshot(self) -> dict:
        with self._lock:
            return {"usd": round(self.usd, 5), "cap": self.cap, "stopped": self.stopped, **self.counts}
