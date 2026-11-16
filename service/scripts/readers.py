"""The answering models an eval can put in the user's seat (exp-reader-swap-v1). Eval code only: no generative
model runs inside the brain.

Each reader answers with its thinking off, so the readers differ by model and not by a thinking budget:
- haiku: Claude Haiku 5.5, `thinking: disabled` (as brain.providers.HaikuClient sends it).
- sonnet: Claude Sonnet 5.5. It rejects `thinking: disabled`; `between_tools` is its thinking-off setting.
- opus: Claude Opus 5.5. Its thinking cannot be turned off; it runs at effort low, and the max_tokens it is
  given must leave room for that thinking (OPUS_EXTRA is added). Listed for later; no run uses it yet.

No temperature is sent (non-default values are rejected). Prices are dollars per million tokens, checked
2026-11-12; a prompt over a model's long-context threshold is not expected here (packs are a few thousand
tokens) and is not priced apart. A classifier refusal comes back as HTTP 200 with stop_reason "refusal" and no
text: `complete` returns it as an empty answer with that stop reason, for the caller to count.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))

from brain.providers import (ANTHROPIC_URL, ANTHROPIC_VERSION, Completion, ProviderUnavailable,  # noqa: E402
                             default_http, post_json)
from brain.spend import est_tokens  # noqa: E402

OPUS_EXTRA = 2000
READERS = {
    "haiku": {"model": "claude-haiku-5-5", "body": {"thinking": {"type": "disabled"}}, "in": 0.10, "out": 0.50},
    "sonnet": {"model": "claude-sonnet-5-5", "body": {"thinking": {"type": "between_tools"}}, "in": 2.00, "out": 10.00},
    "opus": {"model": "claude-opus-5-5", "body": {"output_config": {"effort": "low"}}, "in": 4.00, "out": 20.00,
             "extra_tokens": OPUS_EXTRA},
}


def cost(name: str, in_tok: int, out_tok: int) -> float:
    r = READERS[name]
    return in_tok / 1e6 * r["in"] + out_tok / 1e6 * r["out"]


class Reader:
    """One answering model. `complete` has HaikuClient's signature, so an eval can use either. Answers are kept
    by prompt (as MemoHaiku does), so the same pack is read once."""

    def __init__(self, name: str, meter, key: str | None, http=None, max_retries: int = 3, sleep=time.sleep):
        if name not in READERS:
            raise ValueError(f"unknown reader {name!r}; one of {sorted(READERS)}")
        self.name, self.meter, self._key = name, meter, key
        self.model = READERS[name]["model"]
        self.http = http or default_http()
        self.max_retries, self.sleep = max_retries, sleep
        self._memo: dict[tuple, Completion] = {}
        self._lock = threading.Lock()
        self.refusals = 0

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        if not self._key:
            raise ProviderUnavailable("haiku", "no API key configured")
        k = (system, user, max_tokens)
        with self._lock:
            if k in self._memo:
                return self._memo[k]
        spec = READERS[self.name]
        limit = max_tokens + spec.get("extra_tokens", 0)
        body: dict = {"model": self.model, "max_tokens": limit, **spec["body"],
                      "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system
        # the meter knows Anthropic's calls as "haiku"; the cost recorded is this model's
        with self.meter.charge("haiku", cost(self.name, est_tokens(system + user), limit)) as ch:
            j = post_json(self.http, "haiku", ANTHROPIC_URL, body,
                          {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION},
                          self.max_retries, self.sleep)
            u = j.get("usage") or {}
            ti, to = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
            ch.record(cost(self.name, ti, to), tokens_in=ti, tokens_out=to)
        text = "".join(b.get("text", "") for b in j.get("content") or [] if b.get("type") == "text").strip()
        out = Completion(text, str(j.get("model") or self.model), j.get("stop_reason"), ti, to)
        with self._lock:
            if out.stop_reason == "refusal":
                self.refusals += 1
            self._memo[k] = out
        return out
