"""Provider clients (Voyage, Jev, Haiku) with retries and spend metering, plus offline stubs.

Request code is ported from live-eval/clients.py and typesafe_cookbooks/jev_client.py (retry with backoff on
429, 529 and 5xx), using httpx with timeouts. Every paid call goes through SpendMeter.charge(): it refuses
with CapReached before the call when the daily cap would be exceeded, and records the real usage after it.

Keys are only ever placed in request headers. They are never logged, printed or put in exception messages.

Offline stubs (BRAIN_OFFLINE=1) never touch the network. They still call SpendMeter.check() with the same
projected cost, so a low cap triggers CapReached offline exactly as it would online; they record no spend.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .spend import SpendMeter, est_tokens, haiku_cost, jev_cost, voyage_cost

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"   # the version the live-eval harness was measured with

VOYAGE_BATCH = 128
VOYAGE_MAX_CHARS = 30_000          # per input, as in live-eval; capture windows are at most 1,500 characters
JEV_OVERHEAD_TOKENS = 150          # per request, on top of the state and questions (live-eval estimate)
JEV_WORKERS = 8


class ProviderError(RuntimeError):
    """A provider call failed (non-retryable status, retries exhausted, or a malformed response)."""

    def __init__(self, provider: str, message: str, status: int | None = None):
        super().__init__(f"{provider}: {message}")
        self.provider, self.status = provider, status


class ProviderUnavailable(ProviderError):
    """No key is configured for this provider. Callers degrade the same way as for an outage."""


def default_http() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0))


def _retry_delay(attempt: int, response: httpx.Response | None) -> float:
    if response is not None:
        ra = response.headers.get("retry-after")
        if ra and ra.replace(".", "", 1).isdigit():
            return min(30.0, float(ra))
    return min(30.0, 2 ** attempt + random.random())


def post_json(http: httpx.Client, provider: str, url: str, body: dict, headers: dict,
              max_retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> dict:
    """POST JSON; retry 429, 529, 5xx and network errors with exponential backoff (honoring Retry-After)."""
    last: ProviderError | None = None
    for attempt in range(max_retries + 1):
        resp = None
        try:
            resp = http.post(url, json=body, headers=headers)
        except httpx.TransportError as e:
            last = ProviderError(provider, f"network error ({type(e).__name__})")
        else:
            if resp.status_code < 400:
                try:
                    return resp.json()
                except ValueError:
                    raise ProviderError(provider, "response was not JSON", resp.status_code) from None
            err = ProviderError(provider, f"HTTP {resp.status_code}: {resp.text[:200]!r}", resp.status_code)
            if resp.status_code not in (429, 529) and resp.status_code < 500:
                raise err
            last = err
        if attempt < max_retries:
            sleep(_retry_delay(attempt, resp))
    assert last is not None
    raise ProviderError(provider, f"retries exhausted; last error: {last}", last.status)


# ---------------------------------------------------------------- Voyage

class _LRU:
    def __init__(self, size: int):
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, k):
        with self._lock:
            if k in self._d:
                self._d.move_to_end(k)
                return self._d[k]
        return None

    def put(self, k, v) -> None:
        with self._lock:
            self._d[k] = v
            self._d.move_to_end(k)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


class VoyageEmbedder:
    name = "voyage"

    def __init__(self, meter: SpendMeter, key: str | None, model: str = "voyage-4-lite", dim: int = 1024,
                 http: httpx.Client | None = None, max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep, query_cache: int = 1024):
        self.meter, self._key, self.model, self.dim = meter, key, model, dim
        self.http = http or default_http()
        self.max_retries, self.sleep = max_retries, sleep
        self._queries = _LRU(query_cache)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        hit = self._queries.get(text)
        if hit is not None:
            return hit
        vec = self._embed([text], "query")[0]
        self._queries.put(text, vec)
        return vec

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not self._key:
            raise ProviderUnavailable(self.name, "no API key configured")
        out: list[list[float]] = []
        for s in range(0, len(texts), VOYAGE_BATCH):
            batch = [t[:VOYAGE_MAX_CHARS] for t in texts[s:s + VOYAGE_BATCH]]
            with self.meter.charge(self.name, voyage_cost(sum(est_tokens(t) for t in batch))) as ch:
                j = post_json(self.http, self.name, VOYAGE_URL,
                              {"model": self.model, "input": batch, "input_type": input_type},
                              {"Authorization": f"Bearer {self._key}"}, self.max_retries, self.sleep)
                tokens = int((j.get("usage") or {}).get("total_tokens", 0))
                ch.record(voyage_cost(tokens), tokens_in=tokens)
            vecs: list = [None] * len(batch)
            for item in j.get("data") or []:
                i = item.get("index")
                if isinstance(i, int) and 0 <= i < len(batch):
                    vecs[i] = item.get("embedding")
            if any(v is None or len(v) != self.dim for v in vecs):
                raise ProviderError(self.name, f"expected {len(batch)} embeddings of {self.dim} dimensions")
            out.extend(vecs)
        return out


# ---------------------------------------------------------------- Jev

@dataclass(frozen=True)
class JevResult:
    answers: dict              # raw answers per question name
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def noul(self, name: str) -> float:
        return float(self.answers[name]["noul"])

    def choice(self, name: str) -> tuple[str, dict[str, float]]:
        """(picked option, probability per option) of a Choice question."""
        a = self.answers[name]
        return str(a["choice"]), {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}

    def score(self, name: str) -> tuple[float, dict[int, float]]:
        """(probability-weighted level, probability per level index) of a Score question."""
        a = self.answers[name]
        return float(a["score"]), {int(k): float(v) for k, v in (a.get("probabilities") or {}).items()}


def _validate_answers(provider: str, questions: dict, answers: dict) -> None:
    """A missing or malformed Noul is an error, never a silent 0.0: a 0.0 injection score would let a
    passage through the screen."""
    for name, q in questions.items():
        a = answers.get(name)
        if not isinstance(a, dict):
            raise ProviderError(provider, f"no answer for question {name!r}")
        if q.get("type") == "noul":
            v = a.get("noul")
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= float(v) <= 1.0:
                raise ProviderError(provider, f"bad noul for question {name!r}")
        elif q.get("type") == "choice":
            options = set((q.get("criteria") or {}).keys())
            probs = a.get("probabilities")
            if a.get("choice") not in options or not isinstance(probs, dict) or not probs:
                raise ProviderError(provider, f"bad choice for question {name!r}")
            if set(probs) - options or not all(isinstance(p, (int, float)) and not isinstance(p, bool)
                                               and 0.0 <= float(p) <= 1.0 for p in probs.values()):
                raise ProviderError(provider, f"bad choice probabilities for question {name!r}")
        elif q.get("type") == "score":
            top = len(q.get("criteria") or []) - 1
            v = a.get("score")
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= float(v) <= max(top, 0) + 1e-6:
                raise ProviderError(provider, f"bad score for question {name!r}")


class JevClient:
    name = "jev"

    def __init__(self, meter: SpendMeter, key: str | None, model: str = "jev-1.13.0",
                 http: httpx.Client | None = None, max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep, workers: int = JEV_WORKERS):
        self.meter, self._key, self.model = meter, key, model
        self.http = http or default_http()
        self.max_retries, self.sleep, self.workers = max_retries, sleep, workers

    @staticmethod
    def projected(state: dict, questions: dict) -> float:
        return jev_cost(est_tokens(json.dumps(state) + json.dumps(questions)) + JEV_OVERHEAD_TOKENS)

    def _call(self, state: dict, questions: dict) -> JevResult:
        j = post_json(self.http, self.name, JEV_URL, {"model": self.model, "state": state, "questions": questions},
                      {"Authorization": f"Bearer {self._key}"}, self.max_retries, self.sleep)
        answers = j.get("answers") or {}
        _validate_answers(self.name, questions, answers)
        u = j.get("usage") or {}
        return JevResult(answers, str(j.get("model") or self.model),
                         int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)))

    def ask(self, state: dict, questions: dict) -> JevResult:
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, requests: list[tuple[dict, dict]]) -> list[JevResult]:
        """One Jev request per (state, questions) pair, in parallel. One cap check covers the whole batch.
        Usage of the requests that succeeded is recorded even if another one fails; then the first error
        is raised."""
        if not requests:
            return []
        if not self._key:
            raise ProviderUnavailable(self.name, "no API key configured")
        projected = sum(self.projected(s, q) for s, q in requests)
        with self.meter.charge(self.name, projected) as ch:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(requests))) as ex:
                futures = [ex.submit(self._call, s, q) for s, q in requests]
            results, errors = [], []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as e:  # noqa: BLE001  (re-raised below, after usage is recorded)
                    errors.append(e)
            if results:
                ti = sum(r.input_tokens for r in results)
                to = sum(r.output_tokens for r in results)
                ch.record(jev_cost(ti, to), tokens_in=ti, tokens_out=to, calls=len(results))
            if errors:
                raise errors[0]
        return results


# ---------------------------------------------------------------- Haiku

@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class HaikuClient:
    name = "haiku"

    def __init__(self, meter: SpendMeter, key: str | None, model: str = "claude-haiku-5-5",
                 http: httpx.Client | None = None, max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep):
        self.meter, self._key, self.model = meter, key, model
        self.http = http or default_http()
        self.max_retries, self.sleep = max_retries, sleep

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        """Messages API. Haiku 5.5: thinking disabled, and no temperature (a non-default value is rejected)."""
        if not self._key:
            raise ProviderUnavailable(self.name, "no API key configured")
        body: dict = {"model": self.model, "max_tokens": max_tokens, "thinking": {"type": "disabled"},
                      "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system
        with self.meter.charge(self.name, haiku_cost(est_tokens(system + user), max_tokens)) as ch:
            j = post_json(self.http, self.name, ANTHROPIC_URL, body,
                          {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION},
                          self.max_retries, self.sleep)
            u = j.get("usage") or {}
            ti, to = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
            ch.record(haiku_cost(ti, to), tokens_in=ti, tokens_out=to)
        text = "".join(b.get("text", "") for b in j.get("content") or [] if b.get("type") == "text").strip()
        return Completion(text, str(j.get("model") or self.model), j.get("stop_reason"), ti, to)


# ---------------------------------------------------------------- offline stubs

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are at be did do does for from had has have how i in is it me my of on or so "
                  "that the their there this to was we were what when where which who why will with you your".split())
INJECTION_MARKERS = ("ignore previous instructions", "system prompt")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class StubEmbedder:
    """Deterministic hashed bag-of-words vectors (unit length), so the dense path runs offline."""
    name = "voyage"
    model = "stub-bow"

    def __init__(self, meter: SpendMeter, dim: int = 1024):
        self.meter, self.dim = meter, dim

    def vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0.0:  # pgvector's cosine distance is undefined for a zero vector
            v[0] = 1.0
            return v
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.meter.check(voyage_cost(sum(est_tokens(t) for t in texts)))
        return [self.vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.meter.check(voyage_cost(est_tokens(text)))
        return self.vector(text)


def stub_noul(name: str, state: dict) -> float:
    question = str(state.get("question") or state.get("query") or "")
    passage = str(state.get("candidate_passage") or state.get("passage") or "")
    if name == "answers_query":
        q = {w for w in _words(question) if w not in _STOP}
        return round(len(q & set(_words(passage))) / len(q), 4) if q else 0.0
    if name in ("contains_prompt_injection", "subverts_user"):
        p = passage.lower()
        return 0.9 if any(m in p for m in INJECTION_MARKERS) else 0.05
    return 0.5


def stub_choice(question: dict, state: dict) -> dict:
    """The option whose description shares the most words with the state's question (ties: first option),
    with probability 0.6 and the rest spread evenly. Deterministic; for offline tests only."""
    text = set(_words(str(state.get("question") or "")))
    options = list((question.get("criteria") or {}).items())
    scores = [len(text & set(_words(f"{k} {d or ''}"))) for k, d in options]
    pick = options[scores.index(max(scores))][0]
    rest = 0.4 / max(1, len(options) - 1)
    return {"choice": pick, "probabilities": {k: (0.6 if k == pick else rest) for k, _ in options}}


class StubJev:
    """answers_query = share of the question's content words found in the passage; contains_prompt_injection
    = 0.9 when the passage contains "ignore previous instructions" or "system prompt", else 0.05; any other
    Noul = 0.5."""
    name = "jev"
    model = "stub-jev"

    def __init__(self, meter: SpendMeter):
        self.meter = meter

    def ask(self, state: dict, questions: dict) -> JevResult:
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, requests: list[tuple[dict, dict]]) -> list[JevResult]:
        if not requests:
            return []
        self.meter.check(sum(JevClient.projected(s, q) for s, q in requests))
        def one(n, qq, s):
            if qq.get("type") == "choice":
                return stub_choice(qq, s)
            if qq.get("type") == "score":       # offline: always the lowest level
                return {"score": 0.0, "probabilities": {"0": 1.0}}
            return {"noul": stub_noul(n, s)}
        return [JevResult({n: one(n, qq, s) for n, qq in q.items()}, self.model) for s, q in requests]


class StubHaiku:
    """Echoes the top excerpt (the first line after "Excerpts:" in build_answer_user's format)."""
    name = "haiku"
    model = "stub-haiku"

    def __init__(self, meter: SpendMeter):
        self.meter = meter

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        self.meter.check(haiku_cost(est_tokens(system + user), max_tokens))
        body = user.split("Excerpts:\n", 1)[1] if "Excerpts:\n" in user else user
        first = body.split("\n", 1)[0].strip()
        text = first if first and first != "(no excerpts)" else "UNKNOWN"
        return Completion(text, self.model, "end_turn")


# ---------------------------------------------------------------- bundle

@dataclass
class Providers:
    embedder: VoyageEmbedder | StubEmbedder
    jev: JevClient | StubJev
    haiku: HaikuClient | StubHaiku
    meter: SpendMeter
    offline: bool
    _http: httpx.Client | None = field(default=None, repr=False)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()


def make_providers(settings, meter: SpendMeter, http: httpx.Client | None = None) -> Providers:
    """Stubs when settings.offline; otherwise the real clients. A real client with no key raises
    ProviderUnavailable when called, so the service still starts and degrades."""
    if settings.offline:
        return Providers(StubEmbedder(meter, settings.voyage_dim), StubJev(meter), StubHaiku(meter), meter, True)
    owned = http is None
    http = http or default_http()
    r = settings.provider_retries
    return Providers(
        VoyageEmbedder(meter, settings.voyage_key, settings.voyage_model, settings.voyage_dim, http, r),
        JevClient(meter, settings.jev_key, settings.jev_model, http, r),
        HaikuClient(meter, settings.anthropic_key, settings.answer_model, http, r),
        meter, False, http if owned else None)
