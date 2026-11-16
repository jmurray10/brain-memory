"""API clients (Anthropic, Voyage, Jev) with spend metering, plus offline stubs for dry runs."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import config
from env import REPO_ROOT, anthropic_key, voyage_key
from prices import SpendMeter, haiku_cost
from text import est_tokens

MAX_RETRY = int(os.environ.get("MAX_RETRY", "6"))


def _post(url: str, body: dict, headers: dict, timeout: int = 120) -> dict:
    data = json.dumps(body).encode()
    last = None
    for a in range(1 + MAX_RETRY):
        try:
            req = urllib.request.Request(url, data, {"content-type": "application/json", **headers})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            last = RuntimeError(f"HTTP {e.code} from {url.split('/')[2]}: {msg!r}")
            if e.code in (429, 529) or e.code >= 500:
                time.sleep(min(60, 2 ** a + random.random()))
                continue
            raise last
        except Exception as e:  # network
            last = e
            time.sleep(min(60, 2 ** a))
    raise RuntimeError(f"retries exhausted: {last!r}")


# ---------------- Anthropic ----------------

class HaikuClient:
    def __init__(self, meter: SpendMeter, model: str):
        self.meter, self.model = meter, model
        self.key = anthropic_key()
        if not self.key:
            raise RuntimeError("No Anthropic key (ANTHROPIC_API_KEY)")

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.meter.check(haiku_cost(est_tokens(system + user), max_tokens))
        body = {"model": self.model, "max_tokens": max_tokens,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system
        j = _post("https://api.anthropic.com/v1/messages", body,
                  {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        u = j.get("usage") or {}
        self.meter.add_haiku(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)))
        return "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text").strip()


class StubLLM:
    """Offline stand-in: answers with the first words of the first excerpt; judges 'no'."""
    def complete(self, system: str, user: str, max_tokens: int) -> str:
        if "Answer yes or no only" in user or "CORRECT" in system:
            return "no"
        m = user.split("Excerpts:\n", 1)[-1].split("\n", 1)[0]
        return " ".join(m.split()[:8]) or "UNKNOWN - stub"


# ---------------- Voyage ----------------

class EmbedStore:
    """Embedding cache as a few batch files (one .npz per API batch), indexed in memory at startup.
    Per-text files are too slow on a mounted Windows folder (~3.5 ms per file check)."""

    def __init__(self, root):
        import threading
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        for f in sorted(self.root.glob("batch_*.npz")):
            try:
                z = np.load(f)
                for k, v in zip(z["keys"], z["vecs"]):
                    self.index[str(k)] = v
            except Exception:
                pass  # a torn write from an interrupted run; those texts get re-embedded

    def get(self, key: str):
        return self.index.get(key)

    def put_batch(self, keys: list[str], vecs: np.ndarray):
        with self._lock:
            for k, v in zip(keys, vecs):
                self.index[k] = v
            name = self.root / f"batch_{time.time_ns()}_{random.randint(0, 1 << 30)}.npz"
            tmp = name.with_suffix(".tmp.npz")
            np.savez(tmp, keys=np.array(keys), vecs=vecs.astype(np.float16))
            tmp.rename(name)


class VoyageEmbedder:
    def __init__(self, meter: SpendMeter, model: str = config.VOYAGE_MODEL, batch: int = 128):
        self.meter, self.model, self.batch = meter, model, batch
        self.key = voyage_key()
        if not self.key:
            raise RuntimeError("No Voyage key in .env")
        self.store = EmbedStore(config.EMBED_CACHE)

    def _key(self, text: str, kind: str) -> str:
        return hashlib.sha256(f"{self.model}|{kind}|{text}".encode()).hexdigest()

    def cached(self, text: str, kind: str) -> bool:
        return self.store.get(self._key(text, kind)) is not None

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        keys = [self._key(t, kind) for t in texts]
        out: list = [self.store.get(k) for k in keys]
        miss = [i for i, v in enumerate(out) if v is None]
        for s in range(0, len(miss), self.batch):
            idx = miss[s:s + self.batch]
            chunk = [texts[i][:30000] for i in idx]
            self.meter.check(sum(est_tokens(c) for c in chunk) / 1e6 * 0.02)
            j = _post("https://api.voyageai.com/v1/embeddings",
                      {"model": self.model, "input": chunk, "input_type": kind},
                      {"Authorization": "Bearer " + self.key})
            self.meter.add_voyage(int((j.get("usage") or {}).get("total_tokens", 0)))
            vecs = np.zeros((len(idx), config.VOYAGE_DIM), np.float32)
            for item in j["data"]:
                vecs[item["index"]] = np.asarray(item["embedding"], dtype=np.float32)
            self.store.put_batch([keys[i] for i in idx], vecs)
            for n, i in enumerate(idx):
                out[i] = vecs[n]
        return np.vstack([np.asarray(v, np.float32) for v in out]) if out else np.zeros((0, config.VOYAGE_DIM), np.float32)


class StubEmbedder:
    """Deterministic hashed bag-of-words vectors. Lets the dense path run offline."""
    def cached(self, text: str, kind: str) -> bool:
        return True

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        dim = 256
        out = np.zeros((len(texts), dim), np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        return out


# ---------------- Jev rerank ----------------

class JevReranker:
    def __init__(self, meter: SpendMeter, workers: int = 8):
        os.environ.setdefault("JEV_CACHE_DIR", str(config.JEV_CACHE))
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from typesafe_cookbooks import jev_client  # noqa: F401  (key check)
        from typesafe_cookbooks.rerank import rerank
        if not jev_client.api_key():
            raise RuntimeError("No TYPESAFE_API_KEY / JEV_API_KEY in .env")
        self._rerank, self.meter, self.workers = rerank, meter, workers

    def rerank(self, query: str, cands: list[dict]) -> list[dict]:
        est = sum(est_tokens(c["passage"]) + 150 for c in cands)
        self.meter.check(est / 1e6 * 0.042)
        u: dict = {}
        out = self._rerank(query, cands, workers=self.workers, usage=u)
        self.meter.add_jev(int(u.get("jev_in", 0)), int(u.get("jev_out", 0)),
                           int(u.get("jev_ok", 0)), int(u.get("jev_cache_hits", 0)))
        return out


class NoRerank:
    def rerank(self, query: str, cands: list[dict]) -> list[dict]:
        return cands
