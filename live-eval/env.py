"""Load API keys from the repo-root .env into os.environ. Never prints or logs values."""
from __future__ import annotations

import os
from pathlib import Path

LIVE_EVAL = Path(__file__).resolve().parent
REPO_ROOT = LIVE_EVAL.parents[1]  # live-eval -> brain-v2 -> repo


def load_env(path: str | Path | None = None) -> list[str]:
    """Read KEY=VALUE lines. Existing environment variables win. Returns the names loaded."""
    p = Path(path) if path else REPO_ROOT / ".env"
    if not p.exists():
        return []
    names = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[len("export "):]
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            names.append(k)
    return names


def anthropic_key() -> str | None:
    for n in ("ANTHROPIC_API_KEY",):
        if os.environ.get(n):
            return os.environ[n]
    return None


def voyage_key() -> str | None:
    for n in ("VOYAGE_API_KEY_NEW",
              "VOYAGE_API_KEY_TEST", "VOYAGE_API_KEY"):
        if os.environ.get(n):
            return os.environ[n]
    return None


def jev_key() -> str | None:
    for n in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        if os.environ.get(n):
            return os.environ[n]
    return None


def key_status() -> dict:
    """Which providers have a key. Booleans only."""
    return {"anthropic": bool(anthropic_key()), "voyage": bool(voyage_key()), "jev": bool(jev_key())}
