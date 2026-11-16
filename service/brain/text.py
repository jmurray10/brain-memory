"""Text handling ported from ../live-eval/text.py (the measured chunker, packer and RRF). The Docker image only
contains service/, so the functions are copied here; tests/test_text.py checks they match the originals."""
from __future__ import annotations

import re
from datetime import datetime

from .spend import est_tokens

CHUNK_MAX_CHARS = 1500      # live-eval config.CHUNK_MAX_CHARS
RRF_K = 60


def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split on paragraph, then sentence boundaries, into windows <= max_chars. Concatenation keeps all text."""
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=\n)|(?<=[.!?] )", text)
    out, cur = [], ""
    for p in parts:
        if not p:
            continue
        while len(p) > max_chars:  # a single huge sentence
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_chars])
            p = p[max_chars:]
        if len(cur) + len(p) > max_chars and cur:
            out.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        out.append(cur)
    return out


def render(ts: datetime, role: str, speaker: str | None, text: str) -> str:
    """'[YYYY-MM-DD] speaker: text' (speaker, else role), the form recall was measured with (Chunk.render).
    The date is the turn's own calendar date in the offset the client sent."""
    if role == "agent_reported":
        # What an agent says the user said (migration 014). The speaker field is the client's own text, so it
        # never replaces the label: a turn sent with speaker "user" must not read as the user's own words.
        who = f"agent_reported (as {speaker})" if speaker and speaker != role else role
    else:
        who = speaker if speaker and speaker != role else role
    return f"[{ts.date().isoformat()}] {who}: {text}"


def rrf(rank_lists: list[list], k: int = RRF_K) -> list:
    """Reciprocal rank fusion, 0-based ranks: score 1 / (k + r + 1). Reference for the SQL fusion in core.py."""
    s: dict = {}
    for lst in rank_lists:
        for r, cid in enumerate(lst):
            s[cid] = s.get(cid, 0.0) + 1.0 / (k + r + 1)
    return [c for c, _ in sorted(s.items(), key=lambda x: -x[1])]


def pack(texts: list[str], max_items: int, budget_tokens: int) -> list[int]:
    """Indexes of the texts kept, in the given order, until max_items; skip (not stop at) any that would
    overflow the budget (live-eval pack())."""
    kept, used = [], 0
    for i, t in enumerate(texts):
        if len(kept) >= max_items:
            break
        n = est_tokens(t)
        if used + n > budget_tokens:
            continue
        kept.append(i)
        used += n
    return kept
