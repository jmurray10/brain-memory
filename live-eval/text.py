"""Chunking, packing, BM25. No network."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

STOP = set("a an the and or but of to in on at for with by from is are was were be been am i you he she it we they "
           "me my your his her its our their this that these those do did does what when where who whom which how "
           "why about as if so not no yes can could would should will just".split())


def tokenize(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def est_tokens(s: str) -> int:
    return max(1, len(s) // 4)


@dataclass
class Chunk:
    cid: str
    sid: str
    s_idx: int      # session order in its haystack
    t_idx: int      # turn order within session
    piece: int      # window index within a long turn
    date: str
    role: str
    speaker: str
    tid: str
    text: str
    has_answer: bool

    def render(self) -> str:
        who = self.speaker if self.speaker and self.speaker != self.role else self.role
        return f"[{self.date}] {who}: {self.text}"


def split_text(text: str, max_chars: int) -> list[str]:
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


def build_chunks(sessions, max_chars: int) -> list[Chunk]:
    out = []
    for s_idx, s in enumerate(sessions):
        for t_idx, t in enumerate(s.turns):
            for k, piece in enumerate(split_text(t.text, max_chars)):
                out.append(Chunk(f"{t.tid}#{k}", s.sid, s_idx, t_idx, k, s.date, t.role, t.speaker,
                                 t.tid, piece, t.has_answer))
    return out


def pack(chunks: list[Chunk], max_items: int, budget_tokens: int) -> tuple[list[Chunk], int]:
    """Keep chunks in the given order until max_items; skip (not stop at) any that would overflow the budget.
    Returns (packed, n_skipped)."""
    packed, used, skipped = [], 0, 0
    for c in chunks:
        if len(packed) >= max_items:
            break
        t = est_tokens(c.render())
        if used + t > budget_tokens:
            skipped += 1
            continue
        packed.append(c)
        used += t
    return packed, skipped


def chronological(chunks: list[Chunk]) -> list[Chunk]:
    return sorted(chunks, key=lambda c: (c.s_idx, c.t_idx, c.piece))


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.avg = sum(len(d) for d in docs) / max(1, self.n)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {w: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out

    def topk(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return [i for i in order[:k] if sc[i] > 0]


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    s: dict[str, float] = {}
    for lst in rank_lists:
        for r, cid in enumerate(lst):
            s[cid] = s.get(cid, 0.0) + 1.0 / (k + r + 1)
    return [c for c, _ in sorted(s.items(), key=lambda x: -x[1])]
