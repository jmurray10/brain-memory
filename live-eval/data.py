"""Benchmark adapters. Every benchmark becomes a list of Question objects with:
- haystack: all sessions the retrieval arm searches
- oracle_sessions: the evidence sessions only (oracle arm)
Gold fields (gold, qtype, evidence ids) are for scoring only, never for retrieval or routing.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class Turn:
    tid: str
    role: str
    speaker: str
    text: str
    has_answer: bool = False


@dataclass
class Session:
    sid: str
    date: str
    turns: list[Turn]


@dataclass
class Question:
    bench: str
    qid: str
    qtype: str
    question: str
    gold: str
    question_date: str
    is_abs: bool
    haystack: list[Session]
    oracle_sessions: list[Session]
    evidence_session_ids: list[str] = field(default_factory=list)
    evidence_turn_ids: list[str] = field(default_factory=list)


# ---------- download with pinned hashes ----------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class DownloadIncomplete(RuntimeError):
    pass


def ensure_lme(which: str, max_seconds: float | None = None) -> Path:
    """Download with HTTP Range resume. With max_seconds, stops early and raises DownloadIncomplete;
    call again to continue."""
    import time
    name, want = config.LME_FILES[which]
    config.DATA.mkdir(parents=True, exist_ok=True)
    p = config.DATA / name
    if not p.exists():
        tmp = p.with_suffix(".part")
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(config.LME_BASE + name, headers={"Range": f"bytes={have}-"} if have else {})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:  # server ignored Range: start over
                have = 0
                tmp.unlink(missing_ok=True)
            with open(tmp, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    if max_seconds and time.time() - t0 > max_seconds:
                        raise DownloadIncomplete(f"{name}: {tmp.stat().st_size} bytes so far")
        tmp.rename(p)
    got = sha256_file(p)
    if got != want:
        raise RuntimeError(f"{name}: sha256 {got} != pinned {want}")
    return p


# ---------- LongMemEval ----------

def _lme_sessions(inst: dict) -> list[Session]:
    out = []
    ids = inst.get("haystack_session_ids") or []
    dates = inst.get("haystack_dates") or []
    for i, (sid, sess) in enumerate(zip(ids, inst.get("haystack_sessions") or [])):
        turns = []
        for t_idx, t in enumerate(sess or []):
            if not isinstance(t, dict):
                continue
            role = t.get("role") or "unknown"
            text = t.get("content") or ""
            turns.append(Turn(f"{sid}:{t_idx}", role, role, str(text), bool(t.get("has_answer"))))
        out.append(Session(str(sid), str(dates[i]) if i < len(dates) else "", turns))
    return out


def lme_question(s_inst: dict, oracle_inst: dict | None) -> Question:
    qid = s_inst["question_id"]
    haystack = _lme_sessions(s_inst)
    oracle = _lme_sessions(oracle_inst) if oracle_inst else []
    ev_turns = [t.tid for s in haystack for t in s.turns if t.has_answer]
    return Question(
        bench="lme",
        qid=qid,
        qtype=s_inst["question_type"],
        question=s_inst["question"],
        gold=str(s_inst["answer"]),
        question_date=str(s_inst.get("question_date") or ""),
        is_abs=qid.endswith("_abs"),
        haystack=haystack,
        oracle_sessions=oracle,
        evidence_session_ids=[str(x) for x in (s_inst.get("answer_session_ids") or [])],
        evidence_turn_ids=ev_turns,
    )


def load_lme(qids: list[str] | None = None) -> list[Question]:
    s = json.loads(ensure_lme("s").read_text(encoding="utf-8"))
    o = {r["question_id"]: r for r in json.loads(ensure_lme("oracle").read_text(encoding="utf-8"))}
    by_id = {r["question_id"]: r for r in s}
    ids = qids if qids is not None else list(by_id)
    return [lme_question(by_id[q], o.get(q)) for q in ids]


# ---------- LoCoMo ----------

LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}  # cat 5 excluded


def _locomo_sessions(conv: dict) -> list[Session]:
    n = 1
    out = []
    while f"session_{n}" in conv:
        date = str(conv.get(f"session_{n}_date_time") or "")
        turns = []
        for t in conv[f"session_{n}"]:
            text = str(t.get("text") or "")
            if t.get("blip_caption"):
                text += f" [shares an image: {t['blip_caption']}]"
            turns.append(Turn(str(t["dia_id"]), "user", str(t.get("speaker") or ""), text))
        out.append(Session(f"session_{n}", date, turns))
        n += 1
    return out


def load_locomo(sample_ids: list[str] | None = None, limit_per_conv: int | None = None) -> list[Question]:
    data = json.loads(config.LOCOMO_PATH.read_text(encoding="utf-8"))
    out = []
    for c in data:
        if sample_ids and c["sample_id"] not in sample_ids:
            continue
        sessions = _locomo_sessions(c["conversation"])
        turn_session = {t.tid: s.sid for s in sessions for t in s.turns}
        last_date = sessions[-1].date if sessions else ""
        qas = [qa for qa in c["qa"] if qa.get("category") in LOCOMO_CATS]
        if limit_per_conv:
            qas = qas[:limit_per_conv]
        for i, qa in enumerate(qas):
            ev = []
            for e in qa.get("evidence") or []:
                ev.extend(x.strip() for x in str(e).split(";") if x.strip())
            ev_sess = sorted({turn_session[e] for e in ev if e in turn_session})
            out.append(Question(
                bench="locomo",
                qid=f"{c['sample_id']}:{i}",
                qtype=LOCOMO_CATS[qa["category"]],
                question=qa["question"],
                gold=str(qa.get("answer")),
                question_date=last_date,
                is_abs=False,
                haystack=sessions,
                oracle_sessions=[s for s in sessions if s.sid in ev_sess],
                evidence_session_ids=ev_sess,
                evidence_turn_ids=[e for e in ev if e in turn_session],
            ))
    return out
