"""Stress phase B: retrieval and answers in complex-domain worlds with exact answers.

Each world (evals/worlds/*.json) is loaded through the running API (python -m brain.serve, real Voyage, Jev and
Haiku) into its own namespace, turn by turn with its role, channel, speaker and date. For every question:
- recall with k = 30, packed like the live-eval iso arm (10 chunks within 6,000 tokens): evidence turn recall and
  whether all evidence turns were kept;
- POST /v1/answer (the brain's own answer path: recall k = 10, measured prompts, Haiku);
- a Haiku judge compares the answer with the gold answer, majority of 3 verdicts ("not mentioned" questions are
  correct when the answer says the information is not there).

    python stress/worlds_eval.py            # paid, capped at $5 (expected well under $1)

Writes runs/worlds_<stamp>/ (config, rows.jsonl, summary.json; server.log stays local).
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg

SERVICE = Path(__file__).resolve().parents[1]
WORLDS = SERVICE / "evals" / "worlds"
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))
from brain.text import pack  # noqa: E402

sys.path.insert(0, str(SERVICE / "scripts"))
from testdb import admin_url, db_url  # noqa: E402
DB = "brain_worlds"
PORT = 8770
CAP = 5.0
ROLES = {"user", "assistant", "system", "tool", "attachment"}
CHANNELS = {"chat", "voice", "email", "meeting", "agent_log", "notes", "document", "image"}

JUDGE_SYSTEM = ("You grade answers from a memory assistant. Reply with exactly one word: yes or no.")


def judge_user(q: dict, response: str) -> str:
    if q["answer"].strip().lower() in ("not mentioned", "not mentioned."):
        rule = ("The correct behavior is to say the information is not available or was never mentioned. "
                "Answer yes only if the response does that and does not invent an answer.")
    else:
        rule = ("Answer yes if the response states the gold answer (equivalent wording, units or formatting is "
                "fine; extra correct detail is fine). Answer no if it is missing, wrong, or hedges between options.")
    return f"Question: {q['question']}\nGold answer: {q['answer']}\nResponse: {response}\n\n{rule}"


def ns_of(world: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "-", world.lower())[:63].strip("-") or "world"


def load_worlds() -> list[dict]:
    out = []
    for f in sorted(WORLDS.glob("*.json")):
        w = json.loads(f.read_text(encoding="utf-8"))
        w["_file"] = f.name
        for s in w["sessions"]:
            for t in s["turns"]:
                if t["role"] not in ROLES or t["channel"] not in CHANNELS:
                    sys.exit(f"{f.name} {t['tid']}: role {t['role']!r} / channel {t['channel']!r} not allowed")
        out.append(w)
    return out


def capture_world(h: httpx.Client, w: dict) -> dict[int, str]:
    ns, tid_of = ns_of(w["world"]), {}
    for s in w["sessions"]:
        base = datetime.fromisoformat(s["date"]).replace(hour=9, tzinfo=timezone.utc)
        turns = [{"role": t["role"], "channel": t["channel"], "text": t["text"], "speaker": t.get("speaker") or None,
                  "ts": (base + timedelta(minutes=i)).isoformat(), "meta": {"tid": t["tid"]},
                  "idempotency_key": t["tid"]} for i, t in enumerate(s["turns"])]
        for i in range(0, len(turns), 50):
            r = h.post("/v1/capture", json={"namespace": ns, "session_id": s["session_id"], "turns": turns[i:i + 50]})
            r.raise_for_status()
            for ep, t in zip(r.json()["episodes"], turns[i:i + 50]):
                tid_of[ep["id"]] = t["meta"]["tid"]
    return tid_of


def main() -> int:
    import env
    env.load_env()
    from brain.providers import HaikuClient
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    worlds = load_worlds()
    if not worlds:
        sys.exit("no worlds in evals/worlds")
    s = load()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = SERVICE / "runs" / f"worlds_{stamp}"
    run.mkdir(parents=True)
    with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
        c.execute(f"drop database if exists {DB} with (force)")
        c.execute(f"create database {DB}")
    admin = "worlds-" + secrets.token_hex(20)
    env_srv = {**os.environ, "DATABASE_URL": db_url(DB), "BRAIN_OFFLINE": "0",
               "BRAIN_ADMIN_KEY": admin, "BRAIN_DAILY_CAP_USD": str(CAP), "BRAIN_RATE_PER_MIN": "100000",
               "PORT": str(PORT), "HOST": "127.0.0.1", "BRAIN_LOG_LEVEL": "WARNING"}
    log = open(run / "server.log", "w")
    srv = subprocess.Popen([sys.executable, "-m", "brain.serve"], cwd=SERVICE, env=env_srv, stdout=log,
                           stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{PORT}"
    judge = HaikuClient(SpendMeter(MemorySpendStore(), 1.0), s.anthropic_key, s.answer_model)
    rows = []
    try:
        for _ in range(120):
            try:
                if httpx.get(base + "/healthz", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        with httpx.Client(base_url=base, timeout=120, headers={"Authorization": f"Bearer {admin}"}) as op:
            key = op.post("/v1/admin/accounts", json={"name": "worlds"}).json()["key"]
        with httpx.Client(base_url=base, timeout=180, headers={"Authorization": f"Bearer {key}"}) as h:
            for w in worlds:
                ns = ns_of(w["world"])
                tid_of = capture_world(h, w)
                print(f"{w['world']}: {len(tid_of)} turns captured", flush=True)

                def one(q):
                    rec = h.post("/v1/recall", json={"namespace": ns, "query": q["question"], "k": 30}).json()
                    items = rec["items"]
                    kept = [items[i] for i in pack([it["text"] for it in items], 10, 6000)]
                    got = {tid_of.get(it["episode_id"]) for it in kept}
                    ev = set(q["evidence"])
                    ans = h.post("/v1/answer", json={"namespace": ns, "question": q["question"],
                                                     "question_date": q["question_date"]})
                    text = ans.json().get("answer", "") if ans.status_code == 200 else f"[HTTP {ans.status_code}]"
                    votes = []
                    for _ in range(5):
                        if len([v for v in votes if v is not None]) >= 3:
                            break
                        out = judge.complete(JUDGE_SYSTEM, judge_user(q, text), max_tokens=5).text.strip().lower()
                        votes.append(True if out.startswith("yes") else False if out.startswith("no") else None)
                    valid = [v for v in votes if v is not None][:3]
                    return {"world": w["world"], "domain": w.get("domain"), "qid": q["qid"], "type": q["type"],
                            "question": q["question"], "gold": q["answer"], "answer": text,
                            "evidence": len(ev), "evidence_kept": len(ev & got),
                            "turn_recall": (len(ev & got) / len(ev)) if ev else None,
                            "all_evidence": (ev <= got) if ev else None, "correct": sum(valid) >= 2,
                            "votes": votes, "degraded": rec.get("degraded"), "dense": rec.get("dense")}
                with cf.ThreadPoolExecutor(4) as pool:
                    rows += list(pool.map(one, w["questions"]))
            with httpx.Client(base_url=base, timeout=60, headers={"Authorization": f"Bearer {admin}"}) as op:
                spend = op.get("/v1/admin/spend", params={"days": 1}).json()
    finally:
        srv.terminate()
        try:
            srv.wait(20)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()

    def mean(rs, k):
        v = [float(r[k]) for r in rs if r[k] is not None]
        return round(sum(v) / len(v), 4) if v else None
    summary = {"questions": len(rows), "answer_accuracy": mean(rows, "correct"),
               "turn_recall": mean(rows, "turn_recall"), "all_evidence": mean(rows, "all_evidence"),
               "by_world": {w: {"n": len(rs), "accuracy": mean(rs, "correct"), "turn_recall": mean(rs, "turn_recall")}
                            for w in sorted({r["world"] for r in rows}) for rs in [[r for r in rows if r["world"] == w]]},
               "by_type": {t: {"n": len(rs), "accuracy": mean(rs, "correct"), "turn_recall": mean(rs, "turn_recall")}
                           for t in sorted({r["type"] for r in rows}) for rs in [[r for r in rows if r["type"] == t]]},
               "degraded": sum(1 for r in rows if r["degraded"]), "judge_parse_fail": sum(
                   1 for r in rows if len([v for v in r["votes"] if v is not None]) < 3),
               "spend": {"service_usd": spend.get("today_usd"), "judge_usd": round(judge.meter.spent_today(), 6)}}
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(WORLDS.glob("*.json"))}
    hashes["worlds_eval.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    (run / "config.json").write_text(json.dumps({"hashes": hashes, "cap": CAP, "judge": s.answer_model,
                                                 "votes": 3}, indent=2), encoding="utf-8")
    (run / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", run)
    with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
        c.execute(f"drop database {DB} with (force)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
