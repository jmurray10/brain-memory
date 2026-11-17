"""exp-recall-floor-v1: can a rerank score mark a weak passage without marking the evidence?

    python scripts/recall_floor.py dev             # conversations 0, 2, 4, 6, 8: picks the threshold
    python scripts/recall_floor.py held --t 0.15   # conversations 1, 3, 5, 7, 9: the gate at that threshold

Each LoCoMo conversation is captured into a database of its own on the Supabase test branch: the turns only (no
claims, no write path, no night; recall reads none of them). Each question is then asked of `core.recall` as a
client would (k=10, the shipped settings). For every passage returned the run keeps its rerank score and whether
its turn is one of the question's evidence turns. No reader and no judge: this measures the scores, not answers.
Also counted, with no gate: how the injection screen scored these ordinary turns. Writes
runs/recall_floor_<half>_<stamp>/. Cap $2.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DB = "brain_floor"
CAP = 2.0
PER_CONV = 30
GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
PICK_EVIDENCE_AT_MOST = 0.02     # dev: the largest threshold marking at most this share of evidence passages
GATE_EVIDENCE_AT_MOST = 0.03     # held-out: evidence passages marked, at most
GATE_OTHER_AT_LEAST = 0.40       # held-out: other passages marked, at least
HALVES = {"dev": 0, "held": 1}   # a conversation's position in the file, modulo 2


def floor_table(rows: list[dict], grid=GRID) -> list[dict]:
    """For each threshold: how many returned evidence passages and how many other passages score below it, and
    how many questions would have every evidence passage they were handed marked. A row is one question:
    {"items": [{"rerank": float, "evidence": bool}]}; rows without a rerank (degraded recall) are left out by
    the caller."""
    ev = [it["rerank"] for r in rows for it in r["items"] if it["evidence"]]
    other = [it["rerank"] for r in rows for it in r["items"] if not it["evidence"]]
    with_ev = [[it["rerank"] for it in r["items"] if it["evidence"]] for r in rows]
    with_ev = [s for s in with_ev if s]
    out = []
    for t in grid:
        e, o = sum(s < t for s in ev), sum(s < t for s in other)
        out.append({"t": t, "evidence_below": e, "evidence": len(ev),
                    "evidence_share": round(e / len(ev), 4) if ev else None,
                    "other_below": o, "other": len(other),
                    "other_share": round(o / len(other), 4) if other else None,
                    "questions_all_evidence_below": sum(all(s < t for s in q) for q in with_ev),
                    "questions_with_evidence": len(with_ev)})
    return out


def pick(table: list[dict], at_most: float = PICK_EVIDENCE_AT_MOST) -> float | None:
    """The largest threshold that marks at most `at_most` of the evidence passages; None if none does."""
    ok = [row["t"] for row in table if row["evidence_share"] is not None and row["evidence_share"] <= at_most]
    return max(ok) if ok else None


def gate(table: list[dict], t: float) -> dict:
    """The held-out rule at the threshold the dev half picked: few evidence passages marked, many others."""
    row = next(r for r in table if abs(r["t"] - t) < 1e-9)
    ok = (row["evidence_share"] is not None and row["evidence_share"] <= GATE_EVIDENCE_AT_MOST
          and row["other_share"] is not None and row["other_share"] >= GATE_OTHER_AT_LEAST)
    return {"t": t, "evidence_share": row["evidence_share"], "other_share": row["other_share"], "passed": ok}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in HALVES:
        sys.exit(__doc__)
    half = sys.argv[1]
    t_held = float(sys.argv[sys.argv.index("--t") + 1]) if "--t" in sys.argv else None
    per_conv = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else PER_CONV
    if half == "held" and t_held is None:
        sys.exit("the held-out half needs the threshold the dev half picked: --t 0.15")

    import psycopg

    import env
    env.load_env()
    from locomo_eval import category, gold_turns, session_dt
    from testdb import admin_url, db_url

    from brain import core, worker
    from brain.auth import Caller
    from brain.db import make_pool, migrate, tenant
    from brain.providers import ProviderError, make_providers
    from brain.schemas import TurnIn
    from brain.settings import load
    from brain.spend import PgSpendStore, SpendMeter

    def fresh(name: str) -> str:
        with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
            c.execute(f"drop database if exists {name} with (force)")
            c.execute(f"create database {name}")
        u = db_url(name)
        migrate(u)
        return u

    def drop(name: str) -> None:
        with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
            c.execute(f"drop database if exists {name} with (force)")

    convs = json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8"))
    chosen = [(i, c) for i, c in enumerate(convs) if i % 2 == HALVES[half]]
    meter_pool = make_pool(fresh(DB), max_size=2)
    meter = SpendMeter(PgSpendStore(meter_pool), CAP)
    prov = make_providers(load(), meter)
    try:
        prov.jev.ask({"text": "We moved last spring."},
                     {"q": {"type": "noul", "instructions": "Does the text mention a move?"}})
    except ProviderError as e:
        sys.exit(f"Jev preflight failed, nothing run: {e}")

    def one(item):
        i, conv = item
        qas = [(k, qa) for k, qa in enumerate(conv["qa"]) if category(qa)]
        qas = sorted(qas, key=lambda x: hashlib.sha256(f"{conv['sample_id']}:{x[0]}".encode()).hexdigest())[:per_conv]
        c = conv["conversation"]
        name = f"{DB}_{i:02d}"
        pool = make_pool(fresh(name), max_size=2)
        rows = []
        try:
            with pool.connection() as cn:
                owner = cn.execute("select id from brain.account where name = 'owner'").fetchone()[0]
            caller = Caller("eval", ("*",), ("*",), None, True, owner, "owner", False, True)
            turn_at = {}
            n = 1
            while f"session_{n}" in c:
                base = session_dt(c[f"session_{n}_date_time"])
                turns = []
                for k, t in enumerate(c[f"session_{n}"]):
                    text = str(t.get("text") or "")
                    if t.get("blip_caption"):
                        text += f" [shares an image: {t['blip_caption']}]"
                    if not text.strip():
                        continue
                    ts = base + timedelta(seconds=k)
                    turns.append(TurnIn(role="user", speaker=str(t.get("speaker") or "user"), text=text, ts=ts))
                    turn_at[ts] = t.get("dia_id")
                for b in range(0, len(turns), 100):
                    with tenant(pool, owner) as conn:
                        core.capture(conn, prov, caller, "u", turns[b:b + 100], f"session_{n}", None)
                n += 1
            worker.embed_backfill(pool, prov.embedder, 100_000)
            for k, qa in qas:
                gold = gold_turns(qa.get("evidence"))
                with tenant(pool, owner) as conn:
                    r = core.recall(conn, prov, caller, "u", qa["question"], k=10)
                items = [{"rerank": it["scores"]["rerank"], "injection": it["scores"]["injection"],
                          "turn": turn_at.get(datetime.fromisoformat(it["ts"]).astimezone(timezone.utc))}
                         for it in r["items"]]
                for it in items:
                    it["evidence"] = it["turn"] in gold
                rows.append({"qid": f"{conv['sample_id']}:{k}", "cat": category(qa), "evidence": gold,
                             "degraded": r["degraded"], "screened": r["screened"], "items": items})
            with tenant(pool, owner) as conn:
                screen = conn.execute(
                    """select count(*), count(*) filter (where probability > threshold),
                              count(*) filter (where outcome = 'dropped')
                       from brain.decision_log where stage = 'recall.injection'""").fetchone()
        finally:
            pool.close()
            drop(name)
        print(f"conversation {i}: {len(rows)} questions", flush=True)
        return rows, screen

    rows, screen = [], [0, 0, 0]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for rs, sc in ex.map(one, chosen):
            rows += rs
            screen = [a + b for a, b in zip(screen, sc)]

    scored = [r for r in rows if not r["degraded"] and all(it["rerank"] is not None for it in r["items"])]
    table = floor_table(scored)
    summary = {"half": half, "conversations": [i for i, _ in chosen], "questions": len(rows),
               "scored": len(scored), "degraded": len(rows) - len(scored), "table": table,
               "screen_on_ordinary_turns": {"candidates_scored": screen[0], "injection_above_threshold": screen[1],
                                            "dropped": screen[2]},
               "spend_usd": round(meter.spent_today(), 4)}
    if half == "dev":
        summary["picked_t"] = pick(table)
    else:
        summary["gate"] = gate(table, t_held)
    out = SERVICE / "runs" / f"recall_floor_{half}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    code = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            for f in [SERVICE / "brain/core.py", Path(__file__).resolve()]}
    (out / "config.json").write_text(json.dumps({"half": half, "per_conversation": per_conv, "t": t_held,
                                                 "cap": CAP, "hashes": code}, indent=2), encoding="utf-8")
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)
    prov.close()
    meter_pool.close()
    drop(DB)
    return 0


if __name__ == "__main__":
    sys.exit(main())
