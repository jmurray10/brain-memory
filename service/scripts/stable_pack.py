"""Spike for exp-stable-pack-v1: how much the ten passages `recall` hands over change when the same question is
asked again, and whether a coarser cut makes them steadier without losing evidence. Report only.

    python scripts/stable_pack.py collect [conversation, default conv-30] [questions, default 100] [repeats, default 4]
    python scripts/stable_pack.py rules runs/stable_pack_<stamp>

`collect` reads the database `brain_measure` that scripts/measure_shipped.py leaves behind and asks `recall` for
all 30 scored candidates of each question, `repeats` times (paid: 30 Jev requests per question and repeat; no
memo, since the point is how Jev's scores move). It writes runs/stable_pack_<stamp>/scores.json.

`rules` is free: from the stored scores it applies each way of cutting the top ten and prints, per rule,
how often two repeats of a question give the same ten, and how many of LoCoMo's evidence turns are among them."""
from __future__ import annotations

import hashlib
import itertools
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

DB = "brain_measure"
CAP = 2.0
K = 10


def top(cands: list[dict], band: float = 0.0, k: int = K) -> list[int]:
    """The k chunk ids a rule keeps. `cands` are in fused (RRF) order, each with its Jev score. band 0 is the
    shipped rule: by score, the fused order breaking exact ties. With a band, scores are first cut into steps of
    that width, so passages Jev cannot tell apart are ordered by the fused rank, which does not move."""
    from brain.core import rerank_order

    return [c["chunk_id"] for c in rerank_order([{**c, "scores": {"rerank": c["rerank"]}} for c in cands],
                                               band)[:k]]


def collect() -> None:
    from env import load_env

    load_env()
    from brain import core
    from brain.auth import Caller
    from brain.db import make_pool, tenant
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from locomo_eval import CATS, gold_turns, session_dt
    from testdb import db_url

    want = sys.argv[2] if len(sys.argv) > 2 else "conv-30"
    n_q = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    conv = next(c for c in json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8"))
                if c["sample_id"] == want)
    c, turn_at, n = conv["conversation"], {}, 1
    while f"session_{n}" in c:
        base = session_dt(c[f"session_{n}_date_time"])
        for k, t in enumerate(c[f"session_{n}"]):
            turn_at[(base + timedelta(seconds=k)).isoformat()] = t.get("dia_id")
        n += 1
    meter = SpendMeter(MemorySpendStore(), CAP)
    prov = make_providers(load(), meter)
    pool = make_pool(db_url(DB), max_size=2)
    with pool.connection() as cn:
        owner = cn.execute("select id from brain.account where name = 'owner'").fetchone()[0]
    caller = Caller("stable", ("*",), ("*",), None, True, owner, "owner", False, True)
    qas = [(k, qa) for k, qa in enumerate(conv["qa"]) if qa.get("category") in CATS]
    qas = sorted(qas, key=lambda x: hashlib.sha256(f"{want}:{x[0]}".encode()).hexdigest())[:n_q]
    rows = []
    for k, qa in qas:
        runs = []
        for _ in range(reps):
            with tenant(pool, owner) as conn:
                r = core.recall(conn, prov, caller, "u", qa["question"], k=core.SHORTLIST)
            if r["degraded"]:
                sys.exit(f"recall degraded ({r['degraded']}); nothing written")
            items = sorted(r["items"], key=lambda it: -it["scores"]["rrf"])      # back to fused order
            runs.append([{"chunk_id": it["chunk_id"], "rerank": it["scores"]["rerank"],
                          "turn": turn_at.get(datetime.fromisoformat(it["ts"]).astimezone(timezone.utc).isoformat())}
                         for it in items])
        rows.append({"q": k, "question": qa["question"], "category": qa.get("category"),
                     "evidence": gold_turns(qa.get("evidence")), "runs": runs})
    pool.close()
    run = SERVICE / "runs" / f"stable_pack_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run.mkdir(parents=True)
    (run / "scores.json").write_text(json.dumps({"conversation": want, "repeats": reps, "rows": rows,
                                                 "spend_usd": round(meter.spent_today(), 4)}), encoding="utf-8")
    print(f"{len(rows)} questions, {reps} repeats, ${meter.spent_today():.4f}\nrun dir: {run}")


def rules(run: Path) -> dict:
    data = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    rows, out = data["rows"], {}
    diffs = []
    for row in rows:
        by = [{c["chunk_id"]: c["rerank"] for c in r} for r in row["runs"]]
        for a, b in itertools.combinations(by, 2):
            diffs += [abs(a[k] - b[k]) for k in a if k in b]
    for band in (0.0, 0.02, 0.05, 0.1, 0.2, 1.01):
        same, jac, reached, gold_n, mapped = [], [], 0, 0, 0
        for row in rows:
            tops = [top(r, band) for r in row["runs"]]
            for a, b in itertools.combinations(tops, 2):
                same.append(set(a) == set(b))
                jac.append(len(set(a) & set(b)) / max(1, len(set(a) | set(b))))
            gold = set(row["evidence"])
            for r, t in zip(row["runs"], tops):
                turns = {c["turn"] for c in r if c["chunk_id"] in t}
                mapped += sum(c["turn"] is not None for c in r)
                reached += len(gold & turns)
                gold_n += len(gold)
        name = "shipped (by score)" if band <= 0 else "fused order only" if band > 1 else f"band {band}"
        out[name] = {"same_ten_between_two_asks": round(statistics.mean(same), 3),
                     "overlap_between_two_asks": round(statistics.mean(jac), 3),
                     "evidence_turns_in_the_ten": round(reached / max(1, gold_n), 3), "evidence_reached": reached,
                     "evidence_asked": gold_n}
    out = {"questions": len(rows), "repeats": data["repeats"], "candidates_with_a_turn_id": mapped > 0,
           "score_move_between_two_asks": {"mean": round(statistics.mean(diffs), 4),
                                           "p95": round(sorted(diffs)[int(len(diffs) * 0.95)], 4),
                                           "max": round(max(diffs), 4)}, "rules": out}
    (run / "rules.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "rules":
        print(json.dumps(rules(Path(sys.argv[2])), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect()
    else:
        sys.exit(__doc__)
