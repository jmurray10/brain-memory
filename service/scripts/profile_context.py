"""Where the time of one `context` request goes, step by step, with the settings that are on in production.
Report only.

    python scripts/profile_context.py [conversation, default conv-30] [questions, default 40] [jev workers] [parallel]

Reads the database `brain_measure` that scripts/measure_shipped.py leaves behind (run that first; this script
captures nothing and never writes to another database). The same questions are asked through `core.context`
with timers wrapped around its steps and around every provider round trip. No memo: every call is paid (about
a tenth of a cent per question).

Steps are timed from the outside, so "recall" contains its own embedding, SQL and Jev batch, which are also
reported on their own lines; the step lines add up to the request, the provider lines do not add to them.
With `parallel` (BRAIN_CONTEXT_PARALLEL) the router and ledger steps overlap recall, so the step lines no longer
add up to the request. Each pack's hash is kept, so two runs can be compared pack by pack.
Writes runs/profile_context_<stamp>/summary.json."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

from testdb import db_url  # noqa: E402

DB = "brain_measure"
CAP = 1.0


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 3)


def summary(per_q: list[float], n_q: int) -> dict:
    """per_q: one total per question in which the step ran."""
    return {"ran_in": len(per_q), "of": n_q, "p50_s": pct(per_q, 0.5), "p95_s": pct(per_q, 0.95),
            "mean_s_when_run": round(statistics.mean(per_q), 3),
            "mean_s_per_question": round(sum(per_q) / n_q, 3)} if per_q else {"ran_in": 0, "of": n_q}


def main() -> None:
    from env import load_env

    load_env()
    from brain import core, dates, ledger, preferences, router
    from brain.auth import Caller
    from brain.db import make_pool, tenant
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from locomo_eval import CATS, session_dt

    want = sys.argv[1] if len(sys.argv) > 1 else "conv-30"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    parallel = len(sys.argv) > 4 and sys.argv[4] == "parallel"
    conv = next(c for c in json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8"))
                if c["sample_id"] == want)
    meter = SpendMeter(MemorySpendStore(), CAP)
    prov = make_providers(load(), meter)
    if len(sys.argv) > 3:           # how many Jev requests of one batch are in flight at once
        prov.jev.workers = int(sys.argv[3])
    pool = make_pool(db_url(DB), max_size=2)
    with pool.connection() as cn:
        owner = cn.execute("select id from brain.account where name = 'owner'").fetchone()[0]
        episodes = cn.execute("select count(*) from brain.episode").fetchone()[0]
    if not episodes:
        sys.exit("brain_measure holds no turns; run scripts/measure_shipped.py first")
    caller = Caller("profile", ("*",), ("*",), None, True, owner, "owner", False, True)

    now: dict[str, float] = {}          # step -> seconds within the current question
    batches: list[dict] = []            # every Jev batch: who asked, how many requests, how long

    def timed(obj, name: str, label: str) -> None:
        fn = getattr(obj, name)

        def wrapper(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                now[label] = now.get(label, 0.0) + time.perf_counter() - t0
        setattr(obj, name, wrapper)

    stack: list[str] = []

    def step(obj, name: str, label: str) -> None:
        fn = getattr(obj, name)

        def wrapper(*a, **k):
            stack.append(label)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                stack.pop()
                now[label] = now.get(label, 0.0) + time.perf_counter() - t0
        setattr(obj, name, wrapper)

    for obj, name, label in [(core, "recall", "step.recall"), (router, "route", "step.router"),
                             (ledger, "lookup", "step.ledger_lookup"), (ledger, "facts", "step.ledger_facts"),
                             (ledger, "gather", "step.ledger_gather"),
                             (ledger, "assistant_lines", "step.ledger_assistant"),
                             (preferences, "pack", "step.preferences"), (dates, "annotate", "step.date_notes"),
                             (dates, "block", "step.date_block"), (core, "_ledger_notes", "step.ledger_notes"),
                             (core, "_neighbours", "step.neighbours")]:
        step(obj, name, label)
    timed(prov.embedder, "embed_query", "provider.voyage_embed_query")

    ask_many, call = prov.jev.ask_many, prov.jev._call
    singles: list[float] = []

    def jev_many(requests):
        t0 = time.perf_counter()
        try:
            return ask_many(requests)
        finally:
            dt = time.perf_counter() - t0
            now["provider.jev_batches"] = now.get("provider.jev_batches", 0.0) + dt
            batches.append({"by": stack[-1] if stack else "context", "requests": len(requests), "s": dt})

    def jev_call(state, questions):
        t0 = time.perf_counter()
        try:
            return call(state, questions)
        finally:
            singles.append(time.perf_counter() - t0)

    prov.jev.ask_many, prov.jev._call = jev_many, jev_call

    with tenant(pool, owner) as conn:           # the database round trip from this machine
        pings = []
        for _ in range(20):
            t0 = time.perf_counter()
            conn.execute("select 1").fetchone()
            pings.append(time.perf_counter() - t0)

    qas = [(k, qa) for k, qa in enumerate(conv["qa"]) if qa.get("category") in CATS]
    qas = sorted(qas, key=lambda x: hashlib.sha256(f"{want}:{x[0]}".encode()).hexdigest())[:n_q]
    n, last = 1, ""
    while f"session_{n}" in conv["conversation"]:
        last = conv["conversation"][f"session_{n}_date_time"]
        n += 1
    qdate = session_dt(last).strftime("%Y/%m/%d (%a) %H:%M")

    rows, opens = [], []
    for _, qa in qas:
        now.clear()
        b0 = len(batches)
        t0 = time.perf_counter()
        with tenant(pool, owner) as conn:
            t1 = time.perf_counter()
            ctx = core.context(conn, prov, caller, "u", qa["question"], qdate, "default", None, when_route=False,
                               ledger_route=True, memory_check=False, ledger_notes=True, open_route="full",
                               neighbours=1, date_notes=True, parallel=parallel)
            t2 = time.perf_counter()
        t3 = time.perf_counter()
        opens.append((t1 - t0) + (t3 - t2))
        mine = batches[b0:]
        rows.append({"total": t3 - t0, "context": t2 - t1, "route": str(ctx["route"]), "steps": dict(now),
                     "pack": hashlib.sha256(ctx["prompt"].encode()).hexdigest()[:16],
                     "jev_batches": len(mine), "jev_requests": sum(b["requests"] for b in mine)})

    pool.close()
    nq = len(rows)
    labels = sorted({k for r in rows for k in r["steps"]})
    by_caller: dict[str, list[dict]] = {}
    for b in batches:
        by_caller.setdefault(b["by"], []).append(b)
    steps_sum = [sum(v for k, v in r["steps"].items() if k.startswith("step.")) for r in rows]
    out = {
        "conversation": want, "questions": nq, "jev_workers": prov.jev.workers, "parallel": parallel,
        "request": {"p50_s": pct([r["total"] for r in rows], 0.5), "p95_s": pct([r["total"] for r in rows], 0.95),
                    "mean_s": round(statistics.mean(r["total"] for r in rows), 3)},
        "transaction_open_and_commit": {"mean_s": round(statistics.mean(opens), 3), "p50_s": pct(opens, 0.5)},
        "database_round_trip": {"p50_s": pct(pings, 0.5), "p95_s": pct(pings, 0.95)},
        "inside_context_not_in_a_step": {"mean_s": round(statistics.mean(
            r["context"] - s for r, s in zip(rows, steps_sum)), 3)},
        "steps": {k: summary([r["steps"][k] for r in rows if k in r["steps"]], nq) for k in labels},
        "jev": {"batches_per_question": round(statistics.mean(r["jev_batches"] for r in rows), 1),
                "requests_per_question": round(statistics.mean(r["jev_requests"] for r in rows), 1),
                "one_request": {"n": len(singles), "p50_s": pct(singles, 0.5), "p95_s": pct(singles, 0.95),
                                "max_s": round(max(singles), 3)},
                "batches_by_step": {k: {"batches": len(v), "requests_per_batch": round(statistics.mean(
                    b["requests"] for b in v), 1), "p50_s": pct([b["s"] for b in v], 0.5),
                    "p95_s": pct([b["s"] for b in v], 0.95)} for k, v in sorted(by_caller.items())}},
        "routes": {x: sum(r["route"] == x for r in rows) for x in sorted({r["route"] for r in rows})},
        "packs": [r["pack"] for r in rows],
        "spend_usd": round(meter.spent_today(), 4),
    }
    run = SERVICE / "runs" / f"profile_context_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"run dir: {run}")


if __name__ == "__main__":
    main()
