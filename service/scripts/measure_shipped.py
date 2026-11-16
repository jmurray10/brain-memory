"""DESIGN.md "System" row: write cost per 1,000 turns and recall latency (p50, p95) with the settings that are
on in production. Report only, no gate.

    BRAIN_PLACE_PER_SLOT=1 BRAIN_PLACE_CHECK=1 BRAIN_RELATE_ENDS=1 \\
        python scripts/measure_shipped.py [conversation, default conv-30] [questions, default 40]

One LoCoMo conversation is captured one turn per request (as a chat client would), with the claims a client
model sent for it (read from the eval's cache; a turn not in the cache is sent without claims, and counted),
dates stored at capture. Then the write path runs once, then `context` is asked the questions with the shipped
pack (ledger route, nearby turns 1, date notes). No memo: every provider call is paid and timed.

What is measured is the brain's own spend (Voyage and Jev). The client model's cost of writing claims and
answers is the user's and is not in it. Times are taken on this machine against the Supabase test branch, so
they include this machine's network path to the database and the providers; production runs next to neither.
Writes runs/measure_shipped_<stamp>/summary.json."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

from testdb import admin_url, db_url  # noqa: E402

DB = "brain_measure"
CAP = 2.0


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 3)


def times(xs: list[float]) -> dict:
    return {"n": len(xs), "p50_s": pct(xs, 0.5), "p95_s": pct(xs, 0.95), "max_s": round(max(xs), 3),
            "mean_s": round(statistics.mean(xs), 3)} if xs else {"n": 0}


def main() -> None:
    from env import load_env

    load_env()
    from brain import claims, core, ledger, worker
    from brain.auth import Caller
    from brain.db import make_pool, migrate, tenant
    from brain.providers import ProviderError, make_providers
    from brain.schemas import ClaimIn, TurnIn
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter, utc_today
    from locomo_eval import CACHE, CATS, session_dt

    want = sys.argv[1] if len(sys.argv) > 1 else "conv-30"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    conv = next(c for c in json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8"))
                if c["sample_id"] == want)
    store = MemorySpendStore()
    meter = SpendMeter(store, CAP)
    prov = make_providers(load(), meter)
    try:
        prov.jev.ask({"text": "We moved last spring."}, {"q": {"type": "noul", "instructions": "Does the text mention a move?"}})
    except ProviderError as e:
        sys.exit(f"Jev preflight failed, nothing run: {e}")

    def spend() -> dict:
        return {r["provider"]: {"usd": r["usd"], "calls": r["calls"]} for r in store.rows_since(utc_today())}

    def delta(a: dict, b: dict) -> dict:
        out = {p: {"usd": round(b[p]["usd"] - a.get(p, {"usd": 0})["usd"], 5),
                   "calls": b[p]["calls"] - a.get(p, {"calls": 0})["calls"]} for p in b}
        out["total_usd"] = round(sum(v["usd"] for v in out.values()), 5)
        return out

    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines() if CACHE.exists() else []:
        if line.strip():
            r = json.loads(line)
            cache[r["k"]] = r["claims"]
    system = claims.CLAIMS_CONTRACT + "\nReply with a JSON list of claims only; [] if the turn states none."

    with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
        c.execute(f"drop database if exists {DB} with (force)")
        c.execute(f"create database {DB}")
    url = db_url(DB)
    migrate(url)
    pool = make_pool(url, max_size=2)
    with pool.connection() as cn:
        owner = cn.execute("select id from brain.account where name = 'owner'").fetchone()[0]
    caller = Caller("measure", ("*",), ("*",), None, True, owner, "owner", False, True)
    ns, c = "u", conv["conversation"]

    s0 = spend()
    cap_t, turns_n, no_claims, claims_n, n, last_date = [], 0, 0, 0, 1, ""
    while f"session_{n}" in c:
        last_date = c[f"session_{n}_date_time"]
        base, prev = session_dt(last_date), []
        for k, t in enumerate(c[f"session_{n}"]):
            text = str(t.get("text") or "")
            if t.get("blip_caption"):
                text += f" [shares an image: {t['blip_caption']}]"
            if not text.strip():
                continue
            sp = str(t.get("speaker") or "user")
            key = hashlib.sha256(json.dumps([system, text, sp, prev[-2:], base.date().isoformat()]).encode()).hexdigest()[:24]
            cl = []
            for raw in cache.get(key) or []:
                try:
                    cl.append(ClaimIn(**raw))
                except ValueError:
                    pass
            no_claims += key not in cache
            claims_n += len(cl)
            turn = TurnIn(role="user", speaker=sp, text=text, ts=base + timedelta(seconds=k), claims=cl[:25])
            t0 = time.perf_counter()
            with tenant(pool, owner) as conn:
                core.capture(conn, prov, caller, ns, [turn], f"session_{n}", None, write_path_enabled=True,
                             dates_enabled=True)
            cap_t.append(time.perf_counter() - t0)
            turns_n += 1
            prev.append(f"{sp}: {text[:400]}")
        n += 1
    s1 = spend()
    t0 = time.perf_counter()
    wp = worker.write_path(pool, prov, max_episodes=100_000)
    wp_s = time.perf_counter() - t0
    s2 = spend()

    qas = [(k, qa) for k, qa in enumerate(conv["qa"]) if qa.get("category") in CATS]
    qas = sorted(qas, key=lambda x: hashlib.sha256(f"{want}:{x[0]}".encode()).hexdigest())[:n_q]
    qdate = session_dt(last_date).strftime("%Y/%m/%d (%a) %H:%M")
    ctx_t, routes = [], {}
    for _, qa in qas:
        t0 = time.perf_counter()
        with tenant(pool, owner) as conn:
            ctx = core.context(conn, prov, caller, ns, qa["question"], qdate, "default", None, when_route=False,
                               ledger_route=True, memory_check=False, ledger_notes=True, open_route="full",
                               neighbours=1, date_notes=True)
        ctx_t.append(time.perf_counter() - t0)
        routes[str(ctx["route"])] = routes.get(str(ctx["route"]), 0) + 1
    s3 = spend()
    pool.close()

    cap_d, wp_d, ctx_d = delta(s0, s1), delta(s1, s2), delta(s2, s3)
    per_k = lambda usd: round(usd / turns_n * 1000, 3)  # noqa: E731
    out = {
        "conversation": want, "turns": turns_n, "claims_sent": claims_n, "turns_not_in_claim_cache": no_claims,
        "settings": {"place_per_slot": ledger.PLACE_PER_SLOT, "place_check": getattr(ledger, "PLACE_CHECK", None),
                     "relate_ends": ledger.RELATE_ENDS, "dates_at_capture": True, "neighbours": 1, "date_notes": True},
        "capture": {"per_request": times(cap_t), "spend": cap_d},
        "write_path": {"seconds": round(wp_s, 1), "seconds_per_turn": round(wp_s / turns_n, 3), "spend": wp_d,
                       "counts": {k: v for k, v in wp.items() if k != "ledger"}, "ledger": wp.get("ledger")},
        "write_cost_usd_per_1000_turns": {"capture": per_k(cap_d["total_usd"]), "write_path": per_k(wp_d["total_usd"]),
                                          "total": per_k(cap_d["total_usd"] + wp_d["total_usd"])},
        "context": {"per_request": times(ctx_t), "spend": ctx_d, "routes": routes,
                    "usd_per_1000_recalls": round(ctx_d["total_usd"] / max(1, len(ctx_t)) * 1000, 3)},
        "spend_usd": round(meter.spent_today(), 4),
    }
    run = SERVICE / "runs" / f"measure_shipped_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"run dir: {run}")


if __name__ == "__main__":
    main()
