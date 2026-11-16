"""Build step 3 gate (the experiment log exp-when-v1): LongMemEval temporal slice, open route vs when route.

Each question's whole S haystack is captured once (core.capture with dates on: time phrases -> time_mention), in
its own namespace, then answered twice from the same data: arm `open` (the measured flat answer path) and arm
`when` (router; for when / date-math questions, the code-computed dates block). Graded with the official
LongMemEval judge prompt per type (live-eval prompts.lme_judge_prompt), Haiku, majority of 3 valid verdicts.

    python scripts/temporal_eval.py dev          # 22 temporal dev questions (format check; one tuning round)
    python scripts/temporal_eval.py test         # 111 temporal test questions + a 40-question non-temporal guard

In-process (same core functions as the API), real Voyage / Jev / Haiku, capped by BRAIN_DAILY_CAP_USD-style meter
at $5. Writes runs/when_<split>_<stamp>/.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg

SERVICE = Path(__file__).resolve().parents[1]
LIVE_EVAL = SERVICE.parent / "live-eval"
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(LIVE_EVAL))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from testdb import admin_url, db_url  # noqa: E402  (TEST_DATABASE_URL: the test Supabase branch)
DB = "brain_when"
CAP = 5.0
GUARD_N = 40
DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?: \(\w+\))?(?: (\d{2}):(\d{2}))?")


def session_ts(s: str) -> datetime:
    y, mo, d, hh, mm = DATE_RE.search(s).groups()
    return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), tzinfo=timezone.utc)


def questions(split: str) -> list[dict]:
    import route_eval
    s_rows = {r["question_id"]: r for r in json.loads(route_eval.S_FILE.read_text(encoding="utf-8"))}
    dev, test = route_eval.splits(route_eval.load_questions())
    pick = dev if split == "dev" else test
    temporal = [q for q in pick if q["qtype"] == "temporal-reasoning"]
    out = [{**s_rows[q["qid"]], "set": "temporal"} for q in temporal]
    if split == "test":
        others = sorted((q for q in test if q["qtype"] != "temporal-reasoning" and q.get("in_sample")),
                        key=lambda q: hashlib.sha256(q["qid"].encode()).hexdigest())[:GUARD_N]
        out += [{**s_rows[q["qid"]], "set": "guard"} for q in others]
    return out


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else ""
    if split not in ("dev", "test"):
        sys.exit(__doc__)
    import env
    env.load_env()
    from prompts import lme_judge_prompt, parse_yes_no
    from brain import core
    from brain.auth import Caller
    from brain.db import make_pool, migrate, tenant
    from brain.providers import make_providers
    from brain.schemas import TurnIn
    from brain.settings import load
    from brain.spend import PgSpendStore, SpendMeter

    qs = questions(split)
    print(f"{split}: {len(qs)} questions", {s: sum(q['set'] == s for q in qs) for s in ('temporal', 'guard')}, flush=True)

    def fresh(name: str) -> str:
        with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
            c.execute(f"drop database if exists {name} with (force)")
            c.execute(f"create database {name}")
        url = db_url(name)
        migrate(url)
        return url

    # The meter's own database lives for the whole run. Each question gets its own database, dropped after it is
    # graded: a whole S haystack is thousands of lines, and one shared HNSW index outgrows a small compute's cache
    # (the experiment log stress-c capacity finding), which would make the run measure the database, not the brain.
    meter = SpendMeter(PgSpendStore(make_pool(fresh(DB), max_size=2)), CAP)
    st = load()
    prov = make_providers(st, meter)
    # Preflight: the brain degrades quietly when Jev fails (no dates, no router, no rerank), which would turn this
    # run into a comparison of two identical degraded arms. Stop with the exact error instead.
    from brain.providers import ProviderError
    try:
        prov.jev.ask({"text": "We moved last spring."}, {"q": {"type": "noul", "instructions": "Does the text mention a move?"}})
    except ProviderError as e:
        sys.exit(f"Jev preflight failed, nothing run: {e}")

    def judge(q, pred):
        votes = []
        for _ in range(6):
            raw = prov.haiku.complete("", lme_judge_prompt(q["question_type"], q["question"], str(q["answer"]), pred,
                                                           q["question_id"].endswith("_abs")), 10).text
            v = parse_yes_no(raw)
            if v is not None:
                votes.append(v)
            if len(votes) == 3:
                break
        return (sum(votes) >= 2) if len(votes) == 3 else None, votes

    def one(item):
        i, q = item
        ns, name = f"t{i:03d}", f"{DB}_{i:03d}"
        pool = make_pool(fresh(name), max_size=2)
        try:
            with pool.connection() as c:
                owner = c.execute("select id from brain.account where name = 'owner'").fetchone()[0]
            caller = Caller("eval", ("*",), ("*",), None, True, owner, "owner", False, True)
            for sid, date_s, sess in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
                base = session_ts(date_s)
                turns = [TurnIn(role=t["role"] if t["role"] in ("user", "assistant") else "user", text=t["content"],
                                ts=base + timedelta(seconds=k))
                         for k, t in enumerate(sess or []) if isinstance(t, dict) and (t.get("content") or "").strip()]
                for b in range(0, len(turns), 100):
                    with tenant(pool, owner) as conn:
                        core.capture(conn, prov, caller, ns, turns[b:b + 100], sid, None, False, dates_enabled=True)
            row = {"qid": q["question_id"], "qtype": q["question_type"], "set": q["set"], "question": q["question"],
                   "gold": str(q["answer"])}
            for arm, when in (("open", False), ("when", True)):
                with tenant(pool, owner) as conn:
                    a = core.answer(conn, prov, caller, ns, q["question"], q["question_date"], "auto", None,
                                    when_route=when)
                ok, votes = judge(q, a["answer"])
                row[arm] = {"answer": a["answer"], "correct": ok, "votes": votes, "route": a.get("route")}
            with pool.connection() as c:
                n, r, s = c.execute("select count(*), count(date_from), coalesce(sum(confidence), 0) "
                                    "from brain.time_mention").fetchone()
            row["time_mentions"] = [n, r, float(s)]
            return row
        finally:
            pool.close()
            with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
                c.execute(f"drop database if exists {name} with (force)")

    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        for n, row in enumerate(ex.map(one, enumerate(qs)), 1):
            rows.append(row)
            if n % 10 == 0:
                print(f"{n}/{len(qs)} done, spent ${meter.spent_today():.3f}", flush=True)
    n_tm, r_tm, s_tm = (sum(r["time_mentions"][k] for r in rows) for k in range(3))
    tm = (n_tm, r_tm, round(s_tm / n_tm, 3) if n_tm else 0)

    def acc(rs, arm):
        v = [r[arm]["correct"] for r in rs if r[arm]["correct"] is not None]
        return round(sum(v) / len(v), 4) if v else None
    summary = {"split": split, "n": len(rows), "time_mentions": {"phrases": tm[0], "resolved": tm[1],
                                                                 "avg_confidence": float(tm[2] or 0)}}
    for s in ("temporal", "guard"):
        rs = [r for r in rows if r["set"] == s]
        if not rs:
            continue
        diffs = [int(bool(r["when"]["correct"])) - int(bool(r["open"]["correct"])) for r in rs]
        rnd = random.Random(7)
        boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
        summary[s] = {"n": len(rs), "open": acc(rs, "open"), "when": acc(rs, "when"),
                      "diff": round(sum(diffs) / len(diffs), 4), "ci95": [round(boots[50], 4), round(boots[1949], 4)],
                      "routed_when": sum(r["when"]["route"] == "when" for r in rs),
                      "parse_fail": sum(r[a]["correct"] is None for r in rs for a in ("open", "when"))}
    summary["spend_usd"] = round(meter.spent_today(), 4)
    out = SERVICE / "runs" / f"when_{split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    files = [SERVICE / "brain" / f for f in ("dates.py", "core.py", "router.py", "prompts.py")] + [Path(__file__).resolve()]
    (out / "config.json").write_text(json.dumps({"hashes": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                                                            for f in files}, "cap": CAP, "split": split}, indent=2))
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)
    prov.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
