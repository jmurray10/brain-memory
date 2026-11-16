"""The build step 8 gate, measured: what share of page lines does the cited turn support?

    python scripts/page_audit.py <database> [<database> ...] [--lines N] [--paid] [--seed N]
    python scripts/page_audit.py --prefix brain_eval_ [--lines N] [--paid]      # every database with that prefix

Each <database> is a database on the test server (TEST_DATABASE_URL's server; production is refused) that still
holds a benchmark's claims and pages: run brain_eval.py with BRAIN_EVAL_KEEP=1 to keep its per-question
databases, and drop them when the audit is done (--drop). `--lines` is the sample size per account and database
(default 200).

Without --paid nothing is sent to Jev: the script prints how many page lines exist and how many would be asked.
With --paid it asks Jev one Noul per sampled line (spend is capped at CAP dollars; get the owner's go first, README
rule 7), prints the share supported with a 95% interval, and writes the result under runs/page_audit_<time>/:
summary.json, and handcheck.jsonl (10 lines Jev passed and 10 it failed, or all the failures if fewer, each
with its statement and source turn) for the hand check the pre-registration asks for.
The gate passes at 0.95 (DESIGN.md build order, step 8). Record the result in the experiment log.
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "live-eval"))      # env.py (key loading)

from testdb import admin_url, db_url  # noqa: E402

CAP = 2.0
HANDCHECK = 10


def combine(per_account: list[dict]) -> dict:
    """One result over every sample (lines pooled; the interval is on the pooled share)."""
    from brain.pages import SUPPORT_GATE, _wilson
    n = sum(r["lines"] for r in per_account)
    ok = sum(r["supported"] for r in per_account)
    share = round(ok / n, 4) if n else None
    return {"samples": len(per_account), "population": sum(r["population"] for r in per_account), "lines": n,
            "supported": ok, "share": share, "ci95": _wilson(ok, n),
            "quote_found": sum(r["quote_found"] for r in per_account), "bar": SUPPORT_GATE,
            "gate": "no lines" if not n else ("passed" if share >= SUPPORT_GATE else "not passed")}


def failures(details: list[dict]) -> list[dict]:
    """Every line that failed, with the statement Jev judged ("entity: slot = value"). The summary keeps only the
    rendered sentence, which hides the slot: exp-page-audit-v1's 192 failures could not be read without it."""
    return [{k: v for k, v in d.items() if k != "source"} for d in details if d["reason"]]


def handcheck_sample(details: list[dict], seed: int = 0, each: int = HANDCHECK) -> list[dict]:
    """Up to `each` lines Jev passed and `each` it failed (code-check failures are not Jev's judgement)."""
    judged = [d for d in details if d["p"] is not None]
    rng = random.Random(seed)
    passed = [d for d in judged if d["reason"] is None]
    failed = [d for d in judged if d["reason"] is not None]
    rng.shuffle(passed)
    rng.shuffle(failed)
    return passed[:each] + failed[:each]


def _flag(argv: list[str], name: str, default=None):
    if name in argv:
        k = argv.index(name)
        v = argv[k + 1]
        del argv[k:k + 2]
        return v
    return default


def main(argv: list[str]) -> int:
    argv = list(argv)
    seed = int(_flag(argv, "--seed", 0))
    limit = int(_flag(argv, "--lines", 200))
    prefix = _flag(argv, "--prefix")
    paid, drop = "--paid" in argv, "--drop" in argv
    names = [a for a in argv if not a.startswith("--")]
    import psycopg
    if prefix:
        with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
            names += [r[0] for r in c.execute("select datname from pg_database where datname like %s order by 1",
                                              (prefix.replace("_", r"\_") + "%",)).fetchall()]
    if not names:
        print(__doc__)
        return 2

    from brain import pages
    from brain.db import make_pool, tenant

    def accounts_of(pool):
        with pool.connection() as c:
            return [r[0] for r in c.execute("select id from brain.account order by id").fetchall()]

    if drop and not paid:
        with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
            for name in names:
                c.execute(f'drop database if exists "{name}" with (force)')
        print(f"dropped {len(names)} database(s)")
        return 0
    if not paid:
        total = 0
        for name in names:
            pool = make_pool(db_url(name), min_size=1, max_size=1)
            try:
                for acc in accounts_of(pool):
                    with tenant(pool, acc) as c:
                        n = sum(len(r[0] or []) for r in
                                c.execute("select sentence_claims from page where body_md <> ''").fetchall())
                    total += n
                    if n:
                        print(f"{name} account {acc}: {n} page lines, would ask about {min(n, limit)}")
            finally:
                pool.close()
        print(f"{total} page lines in {len(names)} database(s). Nothing was sent to Jev; pass --paid to run the "
              f"audit (cap ${CAP:.2f}).")
        return 0

    import env
    env.load_env()
    from brain.providers import ProviderError, make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    meter = SpendMeter(MemorySpendStore(), CAP)
    prov = make_providers(load(), meter)
    if prov.offline:
        print("BRAIN_OFFLINE is set: the stub Jev answers 0.5 to everything, so the audit would mean nothing.")
        return 2
    try:
        prov.jev.ask({"text": "We moved last spring."},
                     {"q": {"type": "noul", "instructions": "Does the text mention a move?"}})
    except ProviderError as e:
        print(f"Jev preflight failed, nothing run: {e}")
        return 1
    per, details = [], []
    for name in names:
        pool = make_pool(db_url(name), min_size=1, max_size=1)
        try:
            for acc in accounts_of(pool):
                with tenant(pool, acc) as c:
                    r = pages.line_support_audit(c, prov.jev, limit=limit, seed=seed, detail=True)
                for d in r.pop("detail"):
                    details.append({"database": name, **d})
                if r["lines"]:
                    per.append({"database": name, "account": acc, **r})
        finally:
            pool.close()
    out = {"databases": len(names), "seed": seed, "limit": limit, "spent_usd": round(meter.spent_today(), 4),
           **combine(per), "reasons": {}, "per_sample": per}
    for d in details:
        if d["reason"]:
            out["reasons"][d["reason"]] = out["reasons"].get(d["reason"], 0) + 1
    run = HERE.parent / "runs" / f"page_audit_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (run / "handcheck.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False, default=str) + "\n" for d in handcheck_sample(details, seed)),
        encoding="utf-8")
    (run / "failures.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False, default=str) + "\n" for d in failures(details)),
        encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_sample"}, indent=1))
    print("run dir:", run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
