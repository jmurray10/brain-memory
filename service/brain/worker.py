"""Nightly worker (REFERENCE.md): python -m brain.worker --once. Railway runs it as a cron.

A second cron runs only the write path, every 10 minutes (the owner, 2026-11-09: a fact told now should reach the
ledger soon, not the next night): python -m brain.worker --once --steps write_path,embed_backfill. One advisory
lock (ledger_lock) covers every step that changes the ledger: two write_path runs never take the same jobs, and
consolidate never runs while a write_path run is placing claims (room review v2, item 1d).

Steps, in order, each logged as one line of counts. A step that fails is logged and the others still run; the
exit code is 1 if any step failed.

1. write_path: the claims the user's model sent with capture are checked (build step 4, brain/claims.py) and
   placed in the ledger (build step 5, brain/ledger.py), oldest pending write_path job first, up to
   WRITE_PATH_MAX episodes per run, each batch inside its account. Jobs exist only with WRITE_PATH_ENABLED.
2. embed_backfill: embed chunks stored without vectors (oldest first, batches of 128, within the daily cap and
   BRAIN_BACKFILL_MAX), then mark embed_backfill jobs done whose episodes are fully embedded.
3. cleanup: purge forgotten text past BRAIN_FORGET_GRACE_DAYS (core.purge_forgotten); delete chunks of deleted
   episodes; delete decision_log rows older than DECISION_LOG_DAYS (0, the default, keeps them all: they are the
   booster's training data; forget and erase still remove the rows that carry a user's text); delete done or
   skipped jobs after 30 days.
4. report: yesterday (UTC): spend by provider, captures, recalls, degraded recalls, injection exclusions,
   pending jobs, accounts.
5. backup: a backup of the original data to BRAIN_BACKUP_DIR and/or the Supabase Storage bucket
   BRAIN_BACKUP_BUCKET (brain/backup.py); skipped when neither is set.
6. consolidate (DESIGN.md nightly 1, 2, 6 and hardening; brain/nightly.py): settle tentative entity links,
   resolve contested claims, re-ask a sample of the day's decisions, settle confident supported claims. It waits
   up to CONSOLIDATE_LOCK_WAIT_S for the ledger lock and is skipped ("busy") if a write_path run still holds it.
7. promote_preferences (nightly 4, build step 7; brain/preferences.py): standing preferences become directives.
8. render_pages (nightly 3, build step 8; brain/pages.py): re-render dirty pages from claims, in code.
9. structure (nightly 5, build step 9; brain/structure.py): heat scores and the cold tier (off at
   BRAIN_COLD_FRACTION 0).

Maintenance runs as the schema owner, across all accounts (outside RLS). The brain steps (write_path,
consolidate, promote_preferences, render_pages, structure) run inside each account (RLS on); their decisions are
Jev's and code's. No generative model runs anywhere in the worker (DESIGN.md).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time

import psycopg
from datetime import date, datetime, timedelta, timezone

from .settings import load_env_file

if __name__ == "__main__":      # core and ledger read their settings when they are imported, so a .env comes first
    load_env_file()

from . import backup, claims, core, ledger, logs, nightly, pages, preferences, structure
from .db import make_pool, migrate, tenant
from .providers import ProviderError, make_providers
from .settings import Settings, load
from .spend import CapReached, PgSpendStore, SpendMeter

log = logging.getLogger("brain.worker")

BACKFILL_BATCH = 128
JOB_KEEP_DAYS = 30
WRITE_PATH_BATCH = 20
WRITE_PATH_MAX = 2000
WRITE_PATH_ATTEMPTS = 3
WRITE_PATH_LOCK = 0x6272_7731     # pg advisory lock key for every step that changes the ledger ("brw1")
CONSOLIDATE_LOCK_WAIT_S = 600     # the nightly waits this long for a 10-minute write_path run to finish
LOCK_POLL_S = 5


@contextlib.contextmanager
def ledger_lock(pool, wait_s: float = 0, poll_s: float = LOCK_POLL_S):
    """The one lock for steps that change the ledger (write_path, consolidate). Yields True when this run holds
    it, False when another run still held it after wait_s seconds; the caller then does nothing. The lock lives on
    its own connection, outside the pool: holding a pooled one starved the run itself (PoolTimeout with the
    worker's pool of 2)."""
    with psycopg.connect(pool.conninfo, autocommit=True, prepare_threshold=None) as lock:
        deadline = time.monotonic() + wait_s
        held = lock.execute("select pg_try_advisory_lock(%s)", (WRITE_PATH_LOCK,)).fetchone()[0]
        while not held and time.monotonic() < deadline:
            time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
            held = lock.execute("select pg_try_advisory_lock(%s)", (WRITE_PATH_LOCK,)).fetchone()[0]
        try:
            yield held
        finally:
            if held:
                try:
                    lock.execute("select pg_advisory_unlock(%s)", (WRITE_PATH_LOCK,))
                except psycopg.Error:
                    # The lock's connection died while the step ran (a pooler closing an idle session). The
                    # server released the lock with it, so there is nothing to unlock; say so, because for the
                    # rest of that step the other ledger step was not shut out.
                    log.warning("ledger lock connection was lost before unlock; the step ran partly unlocked")
# DESIGN.md "Nightly consolidation", with the build step each one waits for.
def structure_step(pool, settings: Settings) -> dict:
    """DESIGN.md nightly step 5 (build step 9): heat scores and the cold tier, per account (brain/structure.py)."""
    with pool.connection() as c:
        accounts = [r[0] for r in c.execute("select id from brain.account where disabled_at is null order by id")]
    total = {"accounts": 0, "scored": 0, "went_cold": 0, "warmed": 0, "cold": 0, "policy": settings.cold_policy,
             "fraction": settings.cold_fraction}
    for acct in accounts:
        with tenant(pool, acct) as conn:
            o = structure.run(conn, settings.cold_policy, settings.cold_fraction, settings.cold_min_age_days)
        total["accounts"] += 1
        for k in ("scored", "went_cold", "warmed", "cold"):
            total[k] += o[k]
    return total


def consolidate(pool, providers, lock_wait_s: float = CONSOLIDATE_LOCK_WAIT_S) -> dict:
    """DESIGN.md nightly steps 1, 2 and 6 plus hardening (brain/nightly.py), per account, under the ledger lock so
    it never changes claims a write_path run is relating. One run id names the night in every link it writes."""
    with ledger_lock(pool, lock_wait_s) as held:
        if not held:
            log.warning("consolidate skipped: the ledger lock was still held after %.0f s", lock_wait_s)
            return {"accounts": 0, "stopped": "busy"}
        return _consolidate(pool, providers)


def _consolidate(pool, providers) -> dict:
    with pool.connection() as c:
        accounts = [r[0] for r in c.execute("select id from brain.account where disabled_at is null order by id")]
    run_id = ledger.new_run_id("n")
    total: dict = {"run_id": run_id, "accounts": 0, "tentative": 0, "settled_links": 0, "split": 0,
                   "contested_pairs": 0, "corrected": 0, "audited": 0, "flipped": 0, "hardened": 0}
    for acct in accounts:
        try:
            with tenant(pool, acct) as conn:
                o = nightly.consolidate(conn, providers, run_id=run_id)
        except CapReached:
            total["stopped"] = "cap_reached"
            break
        except ProviderError as e:
            total["stopped"] = f"provider_error:{e.provider}"
            break
        total["accounts"] += 1
        total["tentative"] += o["links"]["tentative"]
        total["settled_links"] += o["links"]["settled"]
        total["split"] += o["links"]["split"]
        total["slots_merged"] = total.get("slots_merged", 0) + o["slots"]["merged"]
        if o["slots"].get("held"):                    # over the churn cap: rolled back, waiting for a person
            total["merges_held"] = total.get("merges_held", 0) + 1
            log.warning("merge_slots held for account %s (run %s)", acct, run_id)
        total["contested_pairs"] += o["contested"]["pairs"]
        total["corrected"] += o["contested"]["corrected"]
        total["audited"] += o["stability"]["asked"]
        total["flipped"] += o["stability"]["flipped"]
        total["hardened"] += o["harden"]["settled"]
    return total


def render_pages(pool) -> dict:
    """DESIGN.md nightly step 3 (build step 8): re-render dirty pages from claims (code only, no model)."""
    with pool.connection() as c:
        accounts = [r[0] for r in c.execute("select id from brain.account where disabled_at is null order by id")]
    total = {"accounts": 0, "rendered": 0, "lines": 0}
    for acct in accounts:
        with tenant(pool, acct) as conn:
            out = pages.render_dirty(conn)
        total["accounts"] += 1
        total["rendered"] += out["rendered"]
        total["lines"] += out["lines"]
    return total


def promote_preferences(pool, providers) -> dict:
    """DESIGN.md nightly step 4 (build step 7): per account, learned directives from standing preferences."""
    with pool.connection() as c:
        accounts = [r[0] for r in c.execute("select id from brain.account where disabled_at is null order by id")]
    total = {"accounts": 0, "candidates": 0, "promoted": 0, "pending": 0, "screened_out": 0, "retired": 0}
    for acct in accounts:
        try:
            with tenant(pool, acct) as conn:
                out = preferences.promote(conn, providers)
        except CapReached:
            total["stopped"] = "cap_reached"
            break
        except ProviderError as e:
            total["stopped"] = f"provider_error:{e.provider}"
            break
        total["accounts"] += 1
        for k in ("candidates", "promoted", "pending", "screened_out", "retired"):
            total[k] += out[k]
    return total


def embed_backfill(pool, embedder, max_chunks: int) -> dict:
    embedded, stopped = 0, None
    while embedded < max_chunks:
        with pool.connection() as c:
            rows = c.execute("select id, text from brain.chunk where embedding is null order by id limit %s",
                             (min(BACKFILL_BATCH, max_chunks - embedded),)).fetchall()
        if not rows:
            break
        try:
            vectors = embedder.embed_documents([r[1] for r in rows])
        except CapReached:
            stopped = "cap_reached"
            break
        except ProviderError as e:
            stopped = f"provider_error:{e.provider}"
            break
        with pool.connection() as c, c.cursor() as cur:
            cur.executemany("""update brain.chunk set embedding = %s::vector, embed_model = %s
                               where id = %s and embedding is null""",
                            [(core._vec(v), embedder.model, r[0]) for r, v in zip(rows, vectors)])
        embedded += len(rows)
    with pool.connection() as c:
        jobs = c.execute("""update brain.job j set status = 'done', updated_at = now()
                            where j.kind = 'embed_backfill' and j.status = 'pending'
                              and not exists (select 1 from brain.chunk k where k.kind = 'episode_line'
                                              and k.ref_id = j.ref_id and k.embedding is null)""").rowcount
        remaining = c.execute("select count(*) from brain.chunk where embedding is null").fetchone()[0]
    if stopped is None and remaining and embedded >= max_chunks:
        stopped = "max_per_run"
    return {"embedded": embedded, "jobs_done": jobs, "remaining": remaining, "stopped": stopped}


def write_path(pool, providers, max_episodes: int = WRITE_PATH_MAX) -> dict:
    """Pending write_path jobs -> claims.process, WRITE_PATH_BATCH episodes at a time per account. A provider
    failure leaves the batch's jobs pending (attempts + 1; failed after WRITE_PATH_ATTEMPTS); the cap stops the
    step."""
    with ledger_lock(pool) as held:     # one ledger-changing run at a time (the nightly and the 10-minute cron)
        if not held:
            return {"jobs": 0, "stopped": "busy"}
        return _write_path(pool, providers, max_episodes)


def _write_path(pool, providers, max_episodes: int) -> dict:
    with pool.connection() as c:
        jobs = c.execute("""select id, account_id, ref_id from brain.job where kind = 'write_path' and status = 'pending'
                            order by id limit %s""", (max_episodes,)).fetchall()
    by_acct: dict[int, list[tuple[int, int]]] = {}
    for jid, acct, ref in jobs:
        by_acct.setdefault(acct, []).append((jid, ref))
    total, ledger_total = claims.Outcome(), ledger.LedgerOutcome()
    stopped, failed_jobs = None, 0
    run_id = ledger.new_run_id("w")       # names this run in every link it writes (claim_link.run_id)
    for acct, items in by_acct.items():
        for i in range(0, len(items), WRITE_PATH_BATCH):
            batch = items[i:i + WRITE_PATH_BATCH]
            ids = [j for j, _ in batch]
            try:
                with tenant(pool, acct) as conn:      # step 4 checks and step 5 placement commit together
                    o = claims.process(conn, providers, [r for _, r in batch])
                    lo = ledger.process(conn, providers, [r for _, r in batch], run_id)
            except CapReached:
                stopped = "cap_reached"
                break
            except ProviderError as e:
                with pool.connection() as c:
                    rows = c.execute(
                        """update brain.job set attempts = attempts + 1, last_error = %s, updated_at = now(),
                                  status = case when attempts + 1 >= %s then 'failed' else 'pending' end
                           where id = any(%s) and status = 'pending' returning status""",
                        (f"provider_error:{e.provider}", WRITE_PATH_ATTEMPTS, ids)).fetchall()
                failed_jobs += sum(r[0] == "failed" for r in rows)
                stopped = f"provider_error:{e.provider}"
                break
            with pool.connection() as c:
                c.execute("update brain.job set status = 'done', updated_at = now() where id = any(%s)", (ids,))
            for k, v in o.__dict__.items():
                setattr(total, k, getattr(total, k) + v)
            for k, v in lo.__dict__.items():
                setattr(ledger_total, k, getattr(ledger_total, k) + v)
        if stopped:
            break
    return {"jobs": len(jobs), **total.as_dict(), "ledger": ledger_total.as_dict(), "failed_jobs": failed_jobs,
            "stopped": stopped}


def cleanup(pool, settings: Settings) -> dict:
    with pool.connection() as c:
        purged = core.purge_forgotten(c, settings.forget_grace_days)
        stray = c.execute("""delete from brain.chunk k using brain.episode e
                             where k.kind = 'episode_line' and k.ref_id = e.id and e.deleted_at is not null""").rowcount
        old_logs = 0
        if settings.decision_log_days > 0:
            old_logs = c.execute("delete from brain.decision_log where ts < now() - make_interval(days => %s)",
                                 (settings.decision_log_days,)).rowcount
        old_jobs = c.execute("""delete from brain.job where status in ('done', 'skipped')
                                and updated_at < now() - make_interval(days => %s)""", (JOB_KEEP_DAYS,)).rowcount
    return {"purged_episodes": purged, "stray_chunks": stray, "old_decision_logs": old_logs, "old_jobs": old_jobs}


def report(pool, day: date) -> dict:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rng = (start, start + timedelta(days=1))
    with pool.connection() as c:
        spend = {r[0]: {"usd": round(float(r[1]), 6), "calls": r[2]} for r in c.execute(
            "select provider, usd, calls from brain.spend where day = %s order by provider", (day,)).fetchall()}
        captures = c.execute("select count(*) from brain.episode where recorded_at >= %s and recorded_at < %s",
                             rng).fetchone()[0]
        recalls, degraded, no_dense = c.execute(
            """select count(*), count(*) filter (where output->>'degraded' is not null),
                      count(*) filter (where output->>'dense' = 'false')
               from brain.decision_log where stage = 'recall.route' and ts >= %s and ts < %s""", rng).fetchone()
        injections = c.execute("""select count(*) from brain.decision_log where stage = 'recall.injection'
                                  and outcome = 'dropped' and ts >= %s and ts < %s""", rng).fetchone()[0]
        pending = dict(c.execute("""select kind, count(*) from brain.job where status = 'pending'
                                    group by kind order by kind""").fetchall())
        accounts = c.execute("select count(*) from brain.account where disabled_at is null").fetchone()[0]
    return {"day": day.isoformat(), "spend_usd": round(sum(v["usd"] for v in spend.values()), 6),
            "spend": spend, "captures": captures, "recalls": recalls, "degraded_recalls": degraded,
            "keyword_only_recalls": no_dense, "injection_exclusions": injections, "pending_jobs": pending,
            "active_accounts": accounts}


def run_once(pool, providers, settings: Settings, today: date | None = None,
             only: set[str] | None = None) -> tuple[dict, bool]:
    """Returns (results by step, ok). `only` runs just the named steps, in the usual order."""
    today = today or datetime.now(timezone.utc).date()
    steps = [
        ("write_path", lambda: write_path(pool, providers)),         # first: its claim chunks embed in this run
        ("embed_backfill", lambda: embed_backfill(pool, providers.embedder, settings.backfill_max)),
        ("cleanup", lambda: cleanup(pool, settings)),
        ("report", lambda: report(pool, today - timedelta(days=1))),
        ("backup", lambda: backup.nightly(pool, settings)),
        ("consolidate", lambda: consolidate(pool, providers)),
        ("promote_preferences", lambda: promote_preferences(pool, providers)),
        ("render_pages", lambda: render_pages(pool)),
        ("structure", lambda: structure_step(pool, settings)),
    ]
    if only:
        unknown = only - {n for n, _ in steps}
        if unknown:
            raise ValueError(f"unknown worker steps: {sorted(unknown)}")
        steps = [s for s in steps if s[0] in only]
    results, ok = {}, True
    for name, fn in steps:
        try:
            results[name] = fn()
            log.info("worker %s %s", name, json.dumps(results[name], sort_keys=True))
        except Exception as e:   # log the class only: messages can carry connection details
            ok = False
            results[name] = {"error": type(e).__name__}
            log.error("worker %s failed: %s", name, type(e).__name__)
    return results, ok


EVERY_MIN_SECONDS = 10


def run_every(tick, seconds: int, sleep=time.sleep, rounds: int | None = None) -> bool:
    """Run `tick` (one pass of the steps, returning whether it went well), wait `seconds`, and repeat until the
    process is stopped: for a deployment with no scheduler, such as Docker Compose. A pass that fails or raises is
    logged and the loop goes on, so one bad pass never ends it. `rounds` bounds the loop (tests). Returns whether
    the last pass went well."""
    ok, n = True, 0
    while rounds is None or n < rounds:
        try:
            ok = bool(tick())
        except Exception as e:      # the class only: messages can carry connection details
            ok = False
            log.error("worker pass failed: %s", type(e).__name__)
        n += 1
        if rounds is None or n < rounds:
            sleep(seconds)
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m brain.worker", description="Brain-Memory nightly worker.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run every step once and exit (the cron)")
    mode.add_argument("--every", type=int, metavar="SECONDS",
                      help="run the steps, wait this long, and repeat until stopped (no scheduler needed)")
    p.add_argument("--steps", help="comma-separated steps to run (default: all), e.g. write_path,embed_backfill")
    args = p.parse_args(argv)
    if args.every is not None and args.every < EVERY_MIN_SECONDS:
        p.error(f"--every must be at least {EVERY_MIN_SECONDS} seconds")
    only = {s.strip() for s in args.steps.split(",") if s.strip()} if args.steps else None
    settings = load()
    logs.configure(settings.log_level)
    if not settings.database_url:
        log.error("DATABASE_URL is not set")
        return 2
    if settings.migrate_on_start:
        migrate(settings.database_url)
    pool = make_pool(settings.database_url, max_size=2)
    meter = SpendMeter(PgSpendStore(pool), settings.daily_cap_usd, account_cap_usd=settings.account_cap_usd)
    providers = make_providers(settings, meter)
    try:
        if args.every is None:
            _, ok = run_once(pool, providers, settings, only=only)
        else:
            try:
                ok = run_every(lambda: run_once(pool, providers, settings, only=only)[1], args.every)
            except KeyboardInterrupt:
                ok = True
    finally:
        providers.close()
        pool.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
