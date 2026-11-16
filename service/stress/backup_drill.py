"""Disaster-recovery drill (phase C): restore production's latest backup into a fresh local database and compare.

1. Ask the worker-equivalent code to write a fresh backup of production to the private bucket (backup.nightly with
   the production settings), so the drill tests today's data, then download it.
2. Create a fresh local database, migrate it, load the file (backup.load).
3. Compare with production: rows per account in account, api_client, directive, episode; key hashes; episode
   bodies by sha256. Chunks are rebuilt without vectors (the worker re-embeds them), so only their count is checked.

Prints names, counts and booleans only. Reads secrets from Brain-Memory/.env.brain-deploy.

    python stress/backup_drill.py
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import dataclasses
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psycopg

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from brain import backup  # noqa: E402
from brain.db import make_pool, migrate  # noqa: E402
from brain.settings import load  # noqa: E402
from scripts.deploy_secrets import DEPLOY_FILE, read_kv  # noqa: E402

from scripts.testdb import admin_url, db_url  # noqa: E402
LOCAL_ADMIN = admin_url()
LOCAL = db_url("brain_drill")
FINGERPRINT = """
select 'account' t, a.name, count(*)::bigint, '' from brain.account a group by a.name
union all select 'api_client', a.name, count(*), string_agg(k.key_hash, ',' order by k.key_hash)
  from brain.api_client k join brain.account a on a.id = k.account_id group by a.name
union all select 'directive', a.name, count(*), md5(string_agg(d.text, '|' order by d.id))
  from brain.directive d join brain.account a on a.id = d.account_id group by a.name
union all select 'episode', a.name, count(*), md5(string_agg(md5(e.body) || coalesce(e.deleted_at::text, '-'),
  '|' order by e.id)) from brain.episode e join brain.account a on a.id = e.account_id group by a.name
order by 1, 2"""


def main() -> int:
    kv = read_kv(DEPLOY_FILE)
    prod = kv["DATABASE_URL"]
    if "sslmode=" not in prod:
        prod += ("&" if "?" in prod else "?") + "sslmode=require"
    settings = dataclasses.replace(load(), backup_dir=None, backup_bucket="brain-backups", backup_keep=14,
                                   supabase_url=kv["SUPABASE_URL"], supabase_service_key=kv["SUPABASE_SERVICE_ROLE_KEY"])
    storage = backup.storage_from(settings)
    pool = make_pool(prod, max_size=1)
    try:
        out = backup.nightly(pool, settings, storage)
        with psycopg.connect(prod, prepare_threshold=None) as c:
            want = c.execute(FINGERPRINT).fetchall()
            prod_chunks = c.execute("select count(*) from brain.chunk k join brain.episode e on e.id = k.ref_id "
                                    "where e.deleted_at is null").fetchone()[0]
    finally:
        pool.close()
    print("fresh production backup:", {k: out[k] for k in ("file", "rows", "uploaded")})
    with tempfile.TemporaryDirectory() as tmp:
        path = storage.download(out["file"], tmp)
        storage.close()
        with psycopg.connect(LOCAL_ADMIN, autocommit=True) as c:
            c.execute("drop database if exists brain_drill with (force)")
            c.execute("create database brain_drill")
        migrate(LOCAL)
        loaded = backup.load(LOCAL, str(path))
    with psycopg.connect(LOCAL) as c:
        got = c.execute(FINGERPRINT).fetchall()
        chunks = c.execute("select count(*) from brain.chunk").fetchone()[0]
        jobs = c.execute("select count(*) from brain.job where kind = 'embed_backfill' and status = 'pending'").fetchone()[0]
    same = got == want
    print("loaded:", loaded["rows"], "| chunks rebuilt:", chunks, "(production live chunks:", prod_chunks, ")",
          "| backfill jobs queued:", jobs)
    print("every table matches production (counts, key hashes, directive text, episode bodies and forget state):",
          same)
    if not same:
        diff = sorted(set(want) ^ set(got))
        print("differences (table, account, count):", [(d[0], d[1], d[2]) for d in diff][:10])
    report = {"file": out["file"], "rows": loaded["rows"], "chunks_rebuilt": chunks, "prod_live_chunks": prod_chunks,
              "match": same, "at": datetime.now(timezone.utc).isoformat()}
    run = SERVICE / "runs" / f"stress_backup_drill_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with psycopg.connect(LOCAL_ADMIN, autocommit=True) as c:
        c.execute("drop database brain_drill with (force)")
    return 0 if same and chunks == prod_chunks else 1


if __name__ == "__main__":
    sys.exit(main())
