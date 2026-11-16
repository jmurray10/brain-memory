"""Backup and restore of the brain's original data (decided 2026-11-07; REFERENCE.md).

Accounts, client keys (sha256 hashes, never keys), directives, episodes and the claims clients sent with their
turns (claim_candidate) are original data: the brain has no model of its own to extract claims again. Chunks,
vectors, the ledger (entities, slots, claims), pages and the decision log are rebuilt from them. A backup is one
gzip NDJSON file:

    {"kind": "brain-backup", "version": 1, "created_at": ..., "migrations": [...]}     header
    {"t": "account", "r": {...}}  ...  {"t": "episode", "r": {...}}                    one row per line

Rows are whole table rows (to_jsonb), so later columns come along. Forgotten episodes still inside the grace period
keep their text; purged ones are skeletons. Only the newest BRAIN_BACKUP_KEEP files are kept, so forgotten text
leaves the backups within grace + keep days. A backup holds every account's text: store it privately.

    python -m brain.backup dump DIR          # what the worker does when BRAIN_BACKUP_DIR is set
    python -m brain.backup list              # backups in the Supabase Storage bucket (BRAIN_BACKUP_BUCKET)
    python -m brain.backup fetch NAME DIR    # download one from the bucket
    python -m brain.backup load FILE         # into a migrated, empty database (DATABASE_URL)

Off-database copy (decided 2026-11-08): the worker uploads each backup to a private Supabase Storage bucket
(BRAIN_BACKUP_BUCKET, with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY) under backups/, and keeps the newest
BRAIN_BACKUP_KEEP there. Storage is separate from the database's own backups, so a dropped table or a bad
migration does not take the copies with it; it is the same Supabase project, so it does not cover losing the
project.

`load` rebuilds the chunks without vectors and queues embed_backfill jobs, and sets the claims clients sent back
to "received" with a write_path job for each turn that has some, so it costs nothing itself; the next worker runs
embed the chunks and file the claims into the ledger again within the daily cap (Jev decides again, so a
rebuilt ledger can differ in a detail from the one that was lost).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg
from psycopg.types.json import Jsonb

from .text import render, split_text

VERSION = 1
TABLES = ("account", "api_client", "directive", "episode", "claim_candidate")   # load order (foreign keys)
# Columns added after a backup was taken come back as NULL from jsonb_populate_record, and a column default does
# not apply to an explicit NULL. A directive in a backup older than migration 014 was in force, so it loads as
# confirmed; a key in one never had speaks_as_user.
LOAD_DEFAULTS = {"directive": {"status": "confirmed"}, "api_client": {"speaks_as_user": False}}


def fill_defaults(table: str, row: dict) -> dict:
    return {**LOAD_DEFAULTS.get(table, {}), **row}
PREFIX = "brain-backup-"


def dump(pool, out_dir: str, keep: int = 14) -> dict:
    """Write one backup file into out_dir and keep the newest `keep` there (keep 0: delete none)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    final = out / f"{PREFIX}{stamp.strftime('%Y%m%dT%H%M%SZ')}.ndjson.gz"
    tmp = final.with_suffix(".tmp")
    counts = {}
    with pool.connection() as c:
        migrations = [r[0] for r in c.execute("select name from brain.schema_migrations order by name").fetchall()]
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "brain-backup", "version": VERSION, "created_at": stamp.isoformat(),
                                "migrations": migrations}) + "\n")
            for t in TABLES:
                n = 0
                with c.cursor(name=f"backup_{t}") as cur:
                    cur.itersize = 1000
                    cur.execute(f"select to_jsonb(x) from brain.{t} x order by id")
                    for (row,) in cur:
                        f.write(json.dumps({"t": t, "r": row}, ensure_ascii=False) + "\n")
                        n += 1
                counts[t] = n
    os.replace(tmp, final)
    old = sorted(out.glob(f"{PREFIX}*.ndjson.gz"))[:-keep] if keep > 0 else []
    for p in old:
        p.unlink()
    return {"file": final.name, "bytes": final.stat().st_size, "rows": counts, "deleted_old": len(old)}


class StorageError(RuntimeError):
    pass


class SupabaseStorage:
    """The few Supabase Storage REST calls a backup needs. The service-role key goes only into headers."""

    FOLDER = "backups/"

    def __init__(self, url: str, key: str, bucket: str, http: httpx.Client | None = None):
        self.base = url.rstrip("/") + "/storage/v1"
        self.bucket = bucket
        self.http = http or httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self._headers = {"Authorization": f"Bearer {key}", "apikey": key}

    def _check(self, r: httpx.Response, what: str) -> httpx.Response:
        if r.status_code >= 400:   # status only: bodies can echo request details
            raise StorageError(f"storage {what} failed: HTTP {r.status_code}")
        return r

    def upload(self, path: Path) -> str:
        name = self.FOLDER + path.name
        with open(path, "rb") as f:
            r = self.http.post(f"{self.base}/object/{self.bucket}/{name}", content=f.read(),
                               headers={**self._headers, "Content-Type": "application/gzip", "x-upsert": "false"})
        self._check(r, "upload")
        return name

    def list(self) -> list[str]:
        r = self._check(self.http.post(f"{self.base}/object/list/{self.bucket}", headers=self._headers,
                                       json={"prefix": self.FOLDER, "limit": 1000, "offset": 0,
                                             "sortBy": {"column": "name", "order": "asc"}}), "list")
        return sorted(o["name"] for o in r.json() if str(o.get("name", "")).startswith(PREFIX))

    def delete(self, names: list[str]) -> int:
        if not names:
            return 0
        r = self.http.request("DELETE", f"{self.base}/object/{self.bucket}", headers=self._headers,
                              json={"prefixes": [self.FOLDER + n for n in names]})
        self._check(r, "delete")
        return len(names)

    def download(self, name: str, out_dir: str) -> Path:
        name = Path(name).name
        r = self._check(self.http.get(f"{self.base}/object/{self.bucket}/{self.FOLDER}{name}",
                                      headers=self._headers), "download")
        dest = Path(out_dir) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def close(self) -> None:
        self.http.close()


def storage_from(settings) -> SupabaseStorage | None:
    if settings.backup_bucket and settings.supabase_url and settings.supabase_service_key:
        return SupabaseStorage(settings.supabase_url, settings.supabase_service_key, settings.backup_bucket)
    return None


def nightly(pool, settings, storage: SupabaseStorage | None = None) -> dict:
    """The worker's backup step: dump to BRAIN_BACKUP_DIR (or a temporary folder), upload to the bucket when one
    is configured, and keep the newest BRAIN_BACKUP_KEEP in each place."""
    import tempfile
    storage = storage or storage_from(settings)
    if not settings.backup_dir and storage is None:
        return {"skipped": "no BRAIN_BACKUP_DIR or BRAIN_BACKUP_BUCKET"}
    with tempfile.TemporaryDirectory() as tmp:
        local = settings.backup_dir or tmp
        out = dump(pool, local, settings.backup_keep if settings.backup_dir else 0)
        if storage is not None:
            out["uploaded"] = storage.upload(Path(local) / out["file"])
            names = storage.list()
            old = names[:-settings.backup_keep] if settings.backup_keep > 0 else []
            out["deleted_old_remote"] = storage.delete(old)
        if not settings.backup_dir:
            out["deleted_old"] = 0
    return out


def _empty(conn: psycopg.Connection) -> bool:
    return not conn.execute("""select exists (select 1 from brain.episode) or exists (select 1 from brain.api_client)
                               or exists (select 1 from brain.directive)
                               or exists (select 1 from brain.account where name <> 'owner')""").fetchone()[0]


def load(database_url: str, path: str) -> dict:
    """Load a backup into a migrated, empty database, then rebuild chunks (no vectors) and queue backfill."""
    counts = {t: 0 for t in TABLES}
    with gzip.open(path, "rt", encoding="utf-8") as f, psycopg.connect(database_url, prepare_threshold=None) as conn:
        header = json.loads(f.readline())
        if header.get("kind") != "brain-backup" or header.get("version") != VERSION:
            raise ValueError("not a brain backup this version can read")
        have = {r[0] for r in conn.execute("select name from brain.schema_migrations").fetchall()}
        missing = set(header.get("migrations", [])) - have
        if missing:
            raise ValueError(f"run the newer migrations first: {sorted(missing)}")
        if not _empty(conn):
            raise ValueError("the target database already has data; load only into an empty brain")
        for line in f:
            rec = json.loads(line)
            t = rec["t"]
            if t not in TABLES:
                raise ValueError(f"unknown table {t!r} in backup")
            conflict = "on conflict (id) do nothing" if t == "account" else ""   # the owner row exists already
            conn.execute(f"""insert into brain.{t} overriding system value
                             select * from jsonb_populate_record(null::brain.{t}, %s) {conflict}""",
                         (Jsonb(fill_defaults(t, rec["r"])),))
            counts[t] += 1
        owner_ok = conn.execute("select count(*) from brain.account where name = 'owner'").fetchone()[0] == 1
        if not owner_ok:
            raise ValueError("owner account clash: the backup's owner id differs from this database's")
        for t in TABLES:
            conn.execute(f"""select setval(pg_get_serial_sequence('brain.{t}', 'id'),
                                           greatest((select max(id) from brain.{t}), 1))""")
        chunks = _rebuild_chunks(conn)
        to_file = _requeue_claims(conn)
        conn.commit()
    return {"rows": counts, "chunks": chunks, "claims_to_file": to_file,
            "note": "the next worker runs fill vectors (embed_backfill) and file the claims again (write_path)"}


def _requeue_claims(conn: psycopg.Connection) -> int:
    """The ledger is not in a backup. Every claim a client sent with a turn that still stands goes back to
    'received' (as it arrived), and each such turn gets a write_path job, so the worker checks and files them
    again. A claim that was dropped stays dropped; a forgotten turn's claims were removed when it was forgotten."""
    n = conn.execute("""update brain.claim_candidate k
                        set status = 'received', drop_reason = null, p_hallucinated = null, p_off_target = null,
                            p_incomplete = null
                        from brain.episode e
                        where e.id = k.episode_id and e.deleted_at is null and k.status in ('received', 'pending', 'placed')
                     """).rowcount
    conn.execute("""insert into brain.job (account_id, kind, ref_id, namespace)
                    select distinct k.account_id, 'write_path', k.episode_id, k.namespace
                    from brain.claim_candidate k where k.status = 'received'""")
    return n


def _rebuild_chunks(conn: psycopg.Connection) -> int:
    rows = conn.execute("""select id, account_id, namespace, ts, ts_offset_min, role, speaker, body from brain.episode
                           where deleted_at is null order by id""").fetchall()
    chunk_rows, jobs = [], []
    for eid, acct, ns, ts, off, role, speaker, body in rows:
        local = ts.astimezone(timezone(timedelta(minutes=off or 0)))
        for piece, text in enumerate(split_text(body)):
            chunk_rows.append((acct, ns, eid, piece, render(local, role, speaker, text)))
        jobs.append((acct, eid, ns))
    with conn.cursor() as cur:
        cur.executemany("""insert into brain.chunk (account_id, namespace, kind, ref_id, piece, text)
                           values (%s, %s, 'episode_line', %s, %s, %s)""", chunk_rows)
        cur.executemany("insert into brain.job (account_id, kind, ref_id, namespace) values (%s, 'embed_backfill', %s, %s)",
                        jobs)
    return len(chunk_rows)


def main(argv: list[str] | None = None) -> int:
    from . import logs
    from .db import make_pool
    from .settings import load as load_settings

    p = argparse.ArgumentParser(prog="python -m brain.backup", description="Back up or restore the brain.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("dir")
    d.add_argument("--keep", type=int, default=None, help="backups to keep (default BRAIN_BACKUP_KEEP, 14)")
    lo = sub.add_parser("load")
    lo.add_argument("file")
    sub.add_parser("list", help="backups in the Supabase Storage bucket")
    fe = sub.add_parser("fetch", help="download a backup from the bucket")
    fe.add_argument("name")
    fe.add_argument("dir")
    args = p.parse_args(argv)
    s = load_settings()
    logs.configure(s.log_level)
    if args.cmd in ("list", "fetch"):
        storage = storage_from(s)
        if storage is None:
            print("set BRAIN_BACKUP_BUCKET, SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
            return 2
        try:
            if args.cmd == "list":
                print("\n".join(storage.list()))
            else:
                print(storage.download(args.name, args.dir))
        except StorageError as e:
            print(str(e), file=sys.stderr)
            return 1
        finally:
            storage.close()
        return 0
    if not s.database_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2
    if args.cmd == "dump":
        pool = make_pool(s.database_url, max_size=1)
        try:
            print(json.dumps(dump(pool, args.dir, args.keep if args.keep is not None else s.backup_keep)))
        finally:
            pool.close()
        return 0
    try:
        print(json.dumps(load(s.database_url, args.file)))
    except ValueError as e:
        print(f"load refused: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
