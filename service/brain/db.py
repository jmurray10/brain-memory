"""Connection pool and migrations. Works with Supabase (session pooler URL) and plain Postgres + pgvector."""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

from . import spend

MIGRATIONS = Path(__file__).parent / "migrations"
SEARCH_PATH = "brain, extensions, public"


def _configure(conn: psycopg.Connection) -> None:
    conn.execute(f"set search_path to {SEARCH_PATH}")
    conn.commit()


def make_pool(database_url: str, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    # prepare_threshold=None: Supabase's poolers do not support server-side prepared statements in every mode.
    return ConnectionPool(database_url, min_size=min_size, max_size=max_size, open=True,
                          configure=_configure, kwargs={"prepare_threshold": None})


@contextmanager
def tenant(pool: ConnectionPool, account_id: int | None) -> Iterator[psycopg.Connection]:
    """A pooled connection inside one transaction that runs as role brain_app with brain.tenant set, so the RLS
    policies of migration 003 limit every statement to this account. Both settings are transaction-local: they
    end at commit or rollback, before the connection goes back to the pool."""
    if account_id is None:
        raise RuntimeError("no account for this call")
    with pool.connection() as conn:
        # One round trip; set_config('role', ..., true) is SET LOCAL ROLE. brain.source tags every decision_log
        # row this transaction writes (migration 014).
        conn.execute("""select set_config('role', 'brain_app', true), set_config('brain.tenant', %s, true),
                               set_config('brain.source', %s, true)""", (str(int(account_id)), log_source()))
        before = spend.ACCOUNT.get()
        spend.ACCOUNT.set(int(account_id))      # paid calls made in this block are this account's
        try:
            yield conn
        finally:        # set back, not reset(token): a streamed export ends this block in another context
            spend.ACCOUNT.set(before)


LOG_SOURCES = ("client", "benchmark")


def log_source() -> str:
    """Where this process's decisions come from: 'client' (the service), or 'benchmark' when an eval, stress or
    test-data script set BRAIN_LOG_SOURCE=benchmark. Benchmark rows never train the booster (the booster plan)."""
    v = os.environ.get("BRAIN_LOG_SOURCE", "client").strip().lower()
    return v if v in LOG_SOURCES else "client"


def migrate(database_url: str) -> list[str]:
    """Apply every migrations/*.sql not yet recorded, in name order, each in its own transaction.
    Returns the names applied. Refuses to run if an applied file was edited afterwards."""
    applied_now = []
    with psycopg.connect(database_url, prepare_threshold=None) as conn:
        # The API and the worker may start together; one migrates while the other waits. Session-level lock,
        # released when this connection closes.
        conn.execute("select pg_advisory_lock(hashtext('brain.migrate'))")
        conn.commit()
        conn.execute("create schema if not exists brain")
        conn.execute("""create table if not exists brain.schema_migrations (
                            name text primary key, sha256 text not null, applied_at timestamptz not null default now())""")
        conn.commit()
        done = dict(conn.execute("select name, sha256 from brain.schema_migrations").fetchall())
        for f in sorted(MIGRATIONS.glob("*.sql")):
            sql = f.read_text(encoding="utf-8")
            digest = hashlib.sha256(sql.encode()).hexdigest()
            if f.name in done:
                if done[f.name] != digest:
                    raise RuntimeError(f"migration {f.name} changed after it was applied; add a new file instead")
                continue
            with conn.transaction():
                conn.execute(sql)
                conn.execute("insert into brain.schema_migrations (name, sha256) values (%s, %s)", (f.name, digest))
            applied_now.append(f.name)
    return applied_now


if __name__ == "__main__":  # python -m brain.db  -> apply migrations
    import sys

    from .settings import load

    s = load()
    if not s.database_url:
        sys.exit("DATABASE_URL is not set")
    print("applied:", migrate(s.database_url) or "nothing new")
