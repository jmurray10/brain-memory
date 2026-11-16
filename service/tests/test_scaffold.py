"""Scaffold tests. Needs a Postgres with pgvector: set TEST_DATABASE_URL (see docker-compose.yml).
Skipped when TEST_DATABASE_URL is not set."""
import importlib
import os

import psycopg
import pytest

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="module")
def fresh_db():
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute("drop schema if exists brain cascade")
    yield URL


def test_migrate_is_idempotent(fresh_db):
    from brain.db import MIGRATIONS, migrate
    files = sorted(f.name for f in MIGRATIONS.glob("*.sql"))
    assert files[:3] == ["001_init.sql", "002_forget_grace.sql", "003_accounts.sql"]
    assert migrate(fresh_db) == files
    assert migrate(fresh_db) == []
    with psycopg.connect(fresh_db) as c:
        n = c.execute("select count(*) from pg_tables where schemaname = 'brain'").fetchone()[0]
    assert n == 18  # 10 design tables + account, api_client, spend, job, schema_migrations, claim_candidate, slot,
    #                slot_merge


def test_healthz(fresh_db, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    monkeypatch.setenv("BRAIN_ADMIN_KEY", "scaffold-admin-key-0123456789abcdef")
    import brain.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200 and r.json()["ok"] and r.json()["pgvector"]
        assert client.get("/v1/status").status_code == 401  # M1: status needs a key
        st = client.get("/v1/status", headers={"Authorization": "Bearer scaffold-admin-key-0123456789abcdef"})
        assert st.json()["keys"]["database"] is True
