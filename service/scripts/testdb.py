"""Where tests, stress runs and benchmarks get their database: TEST_DATABASE_URL (env, else .env.deploy).

Since 2026-11-08 this is a temporary Supabase dev branch of brain-memory (`test`), never local Docker and never
the production project. Throwaway databases for single runs live on the same server under their own names.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEPLOY_FILE = Path(__file__).resolve().parents[3] / ".env.deploy"
PROD_REF = "<supabase-project-ref>"     # the production project: refuse it


def admin_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url and DEPLOY_FILE.exists():
        for line in DEPLOY_FILE.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("TEST_DATABASE_URL="):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not url:
        raise SystemExit("TEST_DATABASE_URL is not set (env or .env.deploy)")
    if PROD_REF in url:
        raise SystemExit("TEST_DATABASE_URL points at the production project; use the test branch")
    if "sslmode=" not in url and "localhost" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def db_url(name: str) -> str:
    """The same server, another database name (query string kept)."""
    p = urlsplit(admin_url())
    return urlunsplit((p.scheme, p.netloc, "/" + name, p.query, p.fragment))
