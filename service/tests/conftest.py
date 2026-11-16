import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tests never spend money: stub providers everywhere. Real clients are only built with a mock transport.
os.environ["BRAIN_OFFLINE"] = "1"

TEST_URL = os.environ.get("TEST_DATABASE_URL")
ADMIN_KEY = "test-admin-key-" + "k" * 40


@pytest.fixture
def db_url():
    """A migrated test database with empty tables (except the attribute registry). Skips when TEST_DATABASE_URL is not set."""
    if not TEST_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg

    from brain.db import migrate
    migrate(TEST_URL)
    with psycopg.connect(TEST_URL, autocommit=True) as c:
        c.execute("""truncate brain.api_client, brain.spend, brain.job, brain.decision_log, brain.chunk, brain.page,
                     brain.directive, brain.claim_link, brain.claim, brain.entity, brain.time_mention, brain.episode
                     restart identity cascade""")
        c.execute("delete from brain.account where name <> 'owner'")   # the owner account comes from migration 003
    return TEST_URL


@pytest.fixture
def pool(db_url):
    from brain.db import make_pool
    p = make_pool(db_url, max_size=4)
    yield p
    p.close()


@pytest.fixture
def settings(db_url):
    from brain.settings import Settings
    return Settings(database_url=db_url, admin_key=ADMIN_KEY, daily_cap_usd=5.0, offline=True,
                    migrate_on_start=True, voyage_key=None, jev_key=None, anthropic_key=None)


@pytest.fixture
def make_client(settings):
    """make_client(**overrides) -> a started TestClient on a fresh app."""
    import dataclasses

    from fastapi.testclient import TestClient

    from brain.api import create_app
    opened = []

    def _make(**overrides):
        tc = TestClient(create_app(dataclasses.replace(settings, **overrides)))
        tc.__enter__()
        opened.append(tc)
        return tc
    yield _make
    for tc in opened:
        tc.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


ADMIN = bearer(ADMIN_KEY)
