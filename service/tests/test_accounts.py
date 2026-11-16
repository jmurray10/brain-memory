"""Accounts and row-level security (REFERENCE.md, migration 003).

One account must never read, recall, export, forget, restore or list another's data, even when both use the same
namespace names, session ids and idempotency keys, and even when a query forgets its filter."""
import anyio
import httpx2
import psycopg
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from conftest import ADMIN, bearer


def new_account(client, name, **body):
    r = client.post("/v1/admin/accounts", headers=ADMIN, json={"name": name, **body})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture
def two(client):
    """alice and bob, each with an account-admin key; both write namespace 'notes'."""
    a, b = new_account(client, "alice"), new_account(client, "bob")
    return bearer(a["key"]), bearer(b["key"])


def cap(client, h, text, ns="notes", **turn):
    r = client.post("/v1/capture", headers=h, json={"namespace": ns, "session_id": "s1",
                                                    "turns": [{"role": "user", "text": text, **turn}]})
    assert r.status_code == 200, r.text
    return r.json()


def texts(client, h, query, ns="notes"):
    r = client.post("/v1/recall", headers=h, json={"namespace": ns, "query": query})
    assert r.status_code == 200, r.text
    return [it["text"] for it in r.json()["items"]]


def test_create_account_gives_an_account_admin_key(client):
    out = new_account(client, "carol")
    assert out["account"]["name"] == "carol" and out["account"]["kind"] == "individual"
    assert out["client"]["name"] == "account-admin" and out["client"]["is_admin"]
    assert out["client"]["default_ns"] == "carol" and out["key"].startswith("bk_")
    st = client.get("/v1/status", headers=bearer(out["key"])).json()
    assert (st["account"], st["client"], st["operator"]) == ("carol", "account-admin", False)
    assert client.get("/v1/status", headers=ADMIN).json()["account"] == "owner"
    # default namespace: the account name
    r = client.post("/v1/capture", headers=bearer(out["key"]), json={"turns": [{"role": "user", "text": "Hi."}]})
    assert r.json()["namespace"] == "carol"
    assert client.post("/v1/admin/accounts", headers=ADMIN, json={"name": "carol"}).status_code == 409
    assert client.post("/v1/admin/accounts", headers=ADMIN, json={"name": "Bad Name"}).status_code == 400
    r = client.post("/v1/admin/accounts", headers=ADMIN, json={"name": "acme", "kind": "business"})
    assert r.status_code == 501
    names = [a["name"] for a in client.get("/v1/admin/accounts", headers=ADMIN).json()["accounts"]]
    assert names == ["carol", "owner"]


def test_same_namespace_session_and_idempotency_key_stay_separate(client, two):
    alice, bob = two
    a = cap(client, alice, "Alice's sister lives in Austin.", idempotency_key="k1")
    b = cap(client, bob, "Bob's sister lives in Boston.", idempotency_key="k1")
    assert not a["deduped"] and not b["deduped"]      # the same key in another account is a different turn
    assert len(texts(client, alice, "sister live")) == 1
    assert all("Austin" in t for t in texts(client, alice, "sister live"))
    assert all("Boston" in t for t in texts(client, bob, "sister live"))
    ea = client.get("/v1/export?namespace=notes&format=jsonl", headers=alice).text
    eb = client.get("/v1/export?namespace=notes&format=jsonl", headers=bob).text
    assert "Austin" in ea and "Boston" not in ea and "Boston" in eb and "Austin" not in eb
    ans = client.post("/v1/answer", headers=bob, json={"namespace": "notes", "question": "Where does my sister live?"})
    assert "Austin" not in ans.text


def test_forget_and_restore_cannot_reach_another_account(client, two):
    alice, bob = two
    aid = cap(client, alice, "Alice keeps the spare key under the mat.")["episodes"][0]["id"]
    cap(client, bob, "Bob keeps nothing anywhere.")
    # Episode ids are global; bob naming alice's id, or the same session id, forgets nothing of hers.
    f = client.post("/v1/forget", headers=bob, json={"namespace": "notes", "episode_ids": [aid]}).json()
    assert f["forgotten"]["episodes"] == 0
    f = client.post("/v1/forget", headers=bob, json={"namespace": "notes", "session_id": "s1"}).json()
    assert f["forgotten"]["episodes"] == 1          # bob's own turn only
    assert any("spare key" in t for t in texts(client, alice, "spare key"))
    # alice forgets hers; bob (an admin of his own account) cannot restore it
    client.post("/v1/forget", headers=alice, json={"namespace": "notes", "episode_ids": [aid]})
    r = client.post("/v1/admin/restore", headers=bob, json={"namespace": "notes", "episode_ids": [aid]}).json()
    assert r["restored"] == [] and not texts(client, alice, "spare key")
    r = client.post("/v1/admin/restore", headers=alice, json={"namespace": "notes", "episode_ids": [aid]}).json()
    assert [e["id"] for e in r["restored"]] == [aid] and texts(client, alice, "spare key")


def test_directives_and_session_start_are_per_account(client, two):
    alice, bob = two
    client.post("/v1/directives", headers=alice, json={"namespace": "notes", "text": "Answer in French."})
    assert client.get("/v1/directives?namespace=notes", headers=bob).json()["directives"] == []
    assert client.get("/v1/session_start?namespace=notes", headers=bob).json()["prompt"] == ""
    did = client.get("/v1/directives?namespace=notes", headers=alice).json()["directives"][0]["id"]
    assert client.delete(f"/v1/directives/{did}?namespace=notes", headers=bob).status_code == 404


def test_account_admin_limits_and_client_names_per_account(client, two):
    alice, bob = two
    body = {"name": "claude-code", "read_ns": ["notes"], "write_ns": ["notes"], "default_ns": "notes"}
    ka = client.post("/v1/admin/clients", headers=alice, json=body)
    kb = client.post("/v1/admin/clients", headers=bob, json=body)
    assert ka.status_code == kb.status_code == 201        # the same client name in two accounts
    assert client.post("/v1/admin/clients", headers=alice, json=body).status_code == 409
    listed = client.get("/v1/admin/clients", headers=alice).json()["clients"]
    assert sorted(c["name"] for c in listed) == ["account-admin", "claude-code"]
    # account admins stay inside their account; accounts and provider spend are the operator's
    assert client.post("/v1/admin/clients", headers=alice, json={**body, "name": "x", "account": "bob"}).status_code == 403
    assert client.get("/v1/admin/clients?account=bob", headers=alice).status_code == 403
    assert client.post("/v1/admin/clients/claude-code/revoke?account=bob", headers=alice).status_code == 403
    assert client.get("/v1/admin/accounts", headers=alice).status_code == 403
    assert client.post("/v1/admin/accounts", headers=alice, json={"name": "eve"}).status_code == 403
    assert client.get("/v1/admin/spend", headers=alice).status_code == 403
    # revoking alice's claude-code leaves bob's working
    assert client.post("/v1/admin/clients/claude-code/revoke", headers=alice).status_code == 200
    assert client.get("/v1/status", headers=bearer(ka.json()["key"])).status_code == 401
    assert client.get("/v1/status", headers=bearer(kb.json()["key"])).status_code == 200
    # the operator can act in a named account
    r = client.post("/v1/admin/clients", headers=ADMIN, json={**body, "name": "support", "account": "bob"})
    assert r.status_code == 201
    assert client.get("/v1/status", headers=bearer(r.json()["key"])).json()["account"] == "bob"
    assert client.get("/v1/admin/clients?account=nobody", headers=ADMIN).status_code == 404


def test_disabled_account_keys_stop(client, two):
    alice, _ = two
    assert client.post("/v1/admin/accounts/alice/disable", headers=ADMIN).status_code == 200
    r = client.get("/v1/status", headers=alice)
    assert r.status_code == 403 and r.json()["error"] == "account_disabled"
    assert client.post("/v1/admin/accounts/alice/enable", headers=ADMIN).status_code == 200
    assert client.get("/v1/status", headers=alice).status_code == 200
    assert client.post("/v1/admin/accounts/owner/disable", headers=ADMIN).status_code == 400


def test_rls_holds_when_a_query_forgets_its_filter(client, two, pool):
    """Raw SQL with no account or namespace filter, run the way the service runs every call."""
    from brain.db import tenant
    alice, bob = two
    cap(client, alice, "Alice row.")
    cap(client, bob, "Bob row.")
    client.post("/v1/recall", headers=alice, json={"namespace": "notes", "query": "row"})   # decision_log rows
    with pool.connection() as c:
        ids = dict(c.execute("select name, id from brain.account").fetchall())
    with tenant(pool, ids["alice"]) as c:
        bodies = [r[0] for r in c.execute("select body from brain.episode").fetchall()]
        chunks = [r[0] for r in c.execute("select text from brain.chunk").fetchall()]
        accounts = [r[0] for r in c.execute("select name from brain.account").fetchall()]
        clients = {r[0] for r in c.execute("select account_id from brain.api_client").fetchall()}
        logs = {r[0] for r in c.execute("select account_id from brain.decision_log").fetchall()}
    assert bodies == ["Alice row."] and all("Alice" in t for t in chunks)
    assert accounts == ["alice"] and clients == {ids["alice"]} and logs == {ids["alice"]}
    with tenant(pool, ids["bob"]) as c:
        assert [r[0] for r in c.execute("select body from brain.episode").fetchall()] == ["Bob row."]
        assert c.execute("select count(*) from brain.decision_log").fetchone()[0] == 0
        # an update or delete aimed at everything touches only bob's rows
        assert c.execute("update brain.episode set speaker = 'x'").rowcount == 1
        assert c.execute("delete from brain.chunk").rowcount == 1
        c.rollback()
    # writing a row into another account is refused
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with tenant(pool, ids["bob"]) as c:
            c.execute("""insert into brain.episode (account_id, namespace, client, ts, role, body)
                         values (%s, 'notes', 'x', now(), 'user', 'planted')""", (ids["alice"],))
    # brain_app without a tenant sees nothing and cannot insert (fail closed)
    with pool.connection() as c:
        c.execute("set local role brain_app")
        assert c.execute("select count(*) from brain.episode").fetchone()[0] == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):      # the policy refuses before not-null
            c.execute("insert into brain.episode (namespace, client, ts, role, body) "
                      "values ('notes', 'x', now(), 'user', 'no tenant')")
        c.rollback()
    with pool.connection() as c:
        super_, bypass = c.execute("select rolsuper, rolbypassrls from pg_roles where rolname = 'brain_app'").fetchone()
        assert not super_ and not bypass
        owner = c.execute("select tableowner from pg_tables where schemaname = 'brain' and tablename = 'episode'").fetchone()[0]
        assert owner != "brain_app"
        no_rls = c.execute("""select array_agg(tablename) from pg_tables where schemaname = 'brain'
                              and not rowsecurity""").fetchone()[0]
        assert no_rls is None


def test_mcp_is_scoped_to_the_key_account(settings, client, two):
    import dataclasses

    from brain.api import create_app
    alice, bob = two
    cap(client, alice, "Alice's boat is called Juniper.")
    app = create_app(dataclasses.replace(settings))

    async def main():
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://brain.test",
                                          headers=bob) as http:
                async with Client(streamable_http_client("http://brain.test/mcp", http_client=http)) as c:
                    return await c.call_tool("memory_recall", {"namespace": "notes", "query": "boat Juniper"})
    r = anyio.run(main)
    assert not r.is_error and r.structured_content["items"] == []


def test_migration_003_moves_existing_rows_to_the_owner_account(db_url, tmp_path, monkeypatch, make_client):
    """A database built before accounts: its rows and keys land in the owner account and keep working."""
    import shutil

    import brain.db as dbm
    from brain.auth import hash_key
    with psycopg.connect(db_url, autocommit=True) as c:
        c.execute("drop schema brain cascade")
    old = tmp_path / "migrations"
    old.mkdir()
    for name in ("001_init.sql", "002_forget_grace.sql"):
        shutil.copy(dbm.MIGRATIONS / name, old / name)
    monkeypatch.setattr(dbm, "MIGRATIONS", old)
    assert dbm.migrate(db_url) == ["001_init.sql", "002_forget_grace.sql"]
    with psycopg.connect(db_url) as c:
        c.execute("""insert into brain.api_client (name, key_hash, read_ns, write_ns, default_ns)
                     values ('legacy', %s, '{jeff}', '{jeff}', 'jeff')""", (hash_key("bk_legacy_key"),))
        eid = c.execute("""insert into brain.episode (namespace, client, ts, role, body, idempotency_key)
                           values ('jeff', 'legacy', now(), 'user', 'The old boat was Kestrel.', 'k-old')
                           returning id""").fetchone()[0]
        c.execute("""insert into brain.chunk (namespace, kind, ref_id, text)
                     values ('jeff', 'episode_line', %s, '[2026-11-01] user: The old boat was Kestrel.')""", (eid,))
    monkeypatch.undo()
    assert dbm.migrate(db_url)[0] == "003_accounts.sql"       # then any later migrations
    with psycopg.connect(db_url) as c:
        owner = c.execute("select id from brain.account where name = 'owner'").fetchone()[0]
        for t in ("api_client", "episode", "chunk"):
            assert c.execute(f"select array_agg(distinct account_id) from brain.{t}").fetchone()[0] == [owner]
    client = make_client()
    legacy = bearer("bk_legacy_key")
    assert client.get("/v1/status", headers=legacy).json()["account"] == "owner"
    assert any("Kestrel" in t for t in texts(client, legacy, "old boat", ns="jeff"))
    r = client.post("/v1/capture", headers=legacy, json={"namespace": "jeff", "turns": [
        {"role": "user", "text": "The old boat was Kestrel.", "idempotency_key": "k-old"}]}).json()
    assert r["deduped"] == 1      # the idempotency index moved to (account_id, namespace, key) intact


def test_one_account_cannot_use_up_another_accounts_rate_limit(make_client):
    """Every account's first key is named account-admin, and key names are unique only within an account: the
    limit is kept per account and key, so alice running into hers leaves bob's untouched."""
    c = make_client(rate_per_min=3)
    alice, bob = bearer(new_account(c, "alice")["key"]), bearer(new_account(c, "bob")["key"])
    for _ in range(3):
        assert c.get("/v1/status", headers=alice).status_code == 200
    assert c.get("/v1/status", headers=alice).status_code == 429
    assert c.get("/v1/status", headers=bob).status_code == 200


def test_one_account_cannot_spend_another_accounts_daily_budget(make_client):
    """Release fix B2: with a per-account cap, alice at her cap gets recall without a rerank (degraded) while
    bob's recall is whole, and the operator's report shows who spent what."""
    c = make_client(account_cap_usd=0.0001)
    alice, bob = bearer(new_account(c, "alice")["key"]), bearer(new_account(c, "bob")["key"])
    for h in (alice, bob):
        cap(c, h, "The gate code is 4417.")
    meter = c.app.state.meter
    ids = dict((n, i) for i, n in c.app.state.pool.connection().__enter__().execute("select id, name from brain.account"))
    meter.store.add(meter.today(), "jev", 0.0001, 1, 0, 1, account=ids["alice"])
    ra = c.post("/v1/recall", headers=alice, json={"namespace": "notes", "query": "gate code"}).json()
    rb = c.post("/v1/recall", headers=bob, json={"namespace": "notes", "query": "gate code"}).json()
    assert ra.get("degraded") and not rb.get("degraded")
    report = c.get("/v1/admin/spend", headers=ADMIN).json()
    assert report["account_cap_usd"] == 0.0001
    assert {"account": "alice", "usd": 0.0001} in report["today_by_account"]


def test_an_account_exports_everything_it_holds_and_nothing_of_another(make_client, pool):
    """Release fix B5: the user's own copy of their data, across namespaces, and only theirs."""
    import json

    from brain import worker
    from test_ledger import Providers, _claim
    c = make_client(write_path_enabled=True)
    alice, bob = bearer(new_account(c, "alice")["key"]), bearer(new_account(c, "bob")["key"])
    text = "I moved to Denver last spring."
    r = c.post("/v1/capture", headers=alice, json={"namespace": "notes", "session_id": "s1", "turns": [
        {"role": "user", "text": text, "claims": [_claim("new:home_city", "Denver", text)]}]})
    assert r.status_code == 200, r.text
    cap(c, alice, "Work note: the launch is in March.", ns="work")
    gone = cap(c, alice, "Alice forgot this.")["episodes"][0]["id"]
    c.post("/v1/forget", headers=alice, json={"namespace": "notes", "episode_ids": [gone]})
    c.post("/v1/directives", headers=alice, json={"namespace": "notes", "text": "Use metric units."})
    cap(c, bob, "Bob's secret: the gate code is 4417.")
    worker.write_path(pool, Providers())

    r = c.get("/v1/admin/export", headers=alice)
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    assert 'filename="brain-account-alice.jsonl"' in r.headers["content-disposition"]
    rows = [json.loads(x) for x in r.text.splitlines()]
    assert rows[0]["t"] == "export" and rows[0]["account"] == "alice"
    by = {t: [x for x in rows if x["t"] == t] for t in ("episode", "claim_sent", "fact", "directive")}
    assert sorted((e["namespace"], e["text"]) for e in by["episode"]) == [
        ("notes", text), ("work", "Work note: the launch is in March.")]           # every namespace, live turns only
    assert [(k["slot"], k["value"], k["status"]) for k in by["claim_sent"]] == [("new:home_city", "Denver", "placed")]
    assert [(f["attribute"], f["value"], f["status"]) for f in by["fact"]] == [("home_city", "Denver", "active")]
    assert [d["text"] for d in by["directive"]] == ["Use metric units."]
    assert "4417" not in r.text and "forgot this" not in r.text and "account_id" not in r.text

    key = c.post("/v1/admin/clients", headers=alice, json={"name": "bot", "read_ns": ["notes"], "write_ns": ["notes"],
                                                           "default_ns": "notes"}).json()["key"]
    assert c.get("/v1/admin/export", headers=bearer(key)).status_code == 403      # an admin key, not a client's


def test_status_shows_a_key_its_own_accounts_spend_and_the_operator_the_whole_bill(client, pool, two):
    from brain.spend import PgSpendStore, utc_today
    alice, bob = two
    with pool.connection() as conn:
        ids = dict(conn.execute("select name, id from brain.account").fetchall())
    store = PgSpendStore(pool)
    store.add(utc_today(), "jev", 0.5, 1000, 0, 10, ids["alice"])
    store.add(utc_today(), "voyage", 0.25, 1000, 0, 5, ids["bob"])
    store.add(utc_today(), "voyage", 0.125, 1000, 0, 5)                  # no account: the backfill
    spent = lambda h: client.get("/v1/status", headers=h).json()["spent_today_usd"]      # noqa: E731
    assert spent(alice) == pytest.approx(0.5) and spent(bob) == pytest.approx(0.25)
    assert spent(ADMIN) == pytest.approx(0.875)
