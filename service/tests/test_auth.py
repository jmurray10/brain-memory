"""Auth, namespaces, scopes, admin client keys, limits and error shape (REFERENCE.md).
Unit tests run offline; API tests need TEST_DATABASE_URL."""
import hashlib
import logging

import pytest

from brain.auth import ADMIN_CALLER, Caller, generate_key, hash_key, parse_bearer, valid_ns
from brain.errors import ApiError
from brain.middleware import RateLimiter
from conftest import ADMIN, ADMIN_KEY, bearer


# ---------------------------------------------------------------- unit

def test_key_format_and_hash():
    k = generate_key()
    assert k.startswith("bk_") and len(k) == 3 + 43 and generate_key() != k   # 32 bytes urlsafe = 43 chars
    assert hash_key(k) == hashlib.sha256(k.encode()).hexdigest()


@pytest.mark.parametrize("ns,ok", [("jeff", True), ("work", True), ("a", True), ("t-1_x", True),
                                   ("a" * 63, True), ("a" * 64, False), ("Jeff", False), ("-x", False),
                                   ("_x", False), ("", False), ("a b", False), ("*", False), ("a/b", False)])
def test_namespace_rule(ns, ok):
    assert valid_ns(ns) is ok


def test_parse_bearer():
    assert parse_bearer("Bearer abc") == "abc"
    assert parse_bearer("bearer  abc ") == "abc"
    assert parse_bearer("Basic abc") is None
    assert parse_bearer("Bearer") is None
    assert parse_bearer(None) is None


def test_caller_scopes_and_default_namespace():
    c = Caller("bot", ("jeff", "shared"), ("jeff",), "jeff", False)
    assert c.namespace(None) == "jeff"
    assert c.namespace("shared") == "shared"
    assert c.namespace(None, write=True) == "jeff"
    with pytest.raises(ApiError) as e:
        c.namespace("shared", write=True)
    assert e.value.status == 403
    with pytest.raises(ApiError) as missing:
        c.namespace("nosuch")
    # same answer for a namespace that exists elsewhere and one that never existed
    assert missing.value.status == 403 and missing.value.message == "this key has no read scope for that namespace"
    with pytest.raises(ApiError) as bad:
        c.namespace("Bad Name")
    assert bad.value.status == 400
    with pytest.raises(ApiError) as nodefault:
        Caller("x", ("*",), (), None, False).namespace(None)
    assert nodefault.value.code == "namespace_required"
    assert ADMIN_CALLER.namespace("anything", write=True) == "anything"


def test_rate_limiter_sliding_window():
    t = [0.0]
    rl = RateLimiter(3, clock=lambda: t[0])
    assert [rl.allow("a")[0] for _ in range(3)] == [True] * 3
    ok, retry = rl.allow("a")
    assert not ok and 1 <= retry <= 61
    assert rl.allow("b")[0]              # per key
    t[0] = 59.9
    assert not rl.allow("a")[0]
    t[0] = 60.0
    assert rl.allow("a")[0]


# ---------------------------------------------------------------- API

def create(client, name="claude-code", read=("jeff",), write=("jeff",), default: str | None = "jeff", is_admin=False):
    return client.post("/v1/admin/clients", headers=ADMIN, json={
        "name": name, "read_ns": list(read), "write_ns": list(write), "default_ns": default, "is_admin": is_admin})


def test_no_or_bad_key_is_401_with_error_shape(client):
    for headers in ({}, bearer("bk_nope"), {"Authorization": "Basic x"}):
        r = client.get("/v1/status", headers=headers)
        assert r.status_code == 401
        body = r.json()
        assert body["error"] == "unauthorized" and body["request_id"].startswith("req_")
        assert r.headers["x-request-id"] == body["request_id"]


def test_admin_key_from_env(client):
    r = client.get("/v1/status", headers=ADMIN)
    assert r.status_code == 200
    s = r.json()
    assert s["client"] == "admin" and s["offline"] is True and s["spent_today_usd"] == 0.0
    assert s["keys"] == {"database": True, "admin": True, "voyage": False, "jev": False, "anthropic": False,
                         "backup_storage": False}
    assert ADMIN_KEY not in r.text


def test_create_client_returns_key_once_and_it_works(client, pool):
    r = create(client)
    assert r.status_code == 201
    key, c = r.json()["key"], r.json()["client"]
    assert key.startswith("bk_") and c["name"] == "claude-code" and c["read_ns"] == ["jeff"]
    assert c["default_ns"] == "jeff" and c["is_admin"] is False and c["revoked_at"] is None
    with pool.connection() as conn:
        stored = conn.execute("select key_hash from brain.api_client where name = 'claude-code'").fetchone()[0]
    assert stored == hash_key(key) and key not in stored
    listed = client.get("/v1/admin/clients", headers=ADMIN)
    assert listed.status_code == 200 and key not in listed.text and "key_hash" not in listed.text
    assert [x["name"] for x in listed.json()["clients"]] == ["claude-code"]
    st = client.get("/v1/status", headers=bearer(key))
    assert st.status_code == 200 and st.json()["client"] == "claude-code"


def test_client_key_cannot_use_admin_endpoints(client):
    key = create(client).json()["key"]
    for method, path in (("get", "/v1/admin/clients"), ("post", "/v1/admin/clients/x/revoke"),
                         ("get", "/v1/admin/spend")):
        r = getattr(client, method)(path, headers=bearer(key))
        assert r.status_code == 403 and r.json()["error"] == "forbidden"
    assert create(client, name="other").status_code == 201  # admin still can
    r = client.post("/v1/admin/clients", headers=bearer(key), json={"name": "evil", "read_ns": ["*"]})
    assert r.status_code == 403


def test_client_flagged_admin_can_administer(client):
    key = create(client, name="ops", read=("*",), write=(), default=None, is_admin=True).json()["key"]
    assert client.get("/v1/admin/clients", headers=bearer(key)).status_code == 200


def test_revoke_makes_key_401(client):
    key = create(client).json()["key"]
    r = client.post("/v1/admin/clients/claude-code/revoke", headers=ADMIN)
    assert r.status_code == 200 and r.json()["client"]["revoked_at"]
    assert client.get("/v1/status", headers=bearer(key)).status_code == 401
    assert client.post("/v1/admin/clients/claude-code/revoke", headers=ADMIN).status_code == 404
    assert create(client).status_code == 409          # names are never reused


@pytest.mark.parametrize("body,code", [
    ({"name": "Bad Name"}, "bad_name"),
    ({"name": "admin"}, "bad_name"),
    ({"name": "x", "read_ns": ["Jeff"]}, "bad_namespace"),
    ({"name": "x", "write_ns": ["ok", "no way"]}, "bad_namespace"),
    ({"name": "x", "read_ns": ["a"], "default_ns": "b"}, "bad_default_ns"),
])
def test_create_client_validation(client, body, code):
    r = client.post("/v1/admin/clients", headers=ADMIN, json=body)
    assert r.status_code == 400 and r.json()["error"] == code


def test_unknown_fields_rejected(client):
    r = client.post("/v1/admin/clients", headers=ADMIN, json={"name": "x", "key_hash": "abc"})
    assert r.status_code == 422 and r.json()["error"] == "invalid_request" and r.json()["request_id"]


def test_wildcard_scope_and_default(client):
    r = create(client, name="grok", read=("*",), write=("jeff", "jeff"), default="work")
    assert r.status_code == 201 and r.json()["client"]["write_ns"] == ["jeff"]   # deduped


def test_admin_spend_report(client, pool):
    from datetime import timedelta

    from brain.spend import PgSpendStore, utc_today
    store = PgSpendStore(pool)
    store.add(utc_today(), "jev", 0.0021, 50000, 0, 30)
    store.add(utc_today() - timedelta(days=3), "haiku", 0.01, 2000, 300, 2)
    store.add(utc_today() - timedelta(days=30), "voyage", 1.0, 1, 0, 1)
    r = client.get("/v1/admin/spend?days=7", headers=ADMIN)
    assert r.status_code == 200
    j = r.json()
    assert [d["provider"] for d in j["days"]] == ["jev", "haiku"]
    assert j["total_usd"] == pytest.approx(0.0121) and j["today_usd"] == pytest.approx(0.0021)
    assert j["daily_cap_usd"] == 5.0
    assert client.get("/v1/admin/spend?days=0", headers=ADMIN).status_code == 422
    assert client.get("/v1/status", headers=ADMIN).json()["spent_today_usd"] == pytest.approx(0.0021)


def test_rate_limit_429_per_key(make_client):
    c = make_client(rate_per_min=3)
    for _ in range(3):
        assert c.get("/v1/status", headers=ADMIN).status_code == 200
    r = c.get("/v1/status", headers=ADMIN)
    assert r.status_code == 429 and r.json()["error"] == "rate_limited" and int(r.headers["retry-after"]) >= 1


def test_body_over_limit_is_413(make_client):
    c = make_client(max_body_bytes=1000)
    r = c.post("/v1/admin/clients", headers={**ADMIN, "content-type": "application/json"},
               content=b'{"name": "' + b"x" * 2000 + b'"}')
    assert r.status_code == 413 and r.json()["error"] == "body_too_large" and r.json()["request_id"]

    def chunks():
        yield b'{"name": "'
        yield b"x" * 2000
        yield b'"}'
    r = c.post("/v1/admin/clients", headers={**ADMIN, "content-type": "application/json"}, content=chunks())
    assert r.status_code == 413
    ok = c.post("/v1/admin/clients", headers=ADMIN, json={"name": "small", "read_ns": ["jeff"]})
    assert ok.status_code == 201


def test_default_body_limit_is_1mb(client):
    assert client.app.state.settings.max_body_bytes == 1_000_000


def test_unknown_route_uses_error_shape(client):
    r = client.get("/v1/nope", headers=ADMIN)
    assert r.status_code == 404 and r.json()["error"] == "not_found" and r.json()["request_id"]


def test_access_log_has_no_keys_or_bodies(client, caplog):
    caplog.set_level(logging.INFO, logger="brain.access")
    key = create(client).json()["key"]
    client.get("/v1/status", headers=bearer(key))
    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "brain.access")
    assert "GET /v1/status 200" in text and "client=owner/claude-code" in text and "rid=req_" in text
    assert "POST /v1/admin/clients 201" in text and "client=owner/admin" in text
    assert key not in text and ADMIN_KEY not in text and '"read_ns"' not in text
