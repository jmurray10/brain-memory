"""MCP (REFERENCE.md): the SDK client against /mcp on the in-process app, the auth gate, and the
stdio proxy's tools forwarded to REST. Needs TEST_DATABASE_URL (the db fixtures skip without it)."""
import dataclasses
import json

import anyio
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from conftest import ADMIN, ADMIN_KEY, bearer

TOOLS = {"memory_capture", "memory_recall", "memory_answer", "memory_forget", "memory_session_start",
         "memory_directives", "memory_add_directive", "memory_page", "memory_current", "memory_history", "memory_context",
         "memory_restore"}        # no memory_erase: erasing takes the admin key outside any model (migration 014)


def run_mcp(app, key, fn):
    """Start the app's lifespan in this event loop, connect the SDK client over streamable HTTP, run fn(client)."""
    async def main():
        async with app.router.lifespan_context(app):
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://brain.test",
                                          headers=headers) as http:
                async with Client(streamable_http_client("http://brain.test/mcp", http_client=http)) as c:
                    return await fn(c)
    return anyio.run(main)


@pytest.fixture
def app(settings):
    from brain.api import create_app
    return create_app(dataclasses.replace(settings))


def _make_key(app, **body):
    from fastapi.testclient import TestClient
    with TestClient(app) as tc:
        r = tc.post("/v1/admin/clients", headers=ADMIN, json=body)
        assert r.status_code == 201, r.text
        return r.json()["key"]


def test_tools_listed_and_capture_then_recall(app):
    async def fn(c):
        tools = (await c.list_tools()).tools
        names = {t.name for t in tools}
        r1 = await c.call_tool("memory_capture", {"namespace": "jeff", "session_id": "s1", "turns": [
            {"role": "user", "text": "My sister lives in Austin.", "ts": "2026-11-01T10:00:00Z"}]})
        r2 = await c.call_tool("memory_recall", {"namespace": "jeff", "query": "Where does my sister live?"})
        r3 = await c.call_tool("memory_session_start", {"namespace": "jeff"})
        return names, tools, r1, r2, r3
    names, tools, cap, rec, ss = run_mcp(app, ADMIN_KEY, fn)
    assert names == TOOLS      # the 501 features are not listed (section 7.7)
    recall = next(t for t in tools if t.name == "memory_recall")
    assert "untrusted" in recall.description and recall.annotations.read_only_hint
    assert next(t for t in tools if t.name == "memory_forget").annotations.destructive_hint
    assert not cap.is_error and cap.structured_content["episodes"][0]["chunks"] == 1
    assert not rec.is_error
    items = rec.structured_content["items"]
    assert items and "Austin" in items[0]["text"] and items[0]["text"].startswith("[2026-11-01] user:")
    assert rec.structured_content["request_id"].startswith("req_")
    assert not ss.is_error and ss.structured_content["directives"] == []


def test_mcp_needs_a_valid_key(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as tc:
        init = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        r = tc.post("/mcp", json=init)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"
        r = tc.post("/mcp", json=init, headers=bearer("bk_not_a_real_key"))
        assert r.status_code == 401


def test_mcp_scopes_and_errors(app):
    key = _make_key(app, name="reader", read_ns=["jeff"], write_ns=[], default_ns="jeff")

    async def fn(c):
        denied = await c.call_tool("memory_capture", {"turns": [{"role": "user", "text": "hello"}]})
        other = await c.call_tool("memory_recall", {"namespace": "work", "query": "anything"})
        bad = await c.call_tool("memory_recall", {"query": "x", "k": 500})
        ok = await c.call_tool("memory_recall", {"query": "hello"})   # default namespace
        return denied, other, bad, ok
    denied, other, bad, ok = run_mcp(app, key, fn)
    for r, code in ((denied, "forbidden"), (other, "forbidden"), (bad, "invalid_request")):
        assert r.is_error
        text = r.content[0].text
        assert json.loads(text[text.index("{"):])["error"] == code
    assert not ok.is_error and ok.structured_content["namespaces"] == ["jeff"]


def test_mcp_forget_keeps_grace_period(app, pool):
    async def fn(c):
        cap = await c.call_tool("memory_capture", {"namespace": "jeff", "turns": [
            {"role": "user", "text": "Forget me later."}]})
        eid = cap.structured_content["episodes"][0]["id"]
        tools = {t.name: t for t in (await c.list_tools()).tools}
        f = await c.call_tool("memory_forget", {"namespace": "jeff", "episode_ids": [eid]})
        return eid, tools, f
    eid, tools, f = run_mcp(app, ADMIN_KEY, fn)
    assert "purge" not in tools["memory_forget"].input_schema["properties"]
    assert not f.is_error and f.structured_content["forgotten"]["episodes"] == 1
    assert f.structured_content["archived"] and f.structured_content["restorable"]
    with pool.connection() as c:
        body, purged = c.execute("select body, purged_at from brain.episode where id = %s", (eid,)).fetchone()
    assert body == "Forget me later." and purged is None


def test_mcp_access_log_has_client_and_namespace(app, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="brain.access")

    async def fn(c):
        return await c.call_tool("memory_recall", {"namespace": "jeff", "query": "anything"})
    run_mcp(app, ADMIN_KEY, fn)
    lines = [r.getMessage() for r in caplog.records if r.name == "brain.access" and "/mcp" in r.getMessage()]
    assert any("client=owner/admin" in ln and "ns=jeff" in ln for ln in lines), lines
    assert all(ADMIN_KEY not in ln for ln in lines)


def test_stdio_proxy_tools_forward_to_rest(client):
    """The stdio proxy's tool set, with a RestBackend whose HTTP client is the in-process TestClient."""
    from brain.client import BrainClient
    from brain.mcp_stdio import RestBackend
    from brain.mcp_server import build_server

    server = build_server(RestBackend(BrainClient(key=ADMIN_KEY, http=client)))

    async def main():
        async with Client(server) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            cap = await c.call_tool("memory_capture", {"namespace": "jeff", "turns": [
                {"role": "user", "text": "The boat is called Juniper."}]})
            rec = await c.call_tool("memory_recall", {"namespace": "jeff", "query": "boat name"})
            add = await c.call_tool("memory_add_directive", {"namespace": "jeff", "text": "Use metric units."})
            lst = await c.call_tool("memory_directives", {"namespace": "jeff"})
            bad = await c.call_tool("memory_recall", {"namespace": "Not Valid", "query": "x"})
            return names, cap, rec, add, lst, bad
    names, cap, rec, add, lst, bad = anyio.run(main)
    assert names == TOOLS
    assert not cap.is_error and not rec.is_error
    assert "Juniper" in rec.structured_content["items"][0]["text"]
    assert not add.is_error and [d["text"] for d in lst.structured_content["directives"]] == ["Use metric units."]
    assert bad.is_error and "bad_namespace" in bad.content[0].text
