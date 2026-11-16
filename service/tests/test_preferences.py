"""Build step 7: preference claims, nightly promotion to directives, retirement, and the preference route."""
from brain import preferences
from conftest import ADMIN
from test_ledger import Providers, _cap, _claim, _rows


def test_preferences_become_directives_and_follow_changes(make_client, pool):
    from brain import worker
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "Always give me recipes in metric units.",
         [_claim("new:recipe_units", "metric units", "Always give me recipes in metric units.")])
    _cap(client, "s1", "2023-05-01T10:01:00Z", "I live in Denver.", [_claim("new:home_city", "Denver", "I live in Denver.")])
    worker.write_path(pool, Providers())
    assert _rows(pool, "select value, kind from brain.claim order by id") == [("metric units", "preference"),
                                                                              ("Denver", "fact")]
    out = worker.promote_preferences(pool, Providers())
    assert out["promoted"] == 1 and out["pending"] == 1 and out["screened_out"] == 0 and out["retired"] == 0
    # promotion proposes: pending, unpinned, scoped to the client that captured the turn, served to nobody
    assert _rows(pool, "select status, pinned, scope, screen->>'flagged' from brain.directive") == [
        ("pending", False, "admin", "false")]
    assert client.get("/v1/session_start", headers=ADMIN, params={"namespace": "jeff"}).json()["directives"] == []
    did = _rows(pool, "select id from brain.directive")[0][0]
    r = client.post(f"/v1/admin/directives/{did}/confirm", headers=ADMIN, json={"namespace": "jeff"})
    assert r.status_code == 200 and r.json()["confirmed"]
    s = client.get("/v1/session_start", headers=ADMIN, params={"namespace": "jeff"}).json()
    assert [d["text"] for d in s["directives"]] == ['"Always give me recipes in metric units."']
    assert worker.promote_preferences(pool, Providers())["promoted"] == 0          # nothing promoted twice

    # the preference changes: the old directive retires, the new one is promoted
    _cap(client, "s2", "2023-06-01T10:00:00Z", "Actually, always use imperial units for recipes now.",
         [_claim("new:recipe_units", "imperial units", "Actually, always use imperial units for recipes now.")])
    worker.write_path(pool, Providers())
    out = worker.promote_preferences(pool, Providers())
    assert out["retired"] == 1 and out["promoted"] == 1
    live = _rows(pool, "select text from brain.directive where retired_at is null")
    assert live == [('"Actually, always use imperial units for recipes now."',)]
    assert _rows(pool, "select count(*) from brain.decision_log where stage = 'nightly.promote'") == [(2,)]
    assert _rows(pool, "select count(*), min(outcome) from brain.decision_log where stage = 'directive.screen'") == [
        (2, "pending")]
    did = _rows(pool, "select id from brain.directive where retired_at is null")[0][0]
    assert client.post(f"/v1/admin/directives/{did}/confirm", headers=ADMIN,
                       json={"namespace": "jeff"}).status_code == 200

    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        got = preferences.pack(c, Providers(), ["jeff"], "admin", "Can you suggest a cake recipe?")
    # the claim behind the live directive is listed once (as the directive)
    assert got["directives"] == 1 and got["preferences"] == 0 and "imperial units" in got["block"]


def test_hand_written_directives_are_never_retired_by_the_brain(client, pool):
    from brain import worker
    r = client.post("/v1/directives", headers=ADMIN, json={"namespace": "jeff", "text": "Use British spelling."})
    assert r.status_code == 201
    worker.promote_preferences(pool, Providers())
    assert _rows(pool, "select retired_at from brain.directive") == [(None,)]


def test_pack_skips_what_the_excerpts_already_hold(make_client, pool):
    from brain import worker
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "Always give me recipes in metric units.",
         [_claim("new:recipe_units", "metric units", "Always give me recipes in metric units.")])
    worker.write_path(pool, Providers())
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        assert preferences.pack(c, Providers(), ["jeff"], "admin", "Can you suggest a cake recipe?")["preferences"] == 1
        # the excerpt already carries the quote with its context: no block
        assert preferences.pack(c, Providers(), ["jeff"], "admin", "Can you suggest a cake recipe?",
                                ["[2023-05-01] user: Always give me  recipes in metric units. Thanks!"]) is None
