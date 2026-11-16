"""Forget archives, restore undoes it (2026-11-09): episodes with their claims, and whole entities."""
from conftest import ADMIN
from test_ledger import Providers, _cap, _claim, _rows


def test_restore_brings_claims_back_and_reapplies_supersession(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    boston = _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
                  [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    worker.write_path(pool, Providers())
    client.post("/v1/forget", headers=ADMIN, json={"namespace": "jeff", "episode_ids": [boston]})
    cur = client.get("/v1/current", headers=ADMIN, params={"namespace": "jeff", "entity": "user"}).json()
    assert [r["value"] for r in cur["current"]["home_city"]] == ["Denver"]

    r = client.post("/v1/restore", headers=ADMIN, json={"namespace": "jeff", "episode_ids": [boston]}).json()
    assert r["claims_restored"] == 1
    assert _rows(pool, "select value, status from brain.claim order by id") == [("Denver", "superseded"),
                                                                                ("Boston", "active")]
    cur = client.get("/v1/current", headers=ADMIN, params={"namespace": "jeff", "entity": "user"}).json()
    assert [r["value"] for r in cur["current"]["home_city"]] == ["Boston"]
    assert _rows(pool, "select count(*) from brain.chunk where kind = 'claim'") == [(2,)]


def test_forget_and_restore_an_entity(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "Maya always wants vegetarian restaurants.",
         [_claim("new:diet_preference", "vegetarian restaurants", "Maya always wants vegetarian restaurants.",
                 mention="Maya")])
    worker.write_path(pool, Providers())
    worker.render_pages(pool)
    with pool.connection() as c:       # a directive learned from Maya's preference
        c.execute("""insert into brain.directive (account_id, namespace, text, source_claim, created_by, pinned,
                                                  status)
                     select account_id, namespace, 'Maya: vegetarian', id, 'brain:nightly', true, 'confirmed'
                     from brain.claim""")
    q = {"namespace": "jeff", "entity": "Maya"}
    r = client.post("/v1/forget", headers=ADMIN, json=q).json()
    assert r["archived"] and r["forgotten"] == {"entity": "Maya", "claims": 1, "directives": 1}
    assert client.get("/v1/current", headers=ADMIN, params=q).status_code == 404
    assert client.get("/v1/page", headers=ADMIN, params=q).status_code == 404
    s = client.get("/v1/session_start", headers=ADMIN, params={"namespace": "jeff"}).json()
    assert "Maya" not in s["prompt"] and not s["directives"]
    hits = client.post("/v1/recall", headers=ADMIN, json={"namespace": "jeff", "query": "Maya vegetarian",
                                                          "rerank": False, "screen": False}).json()["items"]
    assert hits                                   # the turn itself stays searchable

    r = client.post("/v1/restore", headers=ADMIN, json=q).json()
    assert r["restored"] == {"entity": "Maya", "directives": 1}
    assert client.get("/v1/current", headers=ADMIN, params=q).json()["current"]["diet_preference"][0]["value"] == \
        "vegetarian restaurants"
    s = client.get("/v1/session_start", headers=ADMIN, params={"namespace": "jeff"}).json()
    assert [d["text"] for d in s["directives"]] == ["Maya: vegetarian"]
    assert client.post("/v1/restore", headers=ADMIN, json=q).status_code == 404          # nothing left to restore
