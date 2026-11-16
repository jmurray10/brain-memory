"""Build step 8: code-rendered pages, the ledger endpoints (current, history, page, resolve_conflict) and session start."""
from brain import pages
from conftest import ADMIN
from test_ledger import Providers, _cap, _claim, _rows


def _build(client, pool):
    from brain import worker
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    _cap(client, "s3", "2023-06-02T10:00:00Z", "My sister Maya lives in Austin.",
         [_claim("new:home_city", "Austin", "My sister Maya lives in Austin.", mention="My sister Maya")])
    worker.write_path(pool, Providers())
    return worker.render_pages(pool)


def test_pages_render_from_claims_and_cite_every_line(make_client, pool):
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    out = _build(client, pool)
    assert out["rendered"] == 2 and out["lines"] == 3
    assert _rows(pool, "select count(*) from brain.page where dirty") == [(0,)]
    p = client.get("/v1/page", headers=ADMIN, params={"namespace": "jeff", "entity": "user"}).json()
    assert p["summary"] == "user: home city Boston"
    body = p["body_md"]
    assert body.startswith("# user") and "## home city" in body
    assert body.index("- Boston (now; since 2023-06-01)") < body.index("- Denver (earlier, until 2023-06-01")
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        pid = c.execute("select id from brain.page order by id limit 1").fetchone()[0]
        check = pages.citation_check(c, pid)
    assert check["lines"] == check["supported"] == 2
    assert _rows(pool, "select count(*) from brain.chunk where kind = 'page'") == [(2,)]

    s = client.get("/v1/session_start", headers=ADMIN, params={"namespace": "jeff"}).json()
    assert {x["entity"] for x in s["pages"]} == {"user", "My sister Maya"}
    assert "People and things in the user's memory" in s["prompt"]


def test_current_history_and_resolve_conflict(make_client, pool):
    client = make_client(write_path_enabled=True)
    _build(client, pool)
    cur = client.get("/v1/current", headers=ADMIN, params={"namespace": "jeff", "entity": "user"}).json()
    assert [r["value"] for r in cur["current"]["home_city"]] == ["Boston"]
    hist = client.get("/v1/history", headers=ADMIN,
                      params={"namespace": "jeff", "entity": "user", "slot": "home city"}).json()
    assert [(r["value"], r["status"]) for r in hist["history"]] == [("Denver", "superseded"), ("Boston", "active")]
    maya = client.get("/v1/current", headers=ADMIN, params={"namespace": "jeff", "entity": "my sister maya"}).json()
    assert maya["current"]["home_city"][0]["value"] == "Austin"           # aliases and case match

    # make Maya's city contested, then let the user settle it
    with pool.connection() as c:
        a = c.execute("select id from brain.claim where value = 'Austin'").fetchone()[0]
        b = c.execute("""insert into brain.claim (account_id, namespace, entity_id, slot_id, value, valid_from, status,
                                                  quote, episode_id)
                         select account_id, namespace, entity_id, slot_id, 'Dallas', '2023-06-03', 'contested',
                                'Maya lives in Dallas.', episode_id from brain.claim where id = %s returning id""",
                      (a,)).fetchone()[0]
        c.execute("update brain.claim set status = 'contested' where id = %s", (a,))
        c.execute("""insert into brain.claim_link (account_id, from_claim, to_claim, type)
                     select account_id, %s, %s, 'contradicts' from brain.claim where id = %s""", (b, a, a))
    both = client.get("/v1/current", headers=ADMIN, params={"namespace": "jeff", "entity": "My sister Maya"}).json()
    assert {r["value"] for r in both["current"]["home_city"]} == {"Austin", "Dallas"}      # a conflict shows both
    r = client.post("/v1/resolve_conflict", headers=ADMIN, json={"namespace": "jeff", "keep_claim_id": b})
    assert r.status_code == 200 and r.json()["superseded"] == 1
    now = client.get("/v1/current", headers=ADMIN, params={"namespace": "jeff", "entity": "My sister Maya"}).json()
    assert [x["value"] for x in now["current"]["home_city"]] == ["Dallas"]


class SupportJev:
    """Says a line is supported unless its statement names a value in `deny`."""
    model = "fake-jev"

    def __init__(self, deny=()):
        self.deny, self.asked = tuple(deny), []

    def ask_many(self, requests):
        from brain.providers import JevResult
        out = []
        for state, questions in requests:
            assert list(questions) == ["supported"] and questions["supported"]["type"] == "noul"
            assert state["source"] and " = " in state["statement"]
            self.asked.append(state["statement"])
            p = 0.1 if any(d in state["statement"] for d in self.deny) else 0.9
            out.append(JevResult({"supported": {"noul": p}}, self.model))
        return out


def test_line_support_audit_measures_the_step_8_gate(make_client, pool):
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _build(client, pool)
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    before = _rows(pool, "select count(*) from brain.decision_log")

    with tenant(pool, owner) as c:
        empty = pages.line_support_audit(c, SupportJev(), limit=0)
        every = pages.line_support_audit(c, SupportJev())
        jev = SupportJev(deny=["Denver"])
        some = pages.line_support_audit(c, jev)
        again = pages.line_support_audit(c, SupportJev(deny=["Denver"]))
    assert empty["lines"] == 0 and empty["gate"] == "no lines" and empty["share"] is None
    assert (every["population"], every["lines"], every["supported"], every["gate"]) == (3, 3, 3, "passed")
    assert every["quote_found"] == 3 and every["model"] == "fake-jev" and every["unsupported"] == []
    assert sorted(jev.asked) == ["My sister Maya: home city = Austin", "user: home city = Boston",
                                 "user: home city = Denver"]
    assert (some["lines"], some["supported"], some["share"], some["gate"]) == (3, 2, 0.6667, "not passed")
    assert some["ci95"][0] < 0.6667 < some["ci95"][1] < 0.95
    assert [(u["reason"], u["p"]) for u in some["unsupported"]] == [("the turn does not support the line", 0.1)]
    assert "Denver" in some["unsupported"][0]["sentence"] and some == again        # same seed, same sample
    with tenant(pool, owner) as c:
        full = pages.line_support_audit(c, SupportJev(deny=["Denver"]), detail=True)
    assert {k: v for k, v in full.items() if k != "detail"} == some
    assert sorted(d["statement"] for d in full["detail"]) == sorted(jev.asked)
    assert all(d["source"] and d["p"] is not None for d in full["detail"])
    sys_path_scripts = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"
    __import__("sys").path.insert(0, str(sys_path_scripts))
    import page_audit
    picked = page_audit.handcheck_sample(full["detail"], seed=0, each=1)
    assert [d["reason"] for d in picked] == [None, "the turn does not support the line"]
    failed = page_audit.failures(full["detail"])                     # every failure, with the slot Jev judged
    assert failed and all(d["reason"] and d["statement"] and "source" not in d for d in failed)
    assert len(failed) == sum(1 for d in full["detail"] if d["reason"])
    assert page_audit.combine([some, every])["lines"] == 6 and page_audit.combine([some, every])["supported"] == 5
    assert page_audit.combine([])["gate"] == "no lines"

    # a forgotten turn: the line fails in code and Jev is not asked about it
    with pool.connection() as c:
        c.execute("update brain.episode set deleted_at = now() where body like '%Denver%'")
    with tenant(pool, owner) as c:
        jev = SupportJev()
        gone = pages.line_support_audit(c, jev)
    assert gone["supported"] == 2 and len(jev.asked) == 2
    assert gone["unsupported"][0]["reason"] == "cites a forgotten turn" and gone["unsupported"][0]["p"] is None
    assert _rows(pool, "select count(*) from brain.decision_log") == before       # the audit writes nothing
