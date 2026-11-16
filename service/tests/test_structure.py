"""Build step 9: structural scoring and the cold tier (brain/structure.py, migration 009)."""
from brain import structure
from conftest import ADMIN


def _rows(pool, sql, *args):
    with pool.connection() as c:
        return c.execute(sql, args or None).fetchall()


def _cap(client, ts, text):
    r = client.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "turns": [
        {"role": "user", "text": text, "ts": ts}]})
    return r.json()["episodes"][0]["id"]


def test_recency_policy_cold_tier_and_pinned_lines(client, pool):
    from brain.db import tenant
    oldest = _cap(client, "2023-01-01T10:00:00Z", "The violin teacher is Mrs Okafor.")
    _cap(client, "2023-02-01T10:00:00Z", "Bought basil seeds for the window box.")
    _cap(client, "2023-03-01T10:00:00Z", "The quarterly audit moved to Thursday.")
    _cap(client, "2023-04-01T10:00:00Z", "Booked a ferry to the island.")
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        out = structure.run(c, "recency", 0.5, 30)
    assert out["scored"] == 4 and out["went_cold"] == 2 and out["cold"] == 2
    cold = {t for (t,) in _rows(pool, "select text from brain.chunk where cold")}
    assert {t.split("] user: ")[1] for t in cold} == {"The violin teacher is Mrs Okafor.",
                                                     "Bought basil seeds for the window box."}

    # cold lines leave the dense side but stay in keyword search
    r = client.post("/v1/recall", headers=ADMIN, json={"namespace": "jeff", "query": "violin teacher Okafor",
                                                       "rerank": False, "screen": False}).json()
    hit = next(i for i in r["items"] if "Okafor" in i["text"])
    assert hit["scores"]["dense_rank"] is None and hit["scores"]["keyword_rank"] == 1

    # a line whose turn backs a settled claim is pinned: never cold, even when it is the oldest
    with tenant(pool, owner) as c:
        ent = c.execute("insert into entity (namespace, name) values ('jeff', 'violin teacher') returning id").fetchone()[0]
        slot = c.execute("""insert into slot (namespace, entity_id, name, cardinality)
                            values ('jeff', %s, 'name', 'one') returning id""", (ent,)).fetchone()[0]
        c.execute("""insert into claim (namespace, entity_id, slot_id, value, quote, episode_id, status, settled_at)
                     values ('jeff', %s, %s, 'Mrs Okafor', 'The violin teacher is Mrs Okafor.', %s, 'active', now())""",
                  (ent, slot, oldest))
        out = structure.run(c, "recency", 0.5, 30)
    assert out["warmed"] >= 1
    assert not any("Okafor" in t for (t,) in _rows(pool, "select text from brain.chunk where cold"))

    with tenant(pool, owner) as c:                     # fraction 0: everything warms again
        assert structure.run(c, "structural", 0.0, 30)["cold"] == 0


def test_young_lines_never_go_cold(client, pool):
    from datetime import datetime, timezone

    from brain.db import tenant
    _cap(client, datetime.now(timezone.utc).isoformat(), "Fresh note from today.")
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        assert structure.run(c, "structural", 0.9, 30)["cold"] == 0
