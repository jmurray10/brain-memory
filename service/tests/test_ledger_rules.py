"""Room review v2, group 1: the shared end-value rule (ledger.apply_relation), merge_slots without chains and with
undo, the count route's merge and cap flag, the ledger lock around consolidate, and the one-time re-check."""
from datetime import date, timedelta

import psycopg
from psycopg.types.json import Jsonb

from brain import ledger, nightly, recheck, worker
from brain.db import tenant
from brain.preferences import LEARNED_BY
from brain.providers import JevResult
from test_ledger import Providers, WordEmbedder, _cap, _claim, _rows


class RuleJev:
    """Answers the ledger's own questions by rule. r{j}: `relation` (a word, or a function of (new, existing));
    m{k}: 0.9 when the pair of slot names is in `same` (None: every pair matches); i{j}: always counted."""
    model = "rule-jev"

    def __init__(self, relation="updates", same=None):
        self.relation, self.same, self.states = relation, same, []

    def ask_many(self, reqs):
        out = []
        for state, qs in reqs:
            self.states.append(state)
            ans = {}
            for name in qs:
                k = int(name[1:])
                if name[0] == "r":
                    rel = self.relation(state["new"], state["existing"][k]) if callable(self.relation) else self.relation
                    ans[name] = {"choice": rel, "probabilities": {rel: 0.9}}
                elif name[0] == "m":
                    pair = frozenset((state["pairs"][k]["a"]["name"], state["pairs"][k]["b"]["name"]))
                    ans[name] = {"noul": 0.9 if self.same is None or pair in self.same else 0.1}
                elif name[0] == "i":
                    ans[name] = {"noul": 0.9}
                else:
                    raise AssertionError(f"unexpected question {name}")
            out.append(JevResult(ans, self.model))
        return out


class NoJev:
    model = "no-jev"

    def ask_many(self, reqs):
        raise AssertionError("Jev must not be asked here")


class P:
    def __init__(self, jev=None, embedder=True):
        self.jev = jev or RuleJev()
        if embedder:
            self.embedder = WordEmbedder()


def _owner(pool):
    return _rows(pool, "select id from brain.account where name = 'owner'")[0][0]


def _seed(pool, facts):
    """Rows straight into the tables, one episode per fact, all about the user in namespace jeff.
    facts: (slot, cardinality, value, valid_from or None, episode ts). -> (owner account id, [claim ids])."""
    owner, ids = _owner(pool), []
    with tenant(pool, owner) as c:
        ent = c.execute("select id from entity where namespace = 'jeff' and name = 'user'").fetchone()
        ent = ent[0] if ent else c.execute(
            "insert into entity (namespace, name) values ('jeff', 'user') returning id").fetchone()[0]
        for slot, card, value, day, ts in facts:
            quote = f"My {slot.replace('_', ' ')}: {value}."
            ep = c.execute("""insert into episode (namespace, client, ts, role, body) values ('jeff', 't', %s, 'user', %s)
                              returning id""", (ts, quote)).fetchone()[0]
            sid = c.execute("""insert into slot (namespace, entity_id, name, cardinality) values ('jeff', %s, %s, %s)
                               on conflict (account_id, namespace, entity_id, name) do update set name = excluded.name
                               returning id""", (ent, slot, card)).fetchone()[0]
            cid = c.execute("""insert into claim (namespace, entity_id, slot_id, value, value_norm, valid_from, quote,
                                                  episode_id, status) values ('jeff', %s, %s, %s, %s, %s, %s, %s, 'active')
                               returning id""", (ent, sid, value, value.lower(), day, quote, ep)).fetchone()[0]
            c.execute("insert into chunk (namespace, kind, ref_id, text) values ('jeff', 'claim', %s, %s)",
                      (cid, f"[{day or ts[:10]}] user: {slot.replace('_', ' ')} = {value}"))
            ids.append(cid)
    return owner, ids


DENVER = ("home_city", "one", "Denver", "2023-05-01", "2023-05-01T10:00:00Z")
BOSTON = ("current_city", "one", "Boston", "2023-06-01", "2023-06-01T10:00:00Z")
CLAIMS = """select c.value, c.status, c.valid_to::text, s.name from brain.claim c join brain.slot s on s.id = c.slot_id
            order by c.id"""


# ---- 1a. the shared end-value rule --------------------------------------------------------------------------------

def test_a_past_value_told_late_does_not_close_the_newer_one(make_client, pool):
    """Required test (a): a backfilled old fact goes into history; the newer value stays current."""
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:home_city", "Boston", "I just moved to Boston.")])
    assert worker.write_path(pool, Providers())["ledger"]["placed"] == 1
    _cap(client, "s2", "2023-05-01T10:00:00Z", "I lived in Denver before that.",      # an older turn, imported later
         [_claim("new:home_city", "Denver", "I lived in Denver before that.")])
    res = worker.write_path(pool, Providers())
    assert res["ledger"]["placed"] == 1 and res["ledger"]["superseded"] == 0 and res["ledger"]["history"] == 1
    assert _rows(pool, CLAIMS) == [("Boston", "active", None, "home_city"),
                                   ("Denver", "superseded", "2023-06-01", "home_city")]
    link = _rows(pool, """select n.value, o.value, l.type, l.source, l.run_id from brain.claim_link l
                          join brain.claim n on n.id = l.from_claim join brain.claim o on o.id = l.to_claim""")
    assert [r[:4] for r in link] == [("Boston", "Denver", "supersedes", "write.relate")]
    assert link[0][4].startswith("w")                       # the write_path run that decided it
    applied = _rows(pool, "select output->'applied' from brain.decision_log where stage = 'write.relate'")
    assert applied == [(["history"],)]


def test_undated_claims_are_ordered_by_when_they_were_said(pool):
    """Required test (d): a claim with no stated date starts the day it was said. It is never the oldest by default."""
    owner, (old, new) = _seed(pool, [("home_city", "one", "Boston", None, "2023-06-01T10:00:00Z"),
                                     ("home_city", "one", "Denver", "2023-05-01", "2023-06-10T10:00:00Z")])
    with tenant(pool, owner) as c:
        assert ledger.starts(c, old)[0] == date(2023, 6, 1)
        assert ledger.apply_relation(c, new, old, "updates", "one", "test") == "history"
    assert _rows(pool, CLAIMS) == [("Boston", "active", None, "home_city"),
                                   ("Denver", "superseded", "2023-06-01", "home_city")]

    owner, (old, new) = _seed(pool, [("employer", "one", "Acme", "2023-05-01", "2023-05-01T10:00:00Z"),
                                     ("employer", "one", "Globex", None, "2023-06-10T10:00:00Z")])
    with tenant(pool, owner) as c:
        assert ledger.apply_relation(c, new, old, "updates", "one", "test", "run-1") == "ended"
    assert _rows(pool, CLAIMS)[2:] == [("Acme", "superseded", "2023-06-10", "employer"),
                                       ("Globex", "active", None, "employer")]
    assert _rows(pool, "select source, run_id from brain.claim_link where from_claim = %s", new) == [("test", "run-1")]

    # and an undated claim never ends a dated one that starts after the day it was said
    owner, (old, new) = _seed(pool, [("phone", "one", "Pixel", "2023-07-01", "2023-07-01T10:00:00Z"),
                                     ("phone", "one", "Nokia", None, "2023-06-01T10:00:00Z")])
    with tenant(pool, owner) as c:
        assert ledger.apply_relation(c, new, old, "updates", "one", "test") == "history"
    assert _rows(pool, CLAIMS)[4:] == [("Pixel", "active", None, "phone"), ("Nokia", "superseded", "2023-07-01", "phone")]


def test_only_updates_and_retracts_end_a_value(pool):
    owner, (old, new) = _seed(pool, [DENVER, ("home_city", "one", "Boston", "2023-06-01", "2023-06-01T10:00:00Z")])
    with tenant(pool, owner) as c:
        assert ledger.apply_relation(c, new, old, "adds", "one", "test") == "none"
        assert ledger.apply_relation(c, new, old, "unrelated", "one", "test") == "none"
        assert ledger.apply_relation(c, new, old, "updates", "event", "test") == "none"      # events never end
        assert ledger.apply_relation(c, old, new, "retracts", "one", "test") == "none"       # predates what it retracts
    assert [r[1] for r in _rows(pool, CLAIMS)] == ["active", "active"]
    assert _rows(pool, "select count(*) from brain.claim_link") == [(0,)]
    with tenant(pool, owner) as c:
        assert ledger.apply_relation(c, new, old, "retracts", "one", "test") == "ended"
    assert [r[1] for r in _rows(pool, CLAIMS)] == ["superseded", "active"]


def test_two_claims_of_one_turn_stay_contested_and_a_reversed_dispute_reopens(pool):
    owner, (denver,) = _seed(pool, [DENVER])
    with tenant(pool, owner) as c:
        boston = c.execute("""insert into claim (namespace, entity_id, slot_id, value, value_norm, valid_from, quote,
                                                 episode_id, status)
                              select namespace, entity_id, slot_id, 'Boston', 'boston', valid_from, 'Or Boston.',
                                     episode_id, 'active' from claim where id = %s returning id""", (denver,)).fetchone()[0]
        assert ledger.apply_relation(c, boston, denver, "updates", "one", "test") == "contested"   # no order to go by
        assert nightly.resolve_contested(c, P(NoJev())) == {"pairs": 1, "corrected": 0, "kept_both": 1}
    assert [r[1] for r in _rows(pool, CLAIMS)] == ["contested", "contested"]
    assert _rows(pool, "select type, source from brain.claim_link") == [("contradicts", "test")]
    with tenant(pool, owner) as c:
        assert ledger.reverse_link(c, boston, denver, "contradicts") == "reopened"
        assert ledger.reverse_link(c, boston, denver, "contradicts") == "none"
    assert [r[1] for r in _rows(pool, CLAIMS)] == ["active", "active"]
    assert _rows(pool, "select reversed_at is not null from brain.claim_link") == [(True,)]


# ---- 1b. merge_slots ----------------------------------------------------------------------------------------------

def test_a_chain_of_matches_does_not_merge_its_ends(pool):
    """Required test (c): A~B and B~C, but not A~C. A and C never share a slot."""
    owner, _ = _seed(pool, [DENVER, ("current_city", "many", "Boston", "2023-06-01", "2023-06-01T10:00:00Z"),
                            ("birth_city", "one", "Tulsa", "1990-01-01", "2023-06-02T10:00:00Z")])
    jev = RuleJev(same={frozenset(("home_city", "current_city")), frozenset(("current_city", "birth_city"))})
    with tenant(pool, owner) as c:
        out = nightly.merge_slots(c, P(jev), "n-1")
    assert out["pairs_asked"] == 3 and out["merged"] == 1 and out["held"] is False
    # the kept slot takes the cardinality that ends fewer values
    assert _rows(pool, "select name, cardinality from brain.slot order by name") == [("birth_city", "one"),
                                                                                     ("home_city", "many")]
    assert [(r[0], r[3]) for r in _rows(pool, CLAIMS)] == [("Denver", "home_city"), ("Boston", "home_city"),
                                                           ("Tulsa", "birth_city")]
    outcomes = _rows(pool, """select outcome from brain.decision_log where stage = 'nightly.merge_slots'
                              order by outcome""")
    assert outcomes == [("kept",), ("matched",), ("merged",)]


def test_merge_never_ends_a_value_by_date_alone(pool):
    owner, _ = _seed(pool, [DENVER, BOSTON])
    with tenant(pool, owner) as c:
        out = nightly.merge_slots(c, P(RuleJev(relation="adds")), "n-1")
    assert out["merged"] == 1 and out["superseded"] == 0 and out["churn"] == 0
    assert _rows(pool, CLAIMS) == [("Denver", "active", None, "home_city"), ("Boston", "active", None, "home_city")]
    assert _rows(pool, "select count(*) from brain.claim_link") == [(0,)]


def test_merge_orders_an_undated_claim_by_when_it_was_said(pool):
    """Required test (d), through the merge: the undated newer value ends the dated older one, not the reverse."""
    owner, _ = _seed(pool, [DENVER, ("current_city", "one", "Boston", None, "2023-06-01T10:00:00Z")])
    jev = RuleJev()
    with tenant(pool, owner) as c:
        out = nightly.merge_slots(c, P(jev), "n-1")
    assert out["merged"] == 1 and out["superseded"] == 1 and out["history"] == 0
    assert _rows(pool, CLAIMS) == [("Denver", "superseded", "2023-06-01", "home_city"),
                                   ("Boston", "active", None, "home_city")]
    asked = [s for s in jev.states if "new" in s]
    assert [(s["new"]["value"], s["new"]["date"], s["existing"][0]["date"]) for s in asked] == [
        ("Boston", "2023-06-01", "2023-05-01")]


def test_a_merge_night_is_recorded_and_can_be_undone(pool):
    owner, (denver, boston) = _seed(pool, [DENVER, BOSTON])
    with tenant(pool, owner) as c:
        out = nightly.merge_slots(c, P(), "n-1")
    assert out["merged"] == 1 and out["superseded"] == 1 and out["churn"] == 1 and out["run_id"] == "n-1"
    assert _rows(pool, "select source, run_id from brain.claim_link") == [("nightly.merge_slots", "n-1")]
    assert _rows(pool, "select gone_name, claim_ids, run_id from brain.slot_merge") == [("current_city", [boston], "n-1")]
    assert _rows(pool, "select orig_slot_id is not null, merged_run from brain.claim where id = %s", boston) == [
        (True, "n-1")]

    with tenant(pool, owner) as c:
        undone = nightly.undo_merge_run(c, "n-1")
    assert undone == {"run_id": "n-1", "slots_restored": 1, "claims_moved": 1, "links_reversed": 1, "reopened": 1,
                      "blocked_by": []}
    assert _rows(pool, CLAIMS) == [("Denver", "active", None, "home_city"), ("Boston", "active", None, "current_city")]
    assert _rows(pool, "select orig_slot_id, merged_run from brain.claim where id = %s", boston) == [(None, None)]
    assert _rows(pool, "select reversed_at is not null from brain.claim_link") == [(True,)]
    assert _rows(pool, "select undone_at is not null from brain.slot_merge") == [(True,)]
    # the night's decisions are marked so they never become booster labels
    logged = _rows(pool, """select outcome, reversed from brain.decision_log
                            where stage like 'nightly.merge%%' and input->>'run_id' = 'n-1' order by id""")
    assert [o for o, _ in logged] == ["merged", "updates", "undone"]
    assert [r for _, r in logged] == [True, True, False]
    with tenant(pool, owner) as c:
        assert nightly.undo_merge_run(c, "n-1")["slots_restored"] == 0        # nothing left to take back

    with tenant(pool, owner) as c:                                           # the same decision again: the link is back
        again = nightly.merge_slots(c, P(), "n-2")
    assert again["merged"] == 1 and again["superseded"] == 1
    assert _rows(pool, "select reversed_at, run_id from brain.claim_link") == [(None, "n-2")]
    assert _rows(pool, CLAIMS) == [("Denver", "superseded", "2023-06-01", "home_city"),
                                   ("Boston", "active", None, "home_city")]


def test_merge_runs_are_undone_newest_first(pool):
    owner, _ = _seed(pool, [DENVER, BOSTON])
    with tenant(pool, owner) as c:
        assert nightly.merge_slots(c, P(), "n-1")["merged"] == 1
    _seed(pool, [("city_lived", "one", "Austin", "2023-07-01", "2023-07-01T10:00:00Z")])
    with tenant(pool, owner) as c:
        second = nightly.merge_slots(c, P(), "n-2")
    # the merged slot counts under both of its names: the new slot is asked against each
    assert second["pairs_asked"] == 2 and second["merged"] == 1 and second["superseded"] == 1
    assert _rows(pool, "select count(*) from brain.slot") == [(1,)]
    with tenant(pool, owner) as c:
        blocked = nightly.undo_merge_run(c, "n-1")
    assert blocked["blocked_by"] == ["n-2"] and blocked["slots_restored"] == 0
    assert _rows(pool, "select count(*) from brain.slot") == [(1,)]
    with tenant(pool, owner) as c:
        assert nightly.undo_merge_run(c, "n-2")["slots_restored"] == 1
        assert nightly.undo_merge_run(c, "n-1")["slots_restored"] == 1
    assert _rows(pool, CLAIMS) == [("Denver", "active", None, "home_city"), ("Boston", "active", None, "current_city"),
                                   ("Austin", "active", None, "city_lived")]


def test_a_night_over_the_churn_cap_is_held(pool, monkeypatch):
    monkeypatch.setattr(nightly, "CHURN_MIN", 0)
    monkeypatch.setattr(nightly, "CHURN_SHARE", 0.0)
    owner, _ = _seed(pool, [DENVER, BOSTON])
    with tenant(pool, owner) as c:
        out = nightly.merge_slots(c, P(), "n-1")
    assert out["held"] is True and out["merged"] == 0 and out["would_merge"] == 1 and out["churn"] == 1
    assert _rows(pool, CLAIMS) == [("Denver", "active", None, "home_city"), ("Boston", "active", None, "current_city")]
    assert _rows(pool, "select count(*) from brain.slot_merge") == [(0,)]
    assert _rows(pool, "select count(*) from brain.claim_link") == [(0,)]
    logged = _rows(pool, "select stage, decider, outcome from brain.decision_log order by stage, decider")
    assert logged == [("nightly.merge_relate", "jev", "held"), ("nightly.merge_slots", "code", "held"),
                      ("nightly.merge_slots", "jev", "held")]


# ---- 1c. the count route ------------------------------------------------------------------------------------------

def test_two_events_with_one_value_on_different_days_count_as_two(pool):
    """Required test (b): two 5k runs on different dates are 2; the same run told twice is 1."""
    owner, _ = _seed(pool, [("races_run", "event", "5k run", "2023-05-06", "2023-05-06T18:00:00Z"),
                            ("races_run", "event", "5k run", "2023-05-20", "2023-05-20T18:00:00Z"),
                            ("races_run", "event", "5k run", "2023-05-20", "2023-05-22T09:00:00Z"),
                            DENVER, ("home_city", "one", "Denver", "2023-06-01", "2023-06-01T10:00:00Z")])
    with tenant(pool, owner) as c:
        runs = ledger.gather(c, P(embedder=False), ["jeff"], "How many 5k runs did I do?", date(2023, 7, 1))
        city = ledger.gather(c, P(embedder=False), ["jeff"], "How often did I say Denver?", date(2023, 7, 1))
    assert (runs["items"], runs["count"], runs["capped"]) == (3, 2, False)
    assert runs["block"].count("5k run (") == 2 and "(told 2 times)" in runs["block"]
    assert (city["items"], city["count"]) == (2, 1)           # a state value told twice is still one thing


def test_count_route_says_when_the_search_hit_its_cap(pool):
    """Required test (e): the 40-candidate cap is reported, in the result and in the block the model reads."""
    start = date(2023, 1, 1)
    owner, _ = _seed(pool, [("weddings_attended", "event", f"wedding number {k}", str(start + timedelta(days=k)),
                             f"{start + timedelta(days=k)}T10:00:00Z") for k in range(ledger.GATHER_TOP + 5)])
    with tenant(pool, owner) as c:
        got = ledger.gather(c, P(embedder=False), ["jeff"], "How many weddings did I go to?", date(2023, 7, 1))
    assert got["capped"] is True and got["count"] == ledger.GATHER_TOP and got["items"] == ledger.GATHER_TOP
    assert "treat the count as a minimum" in got["block"]


# ---- 1d. the ledger lock ------------------------------------------------------------------------------------------

def test_consolidate_takes_the_write_path_lock(pool, db_url):
    with psycopg.connect(db_url, autocommit=True) as other:      # a write_path run holds the lock
        other.execute("select pg_advisory_lock(%s)", (worker.WRITE_PATH_LOCK,))
        assert worker.consolidate(pool, P(NoJev()), lock_wait_s=0) == {"accounts": 0, "stopped": "busy"}
    out = worker.consolidate(pool, P(NoJev()), lock_wait_s=0)
    assert "stopped" not in out and out["accounts"] == 1 and out["run_id"].startswith("n")


# ---- 1e. the one-time re-check ------------------------------------------------------------------------------------

def _legacy(pool):
    """Three supersessions as the old rules left them (no source on the link): one Jev now calls "adds", one that
    holds, one whose newer claim starts earlier than the value it ended."""
    owner, ids = _seed(pool, [
        DENVER, ("home_city", "one", "Boston", "2023-06-01", "2023-06-01T10:00:00Z"),
        ("employer", "one", "Acme", "2023-01-01", "2023-01-01T10:00:00Z"),
        ("employer", "one", "Globex", "2023-03-01", "2023-03-01T10:00:00Z"),
        ("phone", "one", "Pixel", "2023-04-01", "2023-04-01T10:00:00Z"),
        ("phone", "one", "iPhone", "2023-02-01", "2023-05-01T10:00:00Z")])
    denver, boston, acme, globex, pixel, iphone = ids
    with tenant(pool, owner) as c:
        c.execute("update claim set quote = 'I also spend part of the year in Boston.' where id = %s", (boston,))
        for new, old, end in ((boston, denver, "2023-06-01"), (globex, acme, "2023-03-01"), (iphone, pixel, "2023-05-01")):
            c.execute("update claim set status = 'superseded', valid_to = %s where id = %s", (end, old))
            c.execute("insert into claim_link (from_claim, to_claim, type) values (%s, %s, 'supersedes')", (new, old))
        ep = c.execute("select episode_id from claim where id = %s", (iphone,)).fetchone()[0]
        log = """insert into decision_log (namespace, stage, decider, input, output, outcome)
                 values ('jeff', %s, 'jev', %s, %s, %s)"""
        c.execute(log, ("write.relate", Jsonb({"task": "W7"}), Jsonb({"new_id": boston}), "updates"))
        c.execute(log, ("write.relate", Jsonb({"task": "W7"}), Jsonb({"new_id": globex}), "updates"))
        c.execute(log, ("write.relate", Jsonb({"task": "W7", "episode_id": ep, "new": {"value": "iPhone"}}),   # old shape
                        Jsonb({"relations": ["updates"]}), "updates"))
        c.execute(log, ("nightly.promote", Jsonb({"task": "N3", "claim_id": acme}), Jsonb({}), "promoted"))
        c.execute(log, ("nightly.promote", Jsonb({"task": "N3", "claim_id": globex}), Jsonb({}), "promoted"))
        for claim, text in ((acme, "Use Acme templates."), (globex, "Use Globex templates.")):
            c.execute("insert into directive (namespace, text, source_claim, created_by) values ('jeff', %s, %s, %s)",
                      (text, claim, LEARNED_BY))
    return owner, ids


def _adds_if_also(new, existing):
    return "adds" if "also" in new["quote"] else "updates"


def test_recheck_survey_and_dry_run_write_nothing(pool):
    owner, _ = _legacy(pool)
    before = {"supersedes_links": {"legacy": 3}, "legacy_to_check": 3, "legacy_out_of_order": 1, "reversed_links": 0,
              "merges_recorded": 0, "legacy_merge_decisions": 0, "learned_directives": 2,
              "learned_directives_stale": 1, "decisions_marked_reversed": 0}
    with tenant(pool, owner) as c:
        assert recheck.survey(c) == before
        dry = recheck.recheck(c, P(RuleJev(relation=_adds_if_also)), run_id="r-1")
    assert dry == {"run_id": "r-1", "applied": False, "checked": 3, "kept": 1, "reversed": 2, "reopened": 0,
                   "relations": {"adds": 1, "updates": 2}, "reapplied": {}, "decisions_marked": 0,
                   "directives_retired": 1}
    with tenant(pool, owner) as c:
        assert recheck.survey(c) == before
    assert _rows(pool, "select count(*) from brain.decision_log where stage = 'recheck.relate'") == [(0,)]


def test_recheck_reverses_what_the_shared_rule_would_not_write(pool):
    owner, (denver, boston, acme, globex, pixel, iphone) = _legacy(pool)
    with tenant(pool, owner) as c:
        out = recheck.recheck(c, P(RuleJev(relation=_adds_if_also)), apply=True, run_id="r-1")
    assert out == {"run_id": "r-1", "applied": True, "checked": 3, "kept": 1, "reversed": 2, "reopened": 2,
                   "relations": {"adds": 1, "updates": 2}, "reapplied": {"none": 1, "history": 1},
                   "decisions_marked": 3, "directives_retired": 1}
    assert _rows(pool, CLAIMS) == [
        ("Denver", "active", None, "home_city"), ("Boston", "active", None, "home_city"),     # "adds": both hold
        ("Acme", "superseded", "2023-03-01", "employer"), ("Globex", "active", None, "employer"),   # kept
        ("Pixel", "active", None, "phone"), ("iPhone", "superseded", "2023-04-01", "phone")]  # the late-told past value
    links = _rows(pool, """select from_claim, to_claim, source, run_id, reversed_at is not null from brain.claim_link
                           order by from_claim, to_claim""")
    assert links == [(boston, denver, None, None, True), (globex, acme, "recheck.kept", "r-1", False),
                     (pixel, iphone, "recheck", "r-1", False), (iphone, pixel, None, None, True)]
    marked = _rows(pool, """select stage, coalesce(output->>'new_id', input->>'claim_id', input->'new'->>'value')
                            from brain.decision_log where reversed order by id""")
    assert marked == [("write.relate", str(boston)), ("write.relate", "iPhone"), ("nightly.promote", str(acme))]
    assert _rows(pool, "select outcome from brain.decision_log where stage = 'recheck.relate' order by id") == [
        ("reversed",), ("kept",), ("reversed",)]
    assert _rows(pool, "select text, retired_at is not null, retired_reason from brain.directive order by id") == [
        ("Use Acme templates.", True, "recheck"), ("Use Globex templates.", False, None)]

    with tenant(pool, owner) as c:                       # a second run finds nothing left to check
        again = recheck.recheck(c, P(NoJev()), apply=True, run_id="r-2")
        after = recheck.survey(c)
    assert again["checked"] == 0 and again["directives_retired"] == 0
    assert after["legacy_to_check"] == 0 and after["reversed_links"] == 2 and after["decisions_marked_reversed"] == 3
    assert after["supersedes_links"] == {"recheck": 1, "recheck.kept": 1}


# ---- independent review of PR #2 (2026-11-10) -----------------------------------------------------------------------

STATUS = "select c.value, c.status from brain.claim c order by c.id"


def test_a_claim_is_never_left_contested_without_a_contested_partner(pool):
    """Review finding 2. In one pass a new claim can be filed as history against one value and then meet another:
    nothing may end up contested alone, because the nightly only resolves pairs with both sides contested."""
    # A: the late-told claim N is filed as history first, then "contradicts" a second value
    owner, (a, b, n) = _seed(pool, [("city", "one", "March", "2023-03-01", "2023-03-01T10:00:00Z"),
                                    ("city", "one", "February", "2023-02-01", "2023-02-01T10:00:00Z"),
                                    ("city", "one", "January", "2023-01-01", "2023-06-01T10:00:00Z")])
    with tenant(pool, owner) as c:
        assert ledger.apply_relation(c, n, a, "updates", "one", "t") == "history"
        assert ledger.apply_relation(c, n, b, "contradicts", "one", "t") == "none"
    assert _rows(pool, STATUS) == [("March", "active"), ("February", "active"), ("January", "superseded")]


def test_ending_one_side_of_a_dispute_frees_the_other(pool):
    """Review finding 2, the other order: N and A are in dispute, then N is ended by a later value."""
    owner, (a, b, n) = _seed(pool, [("city", "one", "A", "2023-03-01", "2023-03-01T10:00:00Z"),
                                    ("city", "one", "B", "2023-05-01", "2023-05-01T10:00:00Z"),
                                    ("city", "one", "N", "2023-04-01", "2023-06-01T10:00:00Z")])
    with tenant(pool, owner) as c:
        assert ledger.apply_relation(c, n, a, "contradicts", "one", "t") == "contested"
        assert ledger.apply_relation(c, n, b, "updates", "one", "t") == "history"       # B starts later than N
    assert _rows(pool, STATUS) == [("A", "active"), ("B", "active"), ("N", "superseded")]
    # a claim with a second, still contested, partner stays contested
    owner, (x, y, z, w) = _seed(pool, [("pet", "one", "X", "2023-03-01", "2023-03-01T10:00:00Z"),
                                       ("pet", "one", "Y", "2023-03-02", "2023-03-02T10:00:00Z"),
                                       ("pet", "one", "Z", "2023-03-03", "2023-03-03T10:00:00Z"),
                                       ("pet", "one", "W", "2023-09-01", "2023-09-01T10:00:00Z")])
    with tenant(pool, owner) as c:
        assert ledger.apply_relation(c, y, x, "contradicts", "one", "t") == "contested"
        assert ledger.apply_relation(c, z, x, "contradicts", "one", "t") == "contested"
        assert ledger.apply_relation(c, w, y, "updates", "one", "t") == "ended"         # Y ends; X still disputes Z
    assert _rows(pool, STATUS)[3:] == [("X", "contested"), ("Y", "superseded"), ("Z", "contested"), ("W", "active")]


def test_the_total_does_not_add_a_replaced_value_to_its_replacement(pool):
    """Review finding 3: replaced values are candidates since this PR, so the code-computed total must skip them."""
    owner, (old, new, other) = _seed(pool, [
        ("savings_balance", "one", "$5,000 savings", "2023-01-01", "2023-01-01T10:00:00Z"),
        ("savings_balance", "one", "$8,000 savings", "2023-06-01", "2023-06-01T10:00:00Z"),
        ("brokerage_savings", "one", "$2,000 savings", "2023-06-02", "2023-06-02T10:00:00Z")])
    with tenant(pool, owner) as c:
        for cid, amount in ((old, 5000), (new, 8000), (other, 2000)):
            c.execute("update claim set amount = %s, unit = 'usd' where id = %s", (amount, cid))
        assert ledger.apply_relation(c, new, old, "updates", "one", "t") == "ended"
        got = ledger.gather(c, P(embedder=False), ["jeff"], "What savings do I have?", date(2023, 7, 1))
    assert got["count"] == 3 and got["total"] == 10000.0          # 8,000 + 2,000; the replaced 5,000 is listed, not added
    assert "Total of the amounts above that were not replaced (code-computed): 10000 usd" in got["block"]
    assert "$5,000 savings" in got["block"]


def test_undo_takes_back_what_a_later_step_did_to_a_merged_pair(pool, monkeypatch):
    """Review finding 1: the merge raises a dispute, the nightly then settles it by ending one claim with its own
    link; undoing the merge must reopen that claim, since the two are no longer in one slot."""
    owner, (denver, boston) = _seed(pool, [DENVER, BOSTON])
    with tenant(pool, owner) as c:
        nightly.merge_slots(c, P(RuleJev("contradicts")), "n-1")
    assert [r[1] for r in _rows(pool, CLAIMS)] == ["contested", "contested"]
    with tenant(pool, owner) as c:            # what resolve_contested writes when Jev says "corrects"
        c.execute("update claim set status = 'active' where id = %s", (boston,))
        c.execute("update claim set status = 'superseded', valid_to = '2023-06-01' where id = %s", (denver,))
        c.execute("""insert into claim_link (from_claim, to_claim, type, source, run_id)
                     values (%s, %s, 'supersedes', 'nightly.contested', 'n-1')""", (boston, denver))
        out = nightly.undo_merge_run(c, "n-1")
    assert out["slots_restored"] == 1 or out["claims_moved"] >= 1
    assert _rows(pool, CLAIMS) == [("Denver", "active", None, "home_city"), ("Boston", "active", None, "current_city")]
    assert _rows(pool, """select count(*) from brain.claim_link where reversed_at is null
                          and type in ('supersedes', 'contradicts')""") == [(0,)]


def test_undo_merge_from_the_script_needs_an_account_or_all_accounts():
    import pytest
    sys_path = __import__("sys").path
    scripts = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    import recheck_ledger
    with pytest.raises(SystemExit) as e:
        recheck_ledger.main(["--undo-merge", "n-1"])
    assert e.value.code == 2
