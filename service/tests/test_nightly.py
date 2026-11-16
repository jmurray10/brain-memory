"""Nightly consolidation (brain/nightly.py): settle tentative links, resolve contested pairs, stability audit, harden."""
from brain import nightly
from brain.providers import JevResult
from test_ledger import FakeJev, _cap, _claim, _rows


class NightJev(FakeJev):
    """The ledger fake, plus: the settle Score (by the entity named in `score_for`), the correction verdict, and a
    re-ask of relate decisions that can be made to flip."""

    def __init__(self, settle_score=2.0, verdict="corrects", flip=False):
        super().__init__()
        self.settle_score, self.verdict, self.flip = settle_score, verdict, flip

    def ask_many(self, reqs):
        out = []
        for state, qs in reqs:
            if "same" in qs:
                out.append(JevResult({"same": {"score": self.settle_score, "probabilities": {"0": 1.0}}}, self.model))
            elif "verdict" in qs:
                out.append(JevResult({"verdict": {"choice": self.verdict, "probabilities": {self.verdict: 0.9}}},
                                     self.model))
            elif self.flip and all(k.startswith("r") for k in qs):
                out.append(JevResult({k: {"choice": "unrelated", "probabilities": {"unrelated": 0.9}} for k in qs},
                                     self.model))
            else:
                out.extend(super().ask_many([(state, qs)]))
        return out


class P:
    def __init__(self, jev):
        self.jev = jev


def _owner(pool):
    return _rows(pool, "select id from brain.account where name = 'owner'")[0][0]


def _setup(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    _cap(client, "s3", "2023-06-02T10:00:00Z", "Maya lives in Austin.",
         [_claim("new:home_city", "Austin", "Maya lives in Austin.", mention="Maya")])
    worker.write_path(pool, P(NightJev()))
    return client


def test_tentative_links_settle_or_split(make_client, pool):
    from brain.db import tenant
    _setup(make_client, pool)
    with pool.connection() as c:
        c.execute("update brain.claim set tentative = true where value = 'Austin'")
    with tenant(pool, _owner(pool)) as c:
        assert nightly.settle_links(c, P(NightJev(settle_score=2.0))) == {"tentative": 1, "settled": 1, "split": 0}
    with pool.connection() as c:
        c.execute("update brain.claim set tentative = true where value = 'Austin'")
    with tenant(pool, _owner(pool)) as c:
        assert nightly.settle_links(c, P(NightJev(settle_score=0.1))) == {"tentative": 1, "settled": 0, "split": 1}
    owner_names = _rows(pool, """select n.name from brain.claim c join brain.entity n on n.id = c.entity_id
                                 where c.value = 'Austin'""")
    assert owner_names == [("Maya",)] and _rows(pool, "select count(*) from brain.entity where name = 'Maya'") == [(2,)]


def test_contested_pair_is_corrected_or_kept(make_client, pool):
    from brain.db import tenant
    _setup(make_client, pool)
    with pool.connection() as c:
        a = c.execute("select id from brain.claim where value = 'Austin'").fetchone()[0]
        b = c.execute("""insert into brain.claim (account_id, namespace, entity_id, slot_id, value, valid_from, status,
                                                  quote, episode_id)
                         select account_id, namespace, entity_id, slot_id, 'Dallas', '2023-06-09', 'contested',
                                'Maya lives in Dallas now.', episode_id from brain.claim where id = %s returning id""",
                      (a,)).fetchone()[0]
        c.execute("update brain.claim set status = 'contested' where id = %s", (a,))
        c.execute("""insert into brain.claim_link (account_id, from_claim, to_claim, type)
                     select account_id, %s, %s, 'contradicts' from brain.claim where id = %s""", (a, b, a))
    with tenant(pool, _owner(pool)) as c:
        assert nightly.resolve_contested(c, P(NightJev(verdict="corrects"))) == {"pairs": 1, "corrected": 1,
                                                                                "kept_both": 0}
    got = dict(_rows(pool, "select value, status from brain.claim where value in ('Austin', 'Dallas')"))
    assert got == {"Dallas": "active", "Austin": "superseded"}          # the later claim (Dallas) corrects


def test_plans_close_when_a_later_claim_settles_them(make_client, pool):
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I'm planning to run the Boston half marathon.",
         [_claim("new:race_plan", "run the Boston half marathon", "I'm planning to run the Boston half marathon.",
                 polarity="planned")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I finished the Boston half marathon in 1:52.",
         [_claim("new:race_result", "finished the Boston half marathon in 1:52",
                 "I finished the Boston half marathon in 1:52.")])
    from brain import worker
    worker.write_path(pool, P(NightJev()))
    assert _rows(pool, "select kind from brain.claim order by id") == [("plan",), ("fact",)]

    class PlanJev(NightJev):
        def ask_many(self, reqs):
            if all("settled" in qs for _, qs in reqs):
                return [JevResult({"settled": {"noul": 0.9}, "outcome": {"choice": "done", "probabilities": {"done": 0.9}},
                                   "which": {"choice": "x0", "probabilities": {"x0": 0.9}}}, self.model) for _ in reqs]
            return super().ask_many(reqs)
    with tenant(pool, _owner(pool)) as c:
        assert nightly.track_plans(c, P(PlanJev())) == {"plans": 1, "closed": 1}
    assert _rows(pool, "select status, valid_to::text from brain.claim order by id") == [
        ("superseded", "2023-06-01"), ("active", None)]


def test_stability_audit_flips_block_hardening(make_client, pool):
    from brain.db import tenant
    _setup(make_client, pool)
    with tenant(pool, _owner(pool)) as c:
        audit = nightly.stability_audit(c, P(NightJev(flip=True)), seed=1)
        assert audit["asked"] == 1 and audit["flipped"] == 1
        hardened = nightly.harden(c, audit["flipped_episodes"])
    settled = dict(_rows(pool, "select value, settled_at is not null from brain.claim"))
    assert settled == {"Denver": False, "Boston": False, "Austin": True}   # Boston's update flipped; Denver superseded
    assert hardened == {"settled": 1}
    with tenant(pool, _owner(pool)) as c:
        assert nightly.stability_audit(c, P(NightJev()), seed=1)["flipped"] == 0


def test_consolidate_skips_merge_slots_by_default(monkeypatch):
    """Stopgap (room review v2): merge_slots stays out of the nightly until the shared end-value rule lands."""
    called = []
    monkeypatch.setattr(nightly, "merge_slots", lambda conn, prov, run_id: called.append(run_id) or {"merged": 1})
    monkeypatch.setattr(nightly, "settle_links", lambda conn, prov: {})
    for name in ("resolve_contested", "track_plans"):
        monkeypatch.setattr(nightly, name, lambda conn, prov, run_id: {})
    monkeypatch.setattr(nightly, "stability_audit", lambda conn, prov: {"flipped_episodes": []})
    monkeypatch.setattr(nightly, "harden", lambda conn, flipped: {})
    out = nightly.consolidate(None, None)
    assert called == [] and out["slots"]["merged"] == 0 and out["slots"]["skipped"] == "BRAIN_MERGE_SLOTS off"
    assert out["run_id"].startswith("n")
    on = nightly.consolidate(None, None, merge_slots_enabled=True, run_id="n-1")
    assert on["slots"] == {"merged": 1} and called == ["n-1"] and on["run_id"] == "n-1"


def _tentative_under_another_name(make_client, pool):
    _setup(make_client, pool)
    with pool.connection() as c:        # the claim a mention of "Maya" made, filed tentatively under an entity "Emma"
        c.execute("update brain.entity set name = 'Emma', aliases = '{emma}' where name = 'Maya'")
        c.execute("update brain.claim set tentative = true where value = 'Austin'")


def _austin(pool):
    return _rows(pool, """select n.name, c.tentative from brain.claim c join brain.entity n on n.id = c.entity_id
                          where c.value = 'Austin'""")


def test_an_unconfirmed_link_stays_attached_as_built(make_client, pool):
    from brain.db import tenant
    _tentative_under_another_name(make_client, pool)
    with tenant(pool, _owner(pool)) as c:
        assert nightly.settle_links(c, P(NightJev(settle_score=1.0))) == {"tentative": 1, "settled": 0, "split": 0}
    assert _austin(pool) == [("Emma", True)]


def test_resolve_names_splits_an_unconfirmed_link_between_two_proper_names(make_client, pool):
    from brain import ledger
    from brain.db import tenant
    _tentative_under_another_name(make_client, pool)
    token = ledger.RESOLVE_NAMES_OVERRIDE.set(True)
    try:
        with tenant(pool, _owner(pool)) as c:
            assert nightly.settle_links(c, P(NightJev(settle_score=1.0))) == {"tentative": 1, "settled": 0, "split": 1}
    finally:
        ledger.RESOLVE_NAMES_OVERRIDE.reset(token)
    assert _austin(pool) == [("Maya", False)]


def test_settle_split_splits_every_unconfirmed_link(make_client, pool, monkeypatch):
    from brain import ledger
    from brain.db import tenant
    _setup(make_client, pool)
    with pool.connection() as c:
        c.execute("update brain.claim set tentative = true where value = 'Austin'")
    monkeypatch.setattr(ledger, "SETTLE_SPLIT", True)
    with tenant(pool, _owner(pool)) as c:
        assert nightly.settle_links(c, P(NightJev(settle_score=1.0))) == {"tentative": 1, "settled": 0, "split": 1}
    assert _austin(pool) == [("Maya", False)]
    assert _rows(pool, "select count(*) from brain.entity where name = 'Maya'") == [(2,)]


def test_two_split_claims_of_one_mention_share_one_entity(make_client, pool):
    """As first built each split claim got a new entity, so two facts about Maya became two Mayas."""
    from brain import worker
    from brain.db import tenant
    client = _setup(make_client, pool)
    _cap(client, "s4", "2023-06-03T10:00:00Z", "Maya has a dog.",
         [_claim("new:pet", "a dog", "Maya has a dog.", mention="Maya")])
    worker.write_path(pool, P(NightJev()))
    with pool.connection() as c:        # both of Maya's claims sit, tentatively, under an entity called Emma
        c.execute("update brain.entity set name = 'Emma', aliases = '{emma}' where name = 'Maya'")
        c.execute("update brain.claim set tentative = true where value in ('Austin', 'a dog')")
    with tenant(pool, _owner(pool)) as c:
        assert nightly.settle_links(c, P(NightJev(settle_score=0.1))) == {"tentative": 2, "settled": 0, "split": 2}
    owners = _rows(pool, """select c.value, n.name, n.id from brain.claim c join brain.entity n on n.id = c.entity_id
                            where c.value in ('Austin', 'a dog') order by c.id""")
    assert [o[:2] for o in owners] == [("Austin", "Maya"), ("a dog", "Maya")] and owners[0][2] == owners[1][2]
    assert _rows(pool, "select count(*) from brain.entity where name = 'Maya'") == [(1,)]
    assert _rows(pool, "select count(*) from brain.slot s join brain.entity n on n.id = s.entity_id where n.name = 'Maya'") == [(2,)]
