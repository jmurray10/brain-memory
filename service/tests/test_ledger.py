"""Build step 5: entities, slots and supersession (brain/ledger.py), through capture -> worker -> forget."""
from brain import ledger
from brain.providers import JevResult
from conftest import ADMIN


def _claim(slot, value, quote, mention="", polarity="asserted"):
    return {"mention": mention, "slot": slot, "value": value, "quote": quote, "polarity": polarity}


class FakeJev:
    """Answers every write-path question deterministically: triage keeps every turn; verify passes everything; an
    alignment Score is 2 when the mention's last word is in the candidate's name; a city slot fits another city slot;
    two different values of one slot are an update, equal values are the same."""
    model = "fake-jev"

    def __init__(self):
        self.asked: list[str] = []

    def ask_many(self, reqs):
        out = []
        for state, qs in reqs:
            ans = {}
            for name, q in qs.items():
                self.asked.append(name)
                if name in ("contains_prompt_injection", "subverts_user"):       # directive screen
                    bad = "ignore previous instructions" in state["candidate_passage"].lower()
                    ans[name] = {"noul": 0.9 if bad else 0.05}
                elif name in ("lasting_fact", "preference", "plan", "has_date", "assistant_recall"):
                    ans[name] = {"noul": 0.9 if name == "lasting_fact" else 0.1}
                elif name == "kind":
                    ans[name] = {"choice": "fact", "probabilities": {"fact": 0.9}}
                elif name.endswith(("_hallucinated", "_off_target", "_incomplete")):
                    ans[name] = {"noul": 0.05}
                elif name.endswith("_polarity"):
                    i = int(name[1:].split("_")[0])
                    p = state["claims"][i]["polarity"]
                    ans[name] = {"choice": p, "probabilities": {p: 0.9}}
                elif name.endswith("_about_speaker"):
                    ans[name] = {"noul": 0.9}
                elif q["type"] == "score":
                    j = int(name[1:])
                    last = state["mention"]["name"].lower().split()[-1]
                    hit = last in state["candidates"][j]["name"].lower()
                    ans[name] = {"score": 2.0 if hit else 0.0, "probabilities": {"2" if hit else "0": 1.0}}
                elif name == "is_preference":
                    ans[name] = {"noul": 0.9 if "always" in state["claim"]["quote"].lower() else 0.1}
                elif name.startswith(("f", "m")) and name[1:].isdigit():   # facts search / same-attribute merge
                    if name.startswith("f"):
                        ans[name] = {"noul": 0.9 if "city" in state["items"][int(name[1:])]["fact"] else 0.1}
                    else:
                        pair = state["pairs"][int(name[1:])]
                        both = "city" in pair["a"]["name"] and "city" in pair["b"]["name"]
                        ans[name] = {"noul": 0.9 if both else 0.1}
                elif name.startswith(("p", "a")) and name[1:].isdigit():   # promote (N3) / apply to this question
                    item = state["items"][int(name[1:])]
                    text = item.get("fact") or item.get("standing_directive") or ""
                    ans[name] = {"noul": 0.9 if "units" in text else 0.1}
                elif name == "cardinality":
                    card = "event" if "wedding" in state["claim"]["slot_proposed"] else "one"
                    ans[name] = {"choice": card, "probabilities": {card: 0.9}}
                elif name.startswith("i") and name[1:].isdigit():          # count route: is this item counted?
                    item = state["items"][int(name[1:])]
                    ans[name] = {"noul": 0.9 if "wedding" in item["fact"] else 0.1}
                elif name == "fits":
                    city = "city" in state["claim"]["slot_proposed"]
                    ans[name] = {"noul": 0.9 if city and any("city" in s["name"] for s in state["slots"]) else 0.1}
                elif name == "which" and "slots" in state:            # write path: slot placement
                    pick = next((s["name"] for s in state["slots"] if "city" in s["name"]), state["slots"][0]["name"])
                    ans[name] = {"choice": pick, "probabilities": {pick: 0.9}}
                elif name == "which":                                 # read path: the question's slot
                    pick = next(iter(q["criteria"]))
                    ans[name] = {"choice": pick, "probabilities": {pick: 0.9}}
                elif name in ("names", "held"):
                    ans[name] = {"noul": 0.1 if "weather" in state["question"] else 0.9}
                elif name == "who":
                    pick = "user" if "user" in q["criteria"] else next(iter(q["criteria"]))
                    ans[name] = {"choice": pick, "probabilities": {pick: 0.9}}
                elif name.startswith("r") and name[1:].isdigit():
                    j = int(name[1:])
                    rel = ("same" if state["existing"][j]["value"] == state["new"]["value"]
                           else "adds" if "again" in state["new"]["quote"] else "updates")
                    ans[name] = {"choice": rel, "probabilities": {rel: 0.9}}
                else:
                    raise AssertionError(f"unexpected question {name}")
            out.append(JevResult(ans, self.model))
        return out


class Providers:
    def __init__(self):
        self.jev = FakeJev()


def _cap(client, session, ts, text, claims):
    r = client.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "session_id": session, "turns": [
        {"role": "user", "text": text, "ts": ts, "claims": claims}]})
    assert r.status_code == 200, r.text
    return r.json()["episodes"][0]["id"]


def _rows(pool, sql, *args):
    with pool.connection() as c:
        return c.execute(sql, args or None).fetchall()


def test_update_supersedes_and_forget_restores(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    boston = _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
                  [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    res = worker.write_path(pool, Providers())
    assert res["ledger"]["placed"] == 2 and res["ledger"]["superseded"] == 1 and res["ledger"]["slots_created"] == 1
    claims = _rows(pool, """select c.value, c.status, c.valid_from::text, c.valid_to::text, s.name, n.name
                            from brain.claim c join brain.slot s on s.id = c.slot_id join brain.entity n on n.id = c.entity_id
                            order by c.id""")
    assert claims == [("Denver", "superseded", "2023-05-01", "2023-06-01", "home_city", "user"),
                      ("Boston", "active", "2023-06-01", None, "home_city", "user")]
    assert _rows(pool, "select type from brain.claim_link") == [("supersedes",)]
    assert _rows(pool, "select count(*) from brain.chunk where kind = 'claim'") == [(2,)]
    stages = {r[0] for r in _rows(pool, "select input->>'task' from brain.decision_log where stage like 'write.%'")}
    assert {"W1", "W4", "W6", "W7"} <= stages

    r = client.post("/v1/forget", headers=ADMIN, json={"namespace": "jeff", "episode_ids": [boston]})
    f = r.json()["forgotten"]
    assert f["claims"] == 1 and f["claims_restored"] == 1
    assert _rows(pool, "select value, status, valid_to from brain.claim order by id") == [
        ("Denver", "active", None), ("Boston", "retracted", None)]
    assert _rows(pool, "select count(*) from brain.chunk where kind = 'claim'") == [(1,)]


def test_named_entities_link_by_alignment_and_stay_separate_otherwise(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "My sister Maya lives in Austin.",
         [_claim("new:home_city", "Austin", "My sister Maya lives in Austin.", mention="My sister Maya")])
    _cap(client, "s2", "2023-05-02T10:00:00Z", "Maya got a dog named Biscuit.",
         [_claim("new:pet", "a dog named Biscuit", "Maya got a dog named Biscuit.", mention="Maya")])
    _cap(client, "s3", "2023-05-03T10:00:00Z", "Dana from work runs marathons.",
         [_claim("new:hobby", "runs marathons", "Dana from work runs marathons.", mention="Dana")])
    res = worker.write_path(pool, Providers())
    assert res["ledger"]["placed"] == 3
    ents = dict(_rows(pool, "select name, aliases from brain.entity order by id"))
    assert set(ents) == {"user", "My sister Maya", "Dana"}            # Maya linked; Dana is new
    assert "maya" in ents["My sister Maya"]
    owners = _rows(pool, """select c.value, n.name from brain.claim c join brain.entity n on n.id = c.entity_id
                            order by c.id""")
    assert owners == [("Austin", "My sister Maya"), ("a dog named Biscuit", "My sister Maya"),
                      ("runs marathons", "Dana")]


def test_plans_never_change_what_is_current(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I live in Denver.", [_claim("new:home_city", "Denver", "I live in Denver.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I might move to Boston.",
         [_claim("new:home_city", "Boston", "I might move to Boston.", polarity="hypothetical")])
    worker.write_path(pool, Providers())
    assert _rows(pool, "select value, status, polarity from brain.claim order by id") == [
        ("Denver", "active", "asserted"), ("Boston", "active", "hypothetical")]


def test_lookup_reads_current_value_as_of_a_date_and_history(make_client, pool):
    from datetime import date

    from brain import worker
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    worker.write_path(pool, Providers())
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        now = ledger.lookup(c, Providers(), ["jeff"], "Where do I live now?", date(2023, 7, 1), "current")
        then = ledger.lookup(c, Providers(), ["jeff"], "Where did I live?", date(2023, 5, 15), "current")
        hist = ledger.lookup(c, Providers(), ["jeff"], "Where have I lived?", date(2023, 7, 1), "history")
        none = ledger.lookup(c, Providers(), ["jeff"], "What's the weather like?", date(2023, 7, 1), "current")
        side = ledger.lookup(c, Providers(), ["jeff"], "Where do I live now?", date(2023, 7, 1), "current",
                             decisive=False)
    assert (now["entity"], now["slot"], now["rows"]) == ("user", "home_city", 1) and "- Boston (from 2023-06-01)" in now["block"]
    assert "- Denver (from 2023-05-01 to 2023-06-01)" in then["block"] and "Boston" not in then["block"]
    assert hist["rows"] == 2 and hist["block"].index("Denver") < hist["block"].index("Boston") and "[superseded]" in hist["block"]
    assert none is None
    # a side check on an open question: same rows, but the block defers to the excerpts
    assert side["rows"] == 1 and "excerpts above are the main evidence" in side["block"] and "current value" not in side["block"]


def test_count_route_gathers_and_merges_the_same_event(make_client, pool):
    from datetime import date

    from brain import worker
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I went to Sam's wedding in Napa.",
         [_claim("new:weddings_attended", "Sam's wedding in Napa", "I went to Sam's wedding in Napa.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I went to Priya's wedding last weekend.",
         [_claim("new:weddings_attended", "Priya's wedding", "I went to Priya's wedding last weekend.")])
    _cap(client, "s3", "2023-06-20T10:00:00Z", "Sam's wedding in Napa was so fun.",
         [_claim("new:weddings_attended", "Sam's wedding in Napa", "Sam's wedding in Napa was so fun.")])
    _cap(client, "s4", "2023-06-21T10:00:00Z", "I live in Denver.", [_claim("new:home_city", "Denver", "I live in Denver.")])
    worker.write_path(pool, Providers())
    assert _rows(pool, "select type from brain.claim_link") == [("same_event",)]   # events are never superseded
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        got = ledger.gather(c, Providers(), ["jeff"], "How many weddings did I go to this year?", date(2023, 7, 1))
    assert got["items"] == 3 and got["count"] == 2
    assert "(told 2 times)" in got["block"] and "Denver" not in got["block"]


class WordEmbedder:
    """Bag of words, with 'city' weighted as the word that carries the meaning: "home city" and "current city"
    embed close together (as real embeddings put them), unrelated names do not."""
    def embed_documents(self, texts):
        vocab = sorted({w for t in texts for w in t.split()})
        return [[(3.0 if w == "city" else 1.0) if w in t.split() else 0.0 for w in vocab] for t in texts]


def _split_city_claims(pool):
    """Denver and Boston filed under two different slots of the user (the fragmentation the merge fixes)."""
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    from brain.db import tenant
    with tenant(pool, owner) as c:
        ent = c.execute("insert into entity (namespace, name) values ('jeff', 'user') returning id").fetchone()[0]
        out = []
        for name, value, day in (("home_city", "Denver", "2023-05-01"), ("current_city", "Boston", "2023-06-01")):
            ep = c.execute("""insert into episode (namespace, client, ts, role, body) values ('jeff', 't', %s, 'user', %s)
                              returning id""", (day, f"I live in {value}.")).fetchone()[0]
            sid = c.execute("""insert into slot (namespace, entity_id, name, cardinality) values ('jeff', %s, %s, 'one')
                               returning id""", (ent, name)).fetchone()[0]
            cid = c.execute("""insert into claim (namespace, entity_id, slot_id, value, value_norm, valid_from, quote,
                                                  episode_id, status) values ('jeff', %s, %s, %s, %s, %s, %s, %s, 'active')
                               returning id""", (ent, sid, value, value.lower(), day, f"I live in {value}.", ep)).fetchone()[0]
            c.execute("insert into chunk (namespace, kind, ref_id, text) values ('jeff', 'claim', %s, %s)",
                      (cid, f"[{day}] user: {name.replace('_', ' ')} = {value}"))
            out.append(cid)
    return owner


def test_facts_search_lists_mis_filed_values_newest_first(client, pool):
    from datetime import date

    from brain.db import tenant
    owner = _split_city_claims(pool)
    with tenant(pool, owner) as c:
        got = ledger.facts(c, Providers(), ["jeff"], "Which city do I live in now?", date(2023, 7, 1))
    assert got["facts"] == 2
    lines = got["block"].splitlines()[1:]
    assert lines[0].startswith("- 2023-06-01: Boston") and lines[1].startswith("- 2023-05-01: Denver")


def test_nightly_merges_slots_that_name_the_same_attribute(client, pool):
    from brain import nightly
    from brain.db import tenant
    owner = _split_city_claims(pool)

    class P:
        jev = FakeJev()
        embedder = WordEmbedder()
    with tenant(pool, owner) as c:
        out = nightly.merge_slots(c, P())
    assert out["pairs_asked"] == 1 and out["merged"] == 1 and out["superseded"] == 1
    assert _rows(pool, "select count(*) from brain.slot") == [(1,)]
    assert _rows(pool, "select value, status, valid_to::text from brain.claim order by valid_from") == [
        ("Denver", "superseded", "2023-06-01"), ("Boston", "active", None)]


def test_excerpts_carry_ledger_notes_and_the_evidence_check(make_client, pool):
    from brain import core, worker
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    denver = _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
                  [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    boston = _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
                  [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    worker.write_path(pool, Providers())
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    chosen = [{"episode_id": denver}, {"episode_id": boston}]
    with tenant(pool, owner) as c:
        notes = core._ledger_notes(c, chosen, ["[2023-05-01] user: Denver.", "[2023-06-01] user: Boston."])
    assert notes == ["[2023-05-01] user: Denver. [memory: later replaced: 'Denver' -> 'Boston' on 2023-06-01]",
                     "[2023-06-01] user: Boston."]

    class SuffJev:
        def ask(self, state, qs):
            assert set(qs) == {"enough", "false_premise"} and "Boston" in state["evidence"][0]
            return JevResult({"enough": {"noul": 0.9}, "false_premise": {"noul": 0.05}}, "fake")

    class P:
        jev = SuffJev()
    assert core._sufficiency(P(), "Where do I live?", "2023-07-01", ["Boston."], []) == {"enough": 0.9,
                                                                                         "false_premise": 0.05}


def test_assistant_route_finds_the_line_that_answers(client, pool):
    from brain.db import tenant
    r = client.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "session_id": "tips", "turns": [
        {"role": "user", "text": "Any tips for sleeping better?", "ts": "2023-05-01T10:00:00Z"},
        {"role": "assistant", "text": "1. Keep a fixed bedtime.\n2. No screens an hour before bed.\n3. Keep the room cool.",
         "ts": "2023-05-01T10:00:05Z"}]})
    assert r.status_code == 200

    class LineJev:
        def ask(self, state, qs):
            pick = next(x["id"] for x in state["lines"] if "screens" in x["text"])
            return JevResult({"exists": {"noul": 0.9}, "which": {"choice": pick, "probabilities": {pick: 0.9}}}, "f")

    class P:
        jev = LineJev()
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        got = ledger.assistant_lines(c, P(), ["jeff"], "What was the second tip you gave me?",
                                     [{"session_id": "tips"}])
    assert got["line"] == "L2" and "> [2023-05-01] 2. No screens an hour before bed." in got["block"]
    assert "3. Keep the room cool." in got["block"]


def test_amounts_found_in_values():
    a = ledger.amounts
    assert [(n, u) for n, u, _ in a("spent $1,250.50 on the gift")] == [(1250.5, "usd")]
    assert [(n, u) for n, u, _ in a("drove 4.5 hours to Big Sur")] == [(4.5, "hours")]
    assert [(n, u) for n, u, _ in a("120 stars to reach gold")] == [(120.0, "stars")]
    assert a("graduated in 2015 at 10:30") == []                  # a year and a time of day are not amounts
    assert [(n, u) for n, u, _ in a("leads 5 engineers")] == [(5.0, None)]


def test_count_route_adds_amounts_in_code(make_client, pool):
    from datetime import date

    from brain import worker
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "The wedding gift for Sam cost $120.",
         [_claim("new:wedding_gifts", "$120", "The wedding gift for Sam cost $120.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "Priya's wedding gift was $85.",
         [_claim("new:wedding_gifts", "$85", "Priya's wedding gift was $85.")])
    worker.write_path(pool, Providers())
    assert _rows(pool, "select amount::float, unit from brain.claim order by id") == [(120.0, "usd"), (85.0, "usd")]
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    with tenant(pool, owner) as c:
        got = ledger.gather(c, Providers(), ["jeff"], "How much did I spend on wedding gifts?", date(2023, 7, 1))
    assert got["count"] == 2 and got["total"] == 205.0 and "Total of the amounts above (code-computed): 205 usd" in got["block"]
    assert got["block"].startswith("Memory ledger, items matching the question (code-counted")
    with tenant(pool, owner) as c:      # exp-count-style-v1: the same lines and numbers, worded as evidence
        ev = ledger.gather(c, Providers(), ["jeff"], "How much did I spend on wedding gifts?", date(2023, 7, 1), "evidence")
    assert (ev["count"], ev["total"]) == (2, 205.0) and ev["block"].splitlines()[1:3] == got["block"].splitlines()[1:3]
    head, last = ev["block"].splitlines()[0], ev["block"].splitlines()[-1]
    assert head.startswith("Memory ledger, further statements of the user's") and "code-counted" not in ev["block"]
    assert last == ("Sum of the amounts above (code-added; the answer only if the question asks for all of them "
                    "together): 205 usd")


def test_vector_neighbours_propose_entities_that_share_no_word(make_client, pool):
    from brain import worker

    class SisterEmbedder:                   # "my sister" and "maya" embed together; everything else apart
        def _v(self, t):
            t = t.lower()
            return [1.0, 0.0] if ("sister" in t or "maya" in t) else [0.0, 1.0]

        def embed_documents(self, texts):
            return [self._v(t) + [0.0] * 1022 for t in texts]

        def embed_query(self, text):
            return self._v(text) + [0.0] * 1022

    class AlignJev(FakeJev):
        def ask_many(self, reqs):
            out = []
            for state, qs in reqs:
                if all(q["type"] == "score" for q in qs.values()):
                    out.append(JevResult({k: ({"score": 2.0, "probabilities": {"2": 1.0}}
                                              if state["candidates"][int(k[1:])]["name"] == "My sister"
                                              else {"score": 0.0, "probabilities": {"0": 1.0}}) for k in qs}, self.model))
                else:
                    out.extend(super().ask_many([(state, qs)]))
            return out

    class P:
        jev = AlignJev()
        embedder = SisterEmbedder()
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "My sister lives in Austin.",
         [_claim("new:home_city", "Austin", "My sister lives in Austin.", mention="My sister")])
    _cap(client, "s2", "2023-05-02T10:00:00Z", "Maya got a dog.",
         [_claim("new:pet", "a dog", "Maya got a dog.", mention="Maya")])
    worker.write_path(pool, P())
    owners = _rows(pool, """select c.value, n.name from brain.claim c join brain.entity n on n.id = c.entity_id
                            order by c.id""")
    assert owners == [("Austin", "My sister"), ("a dog", "My sister")]
    assert "maya" in _rows(pool, "select aliases from brain.entity where name = 'My sister'")[0][0]


def test_you_and_assistant_mentions():
    assert "you" in ledger.YOU and "she" in ledger.PRONOUNS
    assert ledger._key("The Theater People.") == "theater people"
    assert ledger._slug("City of Residence!") == "city_of_residence"


def test_an_add_never_supersedes_even_on_a_one_value_slot(make_client, pool):
    """exp-supersede-v1: a second injury is another value, not a replacement of the first."""
    from brain import worker
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I sprained my ankle playing basketball.",
         [_claim("new:ankle_injury", "sprained ankle playing basketball", "I sprained my ankle playing basketball.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I hurt my ankle again, this time on a hike.",
         [_claim("new:ankle_injury", "hurt ankle on a hike", "I hurt my ankle again, this time on a hike.")])
    worker.write_path(pool, Providers())
    assert _rows(pool, "select s.cardinality from brain.slot s where s.name = 'ankle_injury'") == [("one",)]
    assert _rows(pool, "select value, status from brain.claim order by id") == [
        ("sprained ankle playing basketball", "active"), ("hurt ankle on a hike", "active")]


class CheckingJev(FakeJev):
    """FakeJev plus the placement check: the turn supports "home city = X" only when it says "moved to"."""
    def ask_many(self, requests):
        out = []
        for state, questions in requests:
            if set(questions) == {"supported"}:
                assert state["statement"].startswith("user: home city = ")
                out.append(JevResult({"supported": {"noul": 0.9 if "moved to" in state["source"] else 0.1}}, self.model))
            else:
                out.extend(super().ask_many([(state, questions)]))
        return out


def _three_cities(make_client, pool, monkeypatch, flag):
    from brain import ledger, worker
    monkeypatch.setattr(ledger, "PLACE_CHECK", flag)
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    _cap(client, "s3", "2023-06-20T10:00:00Z", "I was born in Austin.",
         [_claim("new:birth_city", "Austin", "I was born in Austin.")])
    prov = Providers()
    prov.jev = CheckingJev()
    worker.write_path(pool, prov)
    return _rows(pool, """select c.value, c.status, s.name from brain.claim c join brain.slot s on s.id = c.slot_id
                          order by c.id""")


def test_place_check_keeps_a_sibling_fact_out_of_an_existing_slot(make_client, pool, monkeypatch):
    """exp-page-audit-v1: W6 put a sibling fact under an existing slot; the birth city would end the home city."""
    rows = _three_cities(make_client, pool, monkeypatch, True)
    assert rows == [("Denver", "superseded", "home_city"), ("Boston", "active", "home_city"),   # a real update joins
                    ("Austin", "active", "birth_city")]
    checks = _rows(pool, """select outcome, input->>'statement' from brain.decision_log
                            where stage = 'write.place_check' order by id""")
    assert checks == [("kept:home_city", "user: home city = Boston"), ("refused:home_city", "user: home city = Austin")]


def test_place_check_is_off_by_default(make_client, pool, monkeypatch):
    from brain import ledger
    assert ledger.PLACE_CHECK is False                       # production is unchanged until the experiment says so
    rows = _three_cities(make_client, pool, monkeypatch, False)
    assert [r[2] for r in rows] == ["home_city"] * 3         # as before: all three filed under the one slot
    assert _rows(pool, "select count(*) from brain.decision_log where stage = 'write.place_check'") == [(0,)]


class PerSlotJev(FakeJev):
    """FakeJev plus the per-slot placement questions: city_of_residence is surely home_city (0.9); birth_city is
    a maybe (0.6), which the old rule's 0.5 line would have joined."""
    def ask_many(self, requests):
        out = []
        for state, questions in requests:
            per_slot = {n: q for n, q in questions.items() if n[0] == "s" and n[1:].isdigit()}
            if not per_slot:
                out.extend(super().ask_many([(state, questions)]))
                continue
            assert "fits" not in questions and "which" not in questions      # one request, the old two questions gone
            rest = super().ask_many([(state, {n: q for n, q in questions.items() if n not in per_slot})])[0]
            ans = dict(rest.answers)
            for n, q in per_slot.items():
                slot = state["slots"][int(n[1:])]["name"]
                assert f"'{slot.replace('_', ' ')}'" in q["instructions"]     # the slot is named in its question
                proposed = state["claim"]["slot_proposed"]
                ans[n] = {"noul": 0.1 if slot != "home_city" else 0.9 if proposed == "city_of_residence" else 0.6}
            out.append(JevResult(ans, self.model))
        return out


def test_per_slot_placement_joins_a_sure_slot_and_keeps_a_maybe_apart(make_client, pool, monkeypatch):
    """exp-ledger-stable-v1: one Noul per existing slot; only 0.7 and above joins, a 0.6 opens its own slot."""
    from brain import ledger, worker
    assert ledger.PLACE_PER_SLOT is False                    # off until its gate is met
    monkeypatch.setattr(ledger, "PLACE_PER_SLOT", True)
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    _cap(client, "s3", "2023-06-20T10:00:00Z", "I was born in Austin.",
         [_claim("new:birth_city", "Austin", "I was born in Austin.")])
    prov = Providers()
    prov.jev = PerSlotJev()
    worker.write_path(pool, prov)
    rows = _rows(pool, """select c.value, c.status, s.name from brain.claim c join brain.slot s on s.id = c.slot_id
                          order by c.id""")
    assert rows == [("Denver", "superseded", "home_city"), ("Boston", "active", "home_city"),
                    ("Austin", "active", "birth_city")]
    logged = _rows(pool, """select outcome, (output->>'same')::float, threshold from brain.decision_log
                            where stage = 'write.place' and output ? 'same' order by id""")
    assert logged == [("existing:home_city", 0.9, 0.7), ("new:birth_city", 0.6, 0.7)]


def test_open_route_modes_decide_how_much_of_the_ledger_an_open_question_gets(make_client, pool, monkeypatch):
    """exp-open-route-v1: on a question the router calls open, `quiet` drops the ledger blocks and `plain` also
    drops the notes on excerpts; another route is untouched by the setting."""
    from brain import core, router, worker
    client = make_client(write_path_enabled=True, ledger_route_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    worker.write_path(pool, Providers())
    picked = {"route": "open"}

    def fake_route(providers, question, question_date=None):
        nouls = {n: 0.0 for n in router.COMPANIONS}
        return router.RouteDecision(picked["route"], 0.9, 0.9, {picked["route"]: 0.9}, nouls, "route", False, "fake")
    monkeypatch.setattr(core.router, "route", fake_route)
    monkeypatch.setattr(core.ledger, "lookup", lambda *a, **k: {"entity": "user", "block": "LEDGER LOOKUP BLOCK"})
    monkeypatch.setattr(core.ledger, "facts", lambda *a, **k: {"facts": 1, "block": "LEDGER FACTS BLOCK"})

    def prompt(mode):
        monkeypatch.setattr(core, "OPEN_ROUTE", mode)
        r = client.post("/v1/context", headers=ADMIN, json={"namespace": "jeff", "question": "Where do I live?",
                                                            "question_date": "2023-07-01"})
        assert r.status_code == 200, r.text
        return r.json()
    full, quiet, plain = prompt("full"), prompt("quiet"), prompt("plain")
    note = "later replaced"
    assert "LEDGER LOOKUP BLOCK" in full["prompt"] and "LEDGER FACTS BLOCK" in full["prompt"]
    assert note in full["prompt"] and full["ledger"] == {"entity": "user", "facts": 1}
    assert "LEDGER" not in quiet["prompt"] and note in quiet["prompt"] and quiet["ledger"] is None
    assert "LEDGER" not in plain["prompt"] and note not in plain["prompt"] and plain["route"] == "open"
    assert [x["chunk_id"] for x in plain["excerpts"]] == [x["chunk_id"] for x in full["excerpts"]]   # same retrieval
    picked["route"] = "current"                                # the router asked for the ledger: all of it, any mode
    asked = prompt("plain")
    assert "LEDGER LOOKUP BLOCK" in asked["prompt"] and note in asked["prompt"]
    assert core.OPEN_ROUTE_MODES == ("full", "quiet", "plain")


def test_context_gives_the_same_pack_when_the_router_and_ledger_run_beside_recall(make_client, pool, monkeypatch):
    """profile-context-v1 (BRAIN_CONTEXT_PARALLEL): only the waiting changes. The router and the ledger's reads run
    while recall does, on the request's own connection; the pack, the route and the ledger's blocks are the same.
    A provider error on the side path leaves the open route's pack, as it does one after the other."""
    import threading

    from brain import core, router, worker
    from brain.providers import ProviderError
    client = make_client(write_path_enabled=True, ledger_route_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    worker.write_path(pool, Providers())
    threads = {}
    real_route, real_facts = router.route, core.ledger.facts

    def seen(name, fn):
        def wrapper(*a, **k):
            threads[name] = threading.current_thread() is threading.main_thread() or threading.current_thread().name
            return fn(*a, **k)
        return wrapper
    monkeypatch.setattr(core.router, "route", seen("route", real_route))
    monkeypatch.setattr(core.ledger, "facts", seen("facts", real_facts))

    def ask(question, parallel):
        monkeypatch.setattr(core, "CONTEXT_PARALLEL", parallel)
        threads.clear()
        r = client.post("/v1/context", headers=ADMIN, json={"namespace": "jeff", "question": question,
                                                            "question_date": "2023-07-01"})
        assert r.status_code == 200, r.text
        return {k: v for k, v in r.json().items() if k != "request_id"}, dict(threads)
    for question in ("Where do I live?", "How many cities have I lived in?", "What did I say about Denver?"):
        one_by_one, t1 = ask(question, False)
        beside, t2 = ask(question, True)
        assert beside == one_by_one, question
        assert t1["route"] != t2["route"]                       # the router left the request's own thread
        if "facts" in t2:
            assert t2["facts"] != t2["route"]                   # and the facts search runs beside the lookup

    def down(*a, **k):
        raise ProviderError("jev", "down")
    plain = ask("Where do I live?", True)[0]
    monkeypatch.setattr(core.router, "route", down)
    for parallel in (False, True):
        got = ask("Where do I live?", parallel)[0]
        assert got["route"] is None and got["ledger"] is None and got["excerpts"] == plain["excerpts"]


def test_an_eval_can_turn_per_slot_placement_on_for_its_own_thread_only():
    import threading

    from brain import ledger
    assert ledger._per_slot() is False
    token = ledger.PLACE_PER_SLOT_OVERRIDE.set(True)
    try:
        other = []
        t = threading.Thread(target=lambda: other.append(ledger._per_slot()))
        t.start()
        t.join()
        assert ledger._per_slot() is True and other == [False]
    finally:
        ledger.PLACE_PER_SLOT_OVERRIDE.reset(token)
    assert ledger._per_slot() is False


def test_soft_notes_point_at_the_later_statement_without_saying_it_replaced_anything(make_client, pool):
    """exp-ledger-wording-v1: the full wording states the ledger's verdict; the soft one leaves it to the reader."""
    from brain import core, worker
    from brain.db import tenant
    assert core.NOTE_STYLE == "full"                         # the shipped wording until a gate says otherwise
    client = make_client(write_path_enabled=True)
    denver = _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
                  [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    boston = _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
                  [_claim("new:city_of_residence", "Boston", "I just moved to Boston.")])
    worker.write_path(pool, Providers())
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    chosen = [{"episode_id": denver}, {"episode_id": boston}]
    texts = ["[2023-05-01] user: Denver.", "[2023-06-01] user: Boston."]
    with tenant(pool, owner) as c:
        soft = core._ledger_notes(c, chosen, texts, "soft")
    assert soft == ["[2023-05-01] user: Denver. [memory: see also the later statement of 2023-06-01: 'Boston']",
                    "[2023-06-01] user: Boston."]
    assert "replaced" not in soft[0] and "no longer holds" not in soft[0]


class EndsJev(FakeJev):
    """FakeJev plus the "has the older value stopped being true" Noul: yes for a move, no for anything else."""
    def ask_many(self, requests):
        out = []
        for state, questions in requests:
            ends = {n: q for n, q in questions.items() if n[0] == "e" and n[1:].isdigit()}
            rest = super().ask_many([(state, {n: q for n, q in questions.items() if n not in ends})])[0]
            ans = dict(rest.answers)
            for n, q in ends.items():
                assert f"existing[{n[1:]}]" in q["instructions"] and set(q["criteria"]) == {"true", "false"}
                ans[n] = {"noul": 0.9 if "moved" in state["new"]["quote"] else 0.2}
            out.append(JevResult(ans, self.model))
        return out


def test_an_update_ends_the_older_value_only_when_it_stopped_being_true(make_client, pool, monkeypatch):
    """exp-relation-ends-v1: "updates" alone ends a value today; with the flag the older value must also be judged
    to have stopped. A second win is not the end of the first."""
    from brain import ledger, worker
    assert ledger.RELATE_ENDS is False                       # off until its gate is met
    monkeypatch.setattr(ledger, "RELATE_ENDS", True)
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-05-01T10:00:00Z", "I moved to Denver last spring.",
         [_claim("new:home_city", "Denver", "I moved to Denver last spring.")])
    _cap(client, "s2", "2023-06-01T10:00:00Z", "I just moved to Boston.",
         [_claim("home_city", "Boston", "I just moved to Boston.")])
    _cap(client, "s3", "2023-06-02T10:00:00Z", "I won my second tournament.",
         [_claim("new:tournament_result", "won second tournament", "I won my second tournament.")])
    _cap(client, "s4", "2023-07-02T10:00:00Z", "I won my fourth tournament.",
         [_claim("tournament_result", "won fourth tournament", "I won my fourth tournament.")])
    prov = Providers()
    prov.jev = EndsJev()
    worker.write_path(pool, prov)
    rows = _rows(pool, "select value, status from brain.claim order by id")
    assert rows == [("Denver", "superseded"), ("Boston", "active"),
                    ("won second tournament", "active"), ("won fourth tournament", "active")]
    logged = _rows(pool, """select output->'relations', output->'applied', output->'ends', threshold
                            from brain.decision_log where stage = 'write.relate' order by id""")
    assert logged == [(["updates"], ["ended"], [0.9], 0.7), (["updates"], ["kept_both"], [0.2], 0.7)]


def test_different_names_is_about_two_proper_names_that_share_no_word():
    assert ledger.different_names("Sophia", "Emma", ["emma"])
    assert ledger.different_names("Light Yagami", "Death Note")
    assert not ledger.different_names("Rachel", "Rachel Lee")            # a shared word
    assert not ledger.different_names("Liz", "Elizabeth", ["liz"])       # known by that name already
    assert not ledger.different_names("Max", "my dog")                   # a description can be the named one
    assert not ledger.different_names("my son", "Evan")
    assert not ledger.different_names("Trello", "user")


def _two_women(make_client, pool):
    """Emma, then Sophia and "my neighbour": every alignment Score against another person comes back 0.9 (maybe
    the same, or related); against the user it is 0."""
    from brain import worker

    class Near:                              # every name is every other name's vector neighbour
        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 1023 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 1023

    class MaybeJev(FakeJev):
        def ask_many(self, reqs):
            out = []
            for state, qs in reqs:
                if "mention" in state and all(q["type"] == "score" for q in qs.values()):
                    out.append(JevResult({k: {"score": 0.0 if state["candidates"][int(k[1:])]["name"] == "user" else 0.9,
                                              "probabilities": {"1": 1.0}} for k in qs}, self.model))
                else:
                    out.extend(super().ask_many([(state, qs)]))
            return out

    class P:
        jev = MaybeJev()
        embedder = Near()
    client = make_client(write_path_enabled=True)
    _cap(client, "s1", "2023-04-11T10:00:00Z", "I had lunch with Emma, a freelance writer.",
         [_claim("new:occupation", "freelance writer", "I had lunch with Emma, a freelance writer.", mention="Emma")])
    _cap(client, "s2", "2023-04-11T11:00:00Z", "I met Sophia at a networking event.",
         [_claim("new:met_at", "a networking event", "I met Sophia at a networking event.", mention="Sophia")])
    _cap(client, "s3", "2023-04-11T12:00:00Z", "My neighbour runs marathons.",
         [_claim("new:hobby", "runs marathons", "My neighbour runs marathons.", mention="My neighbour")])
    worker.write_path(pool, P())
    return _rows(pool, """select c.value, n.name, c.tentative from brain.claim c join brain.entity n on n.id = c.entity_id
                          order by c.id""")


def test_a_maybe_score_files_a_differently_named_person_under_the_other_as_built(make_client, pool):
    assert _two_women(make_client, pool) == [("freelance writer", "Emma", False), ("a networking event", "Emma", True),
                                             ("runs marathons", "Emma", True)]


def test_resolve_names_gives_a_different_proper_name_its_own_entity(make_client, pool):
    """exp-resolve-names-v1: Sophia is not Emma on a "maybe"; a description still links tentatively."""
    token = ledger.RESOLVE_NAMES_OVERRIDE.set(True)
    try:
        owners = _two_women(make_client, pool)
    finally:
        ledger.RESOLVE_NAMES_OVERRIDE.reset(token)
    assert owners[:2] == [("freelance writer", "Emma", False), ("a networking event", "Sophia", False)]
    assert owners[2][0] == "runs marathons" and owners[2][2] is True      # tentative, under one of the two
    logged = _rows(pool, """select output->>'rule', outcome from brain.decision_log where stage = 'write.resolve'
                            and input->'mention'->>'name' = 'Sophia'""")
    assert logged == [("different_name", "new")]


def test_count_block_off_leaves_the_gathered_list_out_of_the_pack(monkeypatch):
    """exp-count-block-v1: with the block off a count question is routed as before and the ledger is not read."""
    from brain import core, router

    class D:
        route, nouls = "count", {}
    calls = []
    monkeypatch.setattr(router, "route", lambda providers, question, qdate: D())
    monkeypatch.setattr(ledger, "gather", lambda *a: calls.append(1) or {"count": 2, "block": "two things"})
    d, found, plain_open, err = core._route_and_ledger(None, None, ["u"], "How many trips?", "2023-05-01", True, "full")
    assert (d.route, found, plain_open, err) == ("count", [{"count": 2, "block": "two things"}], False, None)
    d, found, plain_open, err = core._route_and_ledger(None, None, ["u"], "How many trips?", "2023-05-01", True, "full",
                                                       count_block=False)
    assert (d.route, found, plain_open, err) == ("count", [], False, None) and len(calls) == 1


class DisputeJev(EndsJev):
    """EndsJev, with every relation between different values called "contradicts"; the ends Noul is 0.9 only for a
    quote that says "actually" (a correction)."""
    def ask_many(self, requests):
        out = []
        for state, questions in requests:
            got = super().ask_many([(state, questions)])[0]
            ans = dict(got.answers)
            for n in questions:
                if n[0] == "r" and n[1:].isdigit() and ans[n]["choice"] != "same":
                    ans[n] = {"choice": "contradicts", "probabilities": {"contradicts": 0.9}}
                if n[0] == "e" and n[1:].isdigit():
                    ans[n] = {"noul": 0.9 if "actually" in state["new"]["quote"].lower() else 0.2}
            out.append(JevResult(ans, self.model))
        return out


def _two_workshops(make_client, pool, monkeypatch, disputes):
    from brain import ledger, worker
    monkeypatch.setattr(ledger, "RELATE_ENDS", True)
    token = ledger.RELATE_DISPUTES_OVERRIDE.set(disputes)
    try:
        client = make_client(write_path_enabled=True)
        _cap(client, "s1", "2023-02-01T10:00:00Z", "I paid $20 for the mindfulness workshop.",
             [_claim("new:workshop_cost", "$20", "I paid $20 for the mindfulness workshop.")])
        _cap(client, "s2", "2023-02-02T10:00:00Z", "I paid $500 for the marketing workshop.",
             [_claim("workshop_cost", "$500", "I paid $500 for the marketing workshop.")])
        _cap(client, "s3", "2023-02-03T10:00:00Z", "I wake up at 8:30.",
             [_claim("new:wake_time", "8:30", "I wake up at 8:30.")])
        _cap(client, "s4", "2023-02-04T10:00:00Z", "Actually I wake up at 7:30.",
             [_claim("wake_time", "7:30", "Actually I wake up at 7:30.")])
        prov = Providers()
        prov.jev = DisputeJev()
        worker.write_path(pool, prov)
    finally:
        ledger.RELATE_DISPUTES_OVERRIDE.reset(token)
    return (_rows(pool, "select value, status from brain.claim order by id"),
            _rows(pool, """select output->'relations', output->'applied', output->'ends' from brain.decision_log
                           where stage = 'write.relate' order by id"""))


def test_a_contradiction_marks_both_claims_disputed_as_built(make_client, pool, monkeypatch):
    rows, logged = _two_workshops(make_client, pool, monkeypatch, False)
    assert rows == [("$20", "contested"), ("$500", "contested"), ("8:30", "contested"), ("7:30", "contested")]
    assert logged == [(["contradicts"], ["contested"], [0.2]), (["contradicts"], ["contested"], [0.9])]


def test_relate_disputes_keeps_both_when_the_older_value_was_not_shown_to_be_wrong(make_client, pool, monkeypatch):
    """exp-relate-disputes-v1: two workshops' prices are not a dispute; a correction still is."""
    from brain import ledger
    assert ledger.RELATE_DISPUTES is False                   # off until its gate is met
    rows, logged = _two_workshops(make_client, pool, monkeypatch, True)
    assert rows == [("$20", "active"), ("$500", "active"), ("8:30", "contested"), ("7:30", "contested")]
    assert logged == [(["contradicts"], ["kept_both"], [0.2]), (["contradicts"], ["contested"], [0.9])]


def test_near_slots_pairs_names_for_one_attribute_and_not_others():
    assert ledger.near_slots("payment_terms", "payment_term")
    assert ledger.near_slots("shop_monthly_rent", "monthly_rent")
    assert ledger.near_slots("billing_module_owner", "billing_module_owner")
    assert not ledger.near_slots("shipping_carrier", "favourite_film")
    assert not ledger.near_slots("", "role")


def _owners(make_client, pool):
    """Two people named as the owner of one thing, a month apart, and two people's own film opinions."""
    from brain import worker
    client = make_client(write_path_enabled=True)
    said = [("s1", "2023-05-01T10:00:00Z", "Priya owns the billing module.", "Priya", "new:billing_module_owner", "owns the billing module"),
            ("s2", "2023-05-02T10:00:00Z", "Maren thought the film was slow.", "Maren", "new:film_opinion", "the film was slow"),
            ("s3", "2023-06-01T10:00:00Z", "Billing moved to Henrik, he owns the billing module.", "Henrik",
             "new:billing_module_owner", "Billing moved to Henrik"),
            ("s4", "2023-06-02T10:00:00Z", "Isak loved the film.", "Isak", "new:film_opinion", "loved the film")]
    for session, ts, text, who, slot, value in said:
        _cap(client, session, ts, text, [_claim(slot, value, text, mention=who)])
    prov = Providers()
    prov.jev = EndsJev()
    worker.write_path(pool, prov)
    return _rows(pool, """select n.name, c.status, c.valid_to::text from brain.claim c
                          join brain.entity n on n.id = c.entity_id order by c.id""")


def test_a_fact_filed_under_two_names_is_not_compared_as_built(make_client, pool):
    assert ledger.RELATE_ACROSS is False                     # off until its gate is met
    assert [r[1] for r in _owners(make_client, pool)] == ["active"] * 4
    assert _rows(pool, "select count(*) from brain.decision_log where stage = 'write.relate_across'") == [(0,)]


def test_relate_across_ends_a_replaced_value_filed_under_another_name(make_client, pool, monkeypatch):
    """exp-relate-across-v1: the new owner ends the old one's claim; two people's own opinions both stand."""
    monkeypatch.setattr(ledger, "RELATE_ACROSS", True)
    assert _owners(make_client, pool) == [("Priya", "superseded", "2023-06-01"), ("Maren", "active", None),
                                          ("Henrik", "active", None), ("Isak", "active", None)]
    assert _rows(pool, "select type, source from brain.claim_link") == [("supersedes", "write.relate_across")]
    logged = _rows(pool, """select input->>'task', input->'existing'->0->>'about', output->'relations', output->'applied'
                            from brain.decision_log where stage = 'write.relate_across' order by id""")
    assert logged == [("W7x", "Priya", ["updates"], ["ended"]), ("W7x", "Maren", ["updates"], ["none"])]


def test_the_count_route_reads_as_many_claims_a_side_as_it_is_told(make_client, pool):
    """exp-gather-top-v1: 40 a side as built; an eval arm (or BRAIN_GATHER_TOP) may read more, and the limit note
    names the limit in use."""
    from datetime import date

    from brain import worker
    from brain.db import tenant
    assert ledger.GATHER_TOP == 40
    client = make_client(write_path_enabled=True)
    for k, who in enumerate(("Sam", "Priya", "Noor"), 1):
        text = f"I went to {who}'s wedding."
        _cap(client, f"s{k}", f"2023-0{k}-01T10:00:00Z", text, [_claim("new:weddings_attended", f"{who}'s wedding", text)])
    worker.write_path(pool, Providers())
    owner = _rows(pool, "select id from brain.account where name = 'owner'")[0][0]
    question = "How many weddings did I go to this year?"
    with tenant(pool, owner) as c:
        full = ledger.gather(c, Providers(), ["jeff"], question, date(2023, 7, 1))
        one = ledger.gather(c, Providers(), ["jeff"], question, date(2023, 7, 1), top=1)
    assert full["count"] == 3 and not full["capped"]
    assert one["count"] == 1 and one["capped"] and "stops at 1 candidates a side" in one["block"]
