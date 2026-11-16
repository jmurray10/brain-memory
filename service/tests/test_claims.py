"""Build step 4 (DESIGN.md): the user's model sends claims with capture; the brain triages the
turn and checks the claims (exact quote, value in quote, Jev verify). No generative model inside the brain."""
import json

import pytest

from brain import claims
from brain.providers import JevResult, ProviderError
from conftest import ADMIN

GOOD = {"mention": "", "slot": "new:home_city", "value": "Denver", "time_phrase": "last spring", "polarity": "asserted",
        "quote": "I moved to Denver last spring."}
BAD_QUOTE = {**GOOD, "value": "Paris", "quote": "I moved to Paris."}
FAR_VALUE = {**GOOD, "slot": "new:employer", "value": "works at Acme Robotics"}          # value not in its quote
OFF = {**GOOD, "slot": "new:employer", "value": "Denver last spring"}                   # passes code, verify drops


class FakeJev:
    """Triage: 'thanks' turns are small talk, everything else a lasting fact. Verify: the employer slot is off
    target."""
    model = "fake-jev"

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def ask_many(self, reqs):
        if self.fail:
            raise ProviderError("jev", "HTTP 503", 503)
        out = []
        for state, qs in reqs:
            self.calls.append((state, qs))
            if "kind" in qs:
                small = "thanks" in state["turn"]["text"].lower()
                p = 0.05 if small else 0.9
                out.append(JevResult({"lasting_fact": {"noul": p}, "preference": {"noul": 0.1}, "plan": {"noul": 0.1},
                                      "has_date": {"noul": 0.8}, "assistant_recall": {"noul": 0.9},
                                      "kind": {"choice": "smalltalk" if small else "fact",
                                               "probabilities": {"smalltalk" if small else "fact": 0.9}}}, self.model))
            elif "claims" not in state:      # step 5 questions (ledger): new entity, new slot, plain answers
                ans = {}
                for n, q in qs.items():
                    if q["type"] == "score":
                        ans[n] = {"score": 0.0, "probabilities": {"0": 1.0}}
                    elif q["type"] == "choice":
                        first = next(iter(q["criteria"]))
                        ans[n] = {"choice": first, "probabilities": {first: 0.9}}
                    else:
                        ans[n] = {"noul": 0.1}
                out.append(JevResult(ans, self.model))
            else:
                ans = {}
                for i, c in enumerate(state["claims"]):
                    ans[f"c{i}_hallucinated"] = {"noul": 0.04}
                    ans[f"c{i}_off_target"] = {"noul": 0.92 if c["slot"] == "new:employer" else 0.05}
                    ans[f"c{i}_incomplete"] = {"noul": 0.05}
                    pol = "planned" if c["slot"] == "new:move_plan" else c["polarity"]     # Jev disagrees here
                    ans[f"c{i}_polarity"] = {"choice": pol, "probabilities": {pol: 0.9}}
                    if f"c{i}_about_speaker" in qs:
                        ans[f"c{i}_about_speaker"] = {"noul": 0.1 if c["slot"] == "new:project_status" else 0.9}
                out.append(JevResult(ans, self.model))
        return out


class Providers:
    def __init__(self, jev=None):
        self.jev = jev or FakeJev()


def test_parse_claims_normalizes_and_rejects():
    raw = "Here you go:\n```json\n" + json.dumps([
        {**GOOD, "slot": "Home City"},
        {**GOOD, "polarity": "certain"},          # not a polarity
        {**GOOD, "quote": ""},                    # no quote
        {**GOOD, "slot": "new:"}]) + "\n```"
    out, bad = claims.parse_claims(raw)
    assert [c["slot"] for c in out] == ["new:home_city"] and bad == 3
    assert claims.parse_claims("not json at all") == ([], 1)
    assert claims.parse_claims("[{broken") == ([], 1)          # cut off by max_tokens
    assert claims.parse_claims("[]") == ([], 0)


def test_quote_check_tolerates_typography_only():
    text = "She said “the dose is 40 mg”   twice."
    assert claims.quote_ok('She said "the dose is 40 mg" twice.', text)
    assert not claims.quote_ok("the dose is 50 mg", text)
    assert not claims.quote_ok("a", text)


def test_value_must_come_from_the_quote():
    assert claims.value_in_quote("1.6-fold over parental at 100 ng/mL", "At 100 ng/mL it is 1.6-fold over parental.")
    assert claims.value_in_quote("Denver", "I moved to Denver last spring.")
    assert claims.value_in_quote("the", "anything")            # no content words: nothing to check
    # specifics (names, numbers, ids) must be in the quote
    assert not claims.value_in_quote("approved by Valuation Committee", "Subject: VA-4 approved.")
    assert not claims.value_in_quote("draft is with Mara Ellison", "Appendix A draft is with you by 2026-03-20.")
    assert not claims.value_in_quote("attended the Pride fest with Melanie last year", "We had a blast last year at the Pride fest.")
    assert not claims.value_in_quote("kappa = 11, down from 14", "It works.")
    # rewording of ordinary words is fine
    assert claims.value_in_quote("wants to create a more inviting atmosphere in the garden",
                                 "I've actually been thinking about creating a more inviting atmosphere in my garden.")
    assert claims.value_in_quote("formal letter to follow regarding PI step-down",
                                 "Dr. Hannah Cole will step down as PI; formal letter to follow.")
    assert not claims.value_in_quote("family is super important to me", "Yeah, you're right, Caroline.")


def test_double_negation_convention():
    assert claims.normalize_polarity("negated", "no dose reduction") == "asserted"
    assert claims.normalize_polarity("negated", "dose reduction") == "negated"
    assert claims.normalize_polarity("planned", "not before June") == "planned"


def test_slot_normalization():
    assert claims.normalize_slot("Home City") == "new:home_city"
    assert claims.normalize_slot("new:Dose (mg)") == "new:dose_mg"
    assert claims.normalize_slot("person.home.city", ("person.home.city",)) == "person.home.city"
    assert claims.normalize_slot("!!!") == "new:misc"


def _capture(client, turns, ns="jeff"):
    r = client.post("/v1/capture", headers=ADMIN, json={"namespace": ns, "session_id": "s1", "turns": turns})
    assert r.status_code == 200, r.text
    return r.json()["episodes"]


TURNS = [{"role": "user", "text": "Hi there.", "ts": "2023-05-01T10:00:00Z"},
         {"role": "user", "text": "I moved to Denver last spring.", "ts": "2023-05-01T10:01:00Z",
          "claims": [GOOD, BAD_QUOTE, FAR_VALUE, OFF]},
         {"role": "assistant", "text": "Denver is lovely; try the botanic gardens.", "ts": "2023-05-01T10:02:00Z",
          "claims": [GOOD]},
         {"role": "user", "text": "Thanks!", "ts": "2023-05-01T10:03:00Z"}]


def test_client_claims_are_checked_by_code_and_jev(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    eps = _capture(client, TURNS)
    assert eps[1]["claims_received"] == 4 and eps[2]["claims_ignored"] == 1     # assistant turns carry no claims
    res = worker.write_path(pool, Providers())
    assert res["jobs"] == 3 and res["turns"] == 3 and res["stopped"] is None
    assert (res["worthy"], res["received"], res["code_dropped"], res["verify_dropped"], res["stored"]) == (2, 4, 2, 1, 1)
    assert res["no_claims_worthy"] == 1          # "Hi there." passed the fake triage but came with no claims
    with pool.connection() as c:
        rows = c.execute("select status, drop_reason, value, extractor from brain.claim_candidate order by id").fetchall()
        # the good claim is checked (step 4), then placed in the ledger (step 5)
        assert rows == [("placed", None, "Denver", "admin"), ("dropped", "quote_not_in_turn", "Paris", "admin"),
                        ("dropped", "value_not_in_quote", "works at Acme Robotics", "admin"),
                        ("dropped", "verify:off_target", "Denver last spring", "admin")]
        stages = dict(c.execute("""select stage, count(*) from brain.decision_log where stage like 'write.%'
                                   group by stage""").fetchall())
        assert stages == {"write.triage": 3, "write.check": 2, "write.verify": 2, "write.place": 1}   # place: step 5
        tasks = {r[0] for r in c.execute("select input->>'task' from brain.decision_log where stage like 'write.%'")}
        assert tasks == {"W1", "W4", "W6", None}      # code checks carry the claim, not a booster task
        outcomes = dict(c.execute("""select input->>'episode_id', outcome from brain.decision_log
                                     where stage = 'write.triage'""").fetchall())
        assert outcomes == {str(eps[0]["id"]): "worthy", str(eps[1]["id"]): "worthy", str(eps[3]["id"]): "skip"}
    assert worker.write_path(pool, Providers())["jobs"] == 0          # nothing left to do


def test_jev_cross_checks_polarity_and_speaker(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    text = "I moved to Denver last spring. The bridge retrofit is on hold."
    plan = {**GOOD, "slot": "new:move_plan"}                         # client says asserted, Jev picks planned
    proj = {**GOOD, "slot": "new:project_status", "value": "on hold", "quote": "The bridge retrofit is on hold."}
    _capture(client, [{"role": "user", "text": text, "claims": [GOOD, plan, proj]}])
    res = worker.write_path(pool, Providers())
    assert (res["stored"], res["verify_dropped"]) == (1, 2)
    with pool.connection() as c:
        got = dict(c.execute("select slot, coalesce(drop_reason, status) from brain.claim_candidate").fetchall())
    assert got == {"new:home_city": "placed", "new:move_plan": "verify:polarity",
                   "new:project_status": "verify:not_about_speaker"}


def test_claims_are_ignored_while_the_write_path_is_off(client, pool):
    eps = _capture(client, TURNS[:2])
    assert eps[1]["claims_ignored"] == 4
    with pool.connection() as c:
        assert c.execute("select count(*) from brain.claim_candidate").fetchone()[0] == 0
        assert c.execute("select count(*) from brain.job where kind = 'write_path'").fetchone()[0] == 0


def test_claim_fields_are_validated(make_client):
    client = make_client(write_path_enabled=True)
    bad = {**GOOD, "polarity": "certainly"}
    r = client.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "turns": [
        {"role": "user", "text": "I moved to Denver last spring.", "claims": [bad]}]})
    assert r.status_code == 422
    r = client.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "turns": [
        {"role": "user", "text": "x y z", "claims": [GOOD] * 26}]})
    assert r.status_code == 422


def test_candidates_are_isolated_by_account(make_client, pool):
    from brain import worker
    from brain.db import tenant
    client = make_client(write_path_enabled=True)
    _capture(client, TURNS[:2])
    worker.write_path(pool, Providers())
    with pool.connection() as c:
        other = c.execute("insert into brain.account (name) values ('other-co') returning id").fetchone()[0]
        owner = c.execute("select id from brain.account where name = 'owner'").fetchone()[0]
    with tenant(pool, other) as c:
        assert c.execute("select count(*) from brain.claim_candidate").fetchone()[0] == 0
    with tenant(pool, owner) as c:
        assert c.execute("select count(*) from brain.claim_candidate").fetchone()[0] == 4


def test_forget_removes_candidates_and_write_path_logs(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    ids = [e["id"] for e in _capture(client, TURNS[:2])]
    worker.write_path(pool, Providers())
    r = client.post("/v1/forget", headers=ADMIN, json={"namespace": "jeff", "episode_ids": [ids[1]]})
    assert r.status_code == 200 and r.json()["forgotten"]["claim_candidates"] == 4
    with pool.connection() as c:
        assert c.execute("select count(*) from brain.claim_candidate").fetchone()[0] == 0
        left = [r[0] for r in c.execute("select input->>'episode_id' from brain.decision_log where stage like 'write.%'")]
        assert left and set(left) == {str(ids[0])}       # the other turn's triage row stays


def test_provider_failure_leaves_jobs_and_claims_untouched(make_client, pool):
    from brain import worker
    client = make_client(write_path_enabled=True)
    _capture(client, TURNS[:2])
    for attempt in range(1, worker.WRITE_PATH_ATTEMPTS + 1):
        res = worker.write_path(pool, Providers(FakeJev(fail=True)))
        assert res["stopped"] == "provider_error:jev"
        with pool.connection() as c:
            states = {r for r in c.execute("select status, attempts from brain.job where kind = 'write_path'")}
        want = "failed" if attempt == worker.WRITE_PATH_ATTEMPTS else "pending"
        assert states == {(want, attempt)}
    with pool.connection() as c:
        assert {r[0] for r in c.execute("select status from brain.claim_candidate")} == {"received"}


def test_the_brain_calls_no_generative_model():
    import inspect
    src = inspect.getsource(claims)
    assert "haiku" not in src.lower().replace("haiku_", "") and ".complete(" not in src


@pytest.mark.parametrize("role", ["assistant", "system", "tool"])
def test_roles_that_are_never_extracted(role):
    assert role not in claims.EXTRACT_ROLES
