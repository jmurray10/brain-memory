"""scripts/ledger_set.py: the scripted part of the ledger question set (no model, no keys)."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ledger_set as L  # noqa: E402


def test_the_same_arguments_give_the_same_script_and_every_case_is_asked():
    a, b = L.build("contract", "dev", 1), L.build("contract", "dev", 1)
    assert a == b
    assert L.build("contract", "held", 1)["points"] != a["points"]
    for s in L.scripts():
        cats = [q["category"] for q in s["qa"]]
        assert sorted(set(cats)) == sorted(L.CASES) and len(cats) == 13
        ids = {p["id"] for p in s["points"]}
        assert all(set(q["points"]) <= ids and q["points"] for q in s["qa"])
        assert all(0 <= p["session"] < L.SESSIONS for p in s["points"])


def test_gold_answers_follow_from_the_timeline():
    for s in L.scripts():
        at = {p["id"]: p for p in s["points"]}
        days = [date.fromisoformat(d) for d in s["dates"]]
        by_track = {}
        for q in s["qa"]:
            by_track.setdefault(q["track"], {})[q["category"]] = q
        for q in s["qa"]:
            pts = [at[p] for p in q["points"]]
            if q["category"] == "chain":
                last = pts[0]
                assert q["answer"] == last["need"][0]
                # nothing later in the script states this track again: the chain's own earlier values are its `may`
                asof = by_track[q["track"]]["as-of"]
                held = at[asof["points"][0]]
                assert held["session"] < last["session"] and asof["answer"] == held["need"][0] != q["answer"]
                d = next(date(int(y), m, int(dd)) for dd, mname, y in [asof["question"].rstrip("?").split()[-3:]]
                         for m in range(1, 13) if date(2000, m, 1).strftime("%B") == mname)
                assert days[held["session"]] < d < days[last["session"]]
            if q["category"] == "correction":
                first, fix = pts
                assert fix["after"] == first["id"] and fix["session"] >= first["session"]
                assert q["answer"] == fix["need"][0] and first["need"][0] != q["answer"]
            if q["category"] == "contradiction":
                assert pts[0]["speaker"] != pts[1]["speaker"] and all(v in q["answer"] for v in q["must"]) and len(q["must"]) == 2
            if q["category"] == "ended":
                assert pts[1]["session"] >= pts[0]["session"] + 3 and q["answer"].startswith("No")
            if q["category"] == "ongoing":
                assert len(pts) == 1 and q["answer"].startswith("Yes")
                assert sum(x["category"] == "ended" for x in s["qa"]) == 1      # the other of the two did end
            if q["category"] == "count":
                added = [p["need"][0] for p in pts[:5]]
                removed = [p["need"][0] for p in pts[5:]]
                assert set(removed) <= set(added) and q["must"] == [str(len(added) - len(removed))] == ["3"]
                for p in pts[5:]:                                   # an item goes only after it came
                    assert p["session"] > next(x["session"] for x in pts[:5] if x["need"] == p["need"])


def test_a_value_counts_as_said_only_as_a_whole_token():
    assert L.said("$45", "It's $45 now.") and not L.said("$45", "about $450") and not L.said("2.4", "version 12.45")
    assert L.said("30 days", "net 30 days, I think") and not L.said("7.2%", "it was 17.2% up")
    assert L.said("Dr. Yoon", "seeing dr. yoon again") and not L.said("Leeds", "Leedsman")


def test_a_written_session_is_checked_against_its_brief():
    s = L.build("shop", "dev", 1)
    n = next(p["session"] for p in s["points"] if p["need"])
    pts = [p for p in s["points"] if p["session"] == n]
    a, b = s["speakers"]
    good = [{"speaker": p["speaker"], "text": f"By the way, {' and '.join(p['need']) or 'that thing'} it is.", "point": p["id"]} for p in pts]
    good += [{"speaker": (a, b)[k % 2], "text": "The kettle is on.", "point": None} for k in range(14 - len(good))]
    assert L.check(s, n, good) == []
    assert "session" in L.brief(s, n) and all(p["id"] in L.brief(s, n) for p in pts)
    missing = [dict(t, point=None) if t["point"] == pts[0]["id"] else t for t in good]
    assert any("exactly one" in x for x in L.check(s, n, missing))
    other = next(v for v in s["tracked"] if all(v not in p["need"] + p["may"] for p in pts))
    leak = good[:-1] + [{"speaker": a, "text": f"Remember {other}?", "point": None}]
    assert any("tracked subject" in x for x in L.check(s, n, leak))
    # the next turn may repeat what was just said (an acknowledgement); a later turn may not
    val = next(p for p in pts if p["need"] and not p["quiet"])
    k = next(i for i, t in enumerate(good) if t["point"] == val["id"])
    echo = good[:k + 1] + [{"speaker": b, "text": f"Right, {val['need'][0]} it is.", "point": None}] + [t for t in good[k + 1:] if not t["point"]]
    marked = [t for t in good[k + 1:] if t["point"]]
    assert L.check(s, n, echo + marked) == []
    assert any("tracked subject" in x for x in L.check(s, n, echo + marked + [{"speaker": b, "text": f"So, {val['need'][0]} again.", "point": None}]))
    wrong = [dict(t, speaker=b if t["speaker"] == a else a) if t["point"] == pts[0]["id"] else t for t in good]
    assert any("must be said by" in x for x in L.check(s, n, wrong))


def test_the_set_is_written_in_locomo_layout():
    import locomo_eval
    s = L.build("clinic", "dev", 1)
    a, b = s["speakers"]
    sessions = []
    for n in range(L.SESSIONS):
        pts = [p for p in s["points"] if p["session"] == n]
        sessions.append([{"speaker": a, "text": "Hello.", "point": None}]
                        + [{"speaker": p["speaker"], "text": " ".join(p["need"]) or "ok", "point": p["id"]} for p in pts])
    c = L.to_locomo(s, sessions)
    assert c["conversation"]["speaker_a"] == a and len(c["qa"]) == 13
    assert locomo_eval.session_dt(c["conversation"]["session_1_date_time"]).date().isoformat() == s["dates"][0]
    turns = {t["dia_id"] for n in range(1, L.SESSIONS + 1) for t in c["conversation"][f"session_{n}"]}
    assert all(q["evidence"] and set(q["evidence"]) <= turns for q in c["qa"])
    assert locomo_eval.gold_turns(c["qa"][0]["evidence"]) == sorted(c["qa"][0]["evidence"])


def test_the_locomo_eval_reads_a_named_category_only_from_a_data_set():
    import locomo_eval
    assert locomo_eval.category({"category": 2}) == "temporal" and locomo_eval.category({"category": 5}) is None
    assert locomo_eval.category({"category": "chain"}) is None           # LoCoMo itself has no named categories
    assert locomo_eval.category({"category": "chain"}, named=True) == "chain"
    assert locomo_eval.category({"category": 2}, named=True) == "temporal" and locomo_eval.category({}, named=True) is None


def test_the_report_counts_by_case_and_checks_the_literal_values():
    import ledger_report as R
    assert R.holds(["3"], "There are three open.") and R.holds(["3"], "3 (a, b, c)") and not R.holds(["3"], "13 of them")
    assert R.holds(["Anselm", "Marisol"], "Disputed: Anselm or Marisol") and not R.holds(["Anselm", "Marisol"], "Marisol")
    rows = [{"cat": "chain", "must": ["Oslo"], "evidence": ["D1:1"],
             "flat": {"correct": False, "answer": "Bergen", "evidence_reached": [], "ledger": {"facts": 0}},
             "near1seats": {"correct": True, "answer": "Oslo now", "evidence_reached": ["D1:1"], "ledger": {"facts": 2}},
             "near1seats@sonnet": {"correct": True, "answer": "Oslo"}},
            {"cat": "ended", "must": [], "evidence": ["D2:1", "D3:1"],
             "flat": {"correct": True, "answer": "No", "evidence_reached": ["D3:1"], "ledger": {}},
             "near1seats": {"correct": True, "answer": "No", "evidence_reached": ["D2:1", "D3:1"], "ledger": {"facts": 1}}},
            {"cat": "stable", "must": ["Ingrid"], "evidence": ["D4:1"],
             "flat": {"correct": True, "answer": "Ingrid", "evidence_reached": ["D4:1"], "ledger": {}},
             "near1seats": {"correct": True, "answer": "Ingrid", "evidence_reached": ["D4:1"], "ledger": {}}}]
    t = {x["case"]: x for x in R.table(rows, ["flat", "near1seats", "near1seats@sonnet"])}
    assert t["chain"]["flat"] == 0 and t["chain"]["near1seats"] == 1 and t["chain"]["near1seats must"] == "1/1"
    assert t["ended"]["flat must"] == "-" and t["ended"]["flat evidence"] == 0 and t["ended"]["near1seats evidence"] == 1
    assert t["change cases"]["n"] == 2 and t["change cases"]["near1seats"] == 2 and t["change cases"]["near1seats ledger"] == 2
    assert t["all"]["n"] == 3 and t["all"]["near1seats@sonnet"] == 1 and "near1seats@sonnet evidence" not in t["all"]


def test_the_report_reads_the_ledgers_state_against_the_script():
    import ledger_report as R
    state = [{"value": "works at Oslo Freight", "quote": "I'm at Oslo Freight now", "status": "active", "valid_to": None},
             {"value": "works at Bergen Labs", "quote": "I work at Bergen Labs", "status": "superseded", "valid_to": "2025-03-01"},
             {"value": "likes the Oslo office", "quote": "better than Bergen Labs was", "status": "active", "valid_to": None},
             {"entity": "Henrik", "value": "owns billing", "quote": "he owns billing now", "status": "active", "valid_to": None},
             {"value": "drives a Mazda 3", "quote": "my Mazda 3", "status": "active", "valid_to": None},
             {"value": "drives a Kia Niro", "quote": "the Kia Niro", "status": "active", "valid_to": None},
             {"value": "cap is $2 million", "quote": "cap is $2 million", "status": "contested", "valid_to": None},
             {"value": "cap is $5 million", "quote": "it's $5 million", "status": "contested", "valid_to": None}]
    rows = [{"qid": "c:0", "cat": "chain", "now": ["Oslo Freight"], "old": ["Bergen Labs"], "ledger_state": state},
            {"qid": "c:1", "cat": "correction", "now": ["Mazda 3"], "old": ["Kia Niro"]},          # the wrong car is still current
            {"qid": "c:2", "cat": "buried", "now": ["Tromso"], "old": []},                          # never reached the ledger
            {"qid": "c:3", "cat": "contradiction", "now": ["$2 million", "$5 million"], "old": []},
            {"qid": "c:4", "cat": "stable", "now": [], "old": []}]
    t = {x["case"]: x for x in R.audit(rows)}
    # a current claim that only quotes the old value is not the old value's claim; an entity's name counts
    assert (t["chain"]["followed"], t["chain"]["stale"], t["chain"]["missing"]) == (1, 0, 0)
    assert [c["value"] for c in R._holding("Henrik", state)] == ["owns billing"] and t["buried"]["absent"] == 1
    assert (t["correction"]["followed"], t["correction"]["stale"]) == (0, 1)
    assert (t["buried"]["followed"], t["buried"]["missing"]) == (0, 1)
    assert t["contradiction"]["followed"] == 1 and t["contradiction"]["flagged"] == 1
    assert t["all with values"]["n"] == 4 and t["all with values"]["followed"] == 2
