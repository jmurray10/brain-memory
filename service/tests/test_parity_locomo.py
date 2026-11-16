"""scripts/parity_locomo.py: the pieces that need no server, database or keys."""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import parity_locomo as pk  # noqa: E402


def _q(qid="conv-1:0", ev_s=("session_1", "session_2"), ev_t=("D1:1", "D2:1")):
    turns = [NS(tid="D1:1", speaker="Ann", text="I adopted a dog."), NS(tid="D1:2", speaker="Bo", text=" ")]
    hay = [NS(sid="session_1", date="1:56 pm on 8 May, 2023", turns=turns),
           NS(sid="session_2", date="9:05 am on 2 June, 2023", turns=[NS(tid="D2:1", speaker="Bo", text="Me too.")])]
    return NS(qid=qid, haystack=hay, evidence_session_ids=list(ev_s), evidence_turn_ids=list(ev_t))


def test_locomo_dates_turns_and_grouping():
    assert pk.session_dt("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    turns, empty = pk.conv_turns(_q())
    assert empty == 1 and [t["session_id"] for t in turns] == ["session_1", "session_2"]
    assert turns[0]["turn"] == {"role": "user", "speaker": "Ann", "text": "I adopted a dog.",
                                "ts": "2023-05-08T13:56:00+00:00", "meta": {"tid": "D1:1"},
                                "idempotency_key": "D1:1"}
    groups = pk.by_conversation([_q("conv-1:0"), _q("conv-2:0"), _q("conv-1:1")])
    assert {k: [q.qid for q in v] for k, v in groups.items()} == {"conv-1": ["conv-1:0", "conv-1:1"],
                                                                 "conv-2": ["conv-2:0"]}
    assert pk.estimate(groups)["usd"] < 0.01


def test_score_uses_the_iso_arms_metrics():
    item = lambda cid, eid, sid: {"chunk_id": cid, "episode_id": eid, "session_id": sid, "text": "x"}
    tid_of = {1: "D1:1", 2: "D1:2", 3: "D2:1"}
    both = pk.score(_q(), [item(10, 1, "session_1"), item(11, 3, "session_2")], tid_of)
    assert (both["recall_any"], both["recall_all"], both["turn_recall"], both["kept"]) == (True, True, 1.0, 2)
    one = pk.score(_q(), [item(10, 2, "session_1")], tid_of)        # right session, wrong turn
    assert (one["recall_any"], one["recall_all"], one["turn_recall"]) == (True, False, 0.0)
    none = pk.score(_q(ev_s=(), ev_t=()), [item(10, 1, "session_1")], tid_of)
    assert none["recall_all"] is None and none["turn_recall"] is None


def test_reference_reads_the_iso_rows_of_the_same_questions(tmp_path):
    import json
    rows = [{"arm": "iso", "qid": "c:0", "recall_all": True, "turn_recall": 1.0, "err": None},
            {"arm": "iso", "qid": "c:1", "recall_all": False, "turn_recall": 0.5, "err": None},
            {"arm": "oracle", "qid": "c:0", "recall_all": True, "turn_recall": 1.0, "err": None},
            {"arm": "iso", "qid": "c:9", "recall_all": True, "turn_recall": 1.0, "err": None}]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert pk.reference(tmp_path, {"c:0", "c:1"}) == {"recall_all": 0.5, "turn_recall": 0.75}
    import pytest
    with pytest.raises(SystemExit):
        pk.reference(tmp_path, {"c:0", "c:2"})
