"""exp-over-time-v1: the same early questions over a growing memory (scripts/over_time_set.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import locomo_eval  # noqa: E402
import over_time_set  # noqa: E402


def _sample(cid, n):
    conv = {"speaker_a": "A", "speaker_b": "B"}
    for s in range(1, n + 1):
        conv[f"session_{s}"] = [{"speaker": "A", "dia_id": f"D{s}:1", "text": f"{cid} said in session {s}"}]
        conv[f"session_{s}_date_time"] = f"1:00 pm on {s} May, 2023"
    qa = [{"question": f"q{s}", "answer": "x", "evidence": [f"D{s}:1"], "category": 4} for s in range(1, n + 1)]
    qa.append({"question": "both", "answer": "x", "evidence": ["D1:1", f"D{n}:1"], "category": 1})
    qa.append({"question": "trap", "answer": "x", "evidence": ["D1:1"], "category": 5})
    return {"sample_id": cid, "conversation": conv, "qa": qa}


def test_stages_grow_and_ask_the_same_early_questions(monkeypatch):
    monkeypatch.setattr(over_time_set, "CONVS", (("a", "b"),))
    out = over_time_set.build([_sample("a", 8), _sample("b", 3)])
    assert [c["sample_id"] for c in out] == ["a@1", "a@2", "a@3", "a@4"]
    assert [over_time_set.sessions(c["conversation"]) for c in out] == [2, 4, 8, 11]
    asked = [[q["question"] for q in c["qa"]] for c in out]
    assert sorted(asked[0]) == ["q1", "q2"] and all(a == asked[0] for a in asked)   # evidence in the first quarter only
    assert {q["question_date"] for c in out for q in c["qa"]} == {"2023-05-08"}      # the last session's date, always
    added = out[3]["conversation"]["session_9"]
    assert added[0]["dia_id"] == "E1:1" and added[0]["text"] == "b said in session 1"
    assert locomo_eval.gold_turns([t["dia_id"] for t in added]) == []                # never counted as evidence
    assert locomo_eval.category(out[0]["qa"][0], True) == "single-hop"
