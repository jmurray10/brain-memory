"""exp-worlds-shipped-v1: the complex-domain worlds in the layout locomo_eval.py reads (scripts/worlds_set.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import locomo_eval  # noqa: E402
import worlds_set  # noqa: E402

WORLD = {"world": "w", "domain": "law", "sessions": [
    {"session_id": "w-s01", "date": "2026-05-05", "turns": [
        {"tid": "w-s01-t01", "role": "user", "channel": "chat", "speaker": "Dana", "text": "The cap is $2 million."},
        {"tid": "w-s01-t02", "role": "attachment", "channel": "email", "speaker": "Owen", "text": "Cap moved to $5 million."},
        {"tid": "w-s01-t03", "role": "assistant", "channel": "chat", "text": "Noted."}]}],
    "questions": [
        {"qid": "w-q1", "question": "What is the cap now?", "answer": "$5 million", "type": "current",
         "evidence": ["w-s01-t02"], "question_date": "2026-10-01"},
        {"qid": "w-q2", "question": "Who is the judge?", "answer": "Not mentioned", "type": "abstention", "evidence": []}]}


def test_a_world_becomes_a_conversation_the_eval_can_read():
    c = worlds_set.convert(WORLD)
    conv = c["conversation"]
    assert c["sample_id"] == "world-w" and (conv["speaker_a"], conv["speaker_b"]) == ("Dana", "Owen")
    assert locomo_eval.session_dt(conv["session_1_date_time"]).date().isoformat() == "2026-05-05"
    assert [t["dia_id"] for t in conv["session_1"]] == ["D1:1", "D1:2", "D1:3"]
    assert conv["session_1"][0]["text"] == "The cap is $2 million."                 # the user's chat: as written
    assert conv["session_1"][1]["text"].startswith("[attachment, email] ")           # anything else says what it is
    assert conv["session_1"][2] == {"speaker": "assistant", "dia_id": "D1:3", "text": "[assistant, chat] Noted."}
    q1, q2 = c["qa"]
    assert locomo_eval.category(q1, True) == "current" and locomo_eval.gold_turns(q1["evidence"]) == ["D1:2"]
    assert q2["answer"] == worlds_set.NEVER_SAID and q2["evidence"] == []


def test_every_world_converts_with_its_evidence():
    for p in sorted(worlds_set.WORLDS.glob("*.json")):
        w = json.loads(p.read_text(encoding="utf-8"))
        c = worlds_set.convert(w)
        said = {t["dia_id"] for k, v in c["conversation"].items() if isinstance(v, list) for t in v}
        assert len(c["qa"]) == len(w["questions"]) and all(set(q["evidence"]) <= said for q in c["qa"])
