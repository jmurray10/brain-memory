"""exp-gather-top-v1: the count-route questions as a set (scripts/count_route_set.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import count_route_set  # noqa: E402


def test_only_the_questions_routed_to_count_are_kept():
    rows = [{"qid": "c1:0", "near1seats": {"route": "count"}}, {"qid": "c1:1", "near1seats": {"route": "open"}},
            {"qid": "c2:0", "near1seats": {"route": {"route": "count"}}}, {"qid": "c2:1", "near1seats": {"route": None}}]
    convs = [{"sample_id": c, "conversation": {"speaker_a": "A"}, "qa": [
        {"question": "q0", "category": 1, "evidence": ["D1:1"]}, {"question": "q1", "category": 4, "evidence": ["D1:2"]}]}
        for c in ("c1", "c2", "c3")]
    out = count_route_set.build(convs, count_route_set.routed_count(rows))
    assert [(c["sample_id"], [q["question"] for q in c["qa"]]) for c in out] == [("c1", ["q0"]), ("c2", ["q0"])]
    assert out[0]["qa"][0]["category"] == "multi-hop" and out[0]["conversation"] == {"speaker_a": "A"}
