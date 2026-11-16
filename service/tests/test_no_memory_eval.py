"""The no-memory baseline script (scripts/no_memory_eval.py): its prompt and its summary. No database, no keys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import no_memory_eval as nm  # noqa: E402


def _row(group, correct, answer="Denver", abstention=False):
    return {"qid": "q", "group": group, "abstention": abstention,
            nm.ARM: {"answer": answer, "correct": correct, "votes": []}}


def test_the_prompt_is_the_memory_arms_prompt_with_no_excerpts():
    p = nm.no_memory_prompt("Where do I live?", "2023-06-01")
    assert p == "Excerpts:\n(no excerpts)\n\nToday's date: 2023-06-01\nQuestion: Where do I live?\nAnswer:"
    assert "Today's date: unknown" in nm.no_memory_prompt("Where do I live?", None)


def test_summary_counts_scored_rows_and_reports_abstention_apart():
    rows = [_row("knowledge-update", False, "UNKNOWN - no excerpts"), _row("knowledge-update", False),
            _row("knowledge-update", True, "UNKNOWN - never mentioned", abstention=True),
            _row("temporal-reasoning", None), _row("temporal-reasoning", True)]
    s = nm.summarize(rows, "group")
    ku = s["knowledge-update"]
    assert (ku["n"], ku["scored"], ku["correct"], ku[nm.ARM], ku["said_unknown"]) == (3, 3, 1, 0.3333, 2)
    tr = s["temporal-reasoning"]
    assert (tr["n"], tr["scored"], tr[nm.ARM]) == (2, 1, 1.0)          # the unjudged row is not in the denominator
    assert (s["all"]["n"], s["all"]["scored"], s["all"]["correct"]) == (5, 4, 2)
    # an abstention question is answered correctly by saying the memory is missing: reported apart
    assert s["abstention_only"]["correct"] == 1 and s["all_without_abstention"]["correct"] == 1
    assert s["all_without_abstention"][nm.ARM] == 0.3333
    lo, hi = s["all"]["ci95"]
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert "abstention_only" not in nm.summarize([_row("single-hop", True)], "group")
    assert nm.summarize([], "group")["all"][nm.ARM] is None
