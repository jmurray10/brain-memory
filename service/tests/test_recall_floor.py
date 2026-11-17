"""scripts/recall_floor.py (exp-recall-floor-v1): the table, the dev pick and the held-out gate. No keys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def q(*items):
    return {"items": [{"rerank": s, "evidence": e} for s, e in items]}


def test_the_table_counts_evidence_and_other_passages_below_each_threshold():
    from recall_floor import floor_table
    rows = [q((0.96, True), (0.10, False), (0.09, False)),      # evidence on top, two weak others
            q((0.12, True), (0.40, False)),                     # this question's only evidence is weak
            q((0.50, False))]                                   # no evidence returned at all
    t = {r["t"]: r for r in floor_table(rows, grid=(0.05, 0.15))}
    assert (t[0.05]["evidence_below"], t[0.05]["other_below"]) == (0, 0)
    assert (t[0.15]["evidence_below"], t[0.15]["evidence"], t[0.15]["evidence_share"]) == (1, 2, 0.5)
    assert (t[0.15]["other_below"], t[0.15]["other"], t[0.15]["other_share"]) == (2, 4, 0.5)
    # only questions that were handed evidence can lose it
    assert (t[0.15]["questions_all_evidence_below"], t[0.15]["questions_with_evidence"]) == (1, 2)


def test_the_table_of_no_rows_has_no_shares():
    from recall_floor import floor_table, pick
    t = floor_table([], grid=(0.1,))
    assert t[0]["evidence_share"] is None and t[0]["other_share"] is None and pick(t) is None


def test_dev_picks_the_largest_threshold_that_spares_the_evidence():
    from recall_floor import pick
    table = [{"t": 0.05, "evidence_share": 0.0}, {"t": 0.10, "evidence_share": 0.02},
             {"t": 0.15, "evidence_share": 0.021}]
    assert pick(table) == 0.10
    assert pick([{"t": 0.05, "evidence_share": 0.5}]) is None      # no threshold spares it: a null result


def test_the_held_out_gate_needs_both_few_evidence_and_many_others():
    from recall_floor import gate
    row = {"t": 0.10, "evidence_share": 0.03, "other_share": 0.40}
    assert gate([row], 0.10)["passed"] is True
    assert gate([{**row, "evidence_share": 0.031}], 0.10)["passed"] is False
    assert gate([{**row, "other_share": 0.39}], 0.10)["passed"] is False
