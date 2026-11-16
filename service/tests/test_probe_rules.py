"""The offline rule replay of exp-ledger-stable-v1 (scripts/probe_rules.py). No database, no keys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_rules import acts, cardinality_action, place_action, relation_action, unstable  # noqa: E402


def _choice(pick, **probs):
    return {"choice": pick, "probabilities": probs}


def test_a_relation_pick_under_the_bar_ends_nothing():
    weak = _choice("updates", updates=0.45, adds=0.40, unrelated=0.15)
    assert relation_action(weak) == "end"                 # today's rule: the pick alone
    assert relation_action(weak, bar=0.5) == "none"
    assert relation_action(_choice("updates", updates=0.8, adds=0.2), bar=0.7) == "end"
    assert relation_action(_choice("contradicts", contradicts=0.9), bar=0.7) == "contest"
    assert relation_action(_choice("same", same=0.4, adds=0.35), bar=0.7) == "same"    # a supports link is not gated


def test_options_with_the_same_effect_are_summed_under_the_mass_rule():
    split = _choice("adds", adds=0.36, updates=0.34, retracts=0.30)     # the pick says add; 0.64 says the value ended
    assert relation_action(split) == "none"
    assert relation_action(split, mass=True, bar=0.6) == "end"
    assert relation_action(split, mass=True, bar=0.7) == "none"


def test_placement_joins_only_when_both_bars_are_met():
    rep = {"fits": {"noul": 0.62}, "which": _choice("home_city", home_city=0.55, pet=0.45)}
    assert place_action(rep) == "existing:home_city"
    assert place_action(rep, fit_at=0.7) == "new"
    assert place_action(rep, pick_at=0.6) == "new"


def test_an_unsure_cardinality_falls_back_to_many():
    assert cardinality_action(_choice("one", one=0.45, many=0.40, event=0.15), bar=0.5) == "many"
    assert cardinality_action(_choice("one", one=0.45, many=0.40, event=0.15)) == "one"


def test_unstable_counts_requests_whose_repeats_disagree_and_acts_counts_answers():
    per_request = [["end", "end", "none"], ["none", "none", "none"], ["end", "end", "end"]]
    assert round(unstable(per_request), 3) == 0.333
    assert round(acts(per_request, ("none", "same")), 3) == round(5 / 9, 3)
