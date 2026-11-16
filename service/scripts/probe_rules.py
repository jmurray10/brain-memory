"""exp-ledger-stable-v1, step 2 (free): replay candidate decision rules over the probe's saved payloads.

    python scripts/probe_rules.py runs/cache/probe_payloads_<run>.json

For every sampled request the probe kept its 5 repeated raw answers. A rule turns one answer into an ACTION
(join slot X or open a new slot; end a value, contest it, or neither). A request is unstable under a rule when
its repeats do not all give the same action. Printed per rule: the unstable share, and the share of answers on
which the rule acts at all (so a rule that is stable only because it never acts is seen for what it is)."""
import json
import re
import sys
from pathlib import Path

ENDS, CONTESTS = ("updates", "retracts"), ("contradicts",)


def relation_action(answer: dict, bar: float = 0.0, mass: bool = False) -> str:
    """end | contest | same | none. `bar` is on the pick's probability; with `mass`, on the summed probability
    of the options that lead to the same action (updates + retracts both end a value)."""
    probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
    pick = str(answer["choice"])
    if mass and probs:
        groups = {"end": sum(probs.get(o, 0) for o in ENDS), "contest": sum(probs.get(o, 0) for o in CONTESTS),
                  "same": probs.get("same", 0), "none": probs.get("adds", 0) + probs.get("unrelated", 0)}
        act = max(groups, key=lambda g: groups[g])
        return act if act in ("same", "none") or groups[act] >= bar else "none"
    act = "end" if pick in ENDS else "contest" if pick in CONTESTS else "same" if pick == "same" else "none"
    return act if act in ("same", "none") or probs.get(pick, 1.0) >= bar else "none"


def place_action(answers: dict, fit_at: float = 0.5, pick_at: float = 0.5) -> str:
    """existing:<slot> | new, by the write path's rule (ledger._place) with the two bars given."""
    which = answers["which"]
    p = float((which.get("probabilities") or {}).get(str(which["choice"]), 0.0))
    return f"existing:{which['choice']}" if float(answers["fits"]["noul"]) >= fit_at and p >= pick_at else "new"


def cardinality_action(answer: dict, bar: float = 0.0) -> str:
    p = float((answer.get("probabilities") or {}).get(str(answer["choice"]), 1.0))
    return str(answer["choice"]) if p >= bar else "many"


def unstable(actions_per_request: list[list[str]]) -> float:
    return sum(len(set(a)) > 1 for a in actions_per_request) / len(actions_per_request) if actions_per_request else 0.0


def acts(actions_per_request: list[list[str]], idle: tuple) -> float:
    flat = [a for acts_ in actions_per_request for a in acts_]
    return sum(a not in idle for a in flat) / len(flat) if flat else 0.0


def main() -> None:
    payloads = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    relation, place, card = [], [], []
    for reqs in payloads.values():
        for req in reqs:
            names = list(req["questions"])
            for n in names:
                if re.fullmatch(r"r\d+", n) and req["questions"][n].get("type") == "choice" \
                        and "relate" in req["questions"][n].get("instructions", ""):
                    relation.append([rep[n] for rep in req["answers"]])
            if "fits" in names and "which" in names:
                place.append(req["answers"])
            if "cardinality" in names:
                card.append([rep["cardinality"] for rep in req["answers"]])
    print(f"relation questions {len(relation)}, placement requests {len(place)}, cardinality {len(card)}\n")
    print("Relation (acts = ends or contests a value)")
    for label, kw in [("today (pick, no bar)", {}), ("pick >= 0.5", {"bar": 0.5}), ("pick >= 0.6", {"bar": 0.6}),
                      ("pick >= 0.7", {"bar": 0.7}), ("group mass, no bar", {"mass": True}),
                      ("group mass >= 0.5", {"mass": True, "bar": 0.5}),
                      ("group mass >= 0.6", {"mass": True, "bar": 0.6}),
                      ("group mass >= 0.7", {"mass": True, "bar": 0.7})]:
        a = [[relation_action(x, **kw) for x in reps] for reps in relation]
        print(f"  {label:22} unstable {unstable(a):.3f}   acts {acts(a, ('same', 'none')):.3f}")
    print("\nPlacement (acts = joins an existing slot)")
    for f, p in [(0.5, 0.5), (0.6, 0.5), (0.6, 0.6), (0.7, 0.5), (0.7, 0.7), (0.5, 0.7), (0.8, 0.7)]:
        a = [[place_action(rep, f, p) for rep in reps] for reps in place]
        tag = " (today)" if (f, p) == (0.5, 0.5) else ""
        print(f"  fits >= {f}, pick >= {p}{tag:9} unstable {unstable(a):.3f}   acts {acts(a, ('new',)):.3f}")
    print("\nCardinality of a new slot (acts = one or event)")
    for bar in (0.0, 0.5, 0.6, 0.7):
        a = [[cardinality_action(x, bar) for x in reps] for reps in card]
        print(f"  pick >= {bar:<4}{' (today)' if bar == 0 else '':9} unstable {unstable(a):.3f}   acts {acts(a, ('many',)):.3f}")


if __name__ == "__main__":
    main()
