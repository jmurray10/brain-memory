"""exp-relation-ends-v1, spike: on the pairs the write path ended (decision log, stage write.relate, applied
"ended"), ask Jev directly whether the older value stopped being true, 3 repeats, next to the relation Choice
asked again.

    python scripts/relate_replay.py runs/cache/relate_tail.json [last N decisions, default 1400]

Today one six-way Choice (same, updates, adds, contradicts, retracts, unrelated) decides, and "updates" ends the
older value. The candidate adds one Noul per pair in the same request: has the older value stopped being true?
Code would end a value only when the Choice says updates or retracts AND the Noul agrees. Prints every pair
with the Noul so the split can be read by hand; writes the raw answers to runs/cache (git-ignored)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))

CAP = 1.0
REPEATS = 3
ENDS_TRUE = ("The older value is no longer the case after the new statement: it was replaced by a different value, "
             "it stopped, or it was corrected (moved from Denver to Boston; two dogs, now four; was unsure, now decided).")
ENDS_FALSE = ("Both can be true: the new statement is another event or another item of the same kind (a second "
              "tournament won, a different class taken, another game played), a further detail, a later stage that "
              "does not undo the earlier one, or the same thing said again.")


def ends_question(j: int, criteria: bool = True) -> dict:
    """One Noul for one existing claim: did the new statement make the older value untrue?"""
    q = {"type": "noul", "instructions": (
        f"After `new`, has the older value in `existing[{j}]` stopped being true?")}
    if criteria:
        q["criteria"] = {"true": ENDS_TRUE, "false": ENDS_FALSE}
    return q


def ended_pairs(decisions: list[dict]) -> list[dict]:
    """Distinct (older, new) pairs that a relate decision ended, by their quotes."""
    out = {}
    for d in decisions:
        new = d["input"]["new"]
        for j, (rel, did) in enumerate(zip(d["output"]["relations"], d["output"]["applied"])):
            if did == "ended":
                old = d["input"]["existing"][j]
                out[(old["quote"], new["quote"])] = {"relation": rel, "old": old, "new": new,
                                                     "cardinality": d["output"]["cardinality"]}
    return list(out.values())


def main() -> None:
    sys.path.insert(0, str(SERVICE.parent / "live-eval"))
    from env import load_env

    load_env()
    from brain.ledger import relation_question
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter

    decisions = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    pairs = ended_pairs(decisions[-(int(sys.argv[2]) if len(sys.argv) > 2 else 1400):])
    meter = SpendMeter(MemorySpendStore(), CAP)
    jev = make_providers(load(), meter).jev
    requests = [({"new": p["new"], "existing": [p["old"]]},
                 {"r0": relation_question(0), "ends": ends_question(0), "ends_plain": ends_question(0, False)})
                for p in pairs]
    runs = [jev.ask_many(requests) for _ in range(REPEATS)]
    for i, p in enumerate(pairs):
        p["ends"] = [float(run[i].answers["ends"]["noul"]) for run in runs]
        p["ends_plain"] = [float(run[i].answers["ends_plain"]["noul"]) for run in runs]
        p["again"] = [str(run[i].answers["r0"]["choice"]) for run in runs]
    out = SERVICE / "runs" / "cache" / f"relate_replay_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(pairs), encoding="utf-8")
    for p in sorted(pairs, key=lambda p: -sum(p["ends"])):
        m = sum(p["ends"]) / REPEATS
        print(f"{m:.2f} (plain {sum(p['ends_plain']) / REPEATS:.2f}; again {'/'.join(p['again'])}) "
              f"OLD {p['old']['date']} {p['old']['value'][:55]!r}  ->  NEW {p['new']['date']} {p['new']['value'][:55]!r}")
    for name in ("ends", "ends_plain"):
        means = [sum(p[name]) / REPEATS for p in pairs]
        cross = sum(min(p[name]) < 0.7 <= max(p[name]) for p in pairs)
        print(f"\n{name}: {len(pairs)} pairs; at least 0.7: {sum(m >= 0.7 for m in means)}; under 0.3: "
              f"{sum(m < 0.3 for m in means)}; between: {sum(0.3 <= m < 0.7 for m in means)}; "
              f"cross 0.7 between repeats: {cross}")
    print(f"spent ${meter.spent_today():.3f}; raw answers: {out}")


if __name__ == "__main__":
    main()
