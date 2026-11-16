"""exp-ledger-stable-v1, step 3: ask candidate wordings of the slot-placement question on the states the probe
saved, 5 repeats each, next to today's two questions (`fits`, `which`) asked again in the same requests.

    python scripts/place_replay.py runs/cache/probe_payloads_<run>.json [N states, default 200]

Today: one Noul over the whole slot list ("is the claim the same kind of fact as one of `slots`") and a Choice
among the slots. Candidate: one Noul per slot, the slot named in the question (TypeSafe guidance: one judgment
per question, name the subject, put each item in its own question), variant A instructions only, variant B with
true/false criteria that name the confusing case (sibling slots). Code picks the highest slot and joins it at a
bar. Prints, per rule, the share of states whose action differs between repeats and the share that join a slot;
writes the raw answers to runs/cache/place_replay_<stamp>.json (git-ignored)."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_rules import place_action, unstable  # noqa: E402

CAP = 1.0
REPEATS = 5
SAME_SLOT_TRUE = ("Both record the same attribute of the same person or thing under different names (home_city and "
                  "city_of_residence), so the claim's value would replace the slot's value or stand next to it.")
SAME_SLOT_FALSE = ("They record different attributes, even closely related ones (friday_activity and "
                   "sunday_activity; commute_start and commute_destination; job_title and employer).")


def slot_question(name: str, latest, criteria: bool) -> dict:
    """One Noul for one existing slot: is the claim the same attribute as this slot?"""
    q = {"type": "noul", "instructions": (
        f"Does `claim` record the same attribute as the existing slot '{name.replace('_', ' ')}' (latest value: "
        f"'{latest or '-'}'), so that `claim.value` would replace that value or stand next to it?")}
    if criteria:
        q["criteria"] = {"true": SAME_SLOT_TRUE, "false": SAME_SLOT_FALSE}
    return q


def per_slot_action(answers: dict, names: list[str], prefix: str, bar: float, lead: float = 0.0) -> str:
    """existing:<slot> when the highest per-slot Noul reaches `bar` and leads the second by `lead`; else new."""
    ps = sorted(((float(answers[f"{prefix}{j}"]["noul"]), n) for j, n in enumerate(names)), reverse=True)
    if not ps or ps[0][0] < bar or (len(ps) > 1 and ps[0][0] - ps[1][0] < lead):
        return "new"
    return f"existing:{ps[0][1]}"


def majority(actions: list[str]) -> str:
    return Counter(actions).most_common(1)[0][0]


def main() -> None:
    sys.path.insert(0, str(SERVICE.parent / "live-eval"))
    from env import load_env                 # the repo-root .env, as the eval scripts load it; prints no values

    load_env()
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter

    payloads = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    states = [r for reqs in payloads.values() for r in reqs if "fits" in r["questions"] and "which" in r["questions"]]
    random.Random(7).shuffle(states)
    states = states[:n]
    meter = SpendMeter(MemorySpendStore(), CAP)
    jev = make_providers(load(), meter).jev
    requests = []
    for s in states:
        slots = s["state"]["slots"]
        qs = {"fits": s["questions"]["fits"], "which": s["questions"]["which"]}
        for j, sl in enumerate(slots):
            qs[f"a{j}"] = slot_question(sl["name"], sl["latest_value"], False)
            qs[f"b{j}"] = slot_question(sl["name"], sl["latest_value"], True)
        requests.append((s["state"], qs))
    runs = [jev.ask_many(requests) for _ in range(REPEATS)]
    rows = [{"state": s["state"], "answers": [run[i].answers for run in runs]} for i, s in enumerate(states)]
    out = SERVICE / "runs" / "cache" / f"place_replay_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(rows), encoding="utf-8")
    report(rows)
    print(f"\nspent ${meter.spent_today():.3f}; raw answers: {out}")


def report(rows: list[dict]) -> None:
    names = [[sl["name"] for sl in r["state"]["slots"]] for r in rows]
    print(f"{len(rows)} states, {REPEATS} repeats; slots per state: mean {sum(map(len, names)) / len(rows):.1f}\n")
    today = [[place_action(rep) for rep in r["answers"]] for r in rows]
    base = [majority(a) for a in today]

    def line(label: str, acts_: list[list[str]]) -> None:
        maj = [majority(a) for a in acts_]
        join = sum(m != "new" for m in maj) / len(maj)
        agree = sum(m == b for m, b in zip(maj, base)) / len(maj)
        print(f"  {label:34} unstable {unstable(acts_):.3f}   joins {join:.3f}   same action as today {agree:.3f}")

    line("today: fits >= 0.5, pick >= 0.5", today)
    line("today's questions at 0.7 / 0.7", [[place_action(rep, 0.7, 0.7) for rep in r["answers"]] for r in rows])
    for prefix, label in (("a", "per slot, instructions only"), ("b", "per slot, with criteria")):
        print(f"\n{label}")
        tops = [max((float(rep[f"{prefix}{j}"]["noul"]) for j in range(len(ns))), default=0.0)
                for r, ns in zip(rows, names) for rep in r["answers"]]
        mid = sum(0.3 <= t < 0.7 for t in tops) / len(tops)
        print(f"  highest slot probability between 0.3 and 0.7: {mid:.3f} of answers")
        for bar, lead in ((0.5, 0.0), (0.6, 0.0), (0.7, 0.0), (0.8, 0.0), (0.7, 0.1), (0.7, 0.2)):
            line(f"top >= {bar}" + (f", lead >= {lead}" if lead else ""),
                 [[per_slot_action(rep, ns, prefix, bar, lead) for rep in r["answers"]] for r, ns in zip(rows, names)])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    else:
        main()
