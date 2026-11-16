"""exp-ledger-set-v1: a ledger-set run by case.

    python scripts/ledger_report.py runs/locomo_eval_<stamp> [runs/reader_swap_<reader>_<stamp> ...]

For each case: questions, and per arm the answers the judge of record marked right; beside them the code check
(every `must` value is in the answer; cases with no `must` are left out of it), how many questions had all
their evidence turns in the pack, and how many had a ledger fact in it. A reader_swap directory adds that
reader's answers on the same packs (arms named `<arm>@<reader>`). No model, no cost.

Then the ledger's own state against the script (the run saves every claim with its status): for each question
that names values a ledger should hold as current (`now`) and no longer as current (`old`),
  followed  every `now` value is in a current claim and no `old` value is
  stale     an `old` value is still in a current claim
  missing   a `now` value is in no current claim
  absent    a value of the script, current or old, is in no claim at all (it never reached the ledger)
A claim is current when its status is active or contested and it has no end date. For `contradiction`, `flagged`
counts the questions where both values' claims are marked contested.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CHANGE = ("chain", "as-of", "correction", "contradiction", "ended", "buried", "count")
ORDER = CHANGE + ("ongoing", "stable")
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def holds(must: list[str], answer: str) -> bool:
    """Every literal value is in the answer; a bare count also as its word ("3" or "three")."""
    for v in must:
        pats = [rf"(?<![\w$.]){re.escape(v)}(?![\w%]|[.,]\d)"]
        if v.isdigit() and int(v) < len(WORDS):
            pats.append(rf"\b{WORDS[int(v)]}\b")
        if not any(re.search(p, answer or "", re.I) for p in pats):
            return False
    return True


def load(d: Path) -> dict[str, dict]:
    return {r["qid"]: r for r in map(json.loads, (d / "rows.jsonl").read_text(encoding="utf-8").splitlines()) if r}


def table(rows: list[dict], arms: list[str]) -> list[dict]:
    out = []
    for case in list(ORDER) + ["change cases", "all"]:
        rs = [r for r in rows if r["cat"] == case or case == "all" or (case == "change cases" and r["cat"] in CHANGE)]
        if not rs:
            continue
        line = {"case": case, "n": len(rs)}
        for a in arms:
            got = [r[a] for r in rs if a in r]
            line[a] = sum(bool(g.get("correct")) for g in got)
            with_must = [(r, r[a]) for r in rs if a in r and r.get("must")]
            line[f"{a} must"] = f"{sum(holds(r['must'], g.get('answer')) for r, g in with_must)}/{len(with_must)}" if with_must else "-"
            if "@" not in a:
                line[f"{a} evidence"] = sum(bool(r["evidence"]) and set(r["evidence"]) <= set(r[a].get("evidence_reached") or []) for r in rs)
                line[f"{a} ledger"] = sum(((r[a].get("ledger") or {}).get("facts") or 0) > 0 for r in rs)
        out.append(line)
    return out


def _holding(v: str, claims: list[dict]) -> list[dict]:
    """The claims that are about value `v`: those naming it in their entity or value. A quote is not enough when
    such claims exist (a turn that states a dose also yields "has not missed any doses", whose quote holds the
    dose); only when no claim names the value is the quote read, so a paraphrased value is still found."""
    named = [c for c in claims if holds([v], f"{c.get('entity') or ''} | {c.get('value') or ''}")]
    return named or [c for c in claims if holds([v], c.get("quote") or "")]


def _current(c: dict) -> bool:
    return c.get("status") in ("active", "contested") and not c.get("valid_to")


def audit(rows: list[dict], key: str = "ledger_state") -> list[dict]:
    """Per case: how the ledger stands against the values the script says are current and no longer current."""
    state = {}
    for r in rows:
        if key in r:
            state[r["qid"].split(":")[0]] = r[key]
    out = []
    for case in list(CHANGE) + ["all with values"]:
        rs = [r for r in rows if (r["cat"] == case or case == "all with values") and (r.get("now") or r.get("old"))
              and r["qid"].split(":")[0] in state]
        if not rs:
            continue
        line = {"case": case, "n": len(rs), "followed": 0, "stale": 0, "missing": 0}
        for r in rs:
            claims = state[r["qid"].split(":")[0]]
            stale = any(_current(c) for v in r.get("old") or [] for c in _holding(v, claims))
            missing = any(not any(_current(c) for c in _holding(v, claims)) for v in r.get("now") or [])
            line["absent"] = line.get("absent", 0) + any(not _holding(v, claims) for v in (r.get("now") or []) + (r.get("old") or []))
            line["stale"] += stale
            line["missing"] += missing
            line["followed"] += not stale and not missing
            if r["cat"] == "contradiction":
                both = all(any(c.get("status") == "contested" for c in _holding(v, claims)) for v in r["now"])
                line["flagged"] = line.get("flagged", 0) + both
        out.append(line)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = load(Path(sys.argv[1]))
    arms = [k for k, v in next(iter(rows.values())).items() if isinstance(v, dict) and "correct" in v]
    for d in sys.argv[2:]:
        for qid, r in load(Path(d)).items():
            for k, v in r.items():
                if "@" in k:
                    rows[qid][k] = v
                    if k not in arms:
                        arms.append(k)
    lines = table(list(rows.values()), arms)
    cols = list(lines[0].keys())
    print(" | ".join(cols))
    for line in lines:
        print(" | ".join(str(line.get(c, "")) for c in cols))
    checked = audit(list(rows.values()))
    if checked:
        claims = [c for r in rows.values() for c in r.get("ledger_state") or []]
        print(f"\nledger state: {len(claims)} claims, " + ", ".join(
            f"{sum(c['status'] == s for c in claims)} {s}" for s in ("active", "superseded", "contested", "retracted"))
            + f", {sum(bool(c['valid_to']) for c in claims)} with an end date")
        print("case | n | followed | stale | missing | absent | flagged")
        for line in checked:
            print(" | ".join(str(line.get(c, "")) for c in ("case", "n", "followed", "stale", "missing", "absent", "flagged")))


if __name__ == "__main__":
    main()
