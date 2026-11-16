"""exp-judge-dates-v2: judge again the stored answers of one LoCoMo run on the questions whose gold answer is a
phrase before a date: the model picks the date out of each answer and code checks it against the dates the gold
phrase means (locomo_eval.dated_verdict); an answer with no date keeps its stored verdict. Nothing in the run
folder is changed: the new verdicts go to <run>/rejudge_dates_v2.json.

    python scripts/locomo_rejudge.py runs/locomo_eval_<stamp>

Prints, per arm, the old and new count on those questions and every verdict that changed. Result: better than
the stored judge, not good enough to be the yardstick (the experiment log)."""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

CAP = 1.0


def code_check(pred: str, lo: date, hi: date):
    """True/False when the answer's ISO dates are all inside / not all inside the gold dates; None with no ISO date."""
    found = []
    for y, m, d in re.findall(r"(\d{4})-(\d{2})-(\d{2})", pred):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    return all(lo <= x <= hi for x in found) if found else None


def main() -> None:
    from env import load_env

    load_env()
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from locomo_eval import DATE_PICK_SYSTEM, date_pick_user, dated_verdict, gold_dates

    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    arms = [k for k, v in rows[0].items() if isinstance(v, dict) and "votes" in v]
    meter = SpendMeter(MemorySpendStore(), CAP)
    haiku = make_providers(load(), meter).haiku
    memo: dict[str, tuple] = {}

    def judge(q, gold, pred, stored):
        if pred not in memo:
            picks = [haiku.complete(DATE_PICK_SYSTEM, date_pick_user(q, pred), 30).text for _ in range(3)]
            v = dated_verdict(gold, picks)
            memo[pred] = (stored if v is None else v, picks if v is not None else ["kept"] + picks)
        return memo[pred]

    out, old, new, agree, checked = {}, dict.fromkeys(arms, 0), dict.fromkeys(arms, 0), 0, 0
    dated = [r for r in rows if gold_dates(r["gold"])]
    for r in dated:
        lo, hi = gold_dates(r["gold"])
        memo.clear()
        out[r["qid"]] = {"gold": r["gold"], "dates": [lo.isoformat(), hi.isoformat()]}
        for a in arms:
            pred = str(r[a]["answer"])
            ok, votes = judge(r["question"], r["gold"], pred, r[a]["correct"])
            code = code_check(pred, lo, hi)
            out[r["qid"]][a] = {"was": r[a]["correct"], "now": ok, "votes": votes, "code": code}
            old[a] += bool(r[a]["correct"])
            new[a] += bool(ok)
            if code is not None:
                checked += 1
                agree += (code == bool(ok))
            if bool(ok) != bool(r[a]["correct"]):
                print(f"{'now right' if ok else 'now WRONG'} {r['qid']} {a} | gold {r['gold']} = {lo} to {hi} | "
                      f"picked {votes} | {pred[:170]!r}".encode("ascii", "replace").decode())
    (run / "rejudge_dates_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{len(dated)} questions with a gold phrase before a date")
    for a in arms:
        total = sum(bool(r[a]["correct"]) for r in rows)
        print(f"{a}: on these {old[a]} -> {new[a]}; all {len(rows)}: {total} -> {total - old[a] + new[a]}")
    kept = sum(1 for q in out.values() for a in arms if q[a]["votes"][:1] == ["kept"])
    print(f"{kept} of {len(dated) * len(arms)} answers held no date and kept their stored verdict; "
          f"spent ${meter.spent_today():.3f}")


if __name__ == "__main__":
    main()
