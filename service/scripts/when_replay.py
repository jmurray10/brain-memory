"""exp-when-locomo-v1, spike: on the evidence turns of the LoCoMo temporal questions of one run, resolve the time
phrases with the write path's date code (dates.mentions_for: Jev names the parts, code computes the range) and
print them next to the gold answer and the shipped pack's answer, so the resolved ranges can be read by hand.

    python scripts/when_replay.py runs/locomo_eval_<stamp>

Cost: one small Jev request per distinct phrase (cents). Writes nothing to a database."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))

CAP = 0.5


def main() -> None:
    sys.path.insert(0, str(SERVICE.parent / "live-eval"))
    from env import load_env

    load_env()
    from brain import dates
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from locomo_eval import session_dt

    rows = [json.loads(line) for line in (Path(sys.argv[1]) / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["cat"] == "temporal"]
    convs = {c["sample_id"]: c["conversation"] for c in
             json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8"))}
    turn = {}                                    # (conv, dia_id) -> (text, session date)
    for cid, c in convs.items():
        n = 1
        while f"session_{n}" in c:
            day = session_dt(c[f"session_{n}_date_time"]).date()
            for t in c[f"session_{n}"]:
                turn[(cid, t.get("dia_id"))] = (f"{t.get('speaker')}: {t.get('text') or ''}", day)
            n += 1
    meter = SpendMeter(MemorySpendStore(), CAP)
    providers = make_providers(load(), meter)
    todo, keys = [], []
    for r in rows:
        cid = r["qid"].split(":")[0]
        for ev in r["evidence"]:
            if (cid, ev) in turn and (cid, ev) not in keys:
                keys.append((cid, ev))
                todo.append((len(keys) - 1, *turn[(cid, ev)]))
    found = {}
    for i, m in dates.mentions_for(providers, todo):
        found.setdefault(keys[i], []).append(m)
    resolved = sum(1 for ms in found.values() for m in ms if m.date_from)
    print(f"{len(rows)} temporal questions, {len(keys)} evidence turns, "
          f"{sum(len(v) for v in found.values())} phrases, {resolved} resolved; spent ${meter.spent_today():.3f}\n")
    for r in sorted(rows, key=lambda r: (bool(r["near1"]["correct"]), r["qid"])):
        cid = r["qid"].split(":")[0]
        print(f"{'ok ' if r['near1']['correct'] else 'BAD'} {r['qid']} Q: {r['question']}\n    GOLD: {r['gold']}"
              f"\n    ANS: {str(r['near1']['answer'])[:140]!r}")
        for ev in r["evidence"]:
            text, day = turn.get((cid, ev), ("?", None))
            print(f"    {ev} said {day} ({day:%a}): {text[:170]!r}" if day else f"    {ev}: not found")
            for m in found.get((cid, ev), []):
                print(f"        {m.span!r} -> {m.date_from} to {m.date_to} ({m.granularity}, {m.confidence})")


if __name__ == "__main__":
    main()
