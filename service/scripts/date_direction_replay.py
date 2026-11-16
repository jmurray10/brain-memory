"""exp-date-direction-v1, the direct measure: the date resolver alone, with and without BRAIN_DATE_DIRECTION, on
real sentences that name a calendar date or month with no year.

    python scripts/date_direction_replay.py 600        # the first 600 of the shuffled sentences (seed 7)
    python scripts/date_direction_replay.py 600 600    # the next 600: sentences the rule was not written on

Only brain.dates runs (one Jev request per phrase and arm): no capture, no database. Each user turn of the
benchmark's conversations is searched for date phrases (dates.find_phrases); those that name a month and write
no year are resolved twice, against the turn's own date. Writes runs/date_direction_<stamp>/rows.jsonl
with both resolutions and the direction Jev gave, for reading by hand, and prints the counts. Cap $1.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")
SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

from brain import dates  # noqa: E402

CAP = 1.0


def yearless(text: str) -> list[str]:
    """The date phrases of a turn that name a month and write no year."""
    month = dates.re.compile(dates._MO, dates.re.I)
    return [p for p in dates.find_phrases(text) if month.search(p) and not dates._YEAR_WRITTEN.search(p)]


def compare(off: list, on: list) -> Counter:
    """Counts over paired resolutions: (from, to) tuples or None."""
    c = Counter()
    for a, b in zip(off, on):
        c["same" if a == b else "changed" if a and b else "resolved only without" if a else "resolved only with"] += 1
    return c


def lme_turns():
    import brain_eval
    for q in brain_eval.pick("test", 50, ["knowledge-update", "multi-session", "temporal-reasoning"], 35):
        for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
            for t in sess or []:
                if isinstance(t, dict) and t.get("role") == "user" and t.get("content"):
                    yield t["content"], brain_eval.session_ts(date_s).date()


def main():
    import env
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter

    limit = int(sys.argv[1])
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seen, items = set(), []
    for text, anchor in lme_turns():
        for phrase in yearless(text):
            key = (phrase.lower(), dates.sentence_of(text, phrase), anchor)
            if key not in seen:
                seen.add(key)
                items.append((text, anchor, phrase))
    random.Random(7).shuffle(items)
    items = items[offset:offset + limit]
    print(f"{len(seen)} distinct yearless phrases found; resolving {len(items)} both ways", flush=True)
    env.load_env()
    meter = SpendMeter(MemorySpendStore(), CAP)
    prov = make_providers(load(), meter)

    def run(direction: bool) -> dict:
        token = dates.DATE_DIRECTION_OVERRIDE.set(direction)
        try:
            got = dates.mentions_for(prov, [(i, text, anchor) for i, (text, anchor, _) in enumerate(items)])
        finally:
            dates.DATE_DIRECTION_OVERRIDE.reset(token)
        return {(i, m.span.lower()): m for i, m in got}

    off, on = run(False), run(True)
    rows = []
    for i, (text, anchor, phrase) in enumerate(items):
        a, b = off.get((i, phrase.lower())), on.get((i, phrase.lower()))
        sent = dates.sentence_of(text, phrase)
        parts = dates._CACHE.get((phrase, sent, f"{anchor.isoformat()} ({dates.WEEKDAYS[anchor.weekday()]})", True)) or {}
        rows.append({"phrase": phrase, "sentence": sent, "written": anchor.isoformat(),
                     "off": [str(a.date_from), str(a.date_to)] if a and a.date_from else None,
                     "on": [str(b.date_from), str(b.date_to)] if b and b.date_from else None,
                     "direction": list(parts.get("direction", ("", 0))), "year_choice": parts.get("year", ("", 0))[0],
                     "mode": parts.get("mode", ("", 0))[0]})
    counts = compare([r["off"] for r in rows], [r["on"] for r in rows])
    out = SERVICE / "runs" / f"date_direction_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"offset": offset, "found": len(seen), "resolved": len(items), "counts": dict(counts),
               "direction": dict(Counter(r["direction"][0] for r in rows)), "spend_usd": round(meter.spent_today(), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)


if __name__ == "__main__":
    main()
