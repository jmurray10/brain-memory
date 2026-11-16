"""exp-date-counted-v1, the direct measure: the date resolver alone, with and without BRAIN_DATE_COUNTED, on real
sentences that count days, weeks or months ("two weeks ago", "in 3 days", "a few months ago", "a week later").

    python scripts/date_counted_replay.py 300        # the first 300 of the shuffled sentences (seed 7)
    python scripts/date_counted_replay.py 300 300    # the next 300

Only brain.dates runs (one Jev request per phrase; the two arms share Jev's answers, the rule is code): no
capture, no database. The sentences are the user turns of the LongMemEval conversations that
date_direction_replay.py reads. Writes runs/date_counted_<stamp>/rows.jsonl with both resolutions, the count
code read and Jev's own picks, for reading by hand, and prints the counts. Cap $1.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")
SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

from brain import dates  # noqa: E402

CAP = 1.0
_UNIT = re.compile(r"\b(?:days?|weeks?|months?)\b", re.I)


def counting(text: str) -> list[str]:
    """The date phrases of a turn that count days, weeks or months (whether or not the rule reads them)."""
    return [p for p in dates.find_phrases(text) if _UNIT.search(p) and not re.match(r"(?:last|this|next|past|coming)\b", p, re.I)]


def main():
    import env
    from brain.providers import make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from date_direction_replay import compare, lme_turns

    limit = int(sys.argv[1])
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seen, items = set(), []
    for text, anchor in lme_turns():
        for phrase in counting(text):
            key = (phrase.lower(), dates.sentence_of(text, phrase), anchor)
            if key not in seen:
                seen.add(key)
                items.append((text, anchor, phrase))
    random.Random(7).shuffle(items)
    items = items[offset:offset + limit]
    print(f"{len(seen)} distinct counting phrases found; resolving {len(items)} both ways", flush=True)
    env.load_env()
    meter = SpendMeter(MemorySpendStore(), CAP)
    prov = make_providers(load(), meter)
    direction = dates.DATE_DIRECTION_OVERRIDE.set(True)        # as in production

    def run(counted: bool) -> dict:
        token = dates.DATE_COUNTED_OVERRIDE.set(counted)
        try:
            got = dates.mentions_for(prov, [(i, text, anchor) for i, (text, anchor, _) in enumerate(items)])
        finally:
            dates.DATE_COUNTED_OVERRIDE.reset(token)
        return {(i, m.span.lower()): m for i, m in got}

    off, on = run(False), run(True)
    dates.DATE_DIRECTION_OVERRIDE.reset(direction)
    rows = []
    for i, (text, anchor, phrase) in enumerate(items):
        a, b = off.get((i, phrase.lower())), on.get((i, phrase.lower()))
        sent = dates.sentence_of(text, phrase)
        parts = dates._CACHE.get((phrase, sent, f"{anchor.isoformat()} ({dates.WEEKDAYS[anchor.weekday()]})", True)) or {}
        rows.append({"phrase": phrase, "sentence": sent, "written": anchor.isoformat(),
                     "off": [str(a.date_from), str(a.date_to)] if a and a.date_from else None,
                     "on": [str(b.date_from), str(b.date_to)] if b and b.date_from else None,
                     "code_count": dates.counted_offset(phrase), "mode": list(parts.get("mode", ("", 0))),
                     "jev_offsets": {k: list(parts.get(k, ("", 0))) for k in ("offset_days", "week_offset", "month_offset")}})
    counts = compare([r["off"] for r in rows], [r["on"] for r in rows])
    out = SERVICE / "runs" / f"date_counted_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"offset": offset, "found": len(seen), "resolved": len(items), "counts": dict(counts),
               "code_reads_a_count": sum(r["code_count"] is not None for r in rows),
               "spend_usd": round(meter.spent_today(), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)


if __name__ == "__main__":
    main()
