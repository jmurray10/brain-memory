"""exp-resolve-names-v1, free replay: what BRAIN_RESOLVE_NAMES and BRAIN_SETTLE_SPLIT would have decided on the
entity-resolution decisions the eval runs logged (runs/cache/decisions.jsonl). No provider is called.

    python scripts/resolve_replay.py            # counts
    python scripts/resolve_replay.py 30         # and 30 examples of each kind of changed decision

A logged write.resolve decision holds the mention, the candidates and their Scores; a logged nightly.settle
decision holds the mention, the entity and its Score. The rule is brain.ledger.different_names, the code the
write path runs. Each distinct (mention, quote, entity) is counted once: the same turns were ingested by many runs.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))

from brain import ledger  # noqa: E402

DECISIONS = SERVICE / "runs" / "cache" / "decisions.jsonl"


def replay(decisions) -> tuple[Counter, dict]:
    """Counts by (stage, logged outcome, what the flags would do), and the changed decisions by kind."""
    counts, changed, seen = Counter(), {}, set()
    for d in decisions:
        i, o = d.get("input") or {}, d.get("output") or {}
        if not isinstance(i.get("mention"), dict):
            continue                                            # a pronoun decision: not an alignment
        mention, quote = i["mention"].get("name") or "", i["mention"].get("said_in") or ""
        if d["stage"] == "write.resolve" and "scores" in o and i.get("candidates"):
            best = max(range(len(o["scores"])), key=lambda j: o["scores"][j])
            ent, score = i["candidates"][best], o["scores"][best]
        elif d["stage"] == "nightly.settle" and "entity" in i:
            ent, score = i["entity"], o.get("score")
        else:
            continue
        key = (d["stage"], mention, quote, ent["name"])
        if key in seen:
            continue
        seen.add(key)
        apart = ledger.different_names(mention, ent["name"], ent.get("also_called"))
        if d["stage"] == "write.resolve":
            would = "new (different name)" if d["outcome"] == "tentative" and apart else "unchanged"
        else:
            would = ("split (different name)" if d["outcome"] == "still_tentative" and apart
                     else "split (BRAIN_SETTLE_SPLIT only)" if d["outcome"] == "still_tentative" else "unchanged")
        counts[d["stage"], d["outcome"], would] += 1
        if would != "unchanged":
            changed.setdefault((d["stage"], would), []).append((score, mention, ent["name"], quote[:120]))
    return counts, changed


def _lines(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"write.resolve"' in line or '"nightly.settle"' in line:
                try:
                    yield json.loads(line)
                except ValueError:
                    pass


if __name__ == "__main__":
    show = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    counts, changed = replay(_lines(DECISIONS))
    for k in sorted(counts):
        print(f"{k[0]:<15} {k[1]:<16} {k[2]:<32} {counts[k]}")
    for k, rows in sorted(changed.items()):
        if show:
            print(f"\n== {k[0]}: {k[1]} ({len(rows)})")
            for r in rows[:show]:
                print("  ", r)
