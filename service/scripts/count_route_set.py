"""exp-gather-top-v1: LoCoMo's questions that the router sent to the count route, as a set locomo_eval.py reads.

    python scripts/count_route_set.py runs/locomo_eval_<stamp>      # writes evalsets/locomo_count_route_v1.json

The conversations are LoCoMo's, whole; each keeps only the questions whose saved row in the named run (its
near1seats arm) has route "count". No model, no cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
OUT = SERVICE / "evalsets" / "locomo_count_route_v1.json"
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def routed_count(rows: list[dict], arm: str = "near1seats") -> set[str]:
    out = set()
    for r in rows:
        route = r[arm].get("route")
        if (route.get("route") if isinstance(route, dict) else route) == "count":
            out.add(r["qid"])
    return out


def build(convs: list[dict], qids: set[str]) -> list[dict]:
    out = []
    for c in convs:
        qa = [{**q, "category": CATS[q["category"]]} for k, q in enumerate(c["qa"])
              if f"{c['sample_id']}:{k}" in qids and q.get("category") in CATS]
        if qa:
            out.append({"sample_id": c["sample_id"], "conversation": c["conversation"], "qa": qa})
    return out


def main():
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    out = build(json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8")), routed_count(rows))
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT.name}: {len(out)} conversations, {sum(len(c['qa']) for c in out)} questions (from {run.name})")


if __name__ == "__main__":
    main()
