"""Re-grade an existing run's answers with the current judge (majority vote), without re-answering.
Writes rows_rejudged.jsonl and summary_rejudged.{json,md} next to the originals.
  python3 rejudge.py runs/lme100_v1 --sample samples/lme_s_20261106_100.txt --cap 0.5
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from clients import HaikuClient
from env import load_env
from prices import SpendMeter
from run import judge, load_lme_subset, read_ids, write_summary
from data import load_locomo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--sample")
    ap.add_argument("--cap", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_env()
    rd = Path(args.run_dir)
    meta = json.loads((rd / "config.json").read_text())
    rows = [json.loads(l) for l in (rd / "rows.jsonl").read_text().splitlines() if l.strip()]
    latest = {}
    for r in rows:
        latest[(r["arm"], r["qid"])] = r
    if meta["bench"] == "lme":
        qs = {q.qid: q for q in load_lme_subset(read_ids(args.sample))}
    else:
        qs = {q.qid: q for q in load_locomo()}
    meter = SpendMeter(args.cap)
    llm = HaikuClient(meter, config.JUDGE_MODEL)

    def work(r):
        r = dict(r)
        if r.get("err"):
            return r
        v, raw = judge(qs[r["qid"]], r.get("pred", ""), llm)
        r["judge_single"] = r.get("correct")
        r["correct"], r["parse_fail"], r["judge_raw"] = v is True, v is None, raw
        return r

    with ThreadPoolExecutor(args.workers) as ex:
        out = list(ex.map(work, latest.values()))
    sub = rd / "rejudged"
    sub.mkdir(exist_ok=True)
    shutil.copy(rd / "config.json", sub / "config.json")
    (sub / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    snap = meter.snapshot()
    prev = json.loads((rd / "spend.json").read_text()) if (rd / "spend.json").exists() else {}
    snap["note"] = f"judge-only spend; original run spend ${prev.get('usd', 0):.4f}"
    snap["stopped"] = meter.stopped
    write_summary(sub, snap, False)


if __name__ == "__main__":
    main()
