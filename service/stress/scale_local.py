"""System stress, scale (phase C): recall latency and needle recall as one namespace grows to 200,000 lines, with
1,000 small namespaces sharing the same vector index. Offline stubs (no spend); a real server and Postgres.

    python stress/scale_local.py [--max 200000]

Writes runs/stress_scale_<stamp>/report.json. Stub vectors are hashed bags of words, so absolute recall quality
says little; latency and whether planted needles survive the namespace filter at scale are what it measures.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import argparse
import concurrent.futures as cf
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_local as R  # noqa: E402  (server, database and check helpers)

WORDS = ("protein kinase ligand assay buffer titration lemma theorem manifold eigenvalue tort statute "
         "plaintiff appellate clause indemnity covenant beam load shear torque reactor yield catalyst "
         "solvent gradient spectrum isotope genome allele phenotype dosage cohort placebo variance").split()
NEEDLES = [(f"Needle {i}: the calibration constant for rig ZX{i:03d} is {1000 + i * 7}.",
            f"calibration constant rig ZX{i:03d}") for i in range(20)]


def lines(n: int, seed: int) -> list[str]:
    rnd = random.Random(seed)
    return [" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(8, 30))) + f" (note {seed}-{i})"
            for i in range(n)]


def load(h: httpx.Client, ns: str, texts: list[str], pool: cf.Executor) -> None:
    batches = [texts[i:i + 100] for i in range(0, len(texts), 100)]

    def send(b):
        r = h.post("/v1/capture", json={"namespace": ns, "turns": [{"role": "user", "text": t} for t in b]})
        r.raise_for_status()
    list(pool.map(send, batches))


def measure(h: httpx.Client, ns: str, n: int = 40) -> dict:
    """Three query kinds, timed apart: rare (the needles), mixed (common words plus one specific term, as real
    questions are) and common-only (every word appears in most lines)."""
    kinds: dict[str, list[float]] = {"rare": [], "mixed": [], "common": []}
    found = found_mixed = 0
    for i, (text, query) in enumerate(NEEDLES):
        t0 = time.perf_counter()
        r = h.post("/v1/recall", json={"namespace": ns, "query": query, "k": 10})
        kinds["rare"].append((time.perf_counter() - t0) * 1000)
        found += any(it["text"].endswith(text) for it in r.json()["items"])
        mixed = " ".join(random.sample(WORDS, 3)) + f" ZX{i:03d}"
        t0 = time.perf_counter()
        # rerank off: the offline stub reranker scores by word overlap, which favours random common-word lines
        r = h.post("/v1/recall", json={"namespace": ns, "query": mixed, "k": 10, "rerank": False})
        kinds["mixed"].append((time.perf_counter() - t0) * 1000)
        found_mixed += any(it["text"].endswith(text) for it in r.json()["items"])
    for _ in range(max(10, n - 2 * len(NEEDLES))):
        q = " ".join(random.sample(WORDS, 4))
        t0 = time.perf_counter()
        h.post("/v1/recall", json={"namespace": ns, "query": q, "k": 10}).raise_for_status()
        kinds["common"].append((time.perf_counter() - t0) * 1000)
    out = {"needles_found": found, "needles": len(NEEDLES), "needles_found_by_mixed_query": found_mixed}
    allx = sorted(x for v in kinds.values() for x in v)
    out.update({"p50_ms": round(statistics.median(allx)), "p95_ms": round(allx[int(len(allx) * 0.95) - 1]),
                "max_ms": round(allx[-1])})
    for k, v in kinds.items():
        v.sort()
        out[f"{k}_p50_ms"], out[f"{k}_p95_ms"] = round(statistics.median(v)), round(v[int(len(v) * 0.95) - 1])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=200_000)
    p.add_argument("--neighbours", type=int, default=1000)
    args = p.parse_args()
    R.RUN_DIR = R.SERVICE / "runs" / R.RUN_DIR.name.replace("stress_local", "stress_scale")
    R.RUN_DIR.mkdir(parents=True, exist_ok=True)
    R.fresh_db()
    report = {"steps": []}
    with R.Server(BRAIN_DB_POOL_MAX="10"):
        big, small = R.new_account("bigacct"), R.new_account("manyns")
        with R.client(big) as h, R.client(small) as hs, cf.ThreadPoolExecutor(8) as pool:
            t0 = time.time()
            neighbour = [f"n{i:04d}" for i in range(args.neighbours)]          # 1,000 neighbours sharing the index

            def fill(ns):
                r = hs.post("/v1/capture", json={"namespace": ns, "turns": [
                    {"role": "user", "text": t} for t in lines(100, int(ns[1:]))]})
                r.raise_for_status()
            list(pool.map(fill, neighbour))
            print(f"{args.neighbours} namespaces x 100 lines loaded in {time.time() - t0:.0f}s", flush=True)
            load(h, "big", [t for t, _ in NEEDLES], pool)          # needles first: oldest rows
            size = len(NEEDLES)
            for target in sorted({t for t in (10_000, 50_000, 100_000, args.max) if t <= args.max}):
                t1 = time.time()
                load(h, "big", lines(target - size, target), pool)
                size = target
                with psycopg.connect(R.DB_URL, prepare_threshold=None, autocommit=True) as c:
                    c.execute("analyze brain.chunk")      # what autovacuum does as the table grows
                m = measure(h, "big")
                m.update({"lines": size, "load_s": round(time.time() - t1)})
                small_m = measure(hs, "n0007", n=20)
                m["neighbour_p50_ms"], m["neighbour_p95_ms"] = small_m["p50_ms"], small_m["p95_ms"]
                report["steps"].append(m)
                print(json.dumps(m), flush=True)
    last = report["steps"][-1]
    R.check("scale", f"all needles found at {last['lines']:,} lines", last["needles_found"] == last["needles"],
            f"{last['needles_found']}/{last['needles']}")
    for kind in ("rare", "mixed"):
        R.check("scale", f"{kind}-word recall p95 under 1 s at the largest size (offline stubs, local)",
                last[f"{kind}_p95_ms"] < 1000, f"p50 {last[f'{kind}_p50_ms']} ms, p95 {last[f'{kind}_p95_ms']} ms")
    R.check("scale", "mixed queries still find the needles", last["needles_found_by_mixed_query"] >= 18,
            f"{last['needles_found_by_mixed_query']}/{last['needles']}")
    print(f"common-only queries (reported, not gated): p50 {last['common_p50_ms']} ms, p95 {last['common_p95_ms']} ms")
    R.check("scale", "a small namespace stays fast next to a huge one", last["neighbour_p95_ms"] < 500,
            f"p95 {last['neighbour_p95_ms']} ms")
    report["checks"] = R.RESULTS
    (R.RUN_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if all(r["pass"] for r in R.RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
