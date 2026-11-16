"""Booster shadow evaluation at R2 (route): the experiment log exp-booster-route-shadow.

Uses Jev's saved router-v2 decisions on the 1,081 blind-labeled questions and asks the booster the same
questions on its Hugging Face endpoint. Reports booster alone, coverage and accuracy on kept, the cascade and Jev
alone, per set and overall, plus latency. HF_TOKEN from .env (never printed).

    python scripts/booster_shadow.py            # waits for the endpoint to be running first
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
from brain.booster import BoosterClient  # noqa: E402

ENV = Path(r".env")
ENDPOINT = "https://api.endpoints.huggingface.cloud/v2/endpoint/your-account/brain-booster"
RUNS = {"lme_dev": "route_dev_20261108T185357Z", "lme_test": "route_test_20261108T185635Z",
        "locomo": "route_ext_locomo_20261108T202955Z", "domain": "route_ext_domain_20261108T203142Z"}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def token() -> str:
    for line in ENV.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("HF_TOKEN not in .env")


def iso_date(s: str) -> str:
    s = s or ""
    m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", s)
    if m:
        return "-".join(m.groups())
    m = re.search(r"(\d{1,2}) (\w+),? (\d{4})", s)          # LoCoMo: "9:55 am on 22 October, 2023"
    if m and m.group(2).lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    return s


def wait_running(tok: str, minutes: int = 25) -> str:
    h = {"Authorization": f"Bearer {tok}"}
    end = time.time() + minutes * 60
    while time.time() < end:
        st = httpx.get(ENDPOINT, headers=h, timeout=30).json().get("status", {})
        if st.get("state") == "running":
            return st["url"]
        if st.get("state") in ("failed", "updateFailed"):
            sys.exit(f"endpoint state {st.get('state')}: {str(st.get('message'))[:300]}")
        time.sleep(15)
    sys.exit("endpoint not running in time")


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))]) if xs else None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    kept = [r for r in rows if not r["b_deferred"]]
    cascade = [r["b_correct"] if not r["b_deferred"] else r["j_correct"] for r in rows]
    return {"n": n, "jev": round(sum(r["j_correct"] for r in rows) / n, 4),
            "booster_alone": round(sum(r["b_correct"] for r in rows) / n, 4),
            "coverage": round(len(kept) / n, 4),
            "booster_on_kept": round(sum(r["b_correct"] for r in kept) / len(kept), 4) if kept else None,
            "jev_on_same_kept": round(sum(r["j_correct"] for r in kept) / len(kept), 4) if kept else None,
            "cascade": round(sum(cascade) / n, 4),
            "agreement": round(sum(r["b_route"] == r["j_route"] for r in rows) / n, 4),
            "invalid_or_failed": sum(r["b_error"] is not None for r in rows),
            "latency_kept_p50_ms": pct([r["b_ms"] for r in kept], 0.5),
            "latency_kept_p95_ms": pct([r["b_ms"] for r in kept], 0.95)}


def main() -> int:
    tok = token()
    url = wait_running(tok)
    b = BoosterClient.from_hub(url, tok, timeout_s=30.0)
    rows = []
    for name, run in RUNS.items():
        for ln in (SERVICE / "runs" / run / "rows.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            rows.append({"set": name, "question": r["question"], "question_date": iso_date(r.get("question_date")),
                         "accept": r["accept"], "j_route": r["route"], "j_correct": r["route"] in r["accept"]})
    b.route("warm-up question", "2026-01-01")                      # first call after scale-up is slow
    t0 = time.time()

    def one(r):
        a = b.route(r["question"], r["question_date"])
        route = (a.answer or {}).get("route")
        return {**r, "b_route": route, "b_conf": round(a.confidence, 4), "b_deferred": a.deferred,
                "b_correct": route in r["accept"], "b_ms": round(a.latency_ms), "b_error": a.error,
                "b_answer": a.answer}
    with cf.ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, rows))
    summary = {"overall": summarize(rows), "threshold": b.threshold("route"), "minutes": round((time.time() - t0) / 60, 1)}
    for name in ("lme", "locomo", "domain"):
        summary[name] = summarize([r for r in rows if r["set"].startswith(name)])
    o = summary["overall"]
    per_set_ok = all(summary[s]["cascade"] >= summary[s]["jev"] - 0.02 for s in ("lme", "locomo", "domain"))
    summary["promotion"] = {"cascade>=jev overall": o["cascade"] >= o["jev"], "no set more than 0.02 below": per_set_ok}
    summary["promote"] = all(summary["promotion"].values())   # latency check is reported, judged in the write-up
    out = SERVICE / "runs" / f"booster_shadow_route_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)
    b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
