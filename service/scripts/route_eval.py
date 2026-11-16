"""Build step 2 gate: route accuracy of brain/router.py on LongMemEval S question types (DESIGN.md metrics: route
accuracy against question_type, scoring only, never a feature).

Splits (deterministic, see `splits`): dev = 20% of each type's questions outside the 100-question sample (80), used
to tune the router's wording; test = everything else (420, including the whole 100-question sample), scored once. The gold route per question type is fixed in GOLD before any test run (the experiment log, exp-route-v1).

    python scripts/route_eval.py dev            # paid (about $0.002), tune on this
    python scripts/route_eval.py test --go      # paid (about $0.01), the gate; run once per router version
    python scripts/route_eval.py dev --dry      # baselines only, no Jev calls
    python scripts/route_eval.py rescore RUN    # score a saved run again (labels, threshold), no calls

Scoring (exp-route-v2 on): per-question route labels in evals/route_labels_v1.jsonl, written by a separate
labeler from the question text and DESIGN.md's route table only (no ids, types or router output): a route is
correct if it is the label's route or in its also_ok list. The question-type mapping (GOLD) is reported too.

Writes runs/route_<split>_<stamp>/ (config with hashes, rows.jsonl, summary.json).
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
LIVE_EVAL = SERVICE.parent / "live-eval"
sys.path.insert(0, str(SERVICE))
from brain import router  # noqa: E402

S_FILE = LIVE_EVAL / "data" / "longmemeval_s_cleaned.json"
LABELS = SERVICE / "evals" / "route_labels_v1.jsonl"
LABEL_KEY = SERVICE / "evals" / "route_label_key.json"
SAMPLE = LIVE_EVAL / "samples" / "lme_s_20261106_100.txt"
DEV_SHARE = 0.20

# Gold routes per LongMemEval question type (DESIGN.md read-path table). single-session-user asks for a fact the
# user stated once: the current route (a slot value on an entity) or the open route both read a store that holds it.
# Amendment 1 (2026-11-08, after dev round 1, before any test call; the owner's choice): for knowledge-update and
# multi-session the type cannot tell the claim routes apart (current, history and count all read claims: "did I
# increase or decrease X" is history, "how many do I currently own" is current), so any claim route is correct.
# Which claim route is right is measured end to end at build steps 5-6, when those stores exist.
CLAIM_ROUTES = {"current", "history", "count"}
GOLD = {
    "knowledge-update": CLAIM_ROUTES,
    "temporal-reasoning": {"when"},
    "multi-session": CLAIM_ROUTES,
    "single-session-preference": {"preference"},
    "single-session-assistant": {"assistant"},
    "single-session-user": {"current", "open"},
}

# Pre-registered gate (exp-route-v1), on the test split.
GATE = {"accuracy_min": 0.85, "confident_error_max": 0.08, "per_type_min": 0.70, "beats_keyword": True}

_KW = [
    ("assistant", re.compile(r"\byou (said|told|recommended|suggested|mentioned|gave|provided|listed|shared)\b|"
                             r"\bremind me (of )?(what|the|which)\b.*\byou\b|\bour (previous|earlier|last) "
                             r"(conversation|chat)\b", re.I)),
    ("preference", re.compile(r"\b(recommend|suggest|any (tips|advice|ideas)|what should i|ideas for|"
                              r"can you (give|offer|share) me|help me (choose|pick|decide|plan))\b", re.I)),
    ("when", re.compile(r"\b(when|how long|how many (days|weeks|months|years)|ago|first|earlier|later|"
                        r"before or after|which .* (first|last))\b", re.I)),
    ("current", re.compile(r"\b(now|currently|current|still|these days|at the moment)\b", re.I)),
    ("count", re.compile(r"\b(how many|how much|total|in total|altogether)\b", re.I)),
]


def keyword_route(question: str) -> str:
    """Baseline: first matching keyword rule, else open."""
    for name, rx in _KW:
        if rx.search(question):
            return name
    return "open"


def load_questions() -> list[dict]:
    rows = json.loads(S_FILE.read_text(encoding="utf-8"))
    return [{"qid": r["question_id"], "qtype": r["question_type"], "question": r["question"],
             "question_date": r["question_date"], "is_abs": r["question_id"].endswith("_abs")} for r in rows]


def splits(qs: list[dict]) -> tuple[list[dict], list[dict]]:
    sample = {ln.strip() for ln in SAMPLE.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    dev = []
    for t in sorted({q["qtype"] for q in qs}):
        pool = sorted((q for q in qs if q["qtype"] == t and q["qid"] not in sample),
                      key=lambda q: hashlib.sha256(q["qid"].encode()).hexdigest())
        dev += pool[: round(len(pool) * DEV_SHARE)]
    dev_ids = {q["qid"] for q in dev}
    test = [q for q in qs if q["qid"] not in dev_ids]
    for q in test:
        q["in_sample"] = q["qid"] in sample
    return dev, test


def load_labels() -> dict[str, set[str]] | None:
    if not LABELS.exists():
        return None
    key = json.loads(LABEL_KEY.read_text())
    out = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[key[r["n"]]] = {r["route"], *r.get("also_ok", [])}
    return out


def score(rows: list[dict], labels: dict | None, threshold: float | None = None) -> None:
    """Set correct (labels when present, else GOLD), type_correct, kw_correct, and hedge at `threshold`."""
    for r in rows:
        ok = labels[r["qid"]] if labels else GOLD[r["qtype"]]
        r["accept"] = sorted(ok)
        r["kw_correct"] = r["kw_route"] in ok
        if r.get("route"):
            r["correct"] = r["route"] in ok
            r["type_correct"] = r["route"] in GOLD[r["qtype"]]
            if threshold is not None:
                r["hedge"] = r["route"] != "open" and r["confidence"] < threshold


def summarize(rows: list[dict]) -> dict:
    n = len(rows)

    def acc(rs, key="correct"):
        return round(sum(r[key] for r in rs) / len(rs), 4) if rs else None
    by_type = {}
    for t in sorted({r["qtype"] for r in rows}):
        rs = [r for r in rows if r["qtype"] == t]
        routed = [r for r in rs if r.get("route")]
        by_type[t] = {"n": len(rs), "router": acc(routed), "keyword": acc(rs, "kw_correct"),
                      "routes": dict(collections.Counter(r["route"] for r in routed).most_common()),
                      "keyword_routes": dict(collections.Counter(r["kw_route"] for r in rs).most_common())}
    have = [r for r in rows if r.get("route")]
    s = {"n": n, "router_accuracy": acc(have), "keyword_accuracy": acc(rows, "kw_correct"),
         "majority_accuracy": None, "by_type": by_type,
         "type_proxy_accuracy": acc([r for r in rows if r.get("route")], "type_correct") if any(
             r.get("route") for r in rows) else None}
    if have:
        wrong = [r for r in have if not r["correct"]]
        s.update({
            "hedged": round(sum(r["hedge"] for r in have) / len(have), 4),
            "confident_errors": round(sum(not r["hedge"] for r in wrong) / len(have), 4),
            "hedged_errors": sum(r["hedge"] for r in wrong), "errors": len(wrong),
            "accuracy_when_confident": acc([r for r in have if not r["hedge"]]),
            "sample_accuracy": acc([r for r in have if r.get("in_sample")]),
        })
    counts = collections.Counter(a for r in rows for a in r["accept"])
    top = counts.most_common(1)[0][0]
    s["majority_accuracy"] = round(sum(top in r["accept"] for r in rows) / n, 4)
    if have:   # accuracy by confidence band: [0, 0.5), [0.5, 0.6), ..., [0.9, 1.0]
        edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        s["calibration"] = {}
        for lo, hi in zip(edges, edges[1:]):
            band = [r for r in have if lo <= r["confidence"] < hi]
            if band:
                s["calibration"][f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {"n": len(band), "accuracy": acc(band)}
    return s


def gate(s: dict) -> dict:
    per_type_ok = all(v["router"] is not None and v["router"] >= GATE["per_type_min"] for v in s["by_type"].values())
    checks = {"accuracy": s["router_accuracy"] is not None and s["router_accuracy"] >= GATE["accuracy_min"],
              "confident_errors": s.get("confident_errors", 1) <= GATE["confident_error_max"],
              "per_type": per_type_ok,
              "beats_keyword": (s["router_accuracy"] or 0) > (s["keyword_accuracy"] or 0)}
    return {"checks": checks, "pass": all(checks.values()), "bar": GATE}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=["dev", "test", "rescore"])
    p.add_argument("run", nargs="?", help="rescore: the run directory")
    p.add_argument("--threshold", type=float, help="rescore: hedge threshold to apply")
    p.add_argument("--go", action="store_true", help="required for the test split (the gate run)")
    p.add_argument("--dry", action="store_true", help="baselines only; no Jev calls")
    p.add_argument("--cap", type=float, default=1.0)
    args = p.parse_args()
    if args.split == "test" and not (args.go or args.dry):
        sys.exit("the test split is the gate: pass --go to run it")

    labels = load_labels()
    if args.split == "rescore":
        run = Path(args.run)
        rows = [json.loads(ln) for ln in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines() if ln]
        score(rows, labels, args.threshold)
        s = summarize(rows)
        if "test" in run.name:
            s["gate"] = gate(s)
        name = f"summary_labels{'_t' + str(args.threshold) if args.threshold is not None else ''}.json"
        (run / name).write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(json.dumps(s, indent=2))
        return 0
    dev, test = splits(load_questions())
    qs = dev if args.split == "dev" else test
    print(f"{args.split}: {len(qs)} questions; scoring against {'labels' if labels else 'question types'}")
    rows = [{**q, "kw_route": keyword_route(q["question"])} for q in qs]

    spend = None
    if not args.dry:
        sys.path.insert(0, str(LIVE_EVAL))
        import env
        env.load_env()
        from brain.providers import JevClient
        from brain.settings import load
        from brain.spend import MemorySpendStore, SpendMeter
        s = load()
        if not s.jev_key:
            sys.exit("no Jev key")
        meter = SpendMeter(MemorySpendStore(), args.cap)
        jev = JevClient(meter, s.jev_key, s.jev_model)
        results = jev.ask_many([(router.state(r["question"], r["question_date"]), router.ROUTER_QUESTIONS)
                                for r in rows])
        for r, res in zip(rows, results):
            d = router.decide(res)
            r.update({"route": d.route, "confidence": round(d.confidence, 4), "p_route": round(d.p_route, 4),
                      "weakest": d.weakest, "probabilities": {k: round(v, 4) for k, v in d.probabilities.items()},
                      "nouls": {k: round(v, 4) for k, v in d.nouls.items()}, "hedge": d.hedge, "model": d.model})
        spend = meter.by_day(1)

    score(rows, labels)
    summary = summarize(rows)
    if args.split == "test" and not args.dry:
        summary["gate"] = gate(summary)
    summary["spend"] = spend
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SERVICE / "runs" / f"route_{args.split}{'_dry' if args.dry else ''}_{stamp}"
    out.mkdir(parents=True)
    files = [SERVICE / "brain" / "router.py", Path(__file__).resolve(), SAMPLE] + ([LABELS] if labels else [])
    (out / "config.json").write_text(json.dumps({
        "split": args.split, "n": len(rows), "router": router.VERSION, "hedge_threshold": router.ROUTE_HEDGE,
        "scored_against": "labels" if labels else "question types", "gold": {k: sorted(v) for k, v in GOLD.items()},
        "gate": GATE, "hashes": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files},
        "data_sha256": hashlib.sha256(S_FILE.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)
    return 0 if args.split == "dev" or summary.get("gate", {}).get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
