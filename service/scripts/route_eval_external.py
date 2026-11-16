"""Router v2 on question sets beyond LongMemEval (the experiment log exp-route-domains): LoCoMo10 sample and the
complex-domain set. Scored against blind per-question labels; no tuning on these sets.

    python scripts/route_eval_external.py locomo      # evals/route_locomo_sample.jsonl + route_labels_locomo.jsonl
    python scripts/route_eval_external.py domain      # evals/route_domain_v1.jsonl + route_labels_domain.jsonl

Paid: one Jev request per question (about $0.0001 each). Writes runs/route_ext_<set>_<stamp>/.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
EVALS = SERVICE / "evals"
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain import router  # noqa: E402
from route_eval import keyword_route  # noqa: E402

SETS = {
    "locomo": {"questions": "route_locomo_sample.jsonl", "labels": "route_labels_locomo.jsonl",
               "key": "route_label_key_locomo.json", "id": "qid", "group": "qtype"},
    "domain": {"questions": "route_domain_v1.jsonl", "labels": "route_labels_domain.jsonl",
               "key": "route_label_key_domain.json", "id": "id", "group": "domain"},
}
GATE = {"accuracy_min": 0.85, "confident_error_max": 0.08, "per_route_min": 0.70, "per_route_min_n": 15}


def load(name: str) -> list[dict]:
    cfg = SETS[name]
    qs = [json.loads(ln) for ln in (EVALS / cfg["questions"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    key = json.loads((EVALS / cfg["key"]).read_text())
    labels = {}
    for ln in (EVALS / cfg["labels"]).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            labels[key[r["n"]]] = (r["route"], {r["route"], *r.get("also_ok", [])})
    rows = []
    for q in qs:
        primary, ok = labels[q[cfg["id"]]]
        rows.append({**q, "id": q[cfg["id"]], "group": q[cfg["group"]], "label": primary, "accept": sorted(ok),
                     "kw_route": keyword_route(q["question"])})
    return rows


def acc(rs, key="correct"):
    return round(sum(r[key] for r in rs) / len(rs), 4) if rs else None


def summarize(rows: list[dict]) -> dict:
    s = {"n": len(rows), "router_accuracy": acc(rows), "keyword_accuracy": acc(rows, "kw_correct"),
         "confident_errors": round(sum((not r["correct"]) and not r["hedge"] for r in rows) / len(rows), 4),
         "hedged": round(sum(r["hedge"] for r in rows) / len(rows), 4), "errors": sum(not r["correct"] for r in rows)}
    s["by_label_route"] = {t: {"n": len(rs), "router": acc(rs)} for t in router.ROUTES
                           for rs in [[r for r in rows if r["label"] == t]] if rs}
    s["by_group"] = {g: {"n": len(rs), "router": acc(rs)} for g in sorted({r["group"] for r in rows})
                     for rs in [[r for r in rows if r["group"] == g]]}
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    s["calibration"] = {f"{lo:.1f}-{min(hi, 1.0):.1f}": {"n": len(b), "accuracy": acc(b)}
                        for lo, hi in zip(edges, edges[1:]) for b in [[r for r in rows if lo <= r["confidence"] < hi]] if b}
    if rows and "intended_route" in rows[0]:
        s["writer_vs_labeler_agreement"] = round(sum(r["intended_route"] in r["accept"] for r in rows) / len(rows), 4)
        s["router_vs_writer"] = round(sum(r["route"] in {r["intended_route"], *r.get("also_ok", [])}
                                          for r in rows) / len(rows), 4)
        ns = [r for r in rows if r.get("never_said")]
        s["never_said"] = {"n": len(ns), "router": acc(ns)}
    per_route_ok = all(v["router"] >= GATE["per_route_min"] for v in s["by_label_route"].values()
                       if v["n"] >= GATE["per_route_min_n"])
    s["gate"] = {"checks": {"accuracy": s["router_accuracy"] >= GATE["accuracy_min"],
                            "confident_errors": s["confident_errors"] <= GATE["confident_error_max"],
                            "per_route": per_route_ok,
                            "beats_keyword": s["router_accuracy"] > s["keyword_accuracy"]}, "bar": GATE}
    s["gate"]["pass"] = all(s["gate"]["checks"].values())
    return s


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SETS:
        sys.exit(__doc__)
    rows = load(name)
    import env
    env.load_env()
    from brain.providers import JevClient
    from brain.settings import load as load_settings
    from brain.spend import MemorySpendStore, SpendMeter
    st = load_settings()
    meter = SpendMeter(MemorySpendStore(), 1.0)
    jev = JevClient(meter, st.jev_key, st.jev_model)
    results = jev.ask_many([(router.state(r["question"], r.get("question_date")), router.ROUTER_QUESTIONS)
                            for r in rows])
    for r, res in zip(rows, results):
        d = router.decide(res)
        r.update({"route": d.route, "confidence": round(d.confidence, 4), "p_route": round(d.p_route, 4),
                  "weakest": d.weakest, "nouls": {k: round(v, 4) for k, v in d.nouls.items()}, "hedge": d.hedge,
                  "correct": d.route in r["accept"], "kw_correct": r["kw_route"] in r["accept"]})
    s = summarize(rows)
    s["spend"] = meter.by_day(1)
    s["hashes"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in
                   [SERVICE / "brain" / "router.py", Path(__file__).resolve(), EVALS / SETS[name]["questions"],
                    EVALS / SETS[name]["labels"]]}
    out = SERVICE / "runs" / f"route_ext_{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps({k: s[k] for k in s if k not in ("by_group",)}, indent=2))
    print("run dir:", out)
    return 0 if s["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
