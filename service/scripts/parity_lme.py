"""M5 parity check (REFERENCE.md): the service's recall against the live-eval iso arm on LongMemEval S.

For each sampled question, its whole haystack is captured through the running API (python -m brain.serve) into a
throwaway namespace (one turn per episode, session id = LongMemEval session id, ts = session date), then the question
is recalled. The iso arm kept the top 10 of the 30 Jev-reranked candidates within a 6,000-token budget (live-eval
run.py: pack(ranked, KEEP=10, ISO_BUDGET_TOKENS=6000)); here recall returns all 30 (k=30) and the same pack is applied.
Metrics are the iso arm's, on non-abstention questions: recall_all (every evidence session among the kept chunks)
and turn recall (share of evidence turns kept). Gate: within 0.03 of 0.96 and 0.91 (the experiment log, exp-lme100-v1).

    python scripts/parity_lme.py                 # dry run: parse the sample, estimate spend, no network calls
    python scripts/parity_lme.py --go            # paid run (Voyage + Jev), capped by --cap

Keys come from the repo-root .env through live-eval/env.py; only their presence is ever printed. The run writes
runs/m5_parity_<stamp>/ (config with sample and code hashes, rows.jsonl, summary.json; server.log stays local).
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg

SERVICE = Path(__file__).resolve().parents[1]
BRAIN_V2 = SERVICE.parent
LIVE_EVAL = BRAIN_V2 / "live-eval"
sys.path.insert(0, str(SERVICE))
from brain.text import CHUNK_MAX_CHARS, pack, render, split_text  # noqa: E402

ISO = {"recall_all": 0.96, "turn_recall": 0.91}      # exp-lme100-v1, iso arm
TOL = 0.03
KEEP, BUDGET, SHORTLIST = 10, 6000, 30               # live-eval config KEEP, ISO_BUDGET_TOKENS, SHORTLIST
BATCH_TURNS, BATCH_CHARS = 100, 500_000
DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?: \(\w+\))?(?: (\d{2}):(\d{2}))?")


def live_eval_imports():
    sys.path.insert(0, str(LIVE_EVAL))
    import env  # noqa: F401  (live-eval module: loads keys from the repo-root .env)
    from data import lme_question
    return env, lme_question


def load_sample(path: Path):
    _, lme_question = live_eval_imports()
    qids = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    key = hashlib.sha256("\n".join(qids).encode()).hexdigest()[:12]
    sub = LIVE_EVAL / "data" / f"lme_subset_{key}.json"
    if not sub.exists():
        sys.exit(f"{sub.name} missing: run live-eval once for this sample to build the subset")
    d = json.loads(sub.read_text(encoding="utf-8"))
    return [lme_question(d["s"][q], d["oracle"].get(q)) for q in qids], sub


def session_ts(date: str) -> datetime:
    m = DATE_RE.search(date)
    if not m:
        raise ValueError(f"unparsed session date {date!r}")
    y, mo, d, hh, mm = m.groups()
    return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), tzinfo=timezone.utc)


def turns_of(q) -> tuple[list[dict], int]:
    """Capture turns in haystack order: (turn dicts with session_id, skipped empty turns)."""
    out, empty = [], 0
    for s in q.haystack:
        base = session_ts(s.date)
        for t_idx, t in enumerate(s.turns):
            if not t.text.strip():
                empty += 1
                continue
            out.append({"session_id": s.sid, "turn": {
                "role": t.role if t.role in ("user", "assistant") else "user", "text": t.text,
                "ts": (base + timedelta(seconds=t_idx)).isoformat(), "meta": {"tid": t.tid},
                "idempotency_key": t.tid}})
    return out, empty


def estimate(qs) -> dict:
    """Voyage: every rendered window; Jev: 30 candidates per question at about 2,000 characters plus overhead
    (the live-eval estimate). Over-counts: Voyage free credits are ignored."""
    vt = jt = 0
    for q in qs:
        turns, _ = turns_of(q)
        for x in turns:
            t = x["turn"]
            for w in split_text(t["text"], CHUNK_MAX_CHARS):
                vt += max(1, len(render(datetime.fromisoformat(t["ts"]), t["role"], None, w)) // 4)
        jt += SHORTLIST * (500 + 150 + 2 * 40)       # passage, question, two Noul definitions
    return {"voyage_tokens": vt, "jev_tokens": jt, "usd": round(vt / 1e6 * 0.02 + jt / 1e6 * 0.042, 4)}


def batches(turns: list[dict]):
    """Group consecutive turns of one session into capture bodies (<= 100 turns, <= 500K characters)."""
    cur, sid, size = [], None, 0
    for x in turns:
        n = len(x["turn"]["text"])
        if cur and (x["session_id"] != sid or len(cur) >= BATCH_TURNS or size + n > BATCH_CHARS):
            yield sid, cur
            cur, size = [], 0
        cur.append(x["turn"])
        sid, size = x["session_id"], size + n
    if cur:
        yield sid, cur


def run_question(http: httpx.Client, idx: int, q) -> dict:
    ns = f"q{idx:03d}"
    row = {"qid": q.qid, "qtype": q.qtype, "is_abs": q.is_abs, "namespace": ns, "err": None}
    try:
        turns, empty = turns_of(q)
        tid_of: dict[int, str] = {}
        degraded_caps = 0
        for sid, body in batches(turns):
            r = http.post("/v1/capture", json={"namespace": ns, "session_id": sid, "turns": body})
            r.raise_for_status()
            j = r.json()
            degraded_caps += int(j.get("degraded") is not None)
            for ep, t in zip(j["episodes"], body):
                tid_of[ep["id"]] = t["meta"]["tid"]
        r = http.post("/v1/recall", json={"namespace": ns, "query": q.question, "k": SHORTLIST})
        r.raise_for_status()
        rec = r.json()
        items = rec["items"]
        kept = [items[i] for i in pack([it["text"] for it in items], KEEP, BUDGET)]
        ev_s, ev_t = set(q.evidence_session_ids), set(q.evidence_turn_ids)
        got_s = {it["session_id"] for it in kept}
        got_t = {tid_of.get(it["episode_id"]) for it in kept}
        row.update({
            "turns": len(turns), "empty_turns_skipped": empty, "degraded_captures": degraded_caps,
            "degraded": rec["degraded"], "dense": rec["dense"], "screened": rec["screened"],
            "candidates": len(items), "kept": len(kept), "request_id": rec["request_id"],
            "evidence_sessions": len(ev_s), "evidence_turns": len(ev_t),
            "recall_any": (bool(ev_s & got_s) if ev_s else None),
            "recall_all": (ev_s <= got_s if ev_s else None),
            "turn_recall": (len(ev_t & got_t) / len(ev_t) if ev_t else None),
            "evidence_in_shortlist": len(ev_t & {tid_of.get(it["episode_id"]) for it in items}),
            "kept_ids": [it["chunk_id"] for it in kept],
        })
    except Exception as e:   # class and status only; never bodies
        status = getattr(getattr(e, "response", None), "status_code", None)
        row["err"] = f"{type(e).__name__}" + (f" {status}" if status else "")
    return row


def injected_evidence(db_url: str, rows: list[dict], qs) -> None:
    """Per row: candidates dropped by the injection screen, and how many of them were evidence turns."""
    ev = {q.qid: set(q.evidence_turn_ids) for q in qs}
    with psycopg.connect(db_url) as c:
        for r in rows:
            if r.get("err"):
                continue
            tids = [t for (t,) in c.execute(
                """select e.meta->>'tid' from brain.decision_log d
                   join brain.chunk k on k.id = (d.input->>'chunk_id')::bigint
                   join brain.episode e on e.id = k.ref_id
                   where d.request_id = %s and d.stage = 'recall.injection' and d.outcome = 'dropped'""",
                (r["request_id"],)).fetchall()]
            r["injection_dropped"] = len(tids)
            r["injection_dropped_evidence"] = len(set(tids) & ev[r["qid"]])


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if not r["is_abs"] and not r["err"]]

    def mean(key):
        v = [float(r[key]) for r in scored if r.get(key) is not None]
        return round(sum(v) / len(v), 4) if v else None
    by_type: dict = {}
    for r in scored:
        t = by_type.setdefault(r["qtype"], {"n": 0, "_ra": [], "_tr": []})
        t["n"] += 1
        if r["recall_all"] is not None:
            t["_ra"].append(float(r["recall_all"]))
        if r["turn_recall"] is not None:
            t["_tr"].append(float(r["turn_recall"]))
    for t in by_type.values():
        ra, tr = t.pop("_ra"), t.pop("_tr")
        t["recall_all"] = round(sum(ra) / len(ra), 4) if ra else None
        t["turn_recall"] = round(sum(tr) / len(tr), 4) if tr else None
    s = {"n": len(rows), "scored": len(scored), "errors": sum(1 for r in rows if r["err"]),
         "recall_any": mean("recall_any"), "recall_all": mean("recall_all"), "turn_recall": mean("turn_recall"),
         "degraded_recalls": sum(1 for r in rows if r.get("degraded")),
         "keyword_only_recalls": sum(1 for r in rows if r.get("dense") is False),
         "degraded_captures": sum(r.get("degraded_captures", 0) for r in rows),
         "injection_dropped": sum(r.get("injection_dropped", 0) for r in rows),
         "injection_dropped_evidence": sum(r.get("injection_dropped_evidence", 0) for r in rows),
         "by_type": dict(sorted(by_type.items()))}
    s["gate"] = {k: {"service": s[k], "iso": v, "diff": None if s[k] is None else round(s[k] - v, 4),
                     "pass": s[k] is not None and abs(s[k] - v) <= TOL} for k, v in ISO.items()}
    s["pass"] = all(g["pass"] for g in s["gate"].values()) and s["errors"] == 0
    return s


def code_hashes() -> dict:
    files = sorted((SERVICE / "brain").rglob("*.py")) + sorted((SERVICE / "brain" / "migrations").glob("*.sql"))
    files.append(Path(__file__).resolve())
    return {str(p.relative_to(SERVICE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in files}


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=BRAIN_V2, capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default=str(LIVE_EVAL / "samples" / "lme_s_20261106_100.txt"))
    p.add_argument("--limit", type=int)
    p.add_argument("--go", action="store_true", help="spend money: run against Voyage and Jev")
    p.add_argument("--cap", type=float, default=5.0, help="BRAIN_DAILY_CAP_USD for the server")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--admin-db", default=None, help="default: TEST_DATABASE_URL (scripts/testdb.py)")
    p.add_argument("--db", default="brain_parity")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    sample = Path(args.sample)
    qs, subset = load_sample(sample)
    qs = qs[: args.limit] if args.limit else qs
    for q in qs:                                   # fail before spending if any date or turn is unusable
        turns_of(q)
        if any(len(x["turn"]["text"]) > 100_000 for x in turns_of(q)[0]):
            sys.exit(f"{q.qid}: a turn is over 100,000 characters")
    est = estimate(qs)
    print(f"{len(qs)} questions, {sum(len(turns_of(q)[0]) for q in qs)} turns; estimate {est}")
    if est["usd"] > args.cap:
        sys.exit("estimate is over the cap")
    if not args.go:
        print("dry run: add --go to spend")
        return 0

    env, _ = live_eval_imports()
    env.load_env()
    keys = {"voyage": bool(env.voyage_key()),
            "jev": any(os.environ.get(n) for n in ("TYPESAFE_API_KEY", "JEV_API_KEY"))}
    print("keys present:", keys)
    if not all(keys.values()):
        sys.exit("missing provider keys")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = SERVICE / "runs" / f"m5_parity_{stamp}"
    run_dir.mkdir(parents=True)
    from testdb import admin_url, db_url as _db_url
    args.admin_db = args.admin_db or admin_url()
    db_url = _db_url(args.db)
    with psycopg.connect(args.admin_db, autocommit=True) as c:
        c.execute(f'drop database if exists "{args.db}"')
        c.execute(f'create database "{args.db}"')
    admin = "parity-" + secrets.token_hex(24)
    server_env = {**os.environ, "DATABASE_URL": db_url, "BRAIN_OFFLINE": "0", "BRAIN_ADMIN_KEY": admin,
                  "BRAIN_DAILY_CAP_USD": str(args.cap), "BRAIN_RATE_PER_MIN": "1000000", "PORT": str(args.port),
                  "HOST": "127.0.0.1", "BRAIN_LOG_LEVEL": "WARNING"}
    config = {"run": run_dir.name, "sample": sample.name,
              "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
              "subset_sha256": hashlib.sha256(subset.read_bytes()).hexdigest(), "questions": len(qs),
              "git_head": git_head(), "code": code_hashes(), "estimate": est, "cap_usd": args.cap,
              "recall": {"k": SHORTLIST, "pack_keep": KEEP, "pack_budget_tokens": BUDGET},
              "models": {"voyage": "voyage-4-lite", "jev": os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")},
              "iso_reference": ISO, "tolerance": TOL, "started": stamp}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log = open(run_dir / "server.log", "w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "-m", "brain.serve"], cwd=SERVICE, env=server_env, stdout=log,
                              stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    rows: list[dict] = []
    try:
        for _ in range(120):
            try:
                if httpx.get(base + "/healthz", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            sys.exit("server did not start; see server.log")
        with httpx.Client(base_url=base, timeout=300, headers={"Authorization": f"Bearer {admin}"}) as h:
            acct = h.post("/v1/admin/accounts", json={"name": "parity"})
            acct.raise_for_status()
            key = acct.json()["key"]
        started = time.time()
        with httpx.Client(base_url=base, timeout=300, headers={"Authorization": f"Bearer {key}"}) as h, \
                open(run_dir / "rows.jsonl", "w", encoding="utf-8") as out, \
                cf.ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(run_question, h, i, q): i for i, q in enumerate(qs)}
            for n, f in enumerate(cf.as_completed(futs), 1):
                r = f.result()
                rows.append(r)
                out.write(json.dumps(r) + "\n")
                out.flush()
                if n % 10 == 0 or r["err"]:
                    print(f"{n}/{len(qs)} done, {time.time() - started:.0f}s" + (f"; {r['qid']}: {r['err']}"
                                                                                  if r["err"] else ""), flush=True)
        with httpx.Client(base_url=base, timeout=60, headers={"Authorization": f"Bearer {admin}"}) as h:
            spend = h.get("/v1/admin/spend", params={"days": 2}).json()
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    order = {q.qid: i for i, q in enumerate(qs)}
    rows.sort(key=lambda r: order[r["qid"]])
    injected_evidence(db_url, rows, qs)
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = summarize(rows)
    summary["spend"] = {"total_usd": spend.get("total_usd"), "days": spend.get("days")}
    summary["minutes"] = round((time.time() - started) / 60, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("scored", "errors", "recall_all", "turn_recall", "gate", "pass",
                                              "injection_dropped", "injection_dropped_evidence", "spend",
                                              "minutes")}, indent=2))
    print("run dir:", run_dir)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
