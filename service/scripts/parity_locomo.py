"""Keyword parity on LoCoMo (room review v2, group 4): the service's recall against the live-eval iso arm.

M5 checked the service (Postgres `ts_rank_cd` on the keyword side) against the harness (BM25) on LongMemEval and
passed. LoCoMo's turns are short and conversational, so the two keyword rankers could differ there. This runs the
same check on LoCoMo: each conversation is captured once through the running API (one turn per episode, speaker
kept, session id = LoCoMo session, ts = session date), then every question of that conversation is recalled
(k = 30) and packed as the iso arm does (top 10 within 6,000 tokens). Metrics are the iso arm's: recall_all
(every evidence session among the kept chunks) and turn recall (share of evidence turns kept).

The reference is a live-eval run on the same questions:

    cd ../live-eval && python run.py --bench locomo --per-conv 30 --arms iso --run-id locomo_parity_ref
    python scripts/parity_locomo.py --per-conv 30 --ref ../live-eval/runs/locomo_parity_ref          # dry run
    python scripts/parity_locomo.py --per-conv 30 --ref ../live-eval/runs/locomo_parity_ref --go     # paid

Gate: recall_all and turn recall within 0.03 of the reference. Writes runs/locomo_parity_<stamp>/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
import psycopg  # noqa: E402

import parity_lme as pl  # noqa: E402

SERVICE, LIVE_EVAL = pl.SERVICE, pl.LIVE_EVAL


def session_dt(s: str) -> datetime:
    """'1:56 pm on 8 May, 2023' -> a UTC datetime (as scripts/locomo_eval.py)."""
    t, d = s.split(" on ")
    return datetime.strptime(f"{d.strip()} {t.strip()}", "%d %B, %Y %I:%M %p").replace(tzinfo=timezone.utc)


def conv_turns(q) -> tuple[list[dict], int]:
    """One conversation's capture turns in order: (turn dicts with session_id, skipped empty turns)."""
    out, empty = [], 0
    for s in q.haystack:
        base = session_dt(s.date)
        for k, t in enumerate(s.turns):
            if not t.text.strip():
                empty += 1
                continue
            out.append({"session_id": s.sid, "turn": {
                "role": "user", "speaker": t.speaker or None, "text": t.text,
                "ts": (base + timedelta(seconds=k)).isoformat(), "meta": {"tid": t.tid},
                "idempotency_key": t.tid}})
    return out, empty


def by_conversation(qs) -> dict[str, list]:
    out: dict[str, list] = {}
    for q in qs:
        out.setdefault(q.qid.split(":")[0], []).append(q)
    return out


def estimate(convs: dict[str, list]) -> dict:
    """Voyage: every rendered turn once per conversation; Jev: 30 candidates per question."""
    vt = sum(max(1, (len(x["turn"]["text"]) + 30) // 4) for qs in convs.values() for x in conv_turns(qs[0])[0])
    jt = sum(len(qs) for qs in convs.values()) * pl.SHORTLIST * (150 + 150 + 2 * 40)
    return {"voyage_tokens": vt, "jev_tokens": jt, "usd": round(vt / 1e6 * 0.02 + jt / 1e6 * 0.042, 4)}


def score(q, items: list[dict], tid_of: dict[int, str]) -> dict:
    """The iso arm's metrics for one question, from the service's reranked candidates."""
    kept = [items[i] for i in pl.pack([it["text"] for it in items], pl.KEEP, pl.BUDGET)]
    ev_s, ev_t = set(q.evidence_session_ids), set(q.evidence_turn_ids)
    got_s = {it["session_id"] for it in kept}
    got_t = {tid_of.get(it["episode_id"]) for it in kept}
    return {"candidates": len(items), "kept": len(kept), "evidence_sessions": len(ev_s), "evidence_turns": len(ev_t),
            "recall_any": (bool(ev_s & got_s) if ev_s else None),
            "recall_all": (ev_s <= got_s if ev_s else None),
            "turn_recall": (len(ev_t & got_t) / len(ev_t) if ev_t else None),
            "evidence_in_shortlist": len(ev_t & {tid_of.get(it["episode_id"]) for it in items}),
            "kept_ids": [it["chunk_id"] for it in kept]}


def run_conversation(http: httpx.Client, idx: int, qs: list) -> list[dict]:
    ns = f"c{idx:02d}"
    rows = [{"qid": q.qid, "qtype": q.qtype, "is_abs": False, "namespace": ns, "err": None} for q in qs]
    try:
        turns, empty = conv_turns(qs[0])
        tid_of: dict[int, str] = {}
        degraded_caps = 0
        for sid, body in pl.batches(turns):
            r = http.post("/v1/capture", json={"namespace": ns, "session_id": sid, "turns": body})
            r.raise_for_status()
            j = r.json()
            degraded_caps += int(j.get("degraded") is not None)
            for ep, t in zip(j["episodes"], body):
                tid_of[ep["id"]] = t["meta"]["tid"]
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        for row in rows:
            row["err"] = f"capture {type(e).__name__}" + (f" {status}" if status else "")
        return rows
    for q, row in zip(qs, rows):
        try:
            r = http.post("/v1/recall", json={"namespace": ns, "query": q.question, "k": pl.SHORTLIST})
            r.raise_for_status()
            rec = r.json()
            row.update({"turns": len(turns), "empty_turns_skipped": empty, "degraded_captures": degraded_caps,
                        "degraded": rec["degraded"], "dense": rec["dense"], "screened": rec["screened"],
                        "request_id": rec["request_id"], **score(q, rec["items"], tid_of)})
        except Exception as e:   # class and status only; never bodies
            status = getattr(getattr(e, "response", None), "status_code", None)
            row["err"] = f"{type(e).__name__}" + (f" {status}" if status else "")
    return rows


def reference(ref_dir: Path, qids: set[str]) -> dict:
    """recall_all and turn recall of the live-eval iso arm on exactly these questions (from its rows)."""
    ra, tr, seen = [], [], set()
    for line in (ref_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arm") != "iso" or r.get("qid") not in qids or r.get("err"):
            continue
        seen.add(r["qid"])
        if r.get("recall_all") is not None:
            ra.append(float(r["recall_all"]))
        if r.get("turn_recall") is not None:
            tr.append(float(r["turn_recall"]))
    if seen != qids:
        sys.exit(f"the reference run covers {len(seen)} of these {len(qids)} questions; run it on the same sample")
    return {"recall_all": round(sum(ra) / len(ra), 4), "turn_recall": round(sum(tr) / len(tr), 4)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-conv", type=int, default=30, help="first N questions per conversation (as live-eval)")
    p.add_argument("--convs", help="LoCoMo sample ids, comma-separated (default: all 10)")
    p.add_argument("--ref", required=True, help="the live-eval run dir with the iso arm on the same questions")
    p.add_argument("--go", action="store_true", help="spend money: run against Voyage and Jev")
    p.add_argument("--cap", type=float, default=5.0)
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--db", default="brain_parity_locomo")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    sys.path.insert(0, str(LIVE_EVAL))
    import env
    from data import load_locomo
    qs = load_locomo(args.convs.split(",") if args.convs else None, args.per_conv)
    convs = by_conversation(qs)
    for c in convs.values():
        conv_turns(c[0])                         # fail before spending if a date is unusable
    est = estimate(convs)
    ref = reference(Path(args.ref), {q.qid for q in qs})
    print(f"{len(qs)} questions in {len(convs)} conversations; estimate {est}; reference {ref}")
    if est["usd"] > args.cap:
        sys.exit("estimate is over the cap")
    if not args.go:
        print("dry run: add --go to spend")
        return 0

    env.load_env()
    keys = {"voyage": bool(env.voyage_key()), "jev": bool(env.jev_key())}
    print("keys present:", keys)
    if not all(keys.values()):
        sys.exit("missing provider keys")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = SERVICE / "runs" / f"locomo_parity_{stamp}"
    run_dir.mkdir(parents=True)
    from testdb import admin_url, db_url as _db_url
    db_url = _db_url(args.db)
    with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
        c.execute(f'drop database if exists "{args.db}" with (force)')
        c.execute(f'create database "{args.db}"')
    admin = "parity-" + secrets.token_hex(24)
    server_env = {**os.environ, "DATABASE_URL": db_url, "BRAIN_OFFLINE": "0", "BRAIN_ADMIN_KEY": admin,
                  "BRAIN_DAILY_CAP_USD": str(args.cap), "BRAIN_RATE_PER_MIN": "1000000", "PORT": str(args.port),
                  "HOST": "127.0.0.1", "BRAIN_LOG_LEVEL": "WARNING"}
    locomo = LIVE_EVAL.parent.parent / "locomo10.json"
    config = {"run": run_dir.name, "bench": "locomo", "per_conv": args.per_conv, "questions": len(qs),
              "conversations": sorted(convs), "locomo_sha256": hashlib.sha256(locomo.read_bytes()).hexdigest(),
              "git_head": pl.git_head(), "code": pl.code_hashes(), "estimate": est, "cap_usd": args.cap,
              "recall": {"k": pl.SHORTLIST, "pack_keep": pl.KEEP, "pack_budget_tokens": pl.BUDGET},
              "reference_run": Path(args.ref).name, "iso_reference": ref, "tolerance": pl.TOL, "started": stamp}
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
                cf.ThreadPoolExecutor(args.workers) as pool:
            futs = [pool.submit(run_conversation, h, i, c) for i, c in enumerate(convs.values())]
            for n, f in enumerate(cf.as_completed(futs), 1):
                rows += f.result()
                print(f"{n}/{len(convs)} conversations done, {time.time() - started:.0f}s", flush=True)
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
    pl.injected_evidence(db_url, rows, qs)
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    pl.ISO = ref                                    # the gate compares against this run's own reference
    summary = pl.summarize(rows)
    summary["spend"] = {"total_usd": spend.get("total_usd"), "days": spend.get("days")}
    summary["minutes"] = round((time.time() - started) / 60, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("scored", "errors", "recall_all", "turn_recall", "gate", "pass",
                                              "injection_dropped", "injection_dropped_evidence", "spend",
                                              "minutes")}, indent=2))
    print("run dir:", run_dir)
    with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
        c.execute(f'drop database if exists "{args.db}" with (force)')
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
