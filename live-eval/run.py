"""Live benchmark runner: oracle vs flat retrieval (iso), with spend cap and run invariants.

Examples (from brain-v2/live-eval):
  python3 run.py --bench lme --sample samples/lme_s_20261106_100.txt --preflight
  python3 run.py --bench lme --sample samples/lme_s_20261106_100.txt --dry-run --limit 5
  python3 run.py --bench lme --sample samples/lme_s_20261106_100.txt --cap 5 --run-id lme100_v1
  python3 run.py --bench locomo --convs conv-26 --cap 2 --run-id locomo_conv26
  python3 run.py --summarize runs/lme100_v1
Runs are resumable: rerun the same --run-id and finished (arm, question) rows are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import invariants
from clients import HaikuClient, JevReranker, NoRerank, StubEmbedder, StubLLM, VoyageEmbedder
from data import Question, load_locomo, lme_question, ensure_lme
from env import LIVE_EVAL, key_status, load_env
from prices import SpendCapReached, SpendMeter, haiku_cost
from prompts import (ANSWER_DEFAULT, ANSWER_LOCOMO, ANSWER_PREFERENCE, LOCOMO_JUDGE_SYSTEM,
                     build_answer_user, lme_judge_prompt, locomo_judge_user, parse_yes_no)
from retrieval import retrieve
from router import answer_style
from text import build_chunks, chronological, est_tokens, pack

CODE_FILES = ["run.py", "config.py", "prompts.py", "retrieval.py", "text.py", "data.py", "clients.py",
              "router.py", "invariants.py", "prices.py"]


# ---------------- selection ----------------

def read_ids(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def load_lme_subset(qids: list[str]) -> list[Question]:
    """Load only the sampled questions. Caches a small subset file so later runs skip the 277 MB parse."""
    key = hashlib.sha256("\n".join(qids).encode()).hexdigest()[:12]
    sub = config.DATA / f"lme_subset_{key}.json"
    if not sub.exists():
        want = set(qids)

        def pick(path: Path) -> dict:
            try:  # stream: the S file is 277 MB and the machine may have little RAM
                import ijson
                with open(path, "rb") as f:
                    return {r["question_id"]: r for r in ijson.items(f, "item", use_float=True)
                            if r["question_id"] in want}
            except ImportError:
                return {r["question_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))
                        if r["question_id"] in want}

        s_rows = pick(ensure_lme("s"))
        o_rows = pick(ensure_lme("oracle"))
        missing = want - set(s_rows)
        if missing:
            raise RuntimeError(f"{len(missing)} sample ids not in S: {sorted(missing)[:5]}")
        sub.write_text(json.dumps({"s": s_rows, "oracle": o_rows}), encoding="utf-8")
    d = json.loads(sub.read_text(encoding="utf-8"))
    return [lme_question(d["s"][q], d["oracle"].get(q)) for q in qids]


def select(args) -> list[Question]:
    if args.bench == "lme":
        if not args.sample:
            sys.exit("--sample is required for lme")
        qs = load_lme_subset(read_ids(args.sample))
    else:
        convs = [c.strip() for c in args.convs.split(",")] if args.convs else None
        qs = load_locomo(convs, args.per_conv)
    return qs[: args.limit] if args.limit else qs


# ---------------- one question, one arm ----------------

def answer_system(q: Question) -> tuple[str, str]:
    if q.bench == "locomo":
        return ANSWER_LOCOMO, "locomo"
    style = answer_style(q.question)
    return (ANSWER_PREFERENCE if style == "preference" else ANSWER_DEFAULT), style


def judge_once(q: Question, pred: str, llm) -> str:
    if q.bench == "lme":
        return llm.complete("", lme_judge_prompt(q.qtype, q.question, q.gold, pred, q.is_abs), config.JUDGE_MAX_TOKENS)
    return llm.complete(LOCOMO_JUDGE_SYSTEM, locomo_judge_user(q.question, q.gold, pred), config.JUDGE_MAX_TOKENS)


def judge(q: Question, pred: str, llm, votes: int | None = None) -> tuple[bool | None, str]:
    """Majority of `votes` valid yes/no verdicts; calls that do not parse are retried up to JUDGE_MAX_CALLS.
    Returns (verdict, raw summary). None only if no call produced a yes/no."""
    votes = votes or config.JUDGE_VOTES
    got, raws = [], []
    for _ in range(config.JUDGE_MAX_CALLS):
        raw = judge_once(q, pred, llm)
        raws.append(raw[:20])
        v = parse_yes_no(raw)
        if v is not None:
            got.append(v)
        if len(got) >= votes:
            break
    if not got:
        return None, " | ".join(raws)
    yes = sum(got)
    return yes * 2 > len(got) or (yes * 2 == len(got) and got[0]), f"{yes}/{len(got)} yes"


def run_arm(q: Question, arm: str, llm, embedder, reranker) -> dict:
    row = {"arm": arm, "qid": q.qid, "qtype": q.qtype, "is_abs": q.is_abs, "err": None}
    try:
        info = {}
        if arm == "oracle":
            if not q.oracle_sessions:
                raise RuntimeError("no oracle sessions for this question")
            chunks = build_chunks(q.oracle_sessions, config.CHUNK_MAX_CHARS)
            packed, skipped = pack(chunks, config.ORACLE_MAX_ITEMS, config.ORACLE_BUDGET_TOKENS)
            row["truncated"] = len(packed) < len(chunks)
        else:
            chunks = build_chunks(q.haystack, config.CHUNK_MAX_CHARS)
            ranked, info = retrieve(q.question, chunks, embedder, reranker)
            packed, skipped = pack(ranked, config.KEEP, config.ISO_BUDGET_TOKENS)
            row["truncated"] = False
        ordered = chronological(packed)
        system, style = answer_system(q)
        user = build_answer_user(q.question, q.question_date, [c.render() for c in ordered])
        pred = llm.complete(system, user, config.ANSWER_MAX_TOKENS)
        verdict, raw = judge(q, pred, llm)

        ev_s = set(q.evidence_session_ids)
        got_s = {c.sid for c in packed}
        ev_t = set(q.evidence_turn_ids)
        got_t = {c.tid for c in packed}
        row.update({
            "style": style, "pred": pred, "judge_raw": raw[:60],
            "correct": verdict is True, "parse_fail": verdict is None,
            "n_chunks": len(chunks), "n_packed": len(packed), "skipped": skipped,
            "ctx_tokens": est_tokens(user),
            "recall_any": (bool(ev_s & got_s) if ev_s else None),
            "recall_all": (ev_s <= got_s if ev_s else None),
            "turn_recall": (len(ev_t & got_t) / len(ev_t) if ev_t else None),
            "retrieval": info,
        })
    except SpendCapReached as e:
        row.update({"err": f"spend_cap: {e}", "correct": False, "parse_fail": False})
    except Exception as e:
        row.update({"err": f"{type(e).__name__}: {e}"[:300], "correct": False, "parse_fail": False})
    return row


# ---------------- summary ----------------

def summarize_rows(rows: list[dict]) -> dict:
    latest = {}
    for r in rows:
        latest[(r["arm"], r["qid"])] = r
    arms: dict = {}
    for (arm, _), r in latest.items():
        a = arms.setdefault(arm, {"n": 0, "correct": 0, "by_type": {}, "abs": {"n": 0, "correct": 0},
                                  "parse_fail": 0, "errors": 0, "truncated": 0, "_ra": [], "_rall": [], "_tr": [], "_ctx": []})
        a["n"] += 1
        a["correct"] += int(bool(r.get("correct")))
        t = a["by_type"].setdefault(r["qtype"], {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += int(bool(r.get("correct")))
        if r.get("is_abs"):
            a["abs"]["n"] += 1
            a["abs"]["correct"] += int(bool(r.get("correct")))
        a["parse_fail"] += int(bool(r.get("parse_fail")))
        a["errors"] += int(bool(r.get("err")))
        a["truncated"] += int(bool(r.get("truncated")))
        if not r.get("is_abs"):
            for k, key in (("_ra", "recall_any"), ("_rall", "recall_all"), ("_tr", "turn_recall")):
                if r.get(key) is not None:
                    a[k].append(float(r[key]))
        if r.get("ctx_tokens"):
            a["_ctx"].append(r["ctx_tokens"])
    for a in arms.values():
        a["acc"] = a["correct"] / a["n"] if a["n"] else 0.0
        for t in a["by_type"].values():
            t["acc"] = t["correct"] / t["n"]
        a["abs"]["acc"] = a["abs"]["correct"] / a["abs"]["n"] if a["abs"]["n"] else None
        for k, name in (("_ra", "recall_any"), ("_rall", "recall_all"), ("_tr", "turn_recall"), ("_ctx", "ctx_tokens_mean")):
            v = a.pop(k)
            a[name] = round(sum(v) / len(v), 4) if v else None
    return {"arms": arms}


def write_summary(run_dir: Path, meter_snap: dict, manual_checked: bool) -> dict:
    rows = [json.loads(l) for l in (run_dir / "rows.jsonl").read_text().splitlines() if l.strip()]
    s = summarize_rows(rows)
    s["spend"] = meter_snap
    s["invariants"] = invariants.check(s, meter_snap, manual_checked)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
    lines = [f"# {run_dir.name}", "", f"Valid: **{s['invariants']['valid']}**", ""]
    for v in s["invariants"]["violations"]:
        lines.append(f"- VIOLATION: {v}")
    for w in s["invariants"]["warnings"]:
        lines.append(f"- warning: {w}")
    lines += ["", "| arm | n | acc | abstention acc | recall_all | turn_recall | parse fails | errors |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, a in s["arms"].items():
        lines.append(f"| {arm} | {a['n']} | {a['acc']:.3f} | {a['abs']['acc']} | {a['recall_all']} | "
                     f"{a['turn_recall']} | {a['parse_fail']} | {a['errors']} |")
    types = sorted({t for a in s["arms"].values() for t in a["by_type"]})
    lines += ["", "| type | " + " | ".join(s["arms"]) + " |", "| --- |" + " --- |" * len(s["arms"])]
    for t in types:
        cells = []
        for a in s["arms"].values():
            bt = a["by_type"].get(t)
            cells.append(f"{bt['acc']:.3f} (n={bt['n']})" if bt else "")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", f"Spend: ${meter_snap.get('usd', 0):.4f} of ${meter_snap.get('cap', 0):.2f} cap"]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return s


# ---------------- preflight ----------------

def preflight(qs: list[Question], arms: list[str], embedder) -> dict:
    h_in = h_out = jev_in = 0
    h_usd = 0.0  # priced per call: each call is far below the 100K-token tier
    voy_texts = set()
    for q in qs:
        if "oracle" in arms and q.oracle_sessions:
            ch = build_chunks(q.oracle_sessions, config.CHUNK_MAX_CHARS)
            ctx = min(sum(est_tokens(c.render()) for c in ch), config.ORACLE_BUDGET_TOKENS)
            h_in += ctx + 400 + 400
            h_out += 150 + 5
            h_usd += haiku_cost(ctx + 400, 150) + haiku_cost(400, 5)
        if "iso" in arms:
            ch = build_chunks(q.haystack, config.CHUNK_MAX_CHARS)
            for c in ch:
                t = c.render()
                if embedder is None or not embedder.cached(t, "document"):
                    voy_texts.add(t)
            avg = sum(est_tokens(c.render()) for c in ch) / max(1, len(ch))
            jev_in += int(config.SHORTLIST * (min(avg, 500) + 150))
            iso_ctx = int(min(config.KEEP * avg, config.ISO_BUDGET_TOKENS))
            h_in += iso_ctx + 400 + 400
            h_out += 150 + 5
            h_usd += haiku_cost(iso_ctx + 400, 150) + haiku_cost(400, 5)
    voy_tok = sum(est_tokens(t) for t in voy_texts)
    est = {
        "questions": len(qs), "arms": arms,
        "haiku_in_tokens": h_in, "haiku_out_tokens": h_out, "haiku_usd": round(h_usd, 4),
        "voyage_tokens_uncached": voy_tok, "voyage_usd": round(voy_tok / 1e6 * 0.02, 4),
        "jev_in_tokens": jev_in, "jev_usd": round(jev_in / 1e6 * 0.042, 4),
    }
    est["total_usd"] = round(est["haiku_usd"] + est["voyage_usd"] + est["jev_usd"], 4)
    return est


# ---------------- main ----------------

def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["lme", "locomo"])
    ap.add_argument("--sample", help="file of LongMemEval question ids")
    ap.add_argument("--convs", help="LoCoMo sample ids, comma-separated (default: all 10)")
    ap.add_argument("--per-conv", type=int, help="LoCoMo: first N questions per conversation")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--arms", default="oracle,iso")
    ap.add_argument("--cap", type=float, default=config.DEFAULT_CAP_USD)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-id")
    ap.add_argument("--dry-run", action="store_true", help="no API calls: stub model, stub embeddings, no rerank")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="estimate cost only, no paid calls")
    ap.add_argument("--summarize", help="rewrite summary for an existing run dir")
    ap.add_argument("--manual-checked", action="store_true")
    ap.add_argument("--max-seconds", type=float, help="stop starting new questions after this many seconds (resume later)")
    ap.add_argument("--download", action="store_true", help="only download/verify LongMemEval data (resumable)")
    args = ap.parse_args()

    loaded = load_env()
    if args.download:
        from data import DownloadIncomplete
        t0 = time.time()
        try:
            for w in ("oracle", "s"):
                left = (args.max_seconds - (time.time() - t0)) if args.max_seconds else None
                print("ok", ensure_lme(w, left), flush=True)
        except DownloadIncomplete as e:
            print(f"partial: {e}. Run again to continue.", flush=True)
            return 3
        return 0
    if args.summarize:
        rd = Path(args.summarize)
        prev = json.loads((rd / "summary.json").read_text()) if (rd / "summary.json").exists() else {}
        write_summary(rd, prev.get("spend", {}), args.manual_checked)
        return 0
    if not args.bench:
        sys.exit("--bench is required")

    keys = key_status()
    print(f"env: loaded {len(loaded)} names from .env; keys present: {keys}", flush=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    qs = select(args)
    print(f"selected {len(qs)} {args.bench} questions; arms={arms}", flush=True)

    meter = SpendMeter(0.0 if args.preflight else args.cap)
    if args.dry_run:
        llm, embedder, reranker = StubLLM(), StubEmbedder(), NoRerank()
    elif args.preflight:
        embedder = VoyageEmbedder(meter) if keys["voyage"] else None
        est = preflight(qs, arms, embedder)
        print(json.dumps(est, indent=2))
        out = LIVE_EVAL / "runs" / f"preflight_{args.bench}_{len(qs)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(est, indent=2))
        return 0
    else:
        missing = [k for k, v in keys.items() if not v and not (k == "jev" and args.no_rerank)]
        if missing:
            sys.exit(f"missing keys for: {missing}. Add them to the repo-root .env")
        llm, embedder = HaikuClient(meter, config.ANSWER_MODEL), VoyageEmbedder(meter)
        reranker = NoRerank() if args.no_rerank else JevReranker(meter)

    run_id = args.run_id or f"{args.bench}_{time.strftime('%Y%m%d_%H%M%S')}" + ("_dry" if args.dry_run else "")
    run_dir = config.RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for l in rows_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if not (r.get("err") or "").startswith("spend_cap"):
                    done.add((r["arm"], r["qid"]))
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=LIVE_EVAL).stdout.strip()
    except Exception:
        git = ""
    meta = {"run_id": run_id, "bench": args.bench, "arms": arms, "n_questions": len(qs), "cap": args.cap,
            "dry_run": args.dry_run, "no_rerank": args.no_rerank, "git_head": git,
            "sample_sha": file_hash(Path(args.sample)) if args.sample else None,
            "code_sha": {f: file_hash(LIVE_EVAL / f) for f in CODE_FILES},
            "models": {"answer": config.ANSWER_MODEL, "judge": config.JUDGE_MODEL, "embed": config.VOYAGE_MODEL},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))

    # Spend carries across resumed invocations, so the cap is per run, not per process.
    spend_path = run_dir / "spend.json"
    if spend_path.exists():
        prev = json.loads(spend_path.read_text())
        meter.usd = float(prev.get("usd", 0.0))
        meter.counts = {k: v for k, v in prev.items() if k not in ("usd", "cap", "stopped", "wall_s")}
        print(f"resuming: ${meter.usd:.4f} already spent on this run", flush=True)

    lock = threading.Lock()

    t0 = time.time()

    def out_of_time() -> bool:
        return bool(args.max_seconds) and time.time() - t0 > args.max_seconds

    def work(q: Question):
        out = []
        for arm in arms:
            if (arm, q.qid) in done or meter.stopped or out_of_time():
                continue
            r = run_arm(q, arm, llm, embedder, reranker)
            with lock:
                with open(rows_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                if not args.dry_run:
                    spend_path.write_text(json.dumps(meter.snapshot(), indent=2))
            out.append(r)
        return out

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, q) for q in qs]
        for fut in as_completed(futs):
            fut.result()
            n_done += 1
            if n_done % 5 == 0 or n_done == len(qs):
                print(f"{n_done}/{len(qs)} questions  spend ${meter.snapshot()['usd']:.4f}  {time.time() - t0:.0f}s", flush=True)
    snap = meter.snapshot()
    snap["wall_s"] = round(time.time() - t0, 1)
    have = {(json.loads(l)["arm"], json.loads(l)["qid"]) for l in rows_path.read_text().splitlines() if l.strip()}
    snap["incomplete_rows"] = sum(1 for q in qs for a in arms if (a, q.qid) not in have)
    if snap["incomplete_rows"]:
        print(f"INCOMPLETE: {snap['incomplete_rows']} rows still to run. Rerun the same command to continue.", flush=True)
    if not args.dry_run:
        spend_path.write_text(json.dumps(snap, indent=2))
    write_summary(run_dir, snap, args.manual_checked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
