"""The no-memory baseline: the answering model alone, with nothing retrieved.

    python scripts/no_memory_eval.py lme test 20 knowledge-update multi-session temporal-reasoning --skip 20
    python scripts/no_memory_eval.py locomo 40 120

Same questions, same answer prompts, same answer model (Haiku 5.5, thinking off) and same judges as
brain_eval.py and locomo_eval.py (pass the same arguments to get the same questions). The one difference: the
model is given no excerpts, so the prompt reads "Excerpts: (no excerpts)". This is what the model scores from
the question and today's date alone, the floor every memory arm is measured against:

    no memory  <  retrieval only (`flat`)  <=?  full brain (`brain`)

No database, no Jev, no Voyage: only the answer and judge calls. Cap $1 a run. LongMemEval abstention
questions (ids ending `_abs`) are answered correctly by saying the information is missing, so a model with no
memory gets them right; the summary reports accuracy with and without them.
Writes runs/no_memory_<bench>_<stamp>/ (config, rows, summary).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")
SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CAP = 1.0
ARM = "no_memory"
ARM_LABEL = ("no memory: the answering model alone, nothing retrieved; same questions, answer prompt and judge as "
             "the memory arms")
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def no_memory_prompt(question: str, question_date: str | None) -> str:
    """The memory arms' user prompt with no excerpts (brain.prompts.build_answer_user)."""
    from brain.prompts import build_answer_user
    return build_answer_user(question, question_date or "unknown", [])


def summarize(rows: list[dict], group: str) -> dict:
    """Accuracy per group and overall, over the rows the judge could score. `abstention` rows (LongMemEval _abs)
    are also reported apart, because saying "I don't know" is their correct answer."""
    def part(rs):
        scored = [r for r in rs if r[ARM]["correct"] is not None]
        k = sum(bool(r[ARM]["correct"]) for r in scored)
        return {"n": len(rs), "scored": len(scored), "correct": k,
                ARM: round(k / len(scored), 4) if scored else None, "ci95": wilson(k, len(scored)),
                "said_unknown": sum(r[ARM]["answer"].strip().upper().startswith("UNKNOWN") for r in rs)}
    out = {g: part([r for r in rows if r[group] == g]) for g in sorted({r[group] for r in rows})}
    out["all"] = part(rows)
    if any(r.get("abstention") for r in rows):
        out["all_without_abstention"] = part([r for r in rows if not r.get("abstention")])
        out["abstention_only"] = part([r for r in rows if r.get("abstention")])
    return out


def lme_questions(split: str, n: int, types: list[str], skip: int) -> list[dict]:
    import brain_eval
    return [{"qid": q["question_id"], "group": q["question_type"], "question": q["question"],
             "gold": str(q["answer"]), "date": q["question_date"], "abstention": q["question_id"].endswith("_abs")}
            for q in brain_eval.pick(split, n, types, skip)]


def locomo_questions(per_conv: int, skip: int) -> list[dict]:
    import locomo_eval
    convs = json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8"))
    out = []
    for conv in convs:
        c, n, last = conv["conversation"], 1, ""
        while f"session_{n}" in c:
            last = c[f"session_{n}_date_time"]
            n += 1
        qdate = locomo_eval.session_dt(last).date().isoformat() if last else None
        qas = [(k, qa) for k, qa in enumerate(conv["qa"]) if qa.get("category") in LOCOMO_CATS]
        qas = sorted(qas, key=lambda x: hashlib.sha256(f"{conv['sample_id']}:{x[0]}".encode()).hexdigest())
        out += [{"qid": f"{conv['sample_id']}:{k}", "group": LOCOMO_CATS[qa["category"]], "question": qa["question"],
                 "gold": str(qa.get("answer")), "date": qdate, "abstention": False}
                for k, qa in qas[skip:skip + per_conv]]
    return out


def main() -> int:
    args = sys.argv[1:]
    skip = 0
    if "--skip" in args:
        k = args.index("--skip")
        skip = int(args[k + 1])
        del args[k:k + 2]
    if not args or args[0] not in ("lme", "locomo"):
        sys.exit(__doc__)
    bench = args[0]
    import env
    env.load_env()
    from prompts import (ANSWER_LOCOMO, LOCOMO_JUDGE_SYSTEM, lme_judge_prompt, locomo_judge_user, parse_yes_no)

    from brain.prompts import ANSWER_DEFAULT, ANSWER_MAX_TOKENS, ANSWER_PREFERENCE, answer_style
    from brain.providers import ProviderError, make_providers
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter

    if bench == "lme":
        if len(args) < 3 or args[1] not in ("dev", "test"):
            sys.exit(__doc__)
        types = args[3:] or ["knowledge-update", "multi-session", "single-session-preference", "temporal-reasoning"]
        qs = lme_questions(args[1], int(args[2]), types, skip)
        config = {"bench": "lme", "split": args[1], "n_per_type": int(args[2]), "types": types, "skip": skip}
    else:
        per_conv, skip = int(args[1]), int(args[2]) if len(args) > 2 else skip
        qs = locomo_questions(per_conv, skip)
        config = {"bench": "locomo", "per_conversation": per_conv, "skip": skip}
    print(f"no memory, {bench}: {len(qs)} questions", flush=True)

    meter = SpendMeter(MemorySpendStore(), CAP)
    prov = make_providers(load(), meter)
    if prov.offline:
        sys.exit("BRAIN_OFFLINE is set: the stub model answers nothing real")
    try:
        prov.haiku.complete("", "Reply with the single word: ok", 5)
    except ProviderError as e:
        sys.exit(f"Haiku preflight failed, nothing run: {e}")

    def judge(q, pred):
        votes = []
        for _ in range(6):
            if bench == "lme":
                raw = prov.haiku.complete("", lme_judge_prompt(q["group"], q["question"], q["gold"], pred,
                                                               q["abstention"]), 10).text
            else:
                raw = prov.haiku.complete(LOCOMO_JUDGE_SYSTEM, locomo_judge_user(q["question"], q["gold"], pred),
                                          10).text
            v = parse_yes_no(raw)
            if v is not None:
                votes.append(v)
            if len(votes) == 3:
                break
        return (sum(votes) >= 2) if len(votes) == 3 else None, votes

    def one(q):
        if bench == "lme":      # core.answer with style "auto"
            system = ANSWER_PREFERENCE if answer_style(q["question"]) == "preference" else ANSWER_DEFAULT
        else:
            system = ANSWER_LOCOMO
        pred = prov.haiku.complete(system, no_memory_prompt(q["question"], q["date"]), ANSWER_MAX_TOKENS).text
        ok, votes = judge(q, pred)
        return {"qid": q["qid"], "group": q["group"], "question": q["question"], "gold": q["gold"],
                "abstention": q["abstention"], ARM: {"answer": pred, "correct": ok, "votes": votes}}

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, qs))
    summary = {**config, "n": len(rows), "arms": [ARM], "arm_labels": {ARM: ARM_LABEL},
               "by_group": summarize(rows, "group"), "spend_usd": round(meter.spent_today(), 4)}
    config["hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                        for p in (Path(__file__), SERVICE / "brain" / "prompts.py",
                                  SERVICE.parent / "live-eval" / "prompts.py")}
    config["answer_model"] = load().answer_model
    run = SERVICE / "runs" / f"no_memory_{bench}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    (run / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for g, v in summary["by_group"].items():
        print(f"{g:<26} n={v['n']:<4} no memory {v[ARM]}  ci95 {v['ci95']}  said UNKNOWN {v['said_unknown']}")
    print(f"spent ${summary['spend_usd']:.3f}")
    print("run dir:", run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
