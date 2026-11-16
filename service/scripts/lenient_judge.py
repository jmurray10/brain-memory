"""locomo-full-1540: mark a finished LoCoMo run's answers a second time, with the lenient judge prompt that
published LoCoMo numbers commonly use, and report it beside this harness's own verdicts.

    python scripts/lenient_judge.py runs/locomo_eval_<stamp> near1seats [flat ...] [--cap 3]
    python scripts/lenient_judge.py runs/reader_swap_sonnet_<stamp> near1seats@sonnet [--cap 3]

Each key names an answer in the run's rows (an arm, or `arm@reader` in a reader-swap run). Nothing is answered
again: the stored answer, the question and the gold answer go to the judge once.

LENIENT_PROMPT is the "generous" LoCoMo judge prompt found in public evaluation code (the Mem0 paper's
evaluation, copied by other memory benchmarks): an answer that "touches on the same topic as the gold answer"
is correct. Here it is given to Haiku (this harness's judge model), one vote; the published numbers use other
judge models, so this is the same prompt, not the same judge. The judge of record stays LOCOMO_JUDGE_SYSTEM
with three votes.

Writes runs/lenient_judge_<stamp>/ (config, rows, summary). Cap in whole dollars, default 3.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")
SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

LENIENT_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold}
Generated answer: {answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def lenient_user(question: str, gold, answer: str) -> str:
    return LENIENT_PROMPT.format(question=question, gold=gold, answer=answer)


def parse_label(raw: str | None) -> bool | None:
    """True for CORRECT, False for WRONG. None when the reply has neither word, or both (never counted correct)."""
    if not raw:
        return None
    found = set(re.findall(r"\b(CORRECT|WRONG)\b", raw.upper()))
    if len(found) != 1:
        return None
    return found == {"CORRECT"}


def summarise(rows: list[dict], keys: list[str]) -> dict:
    """For each key: answers marked correct by this harness's judge and by the lenient one, by category, and
    how the two disagree."""
    out = {}
    for key in keys:
        got = [(r["cat"], r[key]["correct"] is True, r[key]["lenient"]) for r in rows if "lenient" in r.get(key, {})]
        cats = sorted(Counter(c for c, _, _ in got))
        out[key] = {
            "n": len(got),
            "ours": sum(o for _, o, _ in got),
            "lenient": sum(l is True for _, _, l in got),
            "no_label": sum(l is None for _, _, l in got),
            "lenient_only": sum(l is True and not o for _, o, l in got),
            "ours_only": sum(o and l is not True for _, o, l in got),
            "by_category": {c: {"n": sum(x == c for x, _, _ in got),
                                "ours": sum(o for x, o, _ in got if x == c),
                                "lenient": sum(l is True for x, _, l in got if x == c)} for c in cats},
        }
    return out


def main():
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from brain.providers import make_providers
    from brain_eval import flag_int

    args = sys.argv[1:]
    cap = flag_int(args, "--cap", 3)
    limit = flag_int(args, "--limit", 0)
    if len(args) < 2:
        sys.exit(__doc__)
    run, keys = Path(args[0]), args[1:]
    rows = [json.loads(l) for l in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    missing = [k for k in keys if k not in rows[0]]
    if missing:
        sys.exit(f"no such answer in the rows of {run.name}: {missing}")
    meter = SpendMeter(MemorySpendStore(), float(cap))
    import env
    env.load_env()
    prov = make_providers(load(), meter)
    todo = [(i, k) for i in range(len(rows)) for k in keys]
    print(f"lenient judge on {len(todo)} answers of {run.name} ({', '.join(keys)}); cap ${cap}", flush=True)

    def one(job):
        i, key = job
        r = rows[i]
        raw = prov.haiku.complete("", lenient_user(r["question"], r["gold"], r[key]["answer"]), 120).text
        return i, key, parse_label(raw), raw

    with cf.ThreadPoolExecutor(4) as ex:
        for n, (i, key, label, raw) in enumerate(ex.map(one, todo), 1):
            rows[i][key]["lenient"] = label
            rows[i][key]["lenient_raw"] = raw
            if n % 200 == 0:
                print(f"{n}/{len(todo)} done, spent ${meter.spent_today():.3f}", flush=True)
    summary = {"run": run.name, "keys": keys, "judge": "haiku, one vote, LENIENT_PROMPT",
               "spend_usd": round(meter.spent_today(), 4), "results": summarise(rows, keys)}
    out = SERVICE / "runs" / f"lenient_judge_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"run": run.name, "keys": keys, "cap": cap, "limit": limit,
                                                 "prompt": LENIENT_PROMPT}, indent=2), encoding="utf-8")
    (out / "rows.jsonl").write_text("".join(json.dumps({"qid": r["qid"], "cat": r["cat"], "question": r["question"],
                                                        "gold": r["gold"], **{k: r[k] for k in keys}}) + "\n"
                                            for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)


if __name__ == "__main__":
    main()
