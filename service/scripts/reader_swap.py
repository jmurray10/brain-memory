"""exp-reader-swap-v1: another model reads the packs a finished run handed to Haiku.

    python scripts/reader_swap.py runs/locomo_eval_<stamp> sonnet near1seats flat none [--cap 5] [--limit N]
    python scripts/reader_swap.py runs/brain_eval_reader_test_<stamp> sonnet near1seats flat none

A LoCoMo run, and a LongMemEval run since 2026-11-12, keeps every pack it built (runs/cache/packs_<run>.json, by hash) and each row names the pack each
arm read. This script gives the same prompts to another reader (scripts/readers.py), under the answer prompt
and with the judge of the run's own benchmark (Haiku, three votes), so the only thing that changes is the model that answers. Nothing is
captured again: no database, no Jev, no Voyage.

Arms are arm names of that run, plus `none`: the same question and date with "(no excerpts)", the no-memory
floor for this reader. Haiku's own answers and verdicts on the same packs are in the run's rows and are
reported beside the new ones; they are not asked again.

Writes runs/reader_swap_<reader>_<stamp>/ (config, rows, summary). Cap in whole dollars, default 5.

Every answer is saved as it is judged (runs/cache/partial_reader_swap_<key>.jsonl), so a run stopped by its cap
or an error loses nothing: the same command again asks only what is missing. The key is the run, the reader and
its request; an answer is reused only for the same question, arm and prompt. The cap is per start. The file is
removed when the run is whole.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")
SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE / "scripts"))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))

NO_EXCERPTS = "Excerpts:\n(no excerpts)"


def no_memory_prompt(prompt: str) -> str:
    """The pack's own question and date, with the excerpts and every block of the brain's taken out."""
    tail = prompt.rsplit("\n\nToday's date: ", 1)
    if len(tail) != 2:
        raise ValueError("the pack does not end with a date and a question")
    return f"{NO_EXCERPTS}\n\nToday's date: {tail[1]}"


def _cat(r: dict):
    return r.get("cat", r.get("qtype"))


def jobs(rows: list[dict], packs: dict[str, str], arms: list[str]) -> list[tuple]:
    """(row index, arm, prompt) for every answer to write. `none` takes its question from the row's first arm."""
    out = []
    for i, r in enumerate(rows):
        for arm in arms:
            src = arm if arm != "none" else next(a for a in arms if a != "none")
            prompt = packs[r[src]["context_sha"]]
            out.append((i, arm, no_memory_prompt(prompt) if arm == "none" else prompt, r[src].get("style")))
    return out


def swap_key(run: str, reader: str, request: dict) -> str:
    return hashlib.sha256(json.dumps([run, reader, request], sort_keys=True).encode()).hexdigest()[:12]


def _sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def saved_load(path: Path) -> tuple[dict[tuple, dict], float]:
    """What stopped starts of the same swap already answered: (qid, arm, prompt sha) to the answer, and what
    those starts had spent when they last saved (each start has its own meter)."""
    done, spent = {}, {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["qid"], r["arm"], r["sha"])] = r["got"]
                spent[r["start"]] = max(spent.get(r["start"], 0.0), r["spent"])
    return done, round(sum(spent.values()), 4)


def saved_add(path: Path, start: str, qid: str, arm: str, prompt: str, got: dict, spent: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"start": start, "qid": qid, "arm": arm, "sha": _sha(prompt), "spent": round(spent, 4),
                            "got": got}, ensure_ascii=False) + chr(10))


def summarise(rows: list[dict], arms: list[str], reader: str) -> dict:
    out = {}
    for arm in arms:
        key = f"{arm}@{reader}"
        new = [r[key] for r in rows if key in r]
        s: dict = {"n": len(new), reader: sum(x["correct"] is True for x in new),
             "no_verdict": sum(x["correct"] is None for x in new),
             "refused": sum(x.get("stop_reason") == "refusal" for x in new),
             "said_unknown": sum(x["answer"].strip().upper().startswith("UNKNOWN") for x in new)}
        if arm != "none":
            old = [r[arm] for r in rows if key in r]
            s["haiku"] = sum(x["correct"] is True for x in old)
            s["gained"] = sum(o["correct"] is not True and n["correct"] is True for o, n in zip(old, new))
            s["lost"] = sum(o["correct"] is True and n["correct"] is not True for o, n in zip(old, new))
            s["by_category"] = {str(c): {"n": k, "haiku": sum(r[arm]["correct"] is True for r in rows if key in r and _cat(r) == c),
                                         reader: sum(r[key]["correct"] is True for r in rows if key in r and _cat(r) == c)}
                                for c, k in sorted(Counter(_cat(r) for r in rows if key in r).items(), key=lambda x: str(x[0]))}
        out[arm] = s
    return out


def main():
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from brain.providers import make_providers
    from brain_eval import flag_int
    from brain.core import ANSWER_DEFAULT, ANSWER_MAX_TOKENS, ANSWER_PREFERENCE
    from prompts import ANSWER_LOCOMO, LOCOMO_JUDGE_SYSTEM, lme_judge_prompt, locomo_judge_user, parse_yes_no
    from readers import READERS, Reader

    args = sys.argv[1:]
    cap = flag_int(args, "--cap", 5)
    limit = flag_int(args, "--limit", 0)
    if len(args) < 3 or args[1] not in READERS:
        sys.exit(__doc__)
    run, reader_name, arms = Path(args[0]), args[1], args[2:]
    lme = run.name.startswith("brain_eval_")
    rows = [json.loads(l) for l in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    packs = json.loads((SERVICE / "runs" / "cache" / f"packs_{run.name}.json").read_text(encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    todo = jobs(rows, packs, arms)
    partial = SERVICE / "runs" / "cache" / f"partial_reader_swap_{swap_key(run.name, reader_name, READERS[reader_name]['body'])}.jsonl"
    done, spent_before = saved_load(partial)
    start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    held = 0
    for i, arm, prompt, _ in todo:
        got = done.get((rows[i]["qid"], arm, _sha(prompt)))
        if got is not None:
            rows[i][f"{arm}@{reader_name}"] = got
            held += 1
    todo = [j for j in todo if f"{j[1]}@{reader_name}" not in rows[j[0]]]
    if held:
        print(f"resuming: {held} answers already saved by a stopped start (${spent_before} spent then)", flush=True)
    meter = SpendMeter(MemorySpendStore(), float(cap))
    lock = threading.Lock()
    import env
    env.load_env()
    st = load()
    prov = make_providers(st, meter)
    reader = Reader(reader_name, meter, st.anthropic_key)
    print(f"{reader_name} reads {len(todo)} packs of {run.name} ({', '.join(arms)}); cap ${cap}", flush=True)

    def judge(r, pred):
        votes = []
        for _ in range(6):
            if lme:
                raw = prov.haiku.complete("", lme_judge_prompt(r["qtype"], r["question"], str(r["gold"]), pred,
                                                               r["qid"].endswith("_abs")), 10).text
            else:
                raw = prov.haiku.complete(LOCOMO_JUDGE_SYSTEM, locomo_judge_user(r["question"], r["gold"], pred), 10).text
            v = parse_yes_no(raw)
            if v is not None:
                votes.append(v)
            if len(votes) == 3:
                break
        return (sum(votes) >= 2) if len(votes) == 3 else None, votes

    def one(job):
        i, arm, prompt, style = job
        system = ANSWER_LOCOMO if not lme else ANSWER_PREFERENCE if style == "preference" else ANSWER_DEFAULT
        c = reader.complete(system, prompt, 400 if not lme else ANSWER_MAX_TOKENS)
        ok, votes = judge(rows[i], c.text)
        got = {"answer": c.text, "correct": ok, "votes": votes, "stop_reason": c.stop_reason,
               "tokens_in": c.input_tokens, "tokens_out": c.output_tokens}
        with lock:
            saved_add(partial, start, rows[i]["qid"], arm, prompt, got, meter.spent_today())
        return i, arm, got

    with cf.ThreadPoolExecutor(4) as ex:
        for k, (i, arm, got) in enumerate(ex.map(one, todo), 1):
            rows[i][f"{arm}@{reader_name}"] = got
            if k % 100 == 0:
                print(f"{k}/{len(todo)} done, spent ${meter.spent_today():.3f}", flush=True)
    summary = {"run": run.name, "reader": reader_name, "model": reader.model, "arms": arms,
               "spend_usd": round(meter.spent_today() + spent_before, 4), "results": summarise(rows, arms, reader_name)}
    if held:
        summary["resumed"] = {"answers_from_stopped_starts": held, "their_spend_usd": spent_before}
    out = SERVICE / "runs" / f"reader_swap_{reader_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"run": run.name, "reader": reader_name, "model": reader.model,
                                                 "request": READERS[reader_name]["body"], "arms": arms, "cap": cap,
                                                 "limit": limit}, indent=2), encoding="utf-8")
    keep = ("qid", "qtype" if lme else "cat", "question", "gold")
    (out / "rows.jsonl").write_text("".join(json.dumps({**{k: r[k] for k in keep}, **{a: r[a] for a in arms if a != "none"},
                                                        **{k: v for k, v in r.items() if "@" in k}}) + "\n"
                                            for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)            # the run is whole now; its answers are in the run dir
    print(json.dumps(summary, indent=2))
    print("run dir:", out)


if __name__ == "__main__":
    main()
