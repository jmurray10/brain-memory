"""exp-over-time-v1: the same early questions asked as one memory grows.

    python scripts/over_time_set.py            # writes evalsets/over_time_v1.json; no model, no cost

From LoCoMo (locomo10.json), for each of CONVS: the questions whose evidence all lies in the first quarter of the
sessions (QUESTIONS of them, picked by a hash of the question's place in the file), and four memories that hold
them, each written as its own conversation in the layout locomo_eval.py --data reads:

    <id>@1  the first quarter of the sessions
    <id>@2  the first half
    <id>@3  every session
    <id>@4  every session, then every session of another conversation (unrelated people, the same namespace)

Every question carries the date of the conversation's last session as its question date, at every stage, so the
only thing that differs between stages is how much the memory holds. The other conversation's turns have ids that
are never evidence ("E3:4").
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
SOURCE = SERVICE.parents[1] / "locomo10.json"
OUT = SERVICE / "evalsets" / "over_time_v1.json"
CONVS = (("conv-26", "conv-30"), ("conv-30", "conv-26"))      # (the memory asked about, the unrelated one added last)
QUESTIONS = 20
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def sessions(conv: dict) -> int:
    return sum(1 for k in conv if re.fullmatch(r"session_\d+", k))


def iso(when: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08'."""
    d, m, y = re.fullmatch(r".* on (\d+) (\w+), (\d{4})", when.strip()).groups()
    return f"{y}-{MONTHS.index(m) + 1:02d}-{int(d):02d}"


def early_questions(sample: dict, upto: int, n: int = QUESTIONS) -> list[dict]:
    """The questions answerable from sessions 1..upto alone, n of them in a fixed order."""
    out = []
    for k, q in enumerate(sample["qa"]):
        ev = re.findall(r"D(\d+):\d+", " ".join(str(e) for e in q.get("evidence") or []))
        if q.get("category") in CATS and ev and all(int(s) <= upto for s in ev):
            out.append((hashlib.sha256(f"{sample['sample_id']}:{k}".encode()).hexdigest(), q))
    return [q for _, q in sorted(out, key=lambda x: x[0])[:n]]


def stage(sample: dict, upto: int, extra: dict | None = None) -> dict:
    """The conversation's first `upto` sessions, then (stage 4) all of `extra`'s, renumbered after them."""
    src = sample["conversation"]
    conv = {"speaker_a": src["speaker_a"], "speaker_b": src["speaker_b"]}
    for n in range(1, upto + 1):
        conv[f"session_{n}"] = src[f"session_{n}"]
        conv[f"session_{n}_date_time"] = src[f"session_{n}_date_time"]
    if extra:
        for j in range(1, sessions(extra) + 1):
            conv[f"session_{upto + j}"] = [{**t, "dia_id": "E" + str(t.get("dia_id", ""))[1:]} for t in extra[f"session_{j}"]]
            conv[f"session_{upto + j}_date_time"] = extra[f"session_{j}_date_time"]
    return conv


def build(data: list[dict]) -> list[dict]:
    by_id = {c["sample_id"]: c for c in data}
    out = []
    for cid, other in CONVS:
        s = by_id[cid]
        total = sessions(s["conversation"])
        quarter = math.ceil(total / 4)
        asked_on = iso(s["conversation"][f"session_{total}_date_time"])
        qa = [{"question": q["question"], "answer": q.get("answer"), "evidence": q["evidence"],
               "category": CATS[q["category"]], "question_date": asked_on} for q in early_questions(s, quarter)]
        for k, (upto, extra) in enumerate(((quarter, None), (math.ceil(total / 2), None), (total, None),
                                           (total, by_id[other]["conversation"])), 1):
            out.append({"sample_id": f"{cid}@{k}", "stage": k, "conversation": stage(s, upto, extra), "qa": qa})
    return out


def main():
    out = build(json.loads(SOURCE.read_text(encoding="utf-8")))
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    for c in out:
        turns = sum(len(v) for v in c["conversation"].values() if isinstance(v, list))
        print(f"{c['sample_id']}: {sessions(c['conversation'])} sessions, {turns} turns, {len(c['qa'])} questions")


if __name__ == "__main__":
    main()
