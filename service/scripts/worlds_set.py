"""exp-worlds-shipped-v1: the six complex-domain worlds (evals/worlds/*.json) in the layout locomo_eval.py reads.

    python scripts/worlds_set.py            # writes evalsets/worlds_v1.json; no model, no cost

One world is one conversation: each session keeps its date, each turn its speaker and text, in order. A question
keeps its type as its category and its evidence as turn ids. What the layout cannot carry is said in the turn
itself, in words: a turn that is not chat by the user starts with "[<role>, <channel>]" (an email, a document, the
assistant's reply, a tool's output). The eval captures every turn as a speaker's own, so the service's handling of
roles and channels (the assistant route, attachments) is not what this set measures; retrieval, the ledger and
the pack on dense professional material are.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
WORLDS = SERVICE / "evals" / "worlds"
OUT = SERVICE / "evalsets" / "worlds_v1.json"
NEVER_SAID = "This was never mentioned. The right answer says the information is not available and invents nothing."


def when(d: str) -> str:
    """'2026-05-05' -> '9:00 am on 5 May, 2026' (the layout's session date)."""
    x = date.fromisoformat(d)
    return f"9:00 am on {x.day} {x.strftime('%B')}, {x.year}"


def convert(world: dict) -> dict:
    conv: dict = {}
    ids: dict[str, str] = {}
    speakers: list[str] = []
    for n, s in enumerate(world["sessions"], 1):
        turns = []
        for k, t in enumerate(s["turns"], 1):
            sp = t.get("speaker") or t["role"]
            if sp not in speakers:
                speakers.append(sp)
            plain = t["role"] == "user" and t.get("channel", "chat") == "chat"
            text = t["text"] if plain else f"[{t['role']}, {t.get('channel', 'chat')}] {t['text']}"
            ids[t["tid"]] = f"D{n}:{k}"
            turns.append({"speaker": sp, "dia_id": f"D{n}:{k}", "text": text})
        conv[f"session_{n}"] = turns
        conv[f"session_{n}_date_time"] = when(s["date"])
    conv["speaker_a"], conv["speaker_b"] = (speakers + ["assistant", "assistant"])[:2]
    qa = []
    for q in world["questions"]:
        never = q["answer"].strip().rstrip(".").lower() == "not mentioned"
        qa.append({"question": q["question"], "answer": NEVER_SAID if never else q["answer"], "category": q["type"],
                   "evidence": [ids[e] for e in q.get("evidence") or []], "question_date": q.get("question_date")})
    return {"sample_id": f"world-{world['world']}", "domain": world.get("domain"), "conversation": conv, "qa": qa}


def main():
    out = [convert(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(WORLDS.glob("*.json"))]
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{OUT.name}: {len(out)} worlds, {sum(len(c['qa']) for c in out)} questions, "
          f"{sum(len(v) for c in out for k, v in c['conversation'].items() if isinstance(v, list))} turns")


if __name__ == "__main__":
    main()
