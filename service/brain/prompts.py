"""Answer prompts and the answer-style router, copied from ../live-eval/prompts.py and ../live-eval/router.py
(the measured versions). tests/test_text.py checks they are identical to the originals."""
from __future__ import annotations

import re

ANSWER_DEFAULT = (
    "You answer questions about a user's long chat history using only the excerpts given. "
    "Each excerpt starts with its session date in brackets. Today's date is given with the question. "
    "Answer directly and briefly. When the question asks for a count, a duration or a date, work it out from "
    "the excerpts and today's date, and state the result. If facts changed over time, give the latest one. "
    "If the excerpts do not contain the answer, reply exactly: UNKNOWN - followed by a few words on what is missing."
)

ANSWER_PREFERENCE = (
    "You are the user's assistant and know them only from the chat excerpts given. Each excerpt starts with its "
    "session date in brackets. Answer the request helpfully in 3 to 6 sentences, tailored to the user's own "
    "preferences, situation and past choices as shown in the excerpts. Mention the specific personal details "
    "you are using. If the excerpts say nothing relevant about the user, give a short general answer."
)

ANSWER_MAX_TOKENS = 400       # live-eval config.ANSWER_MAX_TOKENS
ANSWER_BUDGET_TOKENS = 6000   # live-eval config.ISO_BUDGET_TOKENS


def build_answer_user(question: str, question_date: str, excerpts: list[str]) -> str:
    body = "\n".join(excerpts) if excerpts else "(no excerpts)"
    return f"Excerpts:\n{body}\n\nToday's date: {question_date}\nQuestion: {question}\nAnswer:"


_PREF = re.compile(
    r"\b(recommend|recommendations?|suggest|suggestions?|any (tips|advice|ideas)|what should i|"
    r"can you (give|offer|share) me|ideas for|help me (choose|pick|decide|plan|come up)|"
    r"which .{0,40} should i|do you have any (tips|ideas|advice))\b",
    re.I,
)


def answer_style(question: str) -> str:
    return "preference" if _PREF.search(question) else "default"
