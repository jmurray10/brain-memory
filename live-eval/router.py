"""Answer-style router. Uses the question text only, never the gold question type.
v0 is a keyword rule; build step 2 replaces it with a Jev Choice router and measures route accuracy."""
from __future__ import annotations

import re

_PREF = re.compile(
    r"\b(recommend|recommendations?|suggest|suggestions?|any (tips|advice|ideas)|what should i|"
    r"can you (give|offer|share) me|ideas for|help me (choose|pick|decide|plan|come up)|"
    r"which .{0,40} should i|do you have any (tips|ideas|advice))\b",
    re.I,
)


def answer_style(question: str) -> str:
    return "preference" if _PREF.search(question) else "default"
