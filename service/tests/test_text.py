"""The ported text and prompt code must match the measured originals in ../live-eval (REFERENCE.md).
Offline; the parity tests skip when live-eval is not present (e.g. inside the Docker image)."""
import importlib.util
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain import prompts
from brain.text import pack, render, rrf, split_text

LIVE_EVAL = Path(__file__).resolve().parents[2] / "live-eval"


def _load(name):
    path = LIVE_EVAL / f"{name}.py"
    if not path.exists():
        pytest.skip("live-eval not present")
    mod_name = f"live_eval_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod          # dataclasses look their module up while the file executes
    sys.path.insert(0, str(LIVE_EVAL))   # config.py imports its sibling env.py
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(LIVE_EVAL))
    return mod


def _texts():
    rng = random.Random(7)
    words = ["alpha", "beta", "gamma.", "delta!", "eps?", "zeta", "\n", "\n\n", "x" * 40, "y" * 900, "z" * 3100]
    out = ["", "short", "a" * 1500, "a" * 1501, "One. Two. Three.\nFour!" * 200, "no-boundaries" * 400]
    for _ in range(200):
        out.append(" ".join(rng.choice(words) for _ in range(rng.randint(1, 600))))
    return out


def test_split_text_matches_live_eval():
    le = _load("text")
    for t in _texts():
        for m in (1500, 200, 37):
            assert split_text(t, m) == le.split_text(t, m)


def test_split_text_keeps_all_text_and_bounds():
    for t in _texts():
        w = split_text(t)
        assert "".join(w) == t and all(len(x) <= 1500 for x in w)


def test_render_matches_measured_chunk_render():
    le = _load("text")
    ts = datetime(2026, 11, 7, 23, 30, tzinfo=timezone(timedelta(hours=-8)))
    for role, speaker in (("user", None), ("user", "user"), ("assistant", "Grok"), ("attachment", "")):
        c = le.Chunk("c", "s", 0, 0, 0, "2026-11-07", role, speaker, "t", "hello", False)
        assert render(ts, role, speaker, "hello") == c.render()
    # the client's own calendar date, not the UTC date (2026-11-08 07:30 UTC)
    assert render(ts, "user", None, "x").startswith("[2026-11-07] user: ")


def test_rrf_matches_live_eval():
    le = _load("text")
    a, b = ["a", "b", "c", "d"], ["c", "e", "a"]
    assert rrf([a, b]) == le.rrf([a, b])
    assert rrf([a, b])[0] in ("a", "c")


def test_pack_matches_live_eval():
    le = _load("text")
    texts = ["x" * n for n in (4000, 30000, 8000, 100, 12000, 400)]
    chunks = [le.Chunk(str(i), "s", 0, i, 0, "2026-01-01", "user", "", str(i), t, False) for i, t in enumerate(texts)]
    rendered = [c.render() for c in chunks]
    packed, _ = le.pack(chunks, 4, 6000)
    assert [rendered[i] for i in pack(rendered, 4, 6000)] == [c.render() for c in packed]


def test_prompts_match_live_eval():
    le_prompts, le_router = _load("prompts"), _load("router")
    assert prompts.ANSWER_DEFAULT == le_prompts.ANSWER_DEFAULT
    assert prompts.ANSWER_PREFERENCE == le_prompts.ANSWER_PREFERENCE
    assert prompts.build_answer_user("q?", "2026-11-07", ["a", "b"]) == le_prompts.build_answer_user(
        "q?", "2026-11-07", ["a", "b"])
    assert prompts.build_answer_user("q?", "d", []) == le_prompts.build_answer_user("q?", "d", [])
    for q in ("Can you recommend a show?", "Where did I move?", "Which laptop should I buy?",
              "Any tips for my trip?", "What did I say about tips?"):
        assert prompts.answer_style(q) == le_router.answer_style(q)


def test_answer_budget_constants_match_live_eval():
    cfg = _load("config")
    from brain import core
    assert prompts.ANSWER_MAX_TOKENS == cfg.ANSWER_MAX_TOKENS
    assert prompts.ANSWER_BUDGET_TOKENS == cfg.ISO_BUDGET_TOKENS
    assert (core.DENSE_TOP, core.KEYWORD_TOP, core.SHORTLIST, core.ANSWER_K) == (
        cfg.DENSE_TOP, cfg.BM25_TOP, cfg.SHORTLIST, cfg.KEEP)
