"""The Jev questions must be exactly the measured cookbook questions (REFERENCE.md step 2).

The cookbooks live outside this repo, so their questions are pinned in tests/fixtures/cookbook_questions.json
(README rule 8: runnable from this repo). When ../../typesafe_cookbooks is present, the pin is checked against it
too, so a change in the cookbooks fails here instead of drifting silently."""
import json
import sys
from pathlib import Path

import pytest

from brain.questions import ANSWERS_QUERY, CONTAINS_PROMPT_INJECTION, INJECTION_THRESHOLD, RECALL_QUESTIONS

PIN = json.loads((Path(__file__).parent / "fixtures" / "cookbook_questions.json").read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[3]


def test_rerank_question_matches_pin():
    assert ANSWERS_QUERY == PIN["answers_query"]


def test_injection_question_and_threshold_match_pin():
    assert CONTAINS_PROMPT_INJECTION == PIN["contains_prompt_injection"]
    assert INJECTION_THRESHOLD == PIN["injection_max"]


def test_pin_matches_cookbooks_when_present():
    if not (REPO / "typesafe_cookbooks" / "rerank.py").exists():
        pytest.skip("typesafe_cookbooks not next to this repo; the pin is the reference")
    sys.path.insert(0, str(REPO))
    try:
        from typesafe_cookbooks import classify_passages, rerank
    finally:
        sys.path.remove(str(REPO))
    assert rerank.QUESTION["answers_query"] == PIN["answers_query"]
    assert classify_passages.QUESTIONS["contains_prompt_injection"] == PIN["contains_prompt_injection"]
    assert classify_passages.THRESHOLDS["injection_max"] == PIN["injection_max"]


def test_recall_request_has_both_nouls():
    assert set(RECALL_QUESTIONS) == {"answers_query", "contains_prompt_injection", "subverts_user"}
    assert all(q["type"] == "noul" for q in RECALL_QUESTIONS.values())


def test_screen_rule_uses_provenance():
    from brain.questions import screen_drops, trusted_source
    assert trusted_source("user", "chat") and trusted_source("user", "notes") and trusted_source("user", "voice")
    assert not trusted_source("attachment", "email") and not trusted_source("tool", "agent_log")
    assert not trusted_source("user", "document") and not trusted_source("assistant", "chat")
    # the user's own standing preference: the cookbook Noul fires, the subversion Noul does not -> kept
    assert screen_drops(0.91, 0.03, "user", "chat") is False
    # the same text from a vendor's email -> dropped (either Noul is enough outside the user's own words)
    assert screen_drops(0.91, 0.03, "attachment", "email") is True
    assert screen_drops(0.30, 0.80, "tool", "agent_log") is True
    # a real attack in the user's own chat: both fire -> dropped
    assert screen_drops(0.95, 0.90, "user", "chat") is True
    assert screen_drops(0.10, 0.10, "attachment", "document") is False
