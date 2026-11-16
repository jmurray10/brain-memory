"""Offline tests: no network, no keys. Run from live-eval:  python3 -m unittest tests/test_offline.py -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import invariants  # noqa: E402
from data import Session, Turn, lme_question, load_locomo  # noqa: E402
from prices import SpendCapReached, SpendMeter, haiku_cost  # noqa: E402
from prompts import lme_judge_prompt, parse_yes_no  # noqa: E402
from router import answer_style  # noqa: E402
from text import build_chunks, chronological, pack, split_text  # noqa: E402


class TestJudgeParser(unittest.TestCase):
    def test_yes_no(self):
        self.assertIs(parse_yes_no("yes"), True)
        self.assertIs(parse_yes_no("No."), False)
        self.assertIs(parse_yes_no("CORRECT: yes"), True)
        self.assertIs(parse_yes_no("CORRECT: no"), False)   # the old harness counted this as correct
        self.assertIs(parse_yes_no("correct: No"), False)
        self.assertIs(parse_yes_no("**Yes**"), True)

    def test_garbage_is_parse_failure(self):
        for raw in ["", None, "Correct", "I think so", "maybe yes"]:
            self.assertIsNone(parse_yes_no(raw), raw)


class TestText(unittest.TestCase):
    def test_split_keeps_all_text(self):
        t = ("Sentence one is here. " * 200) + "\nTail paragraph."
        parts = split_text(t, 300)
        self.assertEqual("".join(parts), t)
        self.assertTrue(all(len(p) <= 300 for p in parts))

    def test_pack_skips_instead_of_stopping(self):
        s = Session("s1", "2023/01/01", [Turn("a", "user", "user", "x" * 40),
                                          Turn("b", "user", "user", "y" * 4000),
                                          Turn("c", "user", "user", "z" * 40)])
        ch = build_chunks([s], 100000)
        packed, skipped = pack(ch, 10, 100)
        self.assertEqual([c.tid for c in packed], ["a", "c"])
        self.assertEqual(skipped, 1)

    def test_chronological(self):
        s1 = Session("s1", "d1", [Turn("a", "user", "user", "1"), Turn("b", "user", "user", "2")])
        s2 = Session("s2", "d2", [Turn("c", "user", "user", "3")])
        ch = build_chunks([s1, s2], 100)
        self.assertEqual([c.tid for c in chronological(list(reversed(ch)))], ["a", "b", "c"])


class TestPrices(unittest.TestCase):
    def test_haiku_tiers(self):
        self.assertAlmostEqual(haiku_cost(100_000, 0), 0.01)    # at the threshold: short rate 0.10/M
        self.assertAlmostEqual(haiku_cost(0, 1_000_000), 0.50)
        self.assertAlmostEqual(haiku_cost(200_000, 0), 0.10)    # over 100K: long rate 0.50/M

    def test_cap(self):
        m = SpendMeter(0.01)
        m.add_haiku(50_000, 0)  # $0.005
        m.check(0.004)
        with self.assertRaises(SpendCapReached):
            m.check(0.006)
        self.assertTrue(m.stopped)


class TestRouterAndPrompts(unittest.TestCase):
    def test_router(self):
        self.assertEqual(answer_style("Can you recommend some resources for video editing?"), "preference")
        self.assertEqual(answer_style("How many days ago did I attend the workshop?"), "default")

    def test_judge_templates(self):
        for t in ["single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning",
                  "knowledge-update", "single-session-preference"]:
            p = lme_judge_prompt(t, "Q?", "A", "R", False)
            self.assertIn("Answer yes or no only", p)
        self.assertIn("unanswerable", lme_judge_prompt("multi-session", "Q?", "A", "R", True))
        self.assertIn("off-by-one", lme_judge_prompt("temporal-reasoning", "Q?", "A", "R", False))


class TestAdapters(unittest.TestCase):
    def test_lme_adapter(self):
        s_inst = {"question_id": "x1_abs", "question_type": "knowledge-update", "question": "Q?", "answer": "A",
                  "question_date": "2023/05/30 (Tue) 10:00", "answer_session_ids": ["s2"],
                  "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2023/05/01", "2023/05/20"],
                  "haystack_sessions": [[{"role": "user", "content": "hi"}],
                                        [{"role": "user", "content": "I moved", "has_answer": True}]]}
        o_inst = dict(s_inst, haystack_session_ids=["s2"], haystack_dates=["2023/05/20"],
                      haystack_sessions=[s_inst["haystack_sessions"][1]])
        q = lme_question(s_inst, o_inst)
        self.assertTrue(q.is_abs)
        self.assertEqual(len(q.haystack), 2)
        self.assertEqual([s.sid for s in q.oracle_sessions], ["s2"])
        self.assertEqual(q.evidence_turn_ids, ["s2:0"])

    def test_locomo_adapter(self):
        if not config.LOCOMO_PATH.exists():
            self.skipTest("locomo10.json not found")
        qs = load_locomo()
        self.assertEqual(len(qs), 1540)  # categories 1-4, same as the ledger
        q = qs[0]
        self.assertTrue(q.oracle_sessions)
        self.assertTrue(set(q.evidence_session_ids) <= {s.sid for s in q.haystack})


class TestInvariants(unittest.TestCase):
    def _arm(self, acc, n=100):
        return {"n": n, "acc": acc, "by_type": {"t": {"n": n, "acc": acc}}, "parse_fail": 0, "errors": 0, "truncated": 0}

    def test_oracle_below_iso_is_invalid(self):
        r = invariants.check({"arms": {"oracle": self._arm(0.51), "iso": self._arm(0.71)}}, {})
        self.assertFalse(r["valid"])

    def test_too_good_needs_manual_check(self):
        r = invariants.check({"arms": {"oracle": self._arm(0.99), "iso": self._arm(0.90)}}, {})
        self.assertFalse(r["valid"])
        r2 = invariants.check({"arms": {"oracle": self._arm(0.99), "iso": self._arm(0.90)}}, {}, manual_checked=True)
        self.assertTrue(r2["valid"])

    def test_normal_run_is_valid(self):
        r = invariants.check({"arms": {"oracle": self._arm(0.80), "iso": self._arm(0.70)}}, {})
        self.assertTrue(r["valid"])


if __name__ == "__main__":
    unittest.main()
