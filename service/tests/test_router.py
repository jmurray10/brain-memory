"""Router (build step 2) and Jev Choice support: offline, no keys."""
import pytest

from brain import router
from brain.providers import JevResult, ProviderError, StubJev, _validate_answers
from brain.spend import MemorySpendStore, SpendMeter


def result(choice, probs, **nouls):
    base = {n: 0.1 for n in router.COMPANIONS}
    base.update(nouls)
    return JevResult({"route": {"choice": choice, "probabilities": probs},
                      **{n: {"noul": v} for n, v in base.items()}}, "jev-test")


def test_weakest_argument_sets_confidence_and_hedge():
    d = router.decide(result("assistant", {"assistant": 0.95, "open": 0.05}, about_assistant=0.92))
    assert (d.route, d.confidence, d.weakest, d.hedge) == ("assistant", 0.92, "about_assistant", False)
    d = router.decide(result("assistant", {"assistant": 0.95, "open": 0.05}, about_assistant=0.2))
    assert (d.confidence, d.weakest, d.hedge) == (0.2, "about_assistant", True)
    d = router.decide(result("when", {"when": 0.9, "open": 0.1}, time_window=0.3, needs_date_math=0.8))
    assert (d.confidence, d.weakest) == (0.8, "needs_date_math")     # either date signal will do
    d = router.decide(result("current", {"current": 0.55, "history": 0.45}))
    assert (d.confidence, d.weakest, d.hedge) == (0.55, "route", True)
    assert router.decide(result("open", {"open": 0.3, "when": 0.2})).hedge is False   # open already runs


def test_choice_answers_are_validated():
    q = {"route": router.ROUTE_QUESTION}
    _validate_answers("jev", q, {"route": {"choice": "when", "probabilities": {"when": 0.8, "open": 0.2}}})
    for bad in ({"choice": "teleport", "probabilities": {"when": 1.0}},
                {"choice": "when", "probabilities": {}},
                {"choice": "when", "probabilities": {"when": 1.5}},
                {"choice": "when", "probabilities": {"elsewhere": 0.5}},
                {"choice": "when"}):
        with pytest.raises(ProviderError):
            _validate_answers("jev", q, {"route": bad})


def test_stub_router_is_deterministic_and_valid():
    jev = StubJev(SpendMeter(MemorySpendStore(), 1.0))
    st = router.state("How many days between the trip and the workshop?", "2023/05/30")
    r = jev.ask(st, router.ROUTER_QUESTIONS)
    _validate_answers("jev", router.ROUTER_QUESTIONS, r.answers)
    assert router.decide(r).route in router.ROUTES
    assert router.decide(r).route == router.decide(jev.ask(st, router.ROUTER_QUESTIONS)).route


def test_routes_and_question_names_are_stable():
    assert tuple(router.ROUTE_QUESTION["criteria"]) == router.ROUTES
    assert set(router.ROUTER_QUESTIONS) == {"route", *router.COMPANIONS}
    assert set(router.SIGNATURE.values()) <= set(router.COMPANIONS)
