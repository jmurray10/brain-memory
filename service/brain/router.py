"""Question router (DESIGN.md read path, "Understand"; build step 2).

One Jev request reads the question and its date. Following the TypeSafe function-calling pattern (the cookbook
DESIGN.md names for this step): a Choice picks the route, and companion Nouls in the same request carry the
evidence each route needs (a time window, date math, a past state, the assistant's own words, a request for
advice, gathering across conversations). Code combines them by the weakest-argument rule: the decision is only as
sure as its least sure part, so confidence = min(probability of the chosen route, its signature Noul). Below
ROUTE_HEDGE the open route also runs (DESIGN.md), so an unsure route costs a merge, not a wrong answer.
As built since 2026-11-09 the open route runs for every question (core.context), so nothing in the service reads
`hedge`: it is kept as the router's own measure (scripts/route_eval.py scores confident against hedged errors).
exp-jev-probe-v2: the same request crosses ROUTE_HEDGE on one ask in five, which changes nothing while unread.

v1 (exp-route-v1) was a plain Choice with two Nouls; its confidence did not separate right from wrong.
Step 2 only decides and measures (scripts/route_eval.py); recall keeps the open route until the stores behind
the other routes exist (build steps 3-7). Gold labels are for scoring only, never an input.
"""
from __future__ import annotations

from dataclasses import dataclass

VERSION = "v2"
ROUTES = ("current", "history", "when", "count", "preference", "assistant", "open")
ROUTE_HEDGE = 0.75        # set on dev labels (exp-route-v2): lowest threshold reaching the dev error floor

ROUTE_QUESTION = {
    "type": "choice",
    "instructions": (
        "A user asks an assistant that remembers all their past conversations. Which kind of memory lookup "
        "answers the question? Judge how the answer has to be found or computed, not whether it exists."),
    "criteria": {
        "current": {
            "what": "The latest value of a fact about the user or their world that may have changed over time.",
            "examples": ["Where does my sister live now?", "How many bikes do I own?",
                         "What gym do I go to these days?"],
            "not_for": "Asking what it was before, or how it changed (history)."},
        "history": {
            "what": "What a fact was at an earlier time, before a change, or how it changed.",
            "examples": ["What was my previous job?", "What did I use before the Instant Pot?",
                         "Did I increase or decrease my coffee limit?"],
            "not_for": "The latest value only (current)."},
        "when": {
            "what": ("Working with dates: when something happened, how long ago, the number of days, weeks or "
                     "months between events, which event came first, or what happened in a named time window "
                     "such as last week, in March or two months ago."),
            "examples": ["How many days passed between the workshop and the trip?",
                         "Where did I go last Saturday?", "Which did I start first, yoga or running?"],
            "not_for": ("A duration the user stated as a fact ('how long did the move take?' when they said "
                        "'it took a week') is a single detail (open).")},
        "count": {
            "what": ("Gathering things mentioned in several different conversations and counting, adding up or "
                     "listing them."),
            "examples": ["How many weddings did I attend this year?", "How much did I spend on gifts in total?",
                         "List the books I read."],
            "not_for": "One number the user stated once (open)."},
        "preference": {
            "what": "A recommendation, suggestion, tips or advice that should fit the user's own tastes and habits.",
            "examples": ["Any show suggestions for me?", "Can you suggest a dinner idea for tonight?"]},
        "assistant": {
            "what": "What the assistant itself said, recommended, listed or explained in an earlier conversation.",
            "examples": ["What was the third tip you gave me?", "Remind me of the restaurant you recommended."]},
        "open": {
            "what": "A single detail the user mentioned once: a name, place, item, amount or duration.",
            "examples": ["What is my dog's name?", "How long did the move take?",
                         "What did I name my sourdough starter?"]},
    },
}


def _noul(text: str) -> dict:
    return {"type": "noul", "instructions": text}


COMPANIONS = {
    "time_window": _noul("Does the question limit the answer to a period of time, such as last week, in March, "
                         "two months ago or this year?"),
    "needs_date_math": _noul("Does answering require working with dates: counting days, weeks or months between "
                             "events, how long ago something happened, or ordering events by when they happened?"),
    "asks_past_state": _noul("Does the question ask how things were at an earlier time, before a change, rather "
                             "than how they are now?"),
    "about_assistant": _noul("Does the question ask what the assistant itself said, suggested, recommended or "
                             "listed earlier?"),
    "wants_advice": _noul("Does the user ask for a recommendation, suggestion, tips or advice?"),
    "gathers_many": _noul("Does answering require gathering several things the user mentioned in different "
                          "conversations and counting, adding up or listing them?"),
}

ROUTER_QUESTIONS = {"route": ROUTE_QUESTION, **COMPANIONS}

# The Noul that must hold for each route (the route's argument); open and current have none of their own.
SIGNATURE = {"history": "asks_past_state", "count": "gathers_many", "preference": "wants_advice",
             "assistant": "about_assistant"}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float       # weakest argument: min(P(route), its signature Noul)
    p_route: float
    probabilities: dict
    nouls: dict
    weakest: str            # which judgment set the confidence
    hedge: bool             # confidence under ROUTE_HEDGE; scored by route_eval, not read by core.context
    model: str


def state(question: str, question_date: str | None) -> dict:
    return {"question": question, "question_date": question_date or ""}


def _signature(route: str, nouls: dict) -> tuple[str, float] | None:
    if route == "when":     # a date question needs either a time window or date math
        name = max(("time_window", "needs_date_math"), key=lambda n: nouls[n])
        return name, nouls[name]
    name = SIGNATURE.get(route)
    return (name, nouls[name]) if name else None


def decide(result, threshold: float = ROUTE_HEDGE) -> RouteDecision:
    """Turn one Jev result for ROUTER_QUESTIONS into a decision. Code, not the model, applies the rule."""
    pick, probs = result.choice("route")
    nouls = {n: result.noul(n) for n in COMPANIONS}
    p = float(probs.get(pick, 0.0))
    conf, weakest = p, "route"
    sig = _signature(pick, nouls)
    if sig and sig[1] < conf:
        weakest, conf = sig
    return RouteDecision(pick, conf, p, probs, nouls, weakest, pick != "open" and conf < threshold, result.model)


def route(providers, question: str, question_date: str | None = None) -> RouteDecision:
    return decide(providers.jev.ask(state(question, question_date), ROUTER_QUESTIONS))
