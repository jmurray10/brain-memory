"""Claims (DESIGN.md write path steps 2, 4 and 5; build step 4): checking the claims the user's model sends.

The brain runs no generative model (DESIGN.md). The user's own model, already in the
conversation, reads each turn and sends claims with it on capture (schemas.ClaimIn: the W3 contract, whose wording
for clients is CLAIMS_CONTRACT, also the MCP capture tool's description). Capture stores them as 'received'. The
write_path job then, per batch of episodes:

1. triages every turn (one Jev request each, the booster's W1 contract; logged, used later for raw turns that come
   with no claims and as booster training data);
2. checks each received claim in code: its quote must be an exact substring of the turn (after normalizing quote
   marks and spaces) and most of its value's content words must be in the quote;
3. verifies the survivors (one Jev request per turn: W4's hallucinated, off_target and incomplete Nouls); any
   above VERIFY_DROP drops the claim. There is no model inside the brain to re-extract with.

Survivors become 'pending' (checked; build step 5 resolves entity and slot), the rest 'dropped' with a reason.
Every decision goes to decision_log with its input verbatim (the booster's future training data). Jev decides,
code computes, the user's model writes.

Only EXTRACT_ROLES get a write_path job (core.WRITE_PATH_ROLES is this tuple): user turns (any channel, notes
included) and attachments. Assistant, system and tool turns never do: the assistant route reads episode lines.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from psycopg.types.json import Jsonb

log = logging.getLogger("brain.claims")

KEEP_AT = 0.5               # triage: a lasting-fact, preference or plan Noul at or above this marks a worthy turn
VERIFY_DROP = 0.70          # DESIGN.md: any P(wrong) above 0.7 drops the claim
EXTRACT_ROLES = ("user", "attachment")
MAX_TURN_CHARS = 12_000     # longer turns (documents) are cut for Jev's state
POLARITIES = ("asserted", "negated", "planned", "hypothetical", "corrected", "reported")

TRIAGE_QUESTIONS = {   # W1: keep, kind, has time phrase, assistant content worth recalling
    "lasting_fact": {"type": "noul", "instructions": (
        "Does `turn.text` state a lasting fact about the speaker or their world (people, places, things, projects, "
        "work, health, possessions, decisions, results) that could matter in a later conversation?")},
    "preference": {"type": "noul", "instructions": (
        "Does `turn.text` state a preference, taste, habit or standing instruction of the speaker?")},
    "plan": {"type": "noul", "instructions": "Does `turn.text` state a plan, commitment, deadline or intention?"},
    "has_date": {"type": "noul", "instructions": (
        "Does `turn.text` say when something happened or will happen (a date, a day, a relative time like "
        "'last week')?")},
    "assistant_recall": {"type": "noul", "instructions": (
        "Is `turn.text` content from an assistant or a tool that the user may ask about later: a recommendation, "
        "a list, a specific answer, a figure?")},
    "kind": {"type": "choice", "instructions": "What is `turn.text` mainly?", "criteria": {
        "fact": "States facts about the speaker's life, work or world.",
        "preference": "States likes, dislikes, habits or standing instructions.",
        "event": "Reports something that happened or will happen at a time.",
        "plan": "States a plan or commitment.",
        "opinion": "Gives a view or judgment.",
        "smalltalk": "Greetings, thanks, filler, or questions with no lasting content."}},
}

# What the user's model is asked to send with a turn (the W3 contract, written for a client model).
CLAIMS_CONTRACT = """With each turn you save, send the memory claims it states, so the brain can remember them.
A claim is one belief about a person, thing, project or the speaker: what it is about, which attribute, the value, and the exact sentence that states it.

Include:
- facts about the speaker, the user, and the people, places, things, projects, cases, studies, results and decisions in their world;
- preferences, tastes, habits and standing instructions ("always reply in metric units" is a claim: slot new:unit_preference, value "metric units", polarity asserted);
- plans, commitments, deadlines and events, with their time phrases.
Skip greetings, thanks, questions, small talk, and general knowledge that is not about this speaker's world.

Each claim:
- "mention": the words in the turn that name who or what the claim is about, exactly as written. Use "" only for a fact about the speaker personally (their own life, work, actions, requests or preferences). A fact about a project, study, case, document, product, patient or another person names that thing as the turn writes it, even in the speaker's own notes, minutes or emails. When a message states a fact about its recipient ("you are the second contribution"), the mention is "you". A request to the listener ("when you send me protocols, use numbered steps") is the speaker's own preference: mention "".
- "slot": a known slot if one fits, else "new:" plus a short snake_case name (new:home_city, new:dose, new:filing_deadline).
- "value": what the quoted sentence says, in its own words, as short as possible but complete (keep units, negations and conditions). Take every word of the value from the quote; add no names, places or details from other sentences, and do not infer what the turn only implies.
- "time_phrase": the exact words in the turn that say when, or null.
- "polarity": "asserted"; "negated" (the turn says it is not so); "planned" (intended, not yet done); "hypothetical" (maybe, if); "corrected" (the turn corrects an earlier statement); or "reported" (the speaker relays what another named person said, e.g. "Dana says the assay failed"). Facts that a note, email, minutes or document states on its own are "asserted".
- "quote": the one sentence of the turn that states the claim, copied character for character. It must contain the value. If the value is spread over two sentences, make two claims.
Use earlier turns only to understand pronouns and references; claims come from the turn being saved. The brain drops any claim whose quote is not in the turn."""


POLARITY_CRITERIA = {
    "asserted": "The quote states the value as so.",
    "negated": "The quote states the value is not so (no, not, never, no longer).",
    "planned": "The quote states the value as intended or scheduled, not yet done.",
    "hypothetical": "The quote states the value as possible or conditional (maybe, if, might, considering).",
    "corrected": "The quote corrects an earlier statement with the value.",
    "reported": "The quote relays the value as what another named person said.",
}
POLARITY_OVERRULE = 0.70    # Jev's own polarity pick, this sure and different from the client's, drops the claim
SPEAKER_AT_LEAST = 0.30     # an empty mention (about the speaker) needs Jev's about-the-speaker Noul at least this


def verify_questions(claims_: list[dict]) -> dict:
    """W4 for a turn's candidates in one request (DESIGN.md: one Jev request per turn), plus two cross-checks of
    what a client model gets wrong most (exp-claims-v1): Jev's own polarity pick, and for an empty mention whether
    the quote is about the speaker personally."""
    qs = {}
    for i, cl in enumerate(claims_):
        qs[f"c{i}_polarity"] = {"type": "choice", "criteria": POLARITY_CRITERIA, "instructions": (
            f"How does `claims[{i}].quote` state `claims[{i}].value`? `speaker` wrote `turn`.")}
        if not cl.get("mention"):
            qs[f"c{i}_about_speaker"] = {"type": "noul", "instructions": (
                f"Is `claims[{i}].value`, as `claims[{i}].quote` states it, a fact about `speaker` personally (their "
                "own life, work, actions, requests or preferences), rather than about a project, document, case, "
                "the person addressed or someone else?")}
    for i in range(len(claims_)):
        c = f"`claims[{i}]`"
        qs[f"c{i}_hallucinated"] = {"type": "noul", "instructions": (
            f"Is the value of {c} not supported by `turn` (invented, or inferred beyond what the turn says)? "
            "`speaker` wrote `turn`: 'I', 'me', 'my' and an empty mention mean the speaker.")}
        qs[f"c{i}_off_target"] = {"type": "noul", "instructions": (
            f"Is {c} about the wrong person or thing, or filed under a slot that does not describe its value, or is "
            "its quote a sentence that does not contain the value? `speaker` wrote `turn`: an empty mention claims "
            "the fact is about the speaker personally, so it is off target when the fact belongs to a project, "
            "document, case or another person (including 'you', the recipient of a message).")}
        qs[f"c{i}_incomplete"] = {"type": "noul", "instructions": (
            f"Is the value of {c} cut short so that it loses meaning `turn` gives it (a missing qualifier, unit, "
            "negation or condition)?")}
    return qs


@dataclass
class Turn:
    episode_id: int
    namespace: str
    role: str
    speaker: str
    channel: str
    session_date: str
    text: str
    previous: list[str] = field(default_factory=list)     # up to 2 earlier turns of the session, oldest first


def _norm(s: str) -> str:
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", " ": " "})
    return re.sub(r"\s+", " ", (s or "").translate(table)).strip()


def quote_ok(quote: str, text: str) -> bool:
    q = _norm(quote)
    return len(q) >= 3 and q in _norm(text)


_WORD = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)*")
_STOP = frozenset("""a an the and or but of to in on at by for with from as is are was were be been being it its this that
these those i me my we our you your he she they them his her their there here not no yes so do does did have has had will
would can could should may might must than then also just very about into over under up down out off own""".split())
VALUE_GENERAL_SHARE = 0.5  # share of the value's ordinary content words (by stem) that must appear in the quote


def _stem(w: str) -> str:
    return w.lower()[:5]


def value_in_quote(value: str, quote: str) -> bool:
    """The value must come from the quoted sentence: a client model sometimes pulls a value from a neighbouring
    sentence or adds a name, number or id the turn does not give there. Strict on specifics, lenient on wording:
    every number, id or capitalized name in the value must be in the quote; of its other content words, at least
    half (compared by stem, so 'creating' matches 'create') must be."""
    q = {w.lower() for w in _WORD.findall(_norm(quote))}
    q_stems = {_stem(w) for w in q}
    words = _WORD.findall(_norm(value))
    general = []
    for i, w in enumerate(words):
        lw = w.lower()
        specific = any(ch.isdigit() for ch in w) or (w[0].isupper() and i > 0 and lw != "i") or (w.isupper() and len(w) > 1)
        if specific:
            if lw not in q:
                return False
        elif lw not in _STOP:
            general.append(lw)
    if not general:
        return True
    return sum(_stem(w) in q_stems for w in general) / len(general) >= VALUE_GENERAL_SHARE


_NEGATION = re.compile(r"^(?:no|not|never|none|nothing|nobody|no longer|without)\b", re.I)


def normalize_polarity(polarity: str, value: str) -> str:
    """Convention: a negated claim's value states what is not so ('negated' + 'dose reduction'). A value that
    already carries the negation ('no dose reduction') is asserted, or the two negations cancel."""
    return "asserted" if polarity == "negated" and _NEGATION.match(value.strip()) else polarity


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def normalize_slot(slot: str, known: tuple[str, ...] = ()) -> str:
    """A known slot stays; anything else becomes new:<snake_case> (new:misc if nothing usable is left)."""
    slot = (slot or "").strip()
    if slot in known:
        return slot
    return "new:" + (_slug(slot[4:] if slot.startswith("new:") else slot) or "misc")


def parse_claims(raw: str, slots: tuple[str, ...] = ()) -> tuple[list[dict], int]:
    """A client model's JSON reply -> (well-formed claims, count of malformed ones), for client adapters and the
    benchmarks' stand-in client. Tolerates a code fence or prose around the JSON list."""
    s = raw.strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b < a:              # prose instead of a list, or a list cut off by max_tokens
        return [], 1 if s else 0
    try:
        items = json.loads(s[a:b + 1])
    except ValueError:
        return [], 1
    out, bad = [], 0
    for c in items if isinstance(items, list) else []:
        if not isinstance(c, dict) or not str(c.get("value") or "").strip() or not str(c.get("quote") or "").strip():
            bad += 1
            continue
        raw_slot = str(c.get("slot") or "").strip()
        if not raw_slot or raw_slot == "new:":
            bad += 1
            continue
        pol = str(c.get("polarity") or "asserted").strip().lower()
        if pol not in POLARITIES:
            bad += 1
            continue
        tp = c.get("time_phrase")
        out.append({"mention": str(c.get("mention") or "").strip(), "slot": normalize_slot(raw_slot, slots),
                    "value": str(c["value"]).strip(), "time_phrase": str(tp).strip() if tp else None,
                    "polarity": pol, "quote": str(c["quote"]).strip()})
    return out, bad


def load_turns(conn, episode_ids: list[int]) -> list[Turn]:
    rows = conn.execute(
        """select e.id, e.namespace, e.role, coalesce(e.speaker, e.role), e.channel, e.ts::date::text, e.body,
                  array(select coalesce(p.speaker, p.role) || ': ' || left(p.body, 400) from (
                      select p.speaker, p.role, p.body, p.ts, p.id from episode p
                      where p.namespace = e.namespace and p.session_id is not distinct from e.session_id
                        and e.session_id is not null and p.deleted_at is null
                        and (p.ts, p.id) < (e.ts, e.id)
                      order by p.ts desc, p.id desc limit 2) p order by p.ts, p.id)
           from episode e where e.id = any(%s) and e.deleted_at is null order by e.ts, e.id""",
        (episode_ids,)).fetchall()
    return [Turn(r[0], r[1], r[2], r[3], r[4], r[5], r[6], list(r[7])) for r in rows]


def triage_state(t: Turn) -> dict:
    return {"turn": {"role": t.role, "speaker": t.speaker, "session_date": t.session_date,
                     "text": t.text[:MAX_TURN_CHARS]}, "previous": t.previous[-1:]}


def client_input(t: Turn, slots: list[str] | None = None) -> dict:
    """What a client model sees for one turn when it extracts (the W3 input); used by the stand-in client."""
    return {"speaker": t.speaker, "role": t.role, "channel": t.channel, "session_date": t.session_date,
            "previous_turns": t.previous, "known_slots": slots or [], "turn": t.text[:MAX_TURN_CHARS]}


@dataclass
class Outcome:
    turns: int = 0
    worthy: int = 0             # triage says the turn states something lasting
    received: int = 0           # claims the client sent with these turns
    code_dropped: int = 0       # quote not in the turn, or value not in the quote
    verify_dropped: int = 0
    stored: int = 0             # checked claims, now 'pending' for build step 5
    no_claims_worthy: int = 0   # worthy turns the client sent no claims for (the raw-turn fallback's job later)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


_CLAIM_COLS = ("id", "episode_id", "mention", "slot", "value", "time_phrase", "polarity", "quote")


def process(conn, providers, episode_ids: list[int], request_id: str | None = None) -> Outcome:
    """Triage these episodes and check the claims their client sent (inside the caller's tenant transaction).
    Raises CapReached or ProviderError from Jev; nothing is changed then (the transaction rolls back)."""
    out = Outcome()
    turns = load_turns(conn, episode_ids)
    out.turns = len(turns)
    if not turns:
        return out
    jev = providers.jev
    logs: list[tuple] = []

    def logrow(t: Turn, stage, decider, model, inp, output, prob, thr, outcome):
        logs.append((t.namespace, stage, decider, model, Jsonb({"episode_id": t.episode_id, **inp}), Jsonb(output),
                     prob, thr, outcome, request_id))

    received: dict[int, list[dict]] = {}
    for r in conn.execute(f"""select {', '.join(_CLAIM_COLS)} from claim_candidate
                              where episode_id = any(%s) and status = 'received' order by id""",
                          ([t.episode_id for t in turns],)).fetchall():
        c = dict(zip(_CLAIM_COLS, r))
        received.setdefault(c["episode_id"], []).append(c)

    # W1 triage (Jev)
    for t, r in zip(turns, jev.ask_many([(triage_state(t), TRIAGE_QUESTIONS) for t in turns])):
        nouls = {k: round(r.noul(k), 4) for k in TRIAGE_QUESTIONS if k != "kind"}
        kind, _ = r.choice("kind")
        p_keep = max(nouls["lasting_fact"], nouls["preference"], nouls["plan"])
        worthy = p_keep >= KEEP_AT and kind != "smalltalk"
        out.worthy += worthy
        n_recv = len(received.get(t.episode_id, []))
        out.no_claims_worthy += worthy and not n_recv
        logrow(t, "write.triage", "jev", r.model, {"task": "W1", **triage_state(t)}, {**nouls, "kind": kind},
               p_keep, KEEP_AT, "worthy" if worthy else "skip")

    # code checks on what the client sent
    by_id = {t.episode_id: t for t in turns}
    passed: list[tuple[Turn, list[dict]]] = []
    for eid, cs in received.items():
        t = by_id.get(eid)
        if t is None:
            continue
        good = []
        for c in cs:
            out.received += 1
            reason = ("quote_not_in_turn" if not quote_ok(c["quote"], t.text) else
                      "value_not_in_quote" if not value_in_quote(c["value"], c["quote"]) else None)
            if reason:
                out.code_dropped += 1
                _set(conn, c["id"], "dropped", reason, None)
                logrow(t, "write.check", "code", None, {"claim": _public(c)}, None, None, None, f"dropped:{reason}")
            else:
                good.append(c)
        if good:
            passed.append((t, good))

    # W4 verify (Jev, one request per turn)
    if passed:
        reqs = [({"speaker": t.speaker, "session_date": t.session_date, "turn": t.text[:MAX_TURN_CHARS],
                  "context": t.previous, "claims": [_public(c) for c in good]},
                 verify_questions([_public(c) for c in good])) for t, good in passed]
        for (t, good), (state, _), r in zip(passed, reqs, jev.ask_many(reqs)):
            for i, c in enumerate(good):
                p = {f: round(r.noul(f"c{i}_{f}"), 4) for f in ("hallucinated", "off_target", "incomplete")}
                pol, pol_probs = r.choice(f"c{i}_polarity")
                about = round(r.noul(f"c{i}_about_speaker"), 4) if not c["mention"] else None
                worst = max(p, key=lambda k: p[k])
                reason = (f"verify:{worst}" if p[worst] > VERIFY_DROP else
                          "verify:polarity" if pol != c["polarity"] and pol_probs.get(pol, 0) >= POLARITY_OVERRULE else
                          "verify:not_about_speaker" if about is not None and about < SPEAKER_AT_LEAST else None)
                drop = reason is not None
                out.verify_dropped += drop
                out.stored += not drop
                _set(conn, c["id"], "dropped" if drop else "pending", reason, p)
                logrow(t, "write.verify", "jev", r.model,
                       {"task": "W4", "speaker": state["speaker"], "session_date": state["session_date"],
                        "turn": state["turn"], "context": state["context"], "claim": state["claims"][i]},
                       {**p, "polarity": pol, "polarity_p": round(pol_probs.get(pol, 0), 4), "about_speaker": about},
                       p[worst], VERIFY_DROP, reason or "kept")
    if logs:
        with conn.cursor() as cur:
            cur.executemany("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                         probability, threshold, outcome, request_id)
                               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", logs)
    return out


def _public(c: dict) -> dict:
    return {k: c[k] for k in ("mention", "slot", "value", "time_phrase", "polarity", "quote")}


def _set(conn, cid: int, status: str, reason: str | None, p: dict | None) -> None:
    conn.execute("""update claim_candidate set status = %s, drop_reason = %s, p_hallucinated = %s, p_off_target = %s,
                                               p_incomplete = %s where id = %s""",
                 (status, reason, (p or {}).get("hallucinated"), (p or {}).get("off_target"),
                  (p or {}).get("incomplete"), cid))
