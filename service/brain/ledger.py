"""Ledger (DESIGN.md write steps 6-9; build step 5): checked claim candidates -> entities, slots and claims.

For each 'pending' claim_candidate (checked by brain/claims.py), in episode order:

1. Resolve the entity (W5). An empty mention is the speaker (the episode's speaker, or "user"). "you" is the
   person the speaker answers (the previous speaker in the session). A pronoun picks among the entities the session
   has mentioned (Jev Choice, gated by a Noul). A name that matches an entity or alias exactly links in code; any
   other name is compared with up to ALIGN_CANDIDATES look-alike entities by Jev's entity-alignment Score (levels:
   different / maybe / same; cut points 0.5 and 1.5, from the TypeSafe cookbook): >= LINK_AT links, >= REVIEW_AT
   links as tentative, below creates a new entity.
2. Place the slot (W6). Slots grow from use, per entity: the client's slot name matching an existing slot uses it;
   otherwise Jev decides whether one of the entity's slots holds this kind of fact (Noul gate plus Choice), and a
   new slot gets its cardinality from Jev (one value at a time, several at once, or dated events).
3. Relate (W7). One Jev Choice per active claim on that slot: same, updates, adds, contradicts, retracts,
   unrelated. apply_relation() is the one rule for what that does, shared with the nightly slot merge: only an
   update or a retraction ends a value, and only when the new claim starts later (valid_to = the new claim's
   start, a supersedes link); an update that starts earlier goes into history instead (a past value told late);
   an add never ends anything (both hold, exp-supersede-v1); contradicts marks both contested; same adds a
   supports link (same_event on an event slot, which never ends a value). Plans and hypotheticals never change
   what is current. Every link records the step that made it and its run.
4. Store the claim, index it as a chunk (kind 'claim', embedded by the backfill), mark the candidate 'placed'.

Every decision goes to decision_log with its input (booster tasks W5, W6, W7). Jev decides, code computes.
"""
from __future__ import annotations

import contextvars
import logging
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from . import pages

log = logging.getLogger("brain.ledger")

ALIGN_CANDIDATES = 6
LINK_AT, REVIEW_AT = 1.5, 0.5        # entity alignment cut points (TypeSafe entity-alignment cookbook)
SLOT_FIT_AT = 0.5
SLOT_PICK_AT = 0.5
# exp-page-audit-v1: the page lines the cited turn does not support are claims W6 filed under a sibling slot
# (friday_activity under sunday_activity, commute_destination under commute_start), picked with high confidence.
# With the flag on, a claim joins an existing slot under another name only when its turn supports
# "entity: that slot = value" (the page audit's own question); otherwise it keeps the slot it was proposed with.
# Off until an experiment shows it helps (the experiment log exp-place-check-v1).
PLACE_CHECK = os.environ.get("BRAIN_PLACE_CHECK", "0") == "1"
# exp-ledger-stable-v1: asked 5 times, "does the claim fit one of `slots`" plus "which" gave a different placement
# on 12% of 300 real requests (59% of the fit answers fall between 0.3 and 0.7), and the joins it made that a
# per-slot question refused were sibling or unrelated slots on hand reading. With the flag on, one Noul per
# existing slot (the slot named in the question) replaces both, and code joins the highest slot at SLOT_SAME_AT:
# 3.7% differing placements on the same requests. Off until its gate is met (the experiment log).
PLACE_PER_SLOT = os.environ.get("BRAIN_PLACE_PER_SLOT", "0") == "1"
# An eval that ingests the same data both ways in one process sets this for the thread doing the ingest
# (scripts/brain_eval.py mode `place`); None means the setting above.
PLACE_PER_SLOT_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("place_per_slot", default=None)


def _per_slot() -> bool:
    override = PLACE_PER_SLOT_OVERRIDE.get()
    return PLACE_PER_SLOT if override is None else bool(override)


# exp-relation-ends-v1: the relation Choice says "updates" for a second event, another item or a later stage as
# readily as for a replaced value: of 77 values the write path ended on LoCoMo, a hand reading found about a third
# were replaced. With the flag on, each pair also gets one Noul (has the older value stopped being true?) in the
# same request, and "updates" or "retracts" ends the older value only when that Noul reaches ENDS_AT; under it the
# new claim is stored beside the old one, as "adds" is. Off until its gate is met (the experiment log).
RELATE_ENDS = os.environ.get("BRAIN_RELATE_ENDS", "0") == "1"
RELATE_ENDS_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("relate_ends", default=None)
ENDS_AT = 0.7
ENDS_TRUE = ("The older value is no longer the case after the new statement: it was replaced by a different value, "
             "it stopped, or it was corrected (moved from Denver to Boston; two dogs, now four; was unsure, now decided).")
ENDS_FALSE = ("Both can be true: the new statement is another event or another item of the same kind (a second "
              "tournament won, a different class taken, another game played), a further detail, a later stage that "
              "does not undo the earlier one, or the same thing said again.")


def _relate_ends() -> bool:
    override = RELATE_ENDS_OVERRIDE.get()
    return RELATE_ENDS if override is None else bool(override)


# exp-relate-disputes-v1: the relation Choice says "contradicts" for two different things of one kind as readily
# as for a real conflict (the $20 mindfulness workshop and the $500 marketing workshop; a Walgreens and a CVS
# rewards membership), and both claims are then shown to the reader as disputed. Of 50 logged pairs read by hand
# about three in four could both be true. With the flag on (it needs RELATE_ENDS, whose Noul it reads), a
# "contradicts" marks the pair contested only when that Noul reaches ENDS_AT; under it both claims stand, as with
# "adds". Off until its gate is met (the experiment log).
RELATE_DISPUTES = os.environ.get("BRAIN_RELATE_DISPUTES", "0") == "1"
RELATE_DISPUTES_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("relate_disputes", default=None)


def _relate_disputes() -> bool:
    override = RELATE_DISPUTES_OVERRIDE.get()
    return RELATE_DISPUTES if override is None else bool(override)


# exp-resolve-names-v1: the alignment Score's middle level ("may be the same, or closely related") files a claim
# under the related entity as a tentative link. In the logged runs 95 such links joined two different proper names
# (Sophia under Emma, Suresh under Maria, Light Yagami under Death Note), nearly all wrong on hand reading, and
# the misfiled claim then reads as the wrong person's fact in the lookup. With the flag on, a mention that is a
# proper name is never linked tentatively to an entity known by a different proper name: it gets its own entity
# (a Score at LINK_AT or above still links), and the nightly re-check splits such a claim it cannot confirm.
# Off until its gate is met (the experiment log).
RESOLVE_NAMES = os.environ.get("BRAIN_RESOLVE_NAMES", "0") == "1"
RESOLVE_NAMES_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("resolve_names", default=None)
# DESIGN.md nightly step 1 says a link the night cannot settle "stays separate"; as built it stays attached,
# tentative. With this flag on, the nightly re-check splits every link it does not confirm. Off: not measured.
SETTLE_SPLIT = os.environ.get("BRAIN_SETTLE_SPLIT", "0") == "1"


def _resolve_names() -> bool:
    override = RESOLVE_NAMES_OVERRIDE.get()
    return RESOLVE_NAMES if override is None else bool(override)


def is_proper_name(name: str) -> bool:
    """A name as written for a person, place or title: one to three words, each starting with a capital and made
    of letters ("Emma", "Rachel Lee", "Dr. Rodriguez", "Death Note"). "my sister", "user", "the show" are not."""
    words = (name or "").strip().split()
    return 1 <= len(words) <= 3 and all(
        w[0].isupper() and w.strip(".'").replace("-", "").replace("'", "").isalpha() for w in words)


def different_names(mention: str, name: str, aliases: list[str] | None = None) -> bool:
    """Both are proper names and the mention shares no word with the entity's name or aliases."""
    if not (is_proper_name(mention) and is_proper_name(name)):
        return False
    words = lambda s: set(re.findall(r"\w+", (s or "").lower()))       # noqa: E731
    known = words(name).union(*(words(a) for a in aliases or []))
    return not (words(mention) & known)


# exp-relate-across-v1: a relational fact has two honest filings ("Priya owns the billing module" under Priya,
# "Henrik handles billing now" under Henrik, both in a slot named billing_module_owner), and W7 compares a new
# claim only with claims of its own entity and slot, so the older value stays current. With the flag on, a new
# claim is also compared with a few live claims of other entities whose slot name is near its own; Jev's relation
# and its ends Noul decide, and only a pair it judges replaced or disputed changes anything. Nothing is refiled.
# Off until its gate is met (the experiment log).
RELATE_ACROSS = os.environ.get("BRAIN_RELATE_ACROSS", "0") == "1"
ACROSS_MAX = 6            # claims of other entities shown for one new claim
ACROSS_POOL = 40          # newest claims read before the slot names are compared in code


def slot_words(name: str) -> set[str]:
    """A slot name's words, a plural s dropped (payment_terms -> {payment, term})."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", (name or "").lower())}


def near_slots(a: str, b: str) -> bool:
    """Two slot names that may name one attribute: one's words are all in the other's, or half are shared."""
    x, y = slot_words(a), slot_words(b)
    return bool(x and y) and (x <= y or y <= x or len(x & y) / len(x | y) >= 0.5)


def across_question(j: int) -> dict:
    return {"type": "choice", "criteria": RELATIONS, "instructions": (
        f"How does `new` relate to `existing[{j}]`? They are filed under different names (`about`); judge whether "
        "they state the same attribute of the same person or thing.")}


def ends_question(j: int) -> dict:
    """RELATE_ENDS: one Noul for one existing claim: did the new statement make the older value untrue?"""
    return {"type": "noul", "instructions": f"After `new`, has the older value in `existing[{j}]` stopped being true?",
            "criteria": {"true": ENDS_TRUE, "false": ENDS_FALSE}}


SLOT_SAME_AT = 0.7
PRONOUN_PICK_AT = 0.5
MAX_SLOTS_SHOWN = 60
MAX_EXISTING = 10
STATE_POLARITIES = ("asserted", "corrected", "reported", "negated")    # these change what is current

PRONOUNS = frozenset("he she they it him her them his hers their theirs its this that these those".split())
YOU = frozenset(("you", "your", "yours", "you're"))
_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)

ALIGN_LEVELS = [
    "They are two different people or things.",
    "They may be the same person or thing, or closely related ones; the description could fit either.",
    "They are one and the same person or thing.",
]
CARDINALITY = {
    "one": "One value at a time for this person or thing (a home city, a current dose, a status, a deadline); a new "
           "value replaces the old one.",
    "many": "Several values can hold at once (pets, languages, team members, open issues); a new value adds to them.",
    "event": "Dated occurrences that can be counted (trips, visits, purchases, races run, meetings held).",
}
RELATIONS = {
    "same": "The new claim states the same value again, or the same event.",
    "updates": "The new claim gives a newer value for the same thing, so the existing value no longer holds.",
    "adds": "The new claim adds another value alongside the existing one; both hold.",
    "contradicts": "The new claim disputes the existing one (a correction, or another source disagreeing).",
    "retracts": "The new claim says the existing value no longer holds, without giving a new one.",
    "unrelated": "They are about different things.",
}


@dataclass
class LedgerOutcome:
    candidates: int = 0
    placed: int = 0
    unresolved: int = 0
    entities_created: int = 0
    tentative_links: int = 0
    slots_created: int = 0
    superseded: int = 0
    history: int = 0          # claims filed straight into history: a past value told after a newer one
    contested: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def new_run_id(prefix: str) -> str:
    """An id for one run of a step that changes the ledger (w = write path, n = nightly, r = re-check)."""
    return f"{prefix}{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"


def _key(s: str) -> str:
    return _ARTICLE.sub("", re.sub(r"\s+", " ", (s or "").strip().strip(".,;:!?\"'"))).lower()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:60] or "misc"


class _Ctx:
    def __init__(self, conn, providers, request_id):
        self.conn, self.providers, self.jev, self.request_id = conn, providers, providers.jev, request_id
        self.logs: list[tuple] = []
        self.out = LedgerOutcome()

    def log(self, ns, stage, decider, model, inp, output, prob, thr, outcome):
        self.logs.append((ns, stage, decider, model, Jsonb(inp), Jsonb(output), prob, thr, outcome, self.request_id))

    def flush(self):
        if self.logs:
            with self.conn.cursor() as cur:
                cur.executemany("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                             probability, threshold, outcome, request_id)
                                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", self.logs)


# ---------------------------------------------------------------- entities

def _find_exact(conn, ns: str, name: str) -> int | None:
    k = _key(name)
    row = conn.execute("""select id from entity where namespace = %s and merged_into is null and archived_at is null
                          and (lower(name) = %s or %s = any(select lower(a) from unnest(aliases) a))
                          order by mentions desc, id limit 1""", (ns, k, k)).fetchone()
    return row[0] if row else None


def _vec(ctx: _Ctx, text: str, query: bool = False) -> str | None:
    """An embedding for entity names (vector-neighbour candidates); None without an embedder or on failure."""
    emb = getattr(ctx.providers, "embedder", None)
    if emb is None:
        return None
    try:
        v = emb.embed_query(text) if query else emb.embed_documents([text])[0]
    except Exception as e:          # CapReached or ProviderError: name candidates only
        log.warning("entity vector skipped (%s)", type(e).__name__)
        return None
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _create_entity(ctx: _Ctx, ns: str, name: str, kind: str | None = None) -> int:
    ctx.out.entities_created += 1
    return ctx.conn.execute("""insert into entity (namespace, name, aliases, kind, name_vec)
                               values (%s, %s, %s, %s, %s::vector) returning id""",
                            (ns, name.strip()[:200], [_key(name)], kind, _vec(ctx, name.strip()[:200]))).fetchone()[0]


def _named(ctx: _Ctx, ns: str, name: str, kind: str | None = None) -> int:
    return _find_exact(ctx.conn, ns, name) or _create_entity(ctx, ns, name, kind)


def _summary(conn, entity_id: int) -> str:
    rows = conn.execute("""select s.name, c.value from claim c join slot s on s.id = c.slot_id
                           join episode e on e.id = c.episode_id
                           where c.entity_id = %s and c.status = 'active'
                           order by coalesce(c.valid_from, e.ts::date) desc, e.ts desc, c.id desc
                           limit 3""", (entity_id,)).fetchall()
    return "; ".join(f"{n}: {v}" for n, v in rows)


def _look_alikes(conn, ns: str, mention: str) -> list[tuple[int, str, list[str]]]:
    words = [w for w in re.findall(r"[a-z0-9]+", _key(mention)) if len(w) >= 3]
    if not words:
        return []
    pats = [f"%{w}%" for w in words[:6]]
    return conn.execute("""select id, name, aliases from entity where namespace = %s and merged_into is null
                           and archived_at is null
                           and (lower(name) like any(%s) or exists (select 1 from unnest(aliases) a where a like any(%s)))
                           order by mentions desc, id limit %s""", (ns, pats, pats, ALIGN_CANDIDATES)).fetchall()


def _align(ctx: _Ctx, ns: str, c: dict) -> tuple[int, bool]:
    """A named mention -> (entity id, tentative)."""
    conn = ctx.conn
    exact = _find_exact(conn, ns, c["mention"])
    if exact:
        return exact, False
    cands = _look_alikes(conn, ns, c["mention"])
    qv = _vec(ctx, c["mention"], query=True)
    if qv is not None:            # vector neighbours: names that share no word ("my sister" / "Maya")
        seen = {i for i, _, _ in cands}
        cands += [r for r in conn.execute(
            """select id, name, aliases from entity where namespace = %s and merged_into is null and archived_at is null
               and name_vec is not null order by name_vec <=> %s::vector limit %s""",
            (ns, qv, ALIGN_CANDIDATES)).fetchall() if r[0] not in seen]
        cands = cands[:ALIGN_CANDIDATES + 2]
    if not cands:
        return _create_entity(ctx, ns, c["mention"]), False
    state = {"mention": {"name": c["mention"], "said_in": c["quote"]},
             "candidates": [{"name": n, "also_called": a[:5], "known": _summary(conn, i) or "-"} for i, n, a in cands]}
    qs = {f"a{j}": {"type": "score", "criteria": ALIGN_LEVELS,
                    "instructions": f"How do `mention` and `candidates[{j}]` relate as people or things in one user's memory?"}
          for j in range(len(cands))}
    r = ctx.jev.ask_many([(state, qs)])[0]
    scores = [r.score(f"a{j}")[0] for j in range(len(cands))]
    best = max(range(len(cands)), key=lambda j: scores[j])
    s = scores[best]
    outcome = "link" if s >= LINK_AT else "tentative" if s >= REVIEW_AT else "new"
    logged = {"scores": [round(x, 4) for x in scores]}
    if outcome == "tentative" and _resolve_names() and different_names(c["mention"], cands[best][1], cands[best][2]):
        outcome, logged["rule"] = "new", "different_name"
    ctx.log(ns, "write.resolve", "jev", r.model, {"task": "W5", "episode_id": c["episode_id"], **state},
            logged, s, LINK_AT, outcome)
    if outcome == "new":
        return _create_entity(ctx, ns, c["mention"]), False
    eid = cands[best][0]
    if outcome == "link":
        conn.execute("update entity set aliases = array_append(aliases, %s) where id = %s and not (%s = any(aliases))",
                     (_key(c["mention"]), eid, _key(c["mention"])))
    else:
        ctx.out.tentative_links += 1
    return eid, outcome == "tentative"


def _session_entities(conn, ns: str, session_id: str | None, before_ts, speaker_eid: int) -> list[tuple[int, str]]:
    if session_id is None:
        return []
    return conn.execute("""select distinct on (n.id) n.id, n.name from claim c join episode e on e.id = c.episode_id
                           join entity n on n.id = c.entity_id
                           where e.namespace = %s and e.session_id = %s and e.ts <= %s and n.id <> %s
                             and n.merged_into is null and n.archived_at is null
                           order by n.id, e.ts desc limit 8""", (ns, session_id, before_ts, speaker_eid)).fetchall()


def _resolve(ctx: _Ctx, c: dict) -> tuple[int | None, bool]:
    ns, conn = c["namespace"], ctx.conn
    speaker = c["speaker"] or ("user" if c["role"] == "user" else c["role"])
    speaker_eid = _named(ctx, ns, speaker, "person" if c["role"] == "user" else None)
    m = _key(c["mention"])
    if not m:
        return speaker_eid, False
    if m in YOU:
        prev = conn.execute("""select coalesce(speaker, role) from episode where namespace = %s and session_id = %s
                               and (ts, id) < (%s, %s) and coalesce(speaker, role) is distinct from %s
                               and deleted_at is null order by ts desc, id desc limit 1""",
                            (ns, c["session_id"], c["ts"], c["episode_id"], speaker)).fetchone()
        if not prev or prev[0] in ("assistant", "system", "tool", "agent_reported"):
            return None, False                    # "you" is the assistant or unknown: nothing to file it under
        return _named(ctx, ns, prev[0], "person"), False
    if m in PRONOUNS:
        cands = _session_entities(conn, ns, c["session_id"], c["ts"], speaker_eid)
        if not cands:
            return None, False
        state = {"quote": c["quote"], "pronoun": c["mention"], "earlier_turns": c["previous"],
                 "candidates": [n for _, n in cands]}
        qs = {"listed": {"type": "noul", "instructions": "Does `pronoun` in `quote` refer to one of `candidates`?"},
              "who": {"type": "choice", "instructions": "Who or what does `pronoun` in `quote` refer to?",
                      "criteria": {n: None for _, n in cands}}}
        r = ctx.jev.ask_many([(state, qs)])[0]
        listed = r.noul("listed")
        who, probs = r.choice("who")
        ok = listed >= 0.5 and probs.get(who, 0) >= PRONOUN_PICK_AT
        ctx.log(ns, "write.resolve", "jev", r.model, {"task": "W5", "episode_id": c["episode_id"], **state},
                {"listed": round(listed, 4), "who": who, "p": round(probs.get(who, 0), 4)}, listed, 0.5,
                "link" if ok else "unresolved")
        if not ok:
            return None, False
        return next(i for i, n in cands if n == who), probs.get(who, 0) < 0.8
    return _align(ctx, ns, c)


# ---------------------------------------------------------------- slots

PREFERENCE_AT = 0.5


def _kind(card: str, polarity: str, p_pref: float | None) -> str:
    if polarity in ("planned", "hypothetical"):
        return "plan"
    if card == "event":
        return "event"
    return "preference" if (p_pref or 0) >= PREFERENCE_AT else "fact"


def _place_supported(ctx: _Ctx, c: dict, entity_id: int, which: str) -> bool:
    """PLACE_CHECK: does the claim's turn support "entity: `which` = value"? One Jev request, logged as
    write.place_check. True when the flag is off, so placement is as before."""
    if not PLACE_CHECK:
        return True
    name, body = ctx.conn.execute("""select n.name, (select e.body from episode e where e.id = %s)
                                     from entity n where n.id = %s""", (c["episode_id"], entity_id)).fetchone()
    state = {"source": (body or c["quote"])[:pages.SOURCE_MAX],
             "statement": f"{name}: {which.replace('_', ' ')} = {c['value']}"}
    r = ctx.jev.ask_many([(state, {"supported": {"type": "noul", "instructions": pages.SUPPORT_QUESTION}})])[0]
    p = r.noul("supported")
    ok = p >= pages.SUPPORT_MIN
    ctx.log(c["namespace"], "write.place_check", "jev", r.model,
            {"task": "W6c", "episode_id": c["episode_id"], "statement": state["statement"], "quote": c["quote"]},
            {"supported": round(p, 4)}, p, pages.SUPPORT_MIN, f"{'kept' if ok else 'refused'}:{which}")
    return ok


def slot_same_question(name: str, latest) -> dict:
    """PLACE_PER_SLOT: one Noul for one existing slot. The slot is named in the question itself, so the answer
    does not depend on finding it in a list."""
    return {"type": "noul", "instructions": (
        f"Does `claim` record the same attribute as the existing slot '{name.replace('_', ' ')}' (latest value: "
        f"'{latest or '-'}'), so that `claim.value` would replace that value or stand next to it?")}


def _place(ctx: _Ctx, c: dict, entity_id: int) -> tuple[int, str, str]:
    """-> (slot id, cardinality, claim kind)."""
    ns, conn = c["namespace"], ctx.conn
    base = _slug(c["slot"][4:] if c["slot"].startswith("new:") else c["slot"])
    slots = conn.execute("""select s.id, s.name, s.cardinality,
                                   (select c2.value from claim c2 join episode e2 on e2.id = c2.episode_id
                                    where c2.slot_id = s.id and c2.status = 'active'
                                    order by coalesce(c2.valid_from, e2.ts::date) desc, e2.ts desc, c2.id desc
                                    limit 1)
                            from slot s where s.entity_id = %s order by s.id desc limit %s""",
                         (entity_id, MAX_SLOTS_SHOWN)).fetchall()
    for sid, name, card, _ in slots:
        if name == base:     # a known slot: the kind its earlier claims had (no Jev call)
            prev = conn.execute("""select kind from claim where slot_id = %s and kind in ('fact', 'preference', 'event')
                                   order by id desc limit 1""", (sid,)).fetchone()
            return sid, card, _kind(card, c["polarity"], 1.0 if prev and prev[0] == "preference" else 0.0)
    claim = {"slot_proposed": base, "value": c["value"], "quote": c["quote"]}
    qs = {"cardinality": {"type": "choice", "criteria": CARDINALITY, "instructions": (
        "For the kind of fact `claim.slot_proposed` names (here `claim.value`), how many values can one person or "
        "thing hold at a time?")},
        "is_preference": {"type": "noul", "instructions": (
            "Is `claim` a preference, taste, habit or standing instruction (how someone likes things done), rather "
            "than a plain fact or event?")}}
    state: dict = {"claim": claim}
    per_slot = _per_slot()
    if slots and per_slot:
        state["slots"] = [{"name": n, "latest_value": v or "-"} for _, n, _, v in slots]
        qs.update({f"s{j}": slot_same_question(n, v) for j, (_, n, _, v) in enumerate(slots)})
    elif slots:
        state["slots"] = [{"name": n, "latest_value": v or "-"} for _, n, _, v in slots]
        qs["fits"] = {"type": "noul", "instructions": (
            "Is `claim` the same kind of fact as one of `slots`, so that its value would replace or add to that "
            "slot's value (for example home_city and city_of_residence)?")}
        qs["which"] = {"type": "choice", "instructions": "Which of `slots` holds the same kind of fact as `claim`?",
                       "criteria": {n: None for _, n, _, _ in slots}}
    r = ctx.jev.ask_many([(state, qs)])[0]
    card, _ = r.choice("cardinality")
    p_pref = r.noul("is_preference")
    if slots and per_slot:
        ranked = sorted(((r.noul(f"s{j}"), -j) for j in range(len(slots))), reverse=True)   # ties: the newer slot
        top, j = ranked[0][0], -ranked[0][1]
        sid, which, scard, _ = slots[j]
        seen = {"same": round(top, 4), "which": which, "second": round(ranked[1][0], 4) if len(ranked) > 1 else None,
                "preference": round(p_pref, 4)}
        if top >= SLOT_SAME_AT and _place_supported(ctx, c, entity_id, which):
            ctx.log(ns, "write.place", "jev", r.model, {"task": "W6", "episode_id": c["episode_id"], **state},
                    seen, top, SLOT_SAME_AT, f"existing:{which}")
            return sid, scard, _kind(scard, c["polarity"], p_pref)
        ctx.out.slots_created += 1
        sid = conn.execute("""insert into slot (namespace, entity_id, name, cardinality) values (%s, %s, %s, %s)
                              on conflict (account_id, namespace, entity_id, name) do update set name = excluded.name
                              returning id, cardinality""", (ns, entity_id, base, card)).fetchone()
        ctx.log(ns, "write.place", "jev", r.model, {"task": "W6", "episode_id": c["episode_id"], **state},
                {**seen, "cardinality": card}, top, SLOT_SAME_AT, f"new:{base}")
        return sid[0], sid[1], _kind(sid[1], c["polarity"], p_pref)
    if slots:
        fits = r.noul("fits")
        which, probs = r.choice("which")
        if fits >= SLOT_FIT_AT and probs.get(which, 0) >= SLOT_PICK_AT and _place_supported(ctx, c, entity_id, which):
            sid, _, scard, _ = next(s for s in slots if s[1] == which)
            ctx.log(ns, "write.place", "jev", r.model, {"task": "W6", "episode_id": c["episode_id"], **state},
                    {"fits": round(fits, 4), "which": which, "p": round(probs.get(which, 0), 4),
                     "preference": round(p_pref, 4)}, fits, SLOT_FIT_AT, f"existing:{which}")
            return sid, scard, _kind(scard, c["polarity"], p_pref)
    ctx.out.slots_created += 1
    sid = conn.execute("""insert into slot (namespace, entity_id, name, cardinality) values (%s, %s, %s, %s)
                          on conflict (account_id, namespace, entity_id, name) do update set name = excluded.name
                          returning id, cardinality""", (ns, entity_id, base, card)).fetchone()
    ctx.log(ns, "write.place", "jev", r.model, {"task": "W6", "episode_id": c["episode_id"], **state},
            {"cardinality": card, "preference": round(p_pref, 4)}, None, SLOT_FIT_AT, f"new:{base}")
    return sid[0], sid[1], _kind(sid[1], c["polarity"], p_pref)


# ---------------------------------------------------------------- relate and store

# When a claim's value began, as SQL over claim c and its episode e: the stated date (or the day it was said), then
# the time and order it was said. An undated claim is ordered by its episode, never first or last by default.
START_ORDER = "coalesce(c.valid_from, e.ts::date), e.ts, e.id, c.id"


def starts(conn, claim_id: int) -> tuple:
    """(start date, episode time, episode id) of one claim: the key apply_relation compares."""
    return conn.execute("""select coalesce(c.valid_from, e.ts::date), e.ts, e.id
                           from claim c join episode e on e.id = c.episode_id where c.id = %s""", (claim_id,)).fetchone()


def relation_question(j: int) -> dict:
    return {"type": "choice", "criteria": RELATIONS,
            "instructions": f"How does `new` relate to `existing[{j}]` (same person or thing, same attribute)?"}


def apply_relation(conn, new_id: int, old_id: int, rel: str, cardinality: str, source: str,
                   run_id: str | None = None) -> str:
    """The one rule for what Jev's relation between two claims of a slot does to them. The write path (W7) and the
    nightly slot merge both call it, so a value is never ended any other way (room review v2, item 1a).

    - An event slot never ends a value: "same" links the two as the same event, nothing else changes.
    - "same" adds a supports link; "contradicts" marks both contested.
    - Only "updates" and "retracts" end a value, and only when `new` starts later than `old` (starts(): the stated
      date, else the day it was said, then the order it was said). "adds" means both hold, whatever the slot's
      cardinality (exp-supersede-v1: 10,241 "adds" on one-value slots against 1,609 "updates").
    - An update that starts earlier than `old` is a past value told late (a backfill): `new` goes into history,
      closed from `old`'s start, and `old` stays current. A retraction that predates `old` changes nothing.
    - Two claims of one turn with the same date cannot be ordered: both are marked contested.

    Every link carries `source` (the step that decided) and `run_id`, so one run can be found and undone.
    Returns ended | history | contested | same | none."""
    if cardinality == "event":
        if rel != "same":
            return "none"
        _link(conn, new_id, old_id, "same_event", source, run_id)
        return "same"
    if rel == "same":
        _link(conn, new_id, old_id, "supports", source, run_id)
        return "same"
    if rel not in ("updates", "retracts", "contradicts"):
        return "none"
    new_start, old_start = starts(conn, new_id), starts(conn, old_id)
    if rel == "contradicts" or new_start == old_start:
        live = conn.execute("select count(*) from claim where id = any(%s) and status in ('active', 'contested')",
                            ([old_id, new_id],)).fetchone()[0]
        if live < 2:           # one side is already history (ended earlier in this pass): nothing live to dispute,
            return "none"      # and marking the other alone would strand it as contested with no partner
        conn.execute("update claim set status = 'contested' where id = any(%s)", ([old_id, new_id],))
        _link(conn, new_id, old_id, "contradicts", source, run_id)
        return "contested"
    if new_start > old_start:
        ended = conn.execute("""update claim set status = 'superseded', valid_to = %s
                                where id = %s and status in ('active', 'contested')""", (new_start[0], old_id)).rowcount
        if not ended:
            return "none"
        _link(conn, new_id, old_id, "supersedes", source, run_id)
        release_contested(conn, old_id)
        return "ended"
    if rel == "retracts":
        return "none"
    filed = conn.execute("""update claim set status = 'superseded', valid_to = least(coalesce(valid_to, %s), %s)
                            where id = %s and status in ('active', 'contested', 'superseded')""",
                         (old_start[0], old_start[0], new_id)).rowcount
    if not filed:
        return "none"
    _link(conn, old_id, new_id, "supersedes", source, run_id)
    release_contested(conn, new_id)
    return "history"


def release_contested(conn, ended_id: int) -> int:
    """A claim is contested only while a claim it disputes is contested too. When `ended_id` stops being live
    (superseded), each claim it was in dispute with that has no other contested partner holds again. Without this
    the partner stayed contested for good: the nightly only resolves pairs with both sides contested."""
    return conn.execute(
        """update claim c set status = 'active'
           where c.status = 'contested'
             and c.id in (select case when l.from_claim = %(id)s then l.to_claim else l.from_claim end
                          from claim_link l where l.type = 'contradicts' and l.reversed_at is null
                            and %(id)s in (l.from_claim, l.to_claim))
             and not exists (select 1 from claim_link l join claim x
                                 on x.id = case when l.from_claim = c.id then l.to_claim else l.from_claim end
                             where l.type = 'contradicts' and l.reversed_at is null
                               and c.id in (l.from_claim, l.to_claim) and x.status = 'contested')""",
        {"id": ended_id}).rowcount


def _relate(ctx: _Ctx, c: dict, slot_id: int, cardinality: str, new_id: int, valid_from) -> None:
    ns, conn = c["namespace"], ctx.conn
    existing = conn.execute("""select c.id, c.value, coalesce(c.valid_from, e.ts::date), c.quote, c.polarity
                                from claim c join episode e on e.id = c.episode_id
                                where c.slot_id = %s and c.status in ('active', 'contested') and c.id <> %s
                                order by coalesce(c.valid_from, e.ts::date) desc, e.ts desc, e.id desc, c.id desc
                                limit %s""", (slot_id, new_id, MAX_EXISTING)).fetchall()
    if not existing:
        return
    state = {"new": {"value": c["value"], "polarity": c["polarity"], "date": str(valid_from), "quote": c["quote"]},
             "existing": [{"value": v, "date": str(d), "quote": q, "polarity": p} for _, v, d, q, p in existing]}
    check_ends = _relate_ends()
    gated = ("updates", "retracts", "contradicts") if _relate_disputes() else ("updates", "retracts")
    qs = {f"r{j}": relation_question(j) for j in range(len(existing))}
    if check_ends:
        qs.update({f"e{j}": ends_question(j) for j in range(len(existing))})
    r = ctx.jev.ask_many([(state, qs)])[0]
    rels, applied, ends = [], [], []
    for j, row in enumerate(existing):
        rel, _ = r.choice(f"r{j}")
        rels.append(rel)
        if check_ends:
            p_ends = r.noul(f"e{j}")
            ends.append(round(p_ends, 4))
            if rel in gated and p_ends < ENDS_AT:
                applied.append("kept_both")            # the older value was not shown to have stopped: both stand
                continue
        if c["polarity"] not in STATE_POLARITIES:
            applied.append("none")                     # a plan or a maybe never changes what is current
            continue
        did = apply_relation(conn, new_id, row[0], rel, cardinality, "write.relate", ctx.request_id)
        applied.append(did)
        ctx.out.superseded += did == "ended"
        ctx.out.history += did == "history"
        ctx.out.contested += did == "contested"
    ctx.log(ns, "write.relate", "jev", r.model, {"task": "W7", "episode_id": c["episode_id"], **state},
            {"relations": rels, "cardinality": cardinality, "applied": applied, "new_id": new_id,
             "existing_ids": [row[0] for row in existing], **({"ends": ends} if check_ends else {})},
            None, ENDS_AT if check_ends else None, ",".join(rels))


def _relate_across(ctx: _Ctx, c: dict, entity_id: int, slot_id: int, cardinality: str, new_id: int, valid_from) -> None:
    """RELATE_ACROSS: the new claim against live claims of other entities in a slot named like its own."""
    if not RELATE_ACROSS or cardinality == "event" or c["polarity"] not in STATE_POLARITIES:
        return
    ns, conn = c["namespace"], ctx.conn
    entity, slot = conn.execute("select n.name, s.name from slot s join entity n on n.id = s.entity_id where s.id = %s",
                                (slot_id,)).fetchone()
    pats = [f"%{w}%" for w in slot_words(slot) if len(w) >= 3]
    if not pats:
        return
    rows = conn.execute("""select c.id, n.name, s.name, c.value, coalesce(c.valid_from, e.ts::date), c.quote, c.polarity
                           from claim c join slot s on s.id = c.slot_id join entity n on n.id = c.entity_id
                                join episode e on e.id = c.episode_id
                           where c.namespace = %s and c.entity_id <> %s and c.id <> %s
                             and c.status in ('active', 'contested') and c.polarity = any(%s)
                             and s.cardinality <> 'event' and s.name like any(%s)
                           order by coalesce(c.valid_from, e.ts::date) desc, e.ts desc, e.id desc, c.id desc
                           limit %s""", (ns, entity_id, new_id, list(STATE_POLARITIES), pats, ACROSS_POOL)).fetchall()
    existing = [r for r in rows if near_slots(slot, r[2])][:ACROSS_MAX]
    if not existing:
        return
    item = lambda about, attr, value, date, quote, pol: {                  # noqa: E731
        "about": about, "attribute": attr.replace("_", " "), "value": value, "date": str(date), "quote": quote,
        "polarity": pol}
    state = {"new": item(entity, slot, c["value"], valid_from, c["quote"], c["polarity"]),
             "existing": [item(n, s, v, d, q, p) for _, n, s, v, d, q, p in existing]}
    qs = {f"r{j}": across_question(j) for j in range(len(existing))}
    qs.update({f"e{j}": ends_question(j) for j in range(len(existing))})
    r = ctx.jev.ask_many([(state, qs)])[0]
    rels, ends, applied = [], [], []
    for j, row in enumerate(existing):
        rel, _ = r.choice(f"r{j}")
        p_ends = r.noul(f"e{j}")
        rels.append(rel)
        ends.append(round(p_ends, 4))
        if rel not in ("updates", "retracts", "contradicts") or p_ends < ENDS_AT:
            applied.append("none")
            continue
        did = apply_relation(conn, new_id, row[0], rel, "one", "write.relate_across", ctx.request_id)
        applied.append(did)
        ctx.out.superseded += did == "ended"
        ctx.out.history += did == "history"
        ctx.out.contested += did == "contested"
    ctx.log(ns, "write.relate_across", "jev", r.model, {"task": "W7x", "episode_id": c["episode_id"], **state},
            {"relations": rels, "ends": ends, "applied": applied, "new_id": new_id,
             "existing_ids": [row[0] for row in existing]}, None, ENDS_AT, ",".join(rels))


def _link(conn, a: int, b: int, kind: str, source: str | None = None, run_id: str | None = None) -> None:
    """A link from claim a to claim b. A link that a re-check reversed comes back if the same decision is made again."""
    conn.execute("""insert into claim_link (from_claim, to_claim, type, source, run_id) values (%s, %s, %s, %s, %s)
                    on conflict (from_claim, to_claim, type) do update
                        set reversed_at = null, source = excluded.source, run_id = excluded.run_id
                        where claim_link.reversed_at is not null""", (a, b, kind, source, run_id))


def reverse_link(conn, from_claim: int, to_claim: int, kind: str) -> str:
    """Take one link back (a re-check, or a merge run that is undone). The row stays, marked reversed_at, so the
    decision behind it is never read as a label. A claim it had ended holds again unless another standing link
    still ends it; a claim it had marked disputed is current again unless another standing dispute holds it.
    Returns reopened | reversed | none (no standing link)."""
    done = conn.execute("""update claim_link set reversed_at = now()
                           where from_claim = %s and to_claim = %s and type = %s and reversed_at is null""",
                        (from_claim, to_claim, kind)).rowcount
    if not done:
        return "none"
    if kind == "supersedes":
        left = conn.execute(
            """select min(coalesce(n.valid_from, e.ts::date)) from claim_link l join claim n on n.id = l.from_claim
                      join episode e on e.id = n.episode_id
               where l.to_claim = %s and l.type = 'supersedes' and l.reversed_at is null and n.status <> 'retracted'
               having count(*) > 0""", (to_claim,)).fetchone()
        if left:                                   # another claim still ends it: only the end date may move
            conn.execute("update claim set valid_to = %s where id = %s and status = 'superseded'", (left[0], to_claim))
            return "reversed"
        back = conn.execute(
            """update claim o set valid_to = null, status = case when exists (
                          select 1 from claim_link l join claim x
                              on x.id = case when l.from_claim = o.id then l.to_claim else l.from_claim end
                          where l.type = 'contradicts' and l.reversed_at is null and o.id in (l.from_claim, l.to_claim)
                            and x.status = 'contested') then 'contested' else 'active' end
               where o.id = %s and o.status = 'superseded'""", (to_claim,)).rowcount
        return "reopened" if back else "reversed"
    if kind == "contradicts":
        back = conn.execute(
            """update claim c set status = 'active' where c.id = any(%s) and c.status = 'contested'
                 and not exists (select 1 from claim_link l where l.type = 'contradicts' and l.reversed_at is null
                                 and c.id in (l.from_claim, l.to_claim))""", ([from_claim, to_claim],)).rowcount
        return "reopened" if back else "reversed"
    return "reversed"


_UNIT_WORDS = (r"hours?|hrs?|minutes?|mins?|seconds?|days?|weeks?|months?|years?|miles?|km|kilometers?|lbs?|"
               r"pounds|kg|kilograms?|grams?|g|ml|liters?|litres?|stars?|points?|pages?|books?|episodes?|people|"
               r"times|%|percent|dollars?|euros?|usd|feet|ft|inches|in|steps|calories|kcal|mg|units?")
_AMOUNT_RE = re.compile(
    rf"(?P<cur>[$€£])\s?(?P<n1>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<mult>k|m|million|thousand)?"
    rf"|(?P<n2>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<unit>{_UNIT_WORDS})\b"
    rf"|(?P<n3>\b\d+(?:\.\d+)?\b)", re.I)
_CURRENCY = {"$": "usd", "€": "eur", "£": "gbp"}
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}


def amounts(value: str) -> list[tuple[float, str | None, str]]:
    """Every amount code finds in a value: (number, unit, the text it came from). Years and times of day (a
    four-digit 19xx/20xx with no unit, 10:30) are not amounts."""
    out = []
    for m in _AMOUNT_RE.finditer(value or ""):
        if m.group("cur"):
            n = float(m.group("n1").replace(",", "")) * _MULT.get((m.group("mult") or "").lower(), 1)
            out.append((n, _CURRENCY[m.group("cur")], m.group(0)))
        elif m.group("n2"):
            u = m.group("unit").lower()
            u = {"hrs": "hours", "hr": "hours", "hour": "hours", "min": "minutes", "mins": "minutes", "minute": "minutes",
                 "%": "percent", "lb": "pounds", "lbs": "pounds", "dollar": "usd", "dollars": "usd"}.get(u, u)
            out.append((float(m.group("n2").replace(",", "")), u, m.group(0)))
        else:
            s = m.group("n3")
            before, after = value[max(0, m.start() - 1):m.start()], value[m.end():m.end() + 1]
            if re.fullmatch(r"(19|20)\d\d", s) or before in (":", "/", "-") or after in (":", "/", "-"):
                continue
            out.append((float(s), None, s))
    return out


def _amount(ctx: _Ctx, c: dict, slot_name: str) -> tuple[float | None, str | None]:
    found = amounts(c["value"])
    if not found:
        return None, None
    if len(found) == 1:
        return found[0][0], found[0][1]
    opts = {f"a{k}": f"{t}" for k, (_, _, t) in enumerate(found[:20])}
    r = ctx.jev.ask_many([({"fact": slot_name.replace("_", " "), "value": c["value"], "quote": c["quote"]}, {
        "has": {"type": "noul", "instructions": "Is one of the amounts in `value` the value of `fact` itself?"},
        "which": {"type": "choice", "instructions": "Which amount in `value` is the value of `fact`?",
                  "criteria": opts}})])[0]
    which, probs = r.choice("which")
    if r.noul("has") < 0.5 or probs.get(which, 0) < 0.5:
        return None, None
    n, u, _ = found[int(which[1:])]
    return n, u


def _valid_from(conn, c: dict):
    if c["time_phrase"]:
        row = conn.execute("""select date_from from time_mention where episode_id = %s and lower(span) = lower(%s)
                              and date_from is not null limit 1""", (c["episode_id"], c["time_phrase"])).fetchone()
        if row:
            return row[0]
    return c["ts"].date()


# ---------------------------------------------------------------- read side: the current and history routes

ROSTER_MAX = 40
LOOKUP_GATE = 0.5
LOOKUP_PICK = 0.4
LOOKUP_PICK_SIDE = 0.7   # exp-locomo-brain-v1: on open questions a loose slot pick (< 0.7) misled more than it helped


def lookup(conn, providers, ns_list: list[str], question: str, question_date, route: str,
           decisive: bool = True) -> dict | None:
    """The current and history routes (DESIGN.md read path): Jev picks the entity the question is about (from the
    namespace's most-mentioned entities) and then its slot; code reads the ledger as of the question date. Returns
    {entity, slot, rows, block} or None when the question names nothing the ledger holds (the open route answers
    alone). Two Jev requests; raises the provider's error. `decisive` is False when the router did not ask for a
    current value or a history (the lookup runs as a side check on open and when questions): then the pick must be
    surer and the block says the excerpts may hold more (exp-locomo-brain-v1: "checked facts ... current value"
    narrowed descriptive answers to the ledger's one value)."""
    pick_at = LOOKUP_PICK if decisive else LOOKUP_PICK_SIDE
    ents = conn.execute("""select id, name, (select string_agg(s.name, ', ' order by s.id desc) from (
                                   select name, id from slot where entity_id = e.id order by id desc limit 8) s)
                           from entity e where namespace = any(%s) and merged_into is null and archived_at is null
                           order by mentions desc, id limit %s""", (ns_list, ROSTER_MAX)).fetchall()
    if not ents:
        return None
    crit = {}
    for _, name, slots in ents:
        if name not in crit:
            crit[name] = ("The user themself (I, me, my)" if name == "user" else "") + \
                         (f"; known: {slots}" if slots else "")
    # Jev answers each question on its own, so anything a question refers to goes in the shared state.
    state = {"question": question, "question_date": str(question_date),
             "roster": [{"name": k, "about": v or "-"} for k, v in crit.items()]}
    r = providers.jev.ask_many([(state, {
        "names": {"type": "noul", "instructions": (
            "Does `question` ask about one of the people or things in `roster` (the user counts when the question "
            "says I, me or my)?")},
        "who": {"type": "choice", "instructions": "Which person or thing in `roster` is `question` about?",
                "criteria": {k: (v or None) for k, v in crit.items()}}})])[0]
    who, wp = r.choice("who")
    if r.noul("names") < LOOKUP_GATE or wp.get(who, 0) < pick_at:
        return None
    eid = next(i for i, n, _ in ents if n == who)
    slots = conn.execute("""select s.id, s.name, s.cardinality,
                                   (select c.value from claim c join episode e on e.id = c.episode_id
                                    where c.slot_id = s.id and c.status = 'active'
                                    order by coalesce(c.valid_from, e.ts::date) desc, e.ts desc, c.id desc
                                    limit 1)
                            from slot s where s.entity_id = %s order by s.id desc limit %s""",
                         (eid, MAX_SLOTS_SHOWN)).fetchall()
    if not slots:
        return None
    state2 = {"question": question, "about": who,
              "facts": [{"name": n, "latest": v or "-"} for _, n, _, v in slots]}
    r2 = providers.jev.ask_many([(state2, {
        "held": {"type": "noul", "instructions": (
            "Does one of `facts` hold what `question` asks for, or an earlier or later value of it?")},
        "which": {"type": "choice", "instructions": f"Which of `facts` about {who} does `question` ask for?",
                  "criteria": {n: (f"latest: {v}" if v else None) for _, n, _, v in slots}}})])[0]
    which, sp = r2.choice("which")
    if r2.noul("held") < LOOKUP_GATE or sp.get(which, 0) < pick_at:
        return None
    sid, sname, card, _ = next(s for s in slots if s[1] == which)
    rows = conn.execute(f"""select c.value, c.valid_from, c.valid_to, c.status, c.polarity, c.quote
                            from claim c join episode e on e.id = c.episode_id
                            where c.slot_id = %s and c.status <> 'retracted' order by {START_ORDER}""",
                        (sid,)).fetchall()
    qd = question_date
    if route == "current":
        cur = [x for x in rows if x[4] in STATE_POLARITIES and x[3] in ("active", "contested", "superseded")
               and (x[1] is None or qd is None or x[1] <= qd) and (x[2] is None or qd is None or x[2] > qd)]
        if not cur:                              # nothing in force at that date: the latest known instead
            cur = [x for x in rows if x[4] in STATE_POLARITIES][-1:]
        shown = cur if card != "one" else cur[-1:]
    else:
        shown = rows
    if not shown:                      # the slot holds nothing usable: no lookup (the facts search still runs)
        return None
    lines = []
    for v, vf, vt, st, pol, q in shown:
        span = f"from {vf}" + (f" to {vt}" if vt else "")
        tag = "" if st == "active" and pol in STATE_POLARITIES else f" [{st if st != 'active' else pol}]"
        lines.append(f"- {v} ({span}){tag}: \"{q[:240]}\"")
    head = (f"Memory ledger, checked facts about {who}, {sname.replace('_', ' ')} "
            f"({'current value' if route == 'current' else 'history, oldest first'}):") if decisive else (
        f"Memory ledger, a related fact the brain keeps about {who}, {sname.replace('_', ' ')} (oldest first; the "
        "excerpts above are the main evidence and may hold more detail):")
    return {"entity": who, "slot": sname, "cardinality": card, "rows": len(shown), "block": "\n".join([head, *lines]),
            "picks": {"entity_p": round(wp.get(who, 0), 4), "slot_p": round(sp.get(which, 0), 4)}}


# claims from each side of the hybrid search. exp-gather-top-v1: on count questions answered wrong with evidence
# missing, the list hit this limit on 19 of 34 while the whole ledger held a few hundred claims; a wider limit is
# measured there. 40 until its gate is met (the experiment log).
GATHER_TOP = int(os.environ.get("BRAIN_GATHER_TOP", "40"))
GATHER_BATCH = 15         # items per Jev request
GATHER_KEEP = 0.5
GATHER_TAIL = 5           # a kept item this close to the end of a full side: the search may have cut the list short


_GATHER_SQL = """
with q as (
    select case when count(*) = 0 then null
                else to_tsquery('simple', string_agg(quote_literal(x), ' | ')) end as tsq
    from unnest(tsvector_to_array(to_tsvector('english', %(q)s))) x
),
near as (
    select k.ref_id, k.embedding <=> %(qv)s::vector as dist from chunk k
    where k.kind = 'claim' and k.namespace = any(%(ns)s) and %(qv)s::vector is not null and k.embedding is not null
      and not k.cold
    order by k.embedding <=> %(qv)s::vector limit %(top)s
),
dense as (select ref_id, row_number() over (order by dist, ref_id) as rk from near),
hits as (
    select k.ref_id, ts_rank_cd(k.tsv, q.tsq) as score, k.id from chunk k cross join q
    where k.kind = 'claim' and k.namespace = any(%(ns)s) and q.tsq is not null and k.tsv @@ q.tsq
    order by ts_rank_cd(k.tsv, q.tsq) desc, k.id limit %(top)s
),
kw as (select ref_id, row_number() over (order by score desc, id) as rk from hits)
select c.id, n.name, s.name, s.cardinality, c.value, c.valid_from, c.quote, c.polarity, c.value_norm, c.entity_id,
       c.status, c.valid_to, (select min(d.rk) from dense d where d.ref_id = c.id),
       (select min(w.rk) from kw w where w.ref_id = c.id), (select count(*) from dense), (select count(*) from kw)
from claim c join slot s on s.id = c.slot_id join entity n on n.id = c.entity_id
     join episode e on e.id = c.episode_id
where c.id in (select ref_id from dense union select ref_id from kw)
  and c.status = any(%(statuses)s) and c.polarity <> 'negated' and n.archived_at is null
order by coalesce(c.valid_from, e.ts::date), e.ts, e.id, c.id
"""


LINES_MAX = 250           # a Choice takes at most 255 options
LINE_EXISTS_AT = 0.35     # the line-by-line search cookbook's lower threshold


def assistant_lines(conn, providers, ns_list: list[str], question: str, items: list[dict]) -> dict | None:
    """The assistant route (DESIGN.md read path; the line-by-line search cookbook): what did the assistant say?
    The sessions behind the top excerpts are reopened; code numbers the assistant's lines; Jev answers whether one
    of them answers the question (Noul) and which (Choice over the line ids); the pick comes back with its
    neighbours, in the assistant's own words."""
    sessions = list(dict.fromkeys(it["session_id"] for it in items if it.get("session_id")))[:3]
    if not sessions:
        return None
    rows = conn.execute("""select session_id, ts, body from episode where namespace = any(%s) and session_id = any(%s)
                           and role = 'assistant' and deleted_at is null order by ts, id""",
                        (ns_list, sessions)).fetchall()
    lines = []
    for _sid, ts, body in rows:
        for ln in body.splitlines():
            if ln.strip():
                lines.append((f"L{len(lines) + 1}", str(ts.date()), ln.strip()))
            if len(lines) >= LINES_MAX:
                break
    if not lines:
        return None
    state = {"question": question, "lines": [{"id": i, "date": d, "text": t[:300]} for i, d, t in lines]}
    r = providers.jev.ask(state, {
        "exists": {"type": "noul", "instructions": "Does one of `lines` (the assistant's earlier words) answer `question`?"},
        "which": {"type": "choice", "instructions": "Which of `lines` answers `question`?",
                  "criteria": {i: None for i, _, _ in lines}}})
    exists = r.noul("exists")
    which, _ = r.choice("which")
    if exists < LINE_EXISTS_AT:
        return None
    k = next(n for n, (i, _, _) in enumerate(lines) if i == which)
    shown = lines[max(0, k - 1):k + 3]
    return {"line": which, "exists": round(exists, 4), "block": "\n".join(
        ["The assistant's own earlier words that answer the question (the answering line marked >):",
         *[f"{'>' if i == which else ' '} [{d}] {t}" for i, d, t in shown]])}


FACTS_SHOWN = 8


def _query_vec(providers, question: str) -> str | None:
    emb = getattr(providers, "embedder", None)
    if emb is None:
        return None
    try:
        qv = emb.embed_query(question)
    except Exception as e:              # CapReached or ProviderError: keyword side only
        log.warning("ledger search without the dense side (%s)", type(e).__name__)
        return None
    return "[" + ",".join(f"{x:.6f}" for x in qv) + "]"


def facts(conn, providers, ns_list: list[str], question: str, question_date) -> dict | None:
    """Facts search (the current and history routes' hedge): the exact lookup needs the right entity and slot, and
    a client can file one fact under two of them (exp-slices-v1: the F-150 filed apart from the Mustang). So the
    ledger's claims are also searched like passages (hybrid over claim chunks, superseded ones included), Jev keeps
    those that state a value of what the question asks (now or earlier), and code lists them newest first with
    their status, so the newest value leads and a replaced one shows as replaced."""
    rows = conn.execute(_GATHER_SQL, {"q": question, "ns": ns_list, "qv": _query_vec(providers, question),
                                      "top": GATHER_TOP, "statuses": ["active", "contested", "superseded"]}).fetchall()
    if not rows:
        return None
    items = [{"about": n, "fact": s.replace("_", " "), "value": v, "date": str(d), "quote": q[:240]}
             for _, n, s, _, v, d, q, *_ in rows]
    reqs = [({"question": question, "question_date": str(question_date), "items": items[b:b + GATHER_BATCH]},
             {f"f{j}": {"type": "noul", "instructions": (
                 f"Does `items[{j}]` state a value of what `question` asks about (the current one or an earlier one)?")}
              for j in range(len(items[b:b + GATHER_BATCH]))}) for b in range(0, len(items), GATHER_BATCH)]
    keep = []
    for b, r in enumerate(providers.jev.ask_many(reqs)):
        for j in range(len(reqs[b][0]["items"])):
            if r.noul(f"f{j}") >= GATHER_KEEP:
                keep.append(rows[b * GATHER_BATCH + j])
    if not keep:
        return None
    keep.sort(key=lambda r: (str(r[5]), r[0]), reverse=True)
    lines = []
    for _, n, s, _, v, d, q, p, _, _, st, *_ in keep[:FACTS_SHOWN]:
        tag = "" if st == "active" and p in STATE_POLARITIES else f" [{st if st != 'active' else p}]"
        lines.append(f"- {d}: {v} ({n}, {s.replace('_', ' ')}){tag}: \"{q[:200]}\"")
    return {"facts": len(keep), "block": "\n".join(
        ["Memory ledger, checked facts that bear on the question, newest first (a newer value replaces an older "
         "one unless both are current):", *lines])}


def _standing(status: str, valid_to) -> str:
    """How a gathered claim stands today, in words Jev and the reader's model both see."""
    if status == "superseded":
        return f"replaced {valid_to}" if valid_to else "replaced"
    return "disputed" if status == "contested" else "current"


# exp-count-style-v1: the count the block leads with equalled the gold on 2 of 61 count questions, and what the
# block was measured to add is evidence the excerpts missed (exp-count-block-v1). "evidence" presents the same
# lines as further statements, gives no number of items, and says of the sum what it is a sum of. "count" is the
# block as built. Off ("count") until its gate is met (the experiment log).
COUNT_STYLES = ("count", "evidence")
COUNT_STYLE = os.environ.get("BRAIN_COUNT_STYLE", "count")


def gather(conn, providers, ns_list: list[str], question: str, question_date, style: str | None = None,
           top: int | None = None) -> dict | None:
    """The count route (DESIGN.md read path, build step 6): the claims a "how many" or "list" question gathers.
    Hybrid search over claim chunks, replaced values included (each item says how it stands, so a question about
    the past can count them and a question about now does not); one Jev Noul per candidate ("is this one of the
    things the question counts, in its time frame?"); then code merges the same thing told twice and counts.

    Two claims are the same thing when a same_event or supports link joins them, or when they carry the same value
    on the same entity. An event must also share the date: two 5k runs on different days are two runs (room review
    v2, item 1c). Returns {count, items, total, capped, block} or None if nothing matches. `capped` says the search
    may have cut the list short: a side came back full and a kept item sat among its last GATHER_TAIL. `style`
    (COUNT_STYLES) changes the wording of the block only; the returned numbers are the same."""
    style = style if style in COUNT_STYLES else (COUNT_STYLE if COUNT_STYLE in COUNT_STYLES else "count")
    top = top or GATHER_TOP          # candidates a side; an eval arm may pass its own
    rows = conn.execute(_GATHER_SQL, {"q": question, "ns": ns_list, "qv": _query_vec(providers, question),
                                      "top": top, "statuses": ["active", "contested", "superseded"]}).fetchall()
    if not rows:
        return None
    items = [{"about": n, "fact": s.replace("_", " "), "value": v, "date": str(d), "said_as": p,
              "status": _standing(st, vt), "quote": q[:240]}
             for _, n, s, _, v, d, q, p, _, _, st, vt, *_ in rows]
    reqs = []
    for b in range(0, len(items), GATHER_BATCH):
        part = items[b:b + GATHER_BATCH]
        reqs.append(({"question": question, "question_date": str(question_date), "items": part},
                     {f"i{j}": {"type": "noul", "instructions": (
                         f"Is `items[{j}]` one of the things `question` asks to count or list? Judge by what it is, "
                         "who it belongs to, and the question's time frame against `question_date` and the item's "
                         "date; a plan counts only if the question asks about plans, and an item whose status says "
                         "replaced counts only if the question asks about the past or about everything over time.")}
                      for j in range(len(part))}))
    keep = []
    for b, r in enumerate(providers.jev.ask_many(reqs)):
        for j in range(len(reqs[b][0]["items"])):
            if r.noul(f"i{j}") >= GATHER_KEEP:
                keep.append(b * GATHER_BATCH + j)
    if not keep:
        return None
    ids = [rows[i][0] for i in keep]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in conn.execute("""select from_claim, to_claim from claim_link where type in ('same_event', 'supports')
                                and reversed_at is null and from_claim = any(%s) and to_claim = any(%s)""",
                             (ids, ids)).fetchall():
        parent[find(a)] = find(b)
    told: dict[tuple, list[int]] = {}           # (entity, value) -> the kept rows that carry it
    for i in keep:
        cid, _, _, card, _, d, _, _, vnorm, eid, *_ = rows[i]
        for k in told.setdefault((eid, vnorm), []):
            either_event = "event" in (card, rows[k][3])
            if not either_event or (d is not None and d == rows[k][5]):
                parent[find(cid)] = find(rows[k][0])
                break
        told[(eid, vnorm)].append(i)
    groups: dict[int, list[int]] = {}
    for i in keep:
        groups.setdefault(find(rows[i][0]), []).append(i)
    lines, shown = [], []
    for members in groups.values():
        # shown as it stands now: a member still in force speaks for the group, else the latest replaced one
        i = next((m for m in members if rows[m][10] == "active"), members[-1])
        shown.append(i)
        _, n, s, _, v, d, q, p, _, _, st, vt, *_ = rows[i]
        extra = f" (told {len(members)} times)" if len(members) > 1 else ""
        plan = f" [{p}]" if p not in STATE_POLARITIES else ""
        stand = "" if st == "active" else f" [{_standing(st, vt)}]"
        lines.append(f"- {v} ({n}, {s.replace('_', ' ')}, {d}){plan}{stand}{extra}: \"{q[:200]}\"")
    dense_n, kw_n = rows[0][14], rows[0][15]
    capped = any((dense_n >= top and rows[i][12] is not None and rows[i][12] > top - GATHER_TAIL)
                 or (kw_n >= top and rows[i][13] is not None and rows[i][13] > top - GATHER_TAIL)
                 for i in keep)
    head = (f"Memory ledger, items matching the question (code-counted, the same item told twice merged): "
            f"{len(groups)}") if style == "count" else (
        "Memory ledger, further statements of the user's that bear on the question (they add to the excerpts above; "
        "a statement may give a number itself, and the list may miss things or hold a near miss, so count or add "
        "from what the excerpts and these statements say):")
    # The total adds the amount of the claim shown on each line, and only lines that still hold: a value that was
    # replaced is the same quantity at an earlier time, so adding it to its replacement counts it twice.
    reps = [rows[i][0] for i in shown if rows[i][10] != "superseded"]
    left_out = len(shown) - len(reps)
    amts = conn.execute("select id, amount, unit from claim where id = any(%s)", (reps,)).fetchall()
    units = {u for _, a, u in amts if a is not None}
    total = None
    if len(amts) >= 2 and all(a is not None for _, a, _ in amts) and len(units) == 1:
        total = float(sum(a for _, a, _ in amts))
        unit = next(iter(units)) or ""
        which = "the amounts above that were not replaced" if left_out else "the amounts above"
        if style == "count":
            lines.append(f"Total of {which} (code-computed): {total:g} {unit}".rstrip())
        else:
            lines.append(f"Sum of {which} (code-added; the answer only if the question asks for all of them "
                         f"together): {total:g} {unit}".rstrip())
    if capped:
        lines.append(f"The ledger search stops at {top} candidates a side and matches reached that limit, so "
                     "there may be more than the items above" +
                     (": treat the count as a minimum." if style == "count" else "."))
    return {"count": len(groups), "items": len(keep), "total": total, "capped": capped,
            "block": "\n".join([head, *lines])}


_COLS = ("id", "namespace", "episode_id", "mention", "slot", "value", "time_phrase", "polarity", "quote", "confidence",
         "speaker", "role", "session_id", "ts", "previous")


def process(conn, providers, episode_ids: list[int], request_id: str | None = None) -> LedgerOutcome:
    """Place this batch's checked candidates (inside the caller's tenant transaction). Raises CapReached or
    ProviderError from Jev; the caller's transaction then rolls back and the job is retried."""
    ctx = _Ctx(conn, providers, request_id)
    rows = conn.execute(
        """select k.id, k.namespace, k.episode_id, k.mention, k.slot, k.value, k.time_phrase, k.polarity, k.quote,
                  least(1 - coalesce(k.p_hallucinated, 0), 1 - coalesce(k.p_off_target, 0)),
                  e.speaker, e.role, e.session_id, e.ts,
                  array(select coalesce(p.speaker, p.role) || ': ' || left(p.body, 300) from episode p
                        where p.namespace = e.namespace and p.session_id = e.session_id and (p.ts, p.id) < (e.ts, e.id)
                          and p.deleted_at is null order by p.ts desc, p.id desc limit 2)
           from claim_candidate k join episode e on e.id = k.episode_id
           where k.episode_id = any(%s) and k.status = 'pending' and e.deleted_at is null
           order by e.ts, e.id, k.id""", (episode_ids,)).fetchall()
    for row in rows:
        c = dict(zip(_COLS, row))
        ctx.out.candidates += 1
        entity_id, tentative = _resolve(ctx, c)
        if entity_id is None:
            ctx.out.unresolved += 1
            conn.execute("update claim_candidate set drop_reason = 'ledger:unresolved_mention' where id = %s", (c["id"],))
            continue
        slot_id, cardinality, kind = _place(ctx, c, entity_id)
        vf = _valid_from(conn, c)
        slot_name0 = conn.execute("select name from slot where id = %s", (slot_id,)).fetchone()[0]
        amount, unit = _amount(ctx, c, slot_name0)
        new_id = conn.execute(
            """insert into claim (namespace, entity_id, slot_id, value, value_norm, valid_from, status, confidence,
                                  quote, episode_id, candidate_id, polarity, time_phrase, tentative, kind, amount, unit)
               values (%s, %s, %s, %s, %s, %s, 'active', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) returning id""",
            (c["namespace"], entity_id, slot_id, c["value"], _key(c["value"]), vf, c["confidence"], c["quote"],
             c["episode_id"], c["id"], c["polarity"], c["time_phrase"], tentative, kind, amount, unit)).fetchone()[0]
        _relate(ctx, c, slot_id, cardinality, new_id, vf)
        _relate_across(ctx, c, entity_id, slot_id, cardinality, new_id, vf)
        ent =conn.execute("update entity set mentions = mentions + 1, last_seen = greatest(last_seen, %s) "
                           "where id = %s returning name", (c["ts"], entity_id)).fetchone()[0]
        pages.mark_dirty(conn, c["namespace"], entity_id)            # step 8: re-rendered nightly
        slot_name = conn.execute("select name from slot where id = %s", (slot_id,)).fetchone()[0]
        conn.execute("""insert into chunk (namespace, kind, ref_id, text) values (%s, 'claim', %s, %s)""",
                     (c["namespace"], new_id, f"[{vf}] {ent}: {slot_name.replace('_', ' ')} = {c['value']}"
                                              f" (\"{c['quote'][:300]}\")"))
        conn.execute("update claim_candidate set status = 'placed' where id = %s", (c["id"],))
        ctx.out.placed += 1
    ctx.flush()
    return ctx.out
