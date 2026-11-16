"""Preferences and directives (DESIGN.md nightly step 4 and the preference route; build step 7).

Nightly, per account (promote): active preference claims with no live directive are judged by Jev (the booster's
N3 task: a standing preference that should shape future answers, or a one-off wish?); at or above PROMOTE_AT the
claim becomes a directive whose text is the user's own quoted words, served to every client at session start.
A learned directive whose claim no longer holds (superseded or forgotten) is retired. Directives a client added by
hand are never touched.

At read time (pack): the preference route gets the live directives plus the preference claims Jev judges would
change the answer to this question, as a "preferences to apply" block for the user's model.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from .providers import ProviderError
from .questions import CONTAINS_PROMPT_INJECTION, SUBVERT_THRESHOLD, SUBVERTS_USER
from .spend import CapReached

log = logging.getLogger("brain.preferences")

PROMOTE_AT = 0.7
APPLY_AT = 0.7       # exp-slices-v2: at 0.5 loosely related standing preferences pulled answers off
LEARNED_BY = "brain:nightly"
DIRECTIVE_RULE = "directive-v1"     # rejected when subverts_user fires; the injection Noul is recorded for the user
SCREEN_CHARS = 2000
BATCH = 15
MAX_CANDIDATES = 200


def screen_texts(jev, texts: list[str]) -> list[dict | None]:
    """Screen directive texts with the injection and subverts_user Nouls (room review v2, group 2). A directive is
    an instruction to the assistant by design, so the injection Noul fires on honest ones ("always answer in
    metric units"); it is recorded for the user to see when confirming. A text is rejected when subverts_user
    fires (it works against the user). None per text when Jev cannot be asked: the directive stays pending and
    unscreened, which the user sees before confirming."""
    if not texts:
        return []
    try:
        results = jev.ask_many([({"candidate_passage": t[:SCREEN_CHARS]},
                                 {"contains_prompt_injection": CONTAINS_PROMPT_INJECTION,
                                  "subverts_user": SUBVERTS_USER}) for t in texts])
    except (CapReached, ProviderError) as e:
        log.warning("directive screen skipped (%s)", type(e).__name__)
        return [None] * len(texts)
    out = []
    for r in results:
        inj, sub = r.noul("contains_prompt_injection"), r.noul("subverts_user")
        out.append({"injection": round(inj, 4), "subverts_user": round(sub, 4), "flagged": sub > SUBVERT_THRESHOLD,
                    "rule": DIRECTIVE_RULE, "model": getattr(r, "model", None)})
    return out


def promote(conn, providers, request_id: str | None = None) -> dict:
    """One account's nightly promotion (inside its tenant transaction). Since migration 014 a promoted preference
    becomes a PENDING directive: unpinned, scoped to the client that captured the turn, with its screen recorded.
    It takes effect only when the user confirms it (core.confirm_directive). A text the screen rejects is filed
    retired ('screened_out'), so it is not asked about again."""
    retired = conn.execute(
        """update directive d set retired_at = now()
           where d.created_by = %s and d.retired_at is null and d.source_claim is not null
             and not exists (select 1 from claim c where c.id = d.source_claim and c.status = 'active')""",
        (LEARNED_BY,)).rowcount
    rows = conn.execute(
        """select c.id, c.namespace, n.name, s.name, c.value, c.quote, c.valid_from, e.client
           from claim c join entity n on n.id = c.entity_id join slot s on s.id = c.slot_id
                join episode e on e.id = c.episode_id
           where c.kind = 'preference' and c.status = 'active' and c.polarity in ('asserted', 'corrected')
             and e.role = 'user'
             and n.archived_at is null
             and not exists (select 1 from directive d where d.source_claim = c.id)
           order by c.id limit %s""", (MAX_CANDIDATES,)).fetchall()
    promoted = screened_out = 0
    if rows:
        items = [{"about": n, "fact": s.replace("_", " "), "value": v, "quote": q[:300], "date": str(d)}
                 for _, _, n, s, v, q, d, _ in rows]
        reqs = [({"items": items[b:b + BATCH]},
                 {f"p{j}": {"type": "noul", "instructions": (
                     f"Is `items[{j}]` a standing preference or instruction that should shape how an assistant "
                     "answers this person in future conversations, rather than a one-off wish, a passing remark or a "
                     "plain fact?")} for j in range(len(items[b:b + BATCH]))})
                for b in range(0, len(items), BATCH)]
        logs, standing = [], []
        for b, r in enumerate(providers.jev.ask_many(reqs)):
            for j, it in enumerate(reqs[b][0]["items"]):
                cid, ns, *_, client = rows[b * BATCH + j]
                p = r.noul(f"p{j}")
                ok = p >= PROMOTE_AT
                logs.append((ns, "nightly.promote", "jev", r.model, Jsonb({"task": "N3", "claim_id": cid, "item": it}),
                             Jsonb({"standing": round(p, 4)}), p, PROMOTE_AT, "promoted" if ok else "kept", request_id))
                if ok:
                    text = f"{it['about']}: \"{it['quote']}\"" if it["about"] != "user" else f"\"{it['quote']}\""
                    standing.append((cid, ns, client, text))
        for (cid, ns, client, text), screen in zip(standing, screen_texts(providers.jev, [s[3] for s in standing])):
            bad = bool(screen and screen["flagged"])
            conn.execute("""insert into directive (namespace, scope, text, source_claim, pinned, created_by, status,
                                                   screen, retired_at, retired_reason)
                            values (%s, %s, %s, %s, false, %s, %s, %s, %s, %s)""",
                         (ns, client, text, cid, LEARNED_BY, "rejected" if bad else "pending",
                          Jsonb(screen) if screen else None, datetime.now(timezone.utc) if bad else None,
                          "screened_out" if bad else None))
            if screen:
                logs.append((ns, "directive.screen", "jev", screen["model"], Jsonb({"claim_id": cid}),
                             Jsonb({"injection": screen["injection"], "rule": DIRECTIVE_RULE}),
                             screen["subverts_user"], SUBVERT_THRESHOLD, "rejected" if bad else "pending", request_id))
            promoted += not bad
            screened_out += bad
        with conn.cursor() as cur:
            cur.executemany("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                         probability, threshold, outcome, request_id)
                               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", logs)
    return {"candidates": len(rows), "promoted": promoted, "pending": promoted, "screened_out": screened_out,
            "retired": retired}


APPLY_MAX = 4      # a few relevant preferences help; a dump of every one pulls the answer off the question



def _quote_of(cand) -> str:
    """A candidate's quoted words, whitespace-normalized (a directive's text is `"quote"` or `name: "quote"`)."""
    kind, c, _ = cand
    q = c["quote"] if kind == "claim" else c.split('"', 1)[-1]
    return " ".join(q.strip('"').split())


def pack(conn, providers, ns_list: list[str], client: str, question: str,
         seen: list[str] | None = None) -> dict | None:
    """The preference route's block: the directives and preference claims Jev judges would change this answer,
    most relevant first, at most APPLY_MAX. Every directive is served at session start anyway; here only the ones
    that bear on the question (exp-slices spike: ten unrelated directives pulled an answer off its question).
    Claims that already back a live directive are not listed twice, and nothing whose quote is already in the
    excerpts (`seen`) is repeated: the excerpt carries it with its context. The block is supporting context after
    the excerpts, not an order to apply (exp-slices-v2: "answer using them" made the reader apply generic
    preferences over the specific experiences the excerpts held). A directive is offered only to the clients it
    is for: scope 'all' or the calling `client`, the same rule as session start."""
    directives = conn.execute(
        """select text, source_claim from directive
           where namespace = any(%s) and retired_at is null and pinned and status = 'confirmed'
             and scope in ('all', %s)
           order by created_at, id limit 30""", (ns_list, client)).fetchall()
    backed = {sc for _, sc in directives if sc is not None}
    rows = [r for r in conn.execute(
        """select c.id, n.name, s.name, c.value, c.quote, coalesce(c.valid_from, e.ts::date) from claim c
           join entity n on n.id = c.entity_id join slot s on s.id = c.slot_id
           join episode e on e.id = c.episode_id
           where c.namespace = any(%s) and c.kind = 'preference' and c.status = 'active' and n.archived_at is null
             and c.polarity in ('asserted', 'corrected', 'reported')
           order by coalesce(c.valid_from, e.ts::date) desc, e.ts desc, c.id desc limit 60""", (ns_list,)).fetchall() if r[0] not in backed]
    cands = ([("standing", t, f"- standing: {t}") for t, _ in directives]
             + [("claim", {"about": n, "fact": s.replace("_", " "), "value": v, "quote": q[:240]},
                 f"- {n if n != 'user' else 'the user'}, {s.replace('_', ' ')}: {v} ({d}): \"{q[:200]}\"")
                for _, n, s, v, q, d in rows])
    if seen:
        body = " ".join(" ".join(t.split()) for t in seen)
        cands = [c for c in cands if _quote_of(c)[:120] not in body]
    if not cands:
        return None
    items = [{"standing_directive": c} if k == "standing" else c for k, c, _ in cands]
    reqs = [({"question": question, "items": items[b:b + BATCH]},
             {f"a{j}": {"type": "noul", "instructions": (
                 f"Is `items[{j}]` about the specific thing `question` asks for, so that a good answer to it should "
                 "take it into account? A general habit, or a preference about a different activity or topic, is "
                 "not.")}
              for j in range(len(items[b:b + BATCH]))}) for b in range(0, len(items), BATCH)]
    scored = []
    for b, r in enumerate(providers.jev.ask_many(reqs)):
        for j in range(len(reqs[b][0]["items"])):
            p = r.noul(f"a{j}")
            if p >= APPLY_AT:
                scored.append((p, b * BATCH + j))
    if not scored:
        return None
    top = [cands[i] for _, i in sorted(scored, reverse=True)[:APPLY_MAX]]
    lines = ["Other preferences the user has stated that may bear on this (the excerpts above are the main "
             "evidence; use one of these only where it fits the request):"]
    lines += [line for _, _, line in top]
    return {"directives": sum(k == "standing" for k, _, _ in top), "preferences": sum(k == "claim" for k, _, _ in top),
            "block": "\n".join(lines)}
