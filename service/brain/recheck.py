"""One-time re-check of what the old ledger rules decided (room review v2, item 1e).

Before migration 013 a value could be ended in ways the shared rule (ledger.apply_relation) no longer allows: an
"adds" on a one-value slot, the merge of two slots ending the older value by date alone, a late-told past value
closing a newer one. Those links carry no source (the column did not exist), which is how this module finds them.

survey(conn)   counts what there is to check. No Jev call, nothing written.
recheck(conn, providers, apply=False)
               asks Jev the write path's relation question (W7) again for each old supersession and tests the
               answer against the shared rule. A link the rule would write today is kept (and stamped
               'recheck.kept'). Any other link is reversed (ledger.reverse_link: the ended value holds again unless
               another link still ends it), the decision rows behind it are marked reversed so they never become
               booster labels, and the relation Jev gives now is applied through apply_relation with source
               'recheck'. Then learned directives whose claim no longer holds are retired.
               With apply=False it only reports what it would do.

Not re-checked: plan closures (the ended claim is a plan), and pairs that were contested first (the nightly
correction verdict or the user's own choice ended those). Slot merges made before migration 013 cannot be taken
back: the slot a claim came from was never recorded. Only the values those merges ended by date are re-checked.

Each account's re-check runs inside its tenant transaction.
"""
from __future__ import annotations

import logging

from psycopg.types.json import Jsonb

from . import pages
from .ledger import STATE_POLARITIES, apply_relation, new_run_id, relation_question, reverse_link, starts
from .preferences import LEARNED_BY

log = logging.getLogger("brain.recheck")

BATCH = 50          # links per Jev request batch
KEEPS = ("updates", "retracts")

# Old supersessions of one state claim by another, not born from a contested pair.
_LEGACY = """
from claim_link l join claim n on n.id = l.from_claim join claim o on o.id = l.to_claim
     join slot s on s.id = o.slot_id
where l.type = 'supersedes' and l.reversed_at is null and l.source is null
  and o.status = 'superseded' and n.status <> 'retracted'
  and o.polarity = any(%(state)s) and n.polarity = any(%(state)s)
  and not exists (select 1 from claim_link x where x.type = 'contradicts'
                  and (x.from_claim, x.to_claim) in ((l.from_claim, l.to_claim), (l.to_claim, l.from_claim)))
"""
_STALE_DIRECTIVES = """
from directive d where d.created_by = %s and d.retired_at is null and d.source_claim is not null
  and not exists (select 1 from claim c where c.id = d.source_claim and c.status = 'active')
"""


def survey(conn) -> dict:
    """What a re-check would look at, by code alone."""
    p = {"state": list(STATE_POLARITIES)}
    by_source = dict(conn.execute(
        """select coalesce(source, 'legacy'), count(*) from claim_link
           where type = 'supersedes' and reversed_at is null group by 1 order by 1""").fetchall())
    legacy = conn.execute("select count(*) " + _LEGACY, p).fetchone()[0]
    out_of_order = conn.execute(
        "select count(*) " + _LEGACY + """
          and (select row(coalesce(c.valid_from, e.ts::date), e.ts, e.id) from claim c
               join episode e on e.id = c.episode_id where c.id = n.id)
           <= (select row(coalesce(c.valid_from, e.ts::date), e.ts, e.id) from claim c
               join episode e on e.id = c.episode_id where c.id = o.id)""", p).fetchone()[0]
    return {
        "supersedes_links": by_source,
        "legacy_to_check": legacy,
        "legacy_out_of_order": out_of_order,      # the newer claim does not start later: reversed whatever Jev says
        "reversed_links": conn.execute("select count(*) from claim_link where reversed_at is not null").fetchone()[0],
        "merges_recorded": conn.execute("select count(*) from slot_merge where undone_at is null").fetchone()[0],
        "legacy_merge_decisions": conn.execute(
            """select count(*) from decision_log where stage = 'nightly.merge_slots' and outcome = 'merged'
               and input->>'run_id' is null""").fetchone()[0],
        "learned_directives": conn.execute(
            "select count(*) from directive where created_by = %s and retired_at is null", (LEARNED_BY,)).fetchone()[0],
        "learned_directives_stale": conn.execute("select count(*) " + _STALE_DIRECTIVES, (LEARNED_BY,)).fetchone()[0],
        "decisions_marked_reversed": conn.execute("select count(*) from decision_log where reversed").fetchone()[0],
    }


def recheck(conn, providers, apply: bool = False, run_id: str | None = None, limit: int = 2000) -> dict:
    """Re-ask W7 for up to `limit` old supersessions of this account, oldest first. Returns the counts; with
    apply=False nothing is written. Raises the provider's error."""
    run_id = run_id or new_run_id("r")
    rows = conn.execute(
        """select l.from_claim, l.to_claim, o.namespace, o.entity_id, s.cardinality, n.value, n.quote, n.polarity,
                  o.value, o.quote, o.polarity, n.episode_id
        """ + _LEGACY + " order by l.created_at, l.from_claim, l.to_claim limit %(limit)s",
        {"state": list(STATE_POLARITIES), "limit": limit}).fetchall()
    out = {"run_id": run_id, "applied": apply, "checked": len(rows), "kept": 0, "reversed": 0, "reopened": 0,
           "relations": {}, "reapplied": {}, "decisions_marked": 0, "directives_retired": 0}
    logs = []
    for b in range(0, len(rows), BATCH):
        part, reqs, order = rows[b:b + BATCH], [], []
        for new_id, old_id, _, _, _, nv, nq, np_, ov, oq, op, _ in part:
            sn, so = starts(conn, new_id), starts(conn, old_id)
            order.append(sn > so)
            reqs.append(({"new": {"value": nv, "polarity": np_, "date": str(sn[0]), "quote": nq},
                          "existing": [{"value": ov, "date": str(so[0]), "quote": oq, "polarity": op}]},
                         {"r0": relation_question(0)}))
        for (new_id, old_id, ns, eid, card, nv, *_, ep), later, (state, _), r in zip(
                part, order, reqs, providers.jev.ask_many(reqs)):
            rel, probs = r.choice("r0")
            out["relations"][rel] = out["relations"].get(rel, 0) + 1
            holds = rel in KEEPS and later and card != "event"
            did = None
            if holds:
                out["kept"] += 1
                if apply:
                    conn.execute("""update claim_link set source = 'recheck.kept', run_id = %s
                                    where from_claim = %s and to_claim = %s and type = 'supersedes'""",
                                 (run_id, new_id, old_id))
            else:
                out["reversed"] += 1
                if apply:
                    out["reopened"] += reverse_link(conn, new_id, old_id, "supersedes") == "reopened"
                    out["decisions_marked"] += conn.execute(
                        """update decision_log set reversed = true where stage = 'write.relate' and not reversed
                           and (output->>'new_id' = %s or (output->>'new_id' is null
                                and input->>'episode_id' = %s and input->'new'->>'value' = %s))""",
                        (str(new_id), str(ep), nv)).rowcount
                    did = apply_relation(conn, new_id, old_id, rel, card, "recheck", run_id)
                    out["reapplied"][did] = out["reapplied"].get(did, 0) + 1
                    pages.mark_dirty(conn, ns, eid)
            logs.append((ns, "recheck.relate", "jev", r.model,
                         Jsonb({"task": "W7", "run_id": run_id, "new_id": new_id, "existing_ids": [old_id],
                                "episode_id": ep, **state}),
                         Jsonb({"relations": [rel], "cardinality": card, "starts_later": later, "applied": did,
                                "dry_run": not apply}),
                         probs.get(rel), None, "kept" if holds else "reversed"))
    if apply:
        with conn.cursor() as cur:
            cur.executemany("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                         probability, threshold, outcome)
                               values (%s, %s, %s, %s, %s, %s, %s, %s, %s)""", logs)
        stale = [r[0] for r in conn.execute("select d.source_claim " + _STALE_DIRECTIVES, (LEARNED_BY,)).fetchall()]
        if stale:
            out["directives_retired"] = conn.execute(
                "update directive d set retired_at = now(), retired_reason = 'recheck' where d.source_claim = any(%s) "
                "and d.created_by = %s and d.retired_at is null", (stale, LEARNED_BY)).rowcount
            out["decisions_marked"] += conn.execute(
                """update decision_log set reversed = true where stage = 'nightly.promote' and not reversed
                   and outcome = 'promoted' and input->>'claim_id' = any(%s)""", ([str(i) for i in stale],)).rowcount
    else:
        out["directives_retired"] = conn.execute("select count(*) " + _STALE_DIRECTIVES, (LEARNED_BY,)).fetchone()[0]
    return out
