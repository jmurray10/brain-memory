"""Nightly consolidation (DESIGN.md "Nightly consolidation" steps 1, 2 and 6, and hardening), per account.

1. settle_links: each claim linked to its entity tentatively (alignment Score between 0.5 and 1.5) is re-checked
   with full context: the mention and its quote against everything the entity is known by (its name, aliases and
   up to 8 claims). >= LINK_AT settles the link; < REVIEW_AT moves the claim to a new entity of its own; in between
   it stays tentative, unless BRAIN_SETTLE_SPLIT is on (every unconfirmed link is split) or BRAIN_RESOLVE_NAMES
   is on and the mention and the entity are different proper names (that one is split).
2. resolve_contested (the booster's N2): for each pair linked 'contradicts' with both claims contested, Jev decides
   whether the later claim corrects the earlier one (then the earlier is superseded from the later one's date) or
   it is a genuine conflict (both stay, shown side by side).
3. stability_audit: a sample of the last day's relate decisions is asked again with the logged input; a decision
   that flips is logged ('nightly.stability', outcome 'flipped') and its claims do not harden tonight.
4. harden: an active claim settles (settled_at) when its entity link is not tentative, its checks cleared with a
   margin (confidence >= SETTLE_CONFIDENCE), no audit flipped it, and it has support: another claim saying the same
   (a supports link) or a direct statement (asserted or corrected).

merge_slots (off unless BRAIN_MERGE_SLOTS=1) joins slots of one entity that name the same attribute; see its
docstring for the rules. Every step that ends a value does it through ledger.apply_relation or records its link
with a source and the night's run id, so a run can be found and taken back (room review v2, item 1).

Jev decides, code computes; dreaming never writes a claim no episode states.
"""
from __future__ import annotations

import logging
import os
import random

import psycopg
from psycopg.types.json import Jsonb

from . import ledger, pages
from .ledger import (ALIGN_LEVELS, LINK_AT, MAX_EXISTING, RELATIONS, REVIEW_AT, START_ORDER, STATE_POLARITIES, _link,
                     _summary, apply_relation, new_run_id, relation_question, reverse_link, starts)

log = logging.getLogger("brain.nightly")

AUDIT_SAMPLE = 20
SETTLE_CONFIDENCE = 0.8
CORRECTION = {
    "corrects": "The later claim corrects the earlier one: the earlier was a mistake or is out of date.",
    "conflict": "A genuine conflict: two sources disagree, or both could be true; keep both.",
}


def _log(conn, rows):
    if rows:
        with conn.cursor() as cur:
            cur.executemany("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                         probability, threshold, outcome)
                               values (%s, %s, %s, %s, %s, %s, %s, %s, %s)""", rows)


def settle_links(conn, providers) -> dict:
    rows = conn.execute(
        """select c.id, c.namespace, c.entity_id, n.name, n.aliases, k.mention, c.quote
           from claim c join entity n on n.id = c.entity_id left join claim_candidate k on k.id = c.candidate_id
           where c.tentative and c.status <> 'retracted' order by c.id limit 200""").fetchall()
    out = {"tentative": len(rows), "settled": 0, "split": 0}
    if not rows:
        return out
    reqs = [({"mention": {"name": m or n, "said_in": q},
              "entity": {"name": n, "also_called": (a or [])[:8], "known": _summary(conn, eid) or "-"}},
             {"same": {"type": "score", "criteria": ALIGN_LEVELS,
                       "instructions": "How do `mention` and `entity` relate as people or things in one user's memory?"}})
            for _, _, eid, n, a, m, q in rows]
    logs = []
    for (cid, ns, eid, name, aliases, mention, _), (state, _), r in zip(rows, reqs, providers.jev.ask_many(reqs)):
        s, _ = r.score("same")
        apart = ledger.SETTLE_SPLIT or (ledger._resolve_names() and mention
                                        and ledger.different_names(mention, name, aliases))
        if s >= LINK_AT:
            conn.execute("update claim set tentative = false where id = %s", (cid,))
            out["settled"] += 1
            outcome = "settled"
        elif s < REVIEW_AT or apart:
            # the entity this mention already has, if an earlier split (or a later exact mention) made one: as
            # first built every split claim got an entity of its own, so one person became several
            new = ledger._find_exact(conn, ns, mention or name)
            if new is None or new == eid:
                new = conn.execute("""insert into entity (namespace, name, aliases) values (%s, %s, %s) returning id""",
                                   (ns, (mention or name)[:200], [(mention or name).lower()])).fetchone()[0]
            slot = conn.execute("""insert into slot (namespace, entity_id, name, cardinality)
                                   select namespace, %s, name, cardinality from slot where id =
                                       (select slot_id from claim where id = %s)
                                   on conflict (account_id, namespace, entity_id, name) do update set name = excluded.name
                                   returning id""", (new, cid)).fetchone()[0]
            conn.execute("""update claim set entity_id = %s, slot_id = %s, tentative = false, orig_slot_id = null,
                                             merged_run = null where id = %s""", (new, slot, cid))
            pages.mark_dirty(conn, ns, eid)
            pages.mark_dirty(conn, ns, new)
            out["split"] += 1
            outcome = "split"
        else:
            outcome = "still_tentative"
        logs.append((ns, "nightly.settle", "jev", r.model, Jsonb({"task": "W5", "claim_id": cid, **state}),
                     Jsonb({"score": round(s, 4)}), s, LINK_AT, outcome))
    _log(conn, logs)
    return out


MERGE_COSINE = 0.8          # slot names this close (embeddings) are worth asking about
MERGE_AT = 0.7              # Jev's "same attribute" Noul at or above this is a match
MERGE_ENTITIES = 200
MERGE_RELATE_MAX = 40       # per merged slot, the newest claims whose relation to the other slot's claims is asked
CHURN_MIN = 10              # a night that changes more claims than this ...
CHURN_SHARE = float(os.environ.get("BRAIN_MERGE_CHURN_CAP", "0.2"))   # ... and this share of the active ones is held
CARD_RANK = {"one": 0, "many": 1, "event": 2}     # the stricter cardinality is the one that ends fewer values


def _cos(a, b) -> float:
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def _slot_names(conn, eid: int, slots: list) -> dict[int, list[tuple[int, str, str | None]]]:
    """Every name each slot of an entity answers to: its own and the names of slots merged into it on earlier
    nights (slot_merge rows not undone, followed down). Each comes with the latest active value filed under it:
    {slot id: [(source slot id, name, latest value)]}."""
    absorbed: dict[int, list[tuple[int, str]]] = {}
    for keep, gone, name in conn.execute(
            "select keep_slot_id, gone_slot_id, gone_name from slot_merge where entity_id = %s and undone_at is null "
            "order by id", (eid,)).fetchall():
        absorbed.setdefault(keep, []).append((gone, name))
    latest = dict(conn.execute(
        """select distinct on (coalesce(c.orig_slot_id, c.slot_id)) coalesce(c.orig_slot_id, c.slot_id), c.value
           from claim c join episode e on e.id = c.episode_id
           where c.entity_id = %s and c.status = 'active'
           order by coalesce(c.orig_slot_id, c.slot_id), coalesce(c.valid_from, e.ts::date) desc, e.ts desc,
                    e.id desc, c.id desc""", (eid,)).fetchall())
    out = {}
    for sid, name, *_ in slots:
        names, todo = [(sid, name)], [sid]
        while todo:
            for gone, gname in absorbed.get(todo.pop(), []):
                names.append((gone, gname))
                todo.append(gone)
        out[sid] = [(src, n, latest.get(src)) for src, n in names]
    return out


def merge_slots(conn, providers, run_id: str | None = None) -> dict:
    """One fact filed under two slots of the same entity cannot supersede itself (exp-slices-v1: 3,129 slots for
    4,463 claims). Per entity with several slots, slot names are embedded; pairs at or above MERGE_COSINE go to Jev
    ("do these name the same attribute?"), MERGE_AT or above is a match.

    Rules (room review v2, item 1b):
    - No chains. A slot joins a group only if Jev matched it directly with every slot already in the group (the
      largest slot first), and a slot that absorbed others on an earlier night counts under each of its names. So
      two names share a slot only if Jev matched those two names.
    - The merged slot keeps the stricter cardinality of the two (event over many over one: the one that ends
      fewer values).
    - Nothing is ended by date. After a merge, Jev is asked how each claim relates to the older claims that came
      from the other slot (the write path's W7 question), and ledger.apply_relation, the one end-value rule,
      applies the answer.
    - Every change is recorded: slot_merge holds each merged-away slot, claim.orig_slot_id and merged_run the
      move, claim_link.source and run_id each link. undo_merge_run(run_id) takes a night back.
    - A night that would change the status of more than CHURN_MIN claims and more than CHURN_SHARE of the
      account's active claims is not applied: it is rolled back and logged as held, for a person to look at.
    Values and quotes are untouched."""
    run_id = run_id or new_run_id("n")
    ents = conn.execute(
        """select s.entity_id, n.name, n.namespace from slot s join entity n on n.id = s.entity_id
           where n.archived_at is null group by s.entity_id, n.name, n.namespace having count(*) >= 2
           order by count(*) desc limit %s""", (MERGE_ENTITIES,)).fetchall()
    out = {"entities": len(ents), "pairs_asked": 0, "merged": 0, "superseded": 0, "history": 0, "contested": 0,
           "churn": 0, "held": False, "run_id": run_id}
    emb = getattr(providers, "embedder", None)
    if not ents or emb is None:
        return out
    by_ent, nodes_of = {}, {}
    for eid, _, _ in ents:
        by_ent[eid] = conn.execute(
            """select s.id, s.name, s.cardinality, count(c.id) from slot s
               left join claim c on c.slot_id = s.id and c.status <> 'retracted'
               where s.entity_id = %s group by s.id order by count(c.id) desc, s.id""", (eid,)).fetchall()
        nodes_of[eid] = _slot_names(conn, eid, by_ent[eid])
    flat = [n.replace("_", " ") for eid in by_ent for s in by_ent[eid] for _, n, _ in nodes_of[eid][s[0]]]
    vecs = iter(emb.embed_documents(flat))
    reqs, metas = [], []
    for eid, ename, ns in ents:
        slots, nodes = by_ent[eid], nodes_of[eid]
        vec = {s[0]: [next(vecs) for _ in nodes[s[0]]] for s in slots}
        asks, cand = [], {}                      # asks: (node a, node b); cand: slot pair -> its question indexes
        for x in range(len(slots)):
            for y in range(x + 1, len(slots)):
                a, b = slots[x][0], slots[y][0]
                cross = [(i, j) for i in range(len(nodes[a])) for j in range(len(nodes[b]))]
                if all(_cos(vec[a][i], vec[b][j]) >= MERGE_COSINE for i, j in cross):
                    cand[(min(a, b), max(a, b))] = list(range(len(asks), len(asks) + len(cross)))
                    asks += [(nodes[a][i], nodes[b][j]) for i, j in cross]
        if not asks:
            continue
        state = {"entity": ename, "pairs": [{"a": {"name": na, "latest": va or "-"}, "b": {"name": nb, "latest": vb or "-"}}
                                            for (_, na, va), (_, nb, vb) in asks]}
        qs = {f"m{k}": {"type": "noul", "instructions": (
            f"Do `pairs[{k}].a` and `pairs[{k}].b` name the same attribute of `entity`, so that a newer value of one "
            "would replace or add to the other?")} for k in range(len(asks))}
        reqs.append((state, qs))
        metas.append((eid, ns, slots, cand))
    if not reqs:
        return out
    plans, pair_logs = [], []                    # plans: (entity, namespace, main slot row, [slot rows merged into it])
    for (eid, ns, slots, cand), (state, _), r in zip(metas, reqs, providers.jev.ask_many(reqs)):
        p = [r.noul(f"m{k}") for k in range(len(state["pairs"]))]
        out["pairs_asked"] += len(p)
        match = {pair for pair, ks in cand.items() if all(p[k] >= MERGE_AT for k in ks)}
        group_of, free = {}, list(slots)         # slots come largest first
        while free:
            main, group = free.pop(0), []
            for s in list(free):
                if all((min(s[0], g[0]), max(s[0], g[0])) in match for g in [main, *group]):
                    group.append(s)
                    free.remove(s)
            for s in [main, *group]:
                group_of[s[0]] = main[0]
            if group:
                plans.append((eid, ns, main, group))
        for (a, b), ks in cand.items():
            for k in ks:
                pair_logs.append([ns, {"entity_id": eid, "slots": [a, b], "run_id": run_id, "pair": state["pairs"][k]},
                                  p[k], r.model, "merged" if group_of[a] == group_of[b] else
                                  "matched" if p[k] >= MERGE_AT else "kept"])
    relate_logs: list = []
    active = 0
    if plans:
        active = conn.execute("select count(*) from claim where status = 'active'").fetchone()[0]
        with conn.transaction():                 # a savepoint: a night over the churn cap is rolled back whole
            done = _apply_merges(conn, providers, plans, run_id, relate_logs)
            if done["churn"] > CHURN_MIN and done["churn"] > CHURN_SHARE * active:
                out.update(held=True, churn=done["churn"], would_merge=done["merged"])
                raise psycopg.Rollback()
            out.update(done)
    if out["held"]:
        for row in pair_logs:
            row[4] = "held" if row[4] == "merged" else row[4]
        for row in relate_logs:
            row[-1] = "held"
        log.warning("merge_slots held: run %s would change %d of %d active claims", run_id, out["churn"], active)
        _log(conn, [(plans[0][1], "nightly.merge_slots", "code", None,
                     Jsonb({"run_id": run_id, "churn": out["churn"], "active": active, "slots": out["would_merge"]}),
                     Jsonb({"held": True}), None, CHURN_SHARE, "held")])
    _log(conn, [(ns, "nightly.merge_slots", "jev", model, Jsonb(inp), Jsonb({"same": round(pr, 4)}), pr, MERGE_AT, oc)
                for ns, inp, pr, model, oc in pair_logs])
    _log(conn, relate_logs)
    return out


def _apply_merges(conn, providers, plans: list, run_id: str, logs: list) -> dict:
    """Move the claims, record the move, then ask how the claims of the joined slots relate. Runs inside
    merge_slots' savepoint. Returns {merged, superseded, history, contested, churn}."""
    done = {"merged": 0, "superseded": 0, "history": 0, "contested": 0, "churn": 0}
    reqs, metas, touched = [], [], []
    for eid, ns, main, group in plans:
        keep, came_from = main[0], {}
        card = max([main[2], *[g[2] for g in group]], key=CARD_RANK.__getitem__)
        for gone, gname, gcard, _ in group:
            ids = [r[0] for r in conn.execute("select id from claim where slot_id = %s order by id", (gone,)).fetchall()]
            conn.execute("""insert into slot_merge (namespace, run_id, entity_id, gone_slot_id, gone_name,
                                                    gone_cardinality, keep_slot_id, keep_cardinality_before, claim_ids)
                            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                         (ns, run_id, eid, gone, gname, gcard, keep, main[2], ids))
            conn.execute("""update claim set orig_slot_id = coalesce(orig_slot_id, slot_id),
                                             merged_run = coalesce(merged_run, %s), slot_id = %s
                            where slot_id = %s""", (run_id, keep, gone))
            conn.execute("delete from slot where id = %s", (gone,))
            came_from.update({i: gone for i in ids})
            done["merged"] += 1
        conn.execute("update slot set cardinality = %s where id = %s", (card, keep))
        pages.mark_dirty(conn, ns, eid)
        rows = conn.execute(
            f"""select c.id, c.value, coalesce(c.valid_from, e.ts::date), c.quote, c.polarity from claim c
                join episode e on e.id = c.episode_id
                where c.slot_id = %s and c.status in ('active', 'contested') and c.polarity = any(%s)
                order by {START_ORDER}""", (keep, list(STATE_POLARITIES))).fetchall()
        touched += [r[0] for r in rows]
        for t in range(max(1, len(rows) - MERGE_RELATE_MAX), len(rows)):
            new = rows[t]
            older = [rows[u] for u in range(t - 1, -1, -1)
                     if came_from.get(rows[u][0], keep) != came_from.get(new[0], keep)][:MAX_EXISTING]
            if not older:
                continue
            state = {"new": {"value": new[1], "polarity": new[4], "date": str(new[2]), "quote": new[3]},
                     "existing": [{"value": v, "date": str(d), "quote": q, "polarity": p} for _, v, d, q, p in older]}
            reqs.append((state, {f"r{j}": relation_question(j) for j in range(len(older))}))
            metas.append((ns, keep, card, new[0], [o[0] for o in older]))
    before = dict(conn.execute("select id, status from claim where id = any(%s)", (touched,)).fetchall())
    answers = providers.jev.ask_many(reqs) if reqs else []
    for (ns, keep, card, new_id, old_ids), (state, _), r in zip(metas, reqs, answers):   # oldest claim first
        rels = [r.choice(f"r{j}")[0] for j in range(len(old_ids))]
        applied = [apply_relation(conn, new_id, old, rel, card, "nightly.merge_slots", run_id)
                   for old, rel in zip(old_ids, rels)]
        done["superseded"] += applied.count("ended")
        done["history"] += applied.count("history")
        done["contested"] += applied.count("contested")
        ep = conn.execute("select episode_id from claim where id = %s", (new_id,)).fetchone()[0]
        logs.append([ns, "nightly.merge_relate", "jev", r.model,
                     Jsonb({"task": "W7", "run_id": run_id, "slot_id": keep, "episode_id": ep, **state}),
                     Jsonb({"relations": rels, "cardinality": card, "applied": applied, "new_id": new_id,
                            "existing_ids": old_ids}), None, None, ",".join(rels)])
    after = dict(conn.execute("select id, status from claim where id = any(%s)", (touched,)).fetchall())
    done["churn"] = sum(after.get(i) != s for i, s in before.items())
    return done


def undo_merge_run(conn, run_id: str) -> dict:
    """Take one merge_slots run back, inside the caller's tenant transaction: the links it wrote are reversed (a
    value they ended holds again unless another standing link ends it), each merged-away slot is recreated under
    its name, its claims move back, and the kept slot gets back the cardinality it had. Claims written into the
    merged slot after the run stay where they are. The run's decision rows are marked reversed.

    Runs are undone newest first per slot: while a later run that merged into, or merged away, one of this run's
    kept slots stands, the result names it in `blocked_by` and nothing changes."""
    out = {"run_id": run_id, "slots_restored": 0, "claims_moved": 0, "links_reversed": 0, "reopened": 0,
           "blocked_by": []}
    out["blocked_by"] = [r[0] for r in conn.execute(
        """select distinct later.run_id from slot_merge m join slot_merge later
               on later.id > m.id and m.keep_slot_id in (later.gone_slot_id, later.keep_slot_id)
           where m.run_id = %s and m.undone_at is null and later.undone_at is null and later.run_id <> %s
           order by 1""", (run_id, run_id)).fetchall()]
    if out["blocked_by"]:
        return out
    for a, b, kind in conn.execute("""select from_claim, to_claim, type from claim_link
                                      where run_id = %s and source = 'nightly.merge_slots' and reversed_at is null
                                      order by created_at desc, from_claim desc""", (run_id,)).fetchall():
        did = reverse_link(conn, a, b, kind)
        out["links_reversed"] += did != "none"
        out["reopened"] += did == "reopened"
    rows = conn.execute("""select id, namespace, entity_id, gone_slot_id, gone_name, gone_cardinality, keep_slot_id,
                                  claim_ids from slot_merge where run_id = %s and undone_at is null order by id desc""",
                        (run_id,)).fetchall()
    moved: list[int] = []
    for mid, ns, eid, gone, gname, gcard, keep, claim_ids in rows:
        back = conn.execute("""insert into slot (namespace, entity_id, name, cardinality) values (%s, %s, %s, %s)
                               on conflict (account_id, namespace, entity_id, name) do update set name = excluded.name
                               returning id""", (ns, eid, gname, gcard)).fetchone()[0]
        out["claims_moved"] += conn.execute(
            """update claim set slot_id = %s,
                      merged_run = case when orig_slot_id = %s then null else merged_run end,
                      orig_slot_id = case when orig_slot_id = %s then null else orig_slot_id end
               where id = any(%s) and slot_id = %s""", (back, gone, gone, claim_ids, keep)).rowcount
        conn.execute("update slot_merge set keep_slot_id = %s where keep_slot_id = %s", (back, gone))   # its own merges
        conn.execute("update slot_merge set undone_at = now() where id = %s", (mid,))
        moved += list(claim_ids)
    # A later step may have acted on a pair this merge brought together (the nightly resolving a dispute the merge
    # raised ends one of them with its own link). Once the claims are back in their own slots, no standing link
    # may join two claims of different slots: reverse those too, whichever step wrote them.
    if moved:
        for a, b, kind in conn.execute(
                """select l.from_claim, l.to_claim, l.type from claim_link l
                       join claim x on x.id = l.from_claim join claim y on y.id = l.to_claim
                   where l.type in ('supersedes', 'contradicts') and l.reversed_at is null
                     and (l.from_claim = any(%s) or l.to_claim = any(%s)) and x.slot_id <> y.slot_id
                   order by l.created_at desc, l.from_claim desc""", (moved, moved)).fetchall():
            did = reverse_link(conn, a, b, kind)
            out["links_reversed"] += did != "none"
            out["reopened"] += did == "reopened"
        kept = conn.execute("""select keep_cardinality_before, gone_cardinality, undone_at is null from slot_merge
                               where keep_slot_id = %s order by id""", (keep,)).fetchall()
        card = max([kept[0][0], *[g for _, g, live in kept if live]], key=CARD_RANK.__getitem__)
        conn.execute("update slot set cardinality = %s where id = %s", (card, keep))
        pages.mark_dirty(conn, ns, eid)
        out["slots_restored"] += 1
    if rows:
        # the run's Jev decisions were taken back by a person: never booster labels
        conn.execute("""update decision_log set reversed = true
                        where stage in ('nightly.merge_slots', 'nightly.merge_relate') and input->>'run_id' = %s""",
                     (run_id,))
        _log(conn, [(rows[0][1], "nightly.merge_slots", "code", None, Jsonb({"run_id": run_id}),
                     Jsonb({k: v for k, v in out.items() if k != "run_id"}), None, None, "undone")])
    return out


PLAN_SETTLED_AT = 0.7
PLAN_OUTCOMES = {
    "done": "Carried out: a later claim says it happened.",
    "cancelled": "Called off: a later claim says it will not happen.",
    "changed": "Replaced by a different plan.",
}


def track_plans(conn, providers, run_id: str | None = None) -> dict:
    """Open items (the Intelliprompter cookbook): each active plan claim is read against the later claims about the
    same person or thing; when Jev says one of them shows the plan done, cancelled or changed, the plan is closed
    (superseded by that claim, from its date). Only claims dated after the plan are read, so a plan is never closed
    by date alone or by something older."""
    plans = conn.execute(
        """select c.id, c.namespace, c.entity_id, s.name, c.value, coalesce(c.valid_from, e.ts::date), c.quote
           from claim c join slot s on s.id = c.slot_id join entity n on n.id = c.entity_id
                join episode e on e.id = c.episode_id
           where c.kind = 'plan' and c.status = 'active' and n.archived_at is null order by c.id limit 200""").fetchall()
    out = {"plans": len(plans), "closed": 0}
    reqs, metas = [], []
    for pid, ns, eid, sname, value, vf, quote in plans:
        later = conn.execute(
            """select c.id, s.name, c.value, coalesce(c.valid_from, e.ts::date), c.quote
               from claim c join slot s on s.id = c.slot_id join episode e on e.id = c.episode_id
               where c.entity_id = %s and c.id <> %s and c.status <> 'retracted'
                 and coalesce(c.valid_from, e.ts::date) > %s
               order by coalesce(c.valid_from, e.ts::date), e.ts, c.id limit 12""", (eid, pid, vf)).fetchall()
        if not later:
            continue
        state = {"plan": {"fact": sname.replace("_", " "), "value": value, "date": str(vf), "quote": quote[:240]},
                 "later": [{"id": f"x{k}", "fact": s.replace("_", " "), "value": v, "date": str(d), "quote": q[:200]}
                           for k, (_, s, v, d, q) in enumerate(later)]}
        reqs.append((state, {
            "settled": {"type": "noul", "instructions": (
                "Does one of `later` show that `plan` was carried out, called off, or replaced by a different plan?")},
            "outcome": {"type": "choice", "criteria": PLAN_OUTCOMES, "instructions": "What became of `plan`?"},
            "which": {"type": "choice", "instructions": "Which of `later` shows what became of `plan`?",
                      "criteria": {f"x{k}": None for k in range(len(later))}}}))
        metas.append((pid, ns, eid, later))
    if not reqs:
        return out
    logs = []
    for (pid, ns, eid, later), (state, _), r in zip(metas, reqs, providers.jev.ask_many(reqs)):
        p = r.noul("settled")
        outcome, _ = r.choice("outcome")
        which, wp = r.choice("which")
        closed = p >= PLAN_SETTLED_AT and wp.get(which, 0) >= 0.5
        if closed:
            lid, *_rest = later[int(which[1:])]
            conn.execute("update claim set status = 'superseded', valid_to = %s where id = %s", (_rest[2], pid))
            _link(conn, lid, pid, "supersedes", "nightly.plans", run_id)
            pages.mark_dirty(conn, ns, eid)
            out["closed"] += 1
        logs.append((ns, "nightly.plans", "jev", r.model, Jsonb({"claim_id": pid, **state}),
                     Jsonb({"settled": round(p, 4), "outcome": outcome, "which": which}), p, PLAN_SETTLED_AT,
                     outcome if closed else "open"))
    _log(conn, logs)
    return out


def resolve_contested(conn, providers, run_id: str | None = None) -> dict:
    """N2. The later claim of a contested pair is the one that starts later (ledger.starts: the stated date, else
    the day it was said, then the order it was said), never the one with the higher id. Two claims of one turn
    with the same date cannot be ordered: they stay contested and Jev is not asked."""
    pairs = conn.execute(
        """select l.from_claim, l.to_claim, a.namespace, a.value, a.quote, b.value, b.quote, a.entity_id
           from claim_link l join claim a on a.id = l.from_claim join claim b on b.id = l.to_claim
           where l.type = 'contradicts' and l.reversed_at is null and a.status = 'contested' and b.status = 'contested'
           order by l.from_claim limit 200""").fetchall()
    out = {"pairs": len(pairs), "corrected": 0, "kept_both": 0}
    if not pairs:
        return out
    reqs, order = [], []
    for a, b, ns, av, aq, bv, bq, eid in pairs:
        sa, sb = starts(conn, a), starts(conn, b)
        if sa == sb:
            out["kept_both"] += 1
            continue
        later, earlier = ((a, av, sa, aq), (b, bv, sb, bq)) if sa > sb else ((b, bv, sb, bq), (a, av, sa, aq))
        order.append((later[0], earlier[0], later[2][0], ns, eid))
        reqs.append(({"earlier": {"value": earlier[1], "date": str(earlier[2][0]), "quote": earlier[3]},
                      "later": {"value": later[1], "date": str(later[2][0]), "quote": later[3]}},
                     {"verdict": {"type": "choice", "criteria": CORRECTION,
                                  "instructions": "`earlier` and `later` disagree. Does `later` correct `earlier`, or "
                                                  "is it a genuine conflict?"}}))
    logs = []
    answers = providers.jev.ask_many(reqs) if reqs else []
    for (later_id, earlier_id, later_date, ns, eid), (state, _), r in zip(order, reqs, answers):
        v, probs = r.choice("verdict")
        if v == "corrects" and probs.get(v, 0) >= 0.7:
            conn.execute("update claim set status = 'active' where id = %s", (later_id,))
            conn.execute("update claim set status = 'superseded', valid_to = %s where id = %s",
                         (later_date, earlier_id))
            _link(conn, later_id, earlier_id, "supersedes", "nightly.contested", run_id)
            pages.mark_dirty(conn, ns, eid)
            out["corrected"] += 1
        else:
            out["kept_both"] += 1
        logs.append((ns, "nightly.contested", "jev", r.model,
                     Jsonb({"task": "N2", "claims": [later_id, earlier_id], "run_id": run_id, **state}),
                     Jsonb({"verdict": v, "p": round(probs.get(v, 0), 4)}), probs.get(v, 0), 0.7, v))
    _log(conn, logs)
    return out


def stability_audit(conn, providers, sample: int = AUDIT_SAMPLE, seed: int | None = None) -> dict:
    rows = conn.execute(
        """select id, namespace, input, output from decision_log
           where stage = 'write.relate' and not reversed and ts > now() - interval '1 day'
           order by id""").fetchall()
    rnd = random.Random(seed)
    picked = rnd.sample(rows, min(sample, len(rows)))
    out = {"asked": len(picked), "flipped": 0, "flipped_episodes": []}
    if not picked:
        return out
    reqs = []
    for _, _, inp, _ in picked:
        state = {"new": inp["new"], "existing": inp["existing"]}
        reqs.append((state, {f"r{j}": {"type": "choice", "criteria": RELATIONS, "instructions": (
            f"How does `new` relate to `existing[{j}]` (same person or thing, same attribute)?")}
            for j in range(len(inp["existing"]))}))
    logs = []
    for (lid, ns, inp, prev), (state, qs), r in zip(picked, reqs, providers.jev.ask_many(reqs)):
        now = [r.choice(f"r{j}")[0] for j in range(len(qs))]
        flipped = now != list(prev.get("relations") or [])
        if flipped:
            out["flipped"] += 1
            out["flipped_episodes"].append(inp.get("episode_id"))
        logs.append((ns, "nightly.stability", "jev", r.model, Jsonb({"decision_log_id": lid, "episode_id":
                     inp.get("episode_id"), **state}), Jsonb({"before": prev.get("relations"), "now": now}), None,
                     None, "flipped" if flipped else "stable"))
    _log(conn, logs)
    return out


def harden(conn, unstable_episodes: list[int]) -> dict:
    n = conn.execute(
        """update claim c set settled_at = now()
           where c.settled_at is null and c.status = 'active' and not c.tentative
             and coalesce(c.confidence, 0) >= %s and not (c.episode_id = any(%s))
             and (c.polarity in ('asserted', 'corrected')
                  or exists (select 1 from claim_link l where l.type = 'supports' and l.reversed_at is null
                             and c.id in (l.from_claim, l.to_claim)))""",
        (SETTLE_CONFIDENCE, [e for e in unstable_episodes if e is not None])).rowcount
    return {"settled": n}


# merge_slots follows the review's rules (no chains, the shared end-value rule, undo, a churn cap) but stays out of
# consolidate until the owner turns it on: BRAIN_MERGE_SLOTS=1.
MERGE_SLOTS_ENABLED = os.environ.get("BRAIN_MERGE_SLOTS", "0") == "1"
MERGE_SKIPPED = {"entities": 0, "pairs_asked": 0, "merged": 0, "superseded": 0, "held": False,
                 "skipped": "BRAIN_MERGE_SLOTS off"}


def consolidate(conn, providers, merge_slots_enabled: bool | None = None, run_id: str | None = None) -> dict:
    """One account's consolidation, in order. Raises the provider's error (the caller logs and moves on).
    merge_slots runs only when enabled (default: BRAIN_MERGE_SLOTS, off). `run_id` names the night: every link the
    steps write carries it."""
    merging = MERGE_SLOTS_ENABLED if merge_slots_enabled is None else merge_slots_enabled
    run_id = run_id or new_run_id("n")
    out = {"run_id": run_id, "links": settle_links(conn, providers),
           "slots": merge_slots(conn, providers, run_id) if merging else dict(MERGE_SKIPPED),
           "contested": resolve_contested(conn, providers, run_id), "plans": track_plans(conn, providers, run_id)}
    audit = stability_audit(conn, providers)
    out["stability"] = {k: v for k, v in audit.items() if k != "flipped_episodes"}
    out["harden"] = harden(conn, audit["flipped_episodes"])
    return out
