"""Pages (DESIGN.md nightly step 3; build step 8): code-rendered views of an entity's claims.

No model writes a page (DESIGN.md). A page groups the entity's claims by slot: the value in
force (with date and quote), what it replaced, plans, and contested values side by side. Every line is built from
one claim and carries its id (`sentence_claims` maps each line to its claim ids). A citation is not support:
whether the cited turn really says what the line says is measured by `line_support_audit` (the build step 8 gate). The summary (160 characters at most) lists the most recent values. The write path marks a page dirty
when one of its entity's claims is placed or forgotten; the nightly job renders dirty pages and indexes each as a
'page' chunk. Pages are views: deleting one loses nothing.
"""
from __future__ import annotations

import logging
import math
import random

from psycopg.types.json import Jsonb

log = logging.getLogger("brain.pages")

SUMMARY_MAX = 160
SESSION_PAGES = 10
STATE = ("asserted", "corrected", "reported")
SUPPORT_MIN = 0.5        # Jev's probability that the source turn supports the line
SUPPORT_GATE = 0.95      # DESIGN.md build step 8: share of page lines supported
SOURCE_MAX = 6000
SUPPORT_QUESTION = ("`statement` names a subject, a fact about it and the value of that fact. Does `source` state "
                    "`statement`, or directly imply it?")


def mark_dirty(conn, namespace: str, entity_id: int) -> None:
    conn.execute("""insert into page (namespace, entity_id, body_md, dirty) values (%s, %s, '', true)
                    on conflict (namespace, entity_id) do update set dirty = true""", (namespace, entity_id))


def render(conn, page_id: int) -> dict:
    """Render one page from its entity's claims (inside the caller's tenant transaction)."""
    ns, eid = conn.execute("select namespace, entity_id from page where id = %s", (page_id,)).fetchone()
    name, aliases, kind = conn.execute("select name, aliases, kind from entity where id = %s", (eid,)).fetchone()
    rows = conn.execute(
        """select c.id, s.name, s.cardinality, c.value, c.valid_from, c.valid_to, c.status, c.polarity, c.kind, c.quote
           from claim c join slot s on s.id = c.slot_id join episode e on e.id = c.episode_id
           where c.entity_id = %s and c.status <> 'retracted'
           order by s.name, coalesce(c.valid_from, e.ts::date), e.ts, c.id""", (eid,)).fetchall()
    lines, cites, latest = [f"# {name}"], [], []
    other = sorted({a for a in (aliases or []) if a and a != name.lower()})
    if other:
        lines.append(f"Also called: {', '.join(other)}.")
    by_slot: dict[str, list] = {}
    for r in rows:
        by_slot.setdefault(r[1], []).append(r)
    for slot, cs in by_slot.items():
        lines.append(f"\n## {slot.replace('_', ' ')}")
        for cid, _, card, v, vf, vt, st, pol, k, q in reversed(cs):          # newest first
            if pol in ("planned", "hypothetical"):
                label = "plan" if pol == "planned" else "maybe"
            elif pol == "negated":
                label = "not so"
            elif st == "contested":
                label = "contested"
            elif st == "superseded":
                label = f"earlier, until {vt}" if vt else "earlier"
            else:
                label = "now" if card == "one" else ("event" if card == "event" else "current")
            line = f"- {v} ({label}; since {vf}) [claim {cid}]: \"{q[:200]}\""
            lines.append(line)
            cites.append({"sentence": line, "claim_ids": [cid]})
            if st == "active" and pol in STATE and k != "plan":
                latest.append((vf, slot, v))
    latest.sort(key=lambda x: (str(x[0]), x[1]), reverse=True)
    summary = name
    for _, slot, v in latest:
        part = f"{'; ' if summary != name else ': '}{slot.replace('_', ' ')} {v}"
        if len(summary) + len(part) > SUMMARY_MAX:
            break
        summary += part
    body = "\n".join(lines)
    rev = max((r[0] for r in rows), default=0)
    conn.execute("""update page set body_md = %s, summary = %s, sentence_claims = %s, rendered_at_rev = %s,
                    rendered_at = now(), dirty = false where id = %s""",
                 (body, summary[:SUMMARY_MAX], Jsonb(cites), rev, page_id))
    conn.execute("delete from chunk where kind = 'page' and ref_id = %s", (page_id,))
    if rows:
        conn.execute("insert into chunk (namespace, kind, ref_id, text) values (%s, 'page', %s, %s)",
                     (ns, page_id, body[:8000]))
    return {"page_id": page_id, "lines": len(cites), "summary": summary[:SUMMARY_MAX]}


def citation_check(conn, page_id: int) -> dict:
    """Every line must cite claims that exist, are not retracted, and whose value and quote the line repeats."""
    body_cites = conn.execute("select sentence_claims from page where id = %s", (page_id,)).fetchone()[0]
    ok = 0
    for s in body_cites:
        rows = conn.execute("select value, quote, status from claim where id = any(%s)", (s["claim_ids"],)).fetchall()
        ok += bool(rows) and all(st != "retracted" and v in s["sentence"] and q[:200] in s["sentence"]
                                 for v, q, st in rows)
    return {"lines": len(body_cites), "supported": ok}


def render_dirty(conn, limit: int = 500) -> dict:
    ids = [r[0] for r in conn.execute("select id from page where dirty order by id limit %s", (limit,)).fetchall()]
    lines = 0
    for pid in ids:
        lines += render(conn, pid)["lines"]
    return {"rendered": len(ids), "lines": lines}


def get(conn, namespace: str, entity: str) -> dict | None:
    from .ledger import _find_exact
    eid = _find_exact(conn, namespace, entity)
    if eid is None:
        return None
    row = conn.execute("select id, dirty from page where namespace = %s and entity_id = %s", (namespace, eid)).fetchone()
    if row is None:
        return None
    if row[1]:                   # dirty: render now (code only, cheap) rather than serve a stale view
        render(conn, row[0])
    body, summary, at = conn.execute("select body_md, summary, rendered_at from page where id = %s",
                                     (row[0],)).fetchone()
    return {"entity": entity, "summary": summary, "body_md": body, "rendered_at": at.isoformat()}


def session_summaries(conn, namespace: str) -> list[dict]:
    rows = conn.execute("""select n.name, p.summary from page p join entity n on n.id = p.entity_id
                           where p.namespace = %s and p.summary is not null and p.body_md <> '' and n.archived_at is null
                           order by n.mentions desc, n.id limit %s""", (namespace, SESSION_PAGES)).fetchall()
    return [{"entity": n, "summary": s} for n, s in rows]


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _squash(text: str) -> str:
    return " ".join((text or "").lower().split())


def line_support_audit(conn, jev, limit: int = 200, seed: int = 0, detail: bool = False) -> dict:
    """The build step 8 gate, measured: of a random sample of page lines, the share whose cited turn supports them.

    A line is supported when every claim it cites exists, is not retracted, comes from a turn that is not forgotten,
    and Jev says the turn states or directly implies "entity: fact = value" (probability at least SUPPORT_MIN).
    A line that fails the code checks counts as unsupported and is not sent to Jev. `quote_found` counts lines
    whose stored quote is found in the turn (a code check, reported beside the gate, not part of it).
    With detail=True the result also carries `detail`: every sampled line with its statement and source turn,
    for the hand check. Runs inside the caller's tenant transaction and writes nothing."""
    population = []
    for pid, cites in conn.execute("select id, sentence_claims from page where body_md <> '' order by id").fetchall():
        population += [(pid, i, s["sentence"], list(s["claim_ids"])) for i, s in enumerate(cites or [])]
    sample = list(population)
    random.Random(seed).shuffle(sample)
    sample = sample[:max(0, int(limit))]
    reqs, owners, lines = [], [], []
    for pid, i, sentence, cids in sample:
        rows = conn.execute(
            """select c.id, c.value, c.quote, c.status, s.name, n.name, e.body, e.deleted_at
               from claim c join slot s on s.id = c.slot_id join entity n on n.id = c.entity_id
                    join episode e on e.id = c.episode_id
               where c.id = any(%s) order by c.id""", (cids,)).fetchall()
        line = {"page_id": pid, "claim_ids": cids, "sentence": sentence[:300], "reason": None, "p": None,
                "quote_found": bool(rows) and all(_squash(r[2]) in _squash(r[6]) for r in rows)}
        if detail:
            line["statement"] = "; ".join(f"{r[5]}: {r[4].replace('_', ' ')} = {r[1]}" for r in rows)
            line["source"] = " | ".join(r[6][:1500] for r in rows)
        if not cids or len(rows) != len(set(cids)):
            line["reason"] = "cites a claim that does not exist"
        elif any(r[3] == "retracted" for r in rows):
            line["reason"] = "cites a retracted claim"
        elif any(r[7] is not None for r in rows):
            line["reason"] = "cites a forgotten turn"
        else:
            for r in rows:
                owners.append(len(lines))
                reqs.append(({"source": r[6][:SOURCE_MAX], "statement": f"{r[5]}: {r[4].replace('_', ' ')} = {r[1]}"},
                             {"supported": {"type": "noul", "instructions": SUPPORT_QUESTION}}))
        lines.append(line)
    model = None
    for k, r in zip(owners, jev.ask_many(reqs) if reqs else []):
        p = r.noul("supported")
        model = getattr(r, "model", None) or model
        lines[k]["p"] = p if lines[k]["p"] is None else min(lines[k]["p"], p)
    for line in lines:
        if line["reason"] is None and line["p"] < SUPPORT_MIN:
            line["reason"] = "the turn does not support the line"
    ok = sum(line["reason"] is None for line in lines)
    n = len(lines)
    share = round(ok / n, 4) if n else None
    out = {"population": len(population), "lines": n, "supported": ok, "share": share, "ci95": _wilson(ok, n),
           "quote_found": sum(line["quote_found"] for line in lines), "bar": SUPPORT_GATE, "model": model,
           "gate": "no lines" if not n else ("passed" if share >= SUPPORT_GATE else "not passed"),
           "unsupported": [{k: v for k, v in line.items() if k not in ("statement", "source")}
                           for line in lines if line["reason"]][:50]}
    if detail:
        out["detail"] = lines
    return out
