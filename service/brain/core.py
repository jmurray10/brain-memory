"""Core operations (REFERENCE.md): capture, recall, answer, forget, directives, session start,
export. Plain functions taking a connection, the providers and the caller; REST, MCP and the CLI are thin
wrappers. Jev decides, code computes, Haiku writes."""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
import contextvars
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.types.json import Jsonb

from . import dates, ledger, pages, preferences, router
from .auth import Caller, valid_ns
from .claims import EXTRACT_ROLES, normalize_polarity, normalize_slot
from .db import tenant
from .errors import ApiError
from .prompts import (ANSWER_BUDGET_TOKENS, ANSWER_DEFAULT, ANSWER_MAX_TOKENS, ANSWER_PREFERENCE,
                      answer_style, build_answer_user)
from .providers import Providers, ProviderError, ProviderUnavailable
from .questions import (ANSWERS_QUERY, CONTAINS_PROMPT_INJECTION, INJECTION_THRESHOLD, SCREEN_RULE, SUBVERTS_USER,
                        screen_drops, trusted_source)
from .schemas import TurnIn
from .spend import CapReached
from .text import RRF_K, pack, render, split_text

log = logging.getLogger("brain.core")

DENSE_TOP = 60          # live-eval config.DENSE_TOP
KEYWORD_TOP = 60        # live-eval config.BM25_TOP
KEYWORD_COMMON = 5000   # search spaces of at least this many chunks drop common lexemes from the keyword query
KEYWORD_FREQ = 0.05     # common = in more than 5% of chunks (pg_stats); scoring every such row was 1.5 s at 200k
SHORTLIST = 30          # live-eval config.SHORTLIST: all of these go to Jev
SHORTLIST_MAX = 60      # the two candidate lists hold 60 each; a longer shortlist would need longer lists
ANSWER_K = 10
ANSWER_K_MAX = 30


# exp-wide-pack-v1: how many candidates Jev reads and how many excerpts the reader gets. On LoCoMo most evidence
# that never reaches the reader was never a candidate (the experiment log, exp-more-excerpts-v1), and the pack is a
# small part of what a model can read. BRAIN_PACK_WIDE = "S,K": a shortlist of S (30 to 60; S Jev requests per
# recall where the shipped cut makes 30) and K excerpts (10 to 30) within the same token budget. Unset or out
# of range: 30 and 10, as shipped.
def _wide(v) -> tuple[int, int] | None:
    """(shortlist, excerpts) from "S,K" or a pair, or None (as shipped) when unset or out of range."""
    try:
        s, k = (int(x) for x in (v.split(",") if isinstance(v, str) else v or ()))
    except ValueError:
        return None
    return (s, k) if 30 <= s <= SHORTLIST_MAX and 10 <= k <= min(ANSWER_K_MAX, s) else None


PACK_WIDE = _wide(os.environ.get("BRAIN_PACK_WIDE"))
# exp-shortlist-v1: who gets the 30 places Jev reads. As shipped the two lists are fused (RRF) and cut to 30,
# and a passage the vector search ranks 25th loses its place to keyword hits that share a common word with
# the question (of 126 LoCoMo evidence turns that never became candidates, 71 stood in the dense top 60).
# "D,K" seats the dense top D and the keyword top K first; the places left go by the fused rank as before.
# "0,0" (the default) is the shipped cut. The shortlist stays 30, so Jev's cost does not change. With a longer
# shortlist (BRAIN_PACK_WIDE) the seats may fill up to that length.
def _seats(v: str | None) -> tuple[int, int]:
    try:
        d, k = (max(0, int(x)) for x in (v or "0,0").split(","))
    except ValueError:
        return 0, 0
    return (d, k) if d + k <= (PACK_WIDE[0] if PACK_WIDE else SHORTLIST) else (0, 0)


SHORTLIST_SEATS = _seats(os.environ.get("BRAIN_SHORTLIST_SEATS"))
# Room review v2 (group 4): keep the best passage of each session before filling by score. Off until an eval
# shows it helps (BRAIN_SESSION_SLOTS=1 turns it on; the experiment log exp-session-slots-v1).
SESSION_SLOTS = os.environ.get("BRAIN_SESSION_SLOTS", "0") == "1"
SESSION_MIN = 0.5       # only a passage Jev scores at least this high can hold a session's slot
PASSAGE_CHARS = 2000    # live-eval retrieval.py sends render()[:2000] to the reranker
# What the context pack adds when the router sends a question to the open route (a single detail said once):
#   full  (default, as shipped): the ledger's side lookup, the facts search, and the notes on excerpts;
#   quiet: no ledger blocks, notes kept;   plain: no ledger blocks and no notes (retrieval speaks alone).
# In five LoCoMo runs the full brain lost to retrieval only on open-route questions every time (35 wins, 62
# losses) while it won on the other routes; the experiment log exp-open-route-v1 decides the default.
OPEN_ROUTE_MODES = ("full", "quiet", "plain")
OPEN_ROUTE = os.environ.get("BRAIN_OPEN_ROUTE", "full") if os.environ.get("BRAIN_OPEN_ROUTE") in OPEN_ROUTE_MODES     else "full"
# The turns next to each excerpt in its session, handed over as context (the experiment log exp-more-excerpts-v1: of
# the evidence turns recall never made candidates, a third stand within two turns of an excerpt it did hand over).
# 0 is off (the shipped default), at most 2 on each side.
NEIGHBOURS_MAX = 2
NEIGHBOURS = min(NEIGHBOURS_MAX, max(0, int(os.environ.get("BRAIN_NEIGHBOURS", "0") or 0)))
NEIGHBOUR_CHARS = 600
# exp-ledger-wording-v1: how the ledger's notes on excerpts are worded. `full` states what the ledger did ("later
# replaced: 'a' -> 'b' on DATE"); with it the full brain lost 8 LoCoMo questions and gained 1 against the same
# pack without notes, and the replacements read by hand were mostly not replacements. `soft` points at the later
# statement and leaves the judgment to the reader; `off` adds no notes. `full` stays the default until a gate
# says otherwise.
# exp-when-locomo-v1: with dates stored at capture (BRAIN_DATES), write each resolved date beside its phrase in
# the excerpts and nearby turns of every pack. Off until its gate is met (the experiment log).
DATE_NOTES = os.environ.get("BRAIN_DATE_NOTES", "0") == "1"
NOTE_STYLES = ("full", "soft", "off")
# profile-context-v1: the router and the ledger's reads need the question, not the excerpts, yet they waited
# for recall. 1 = `context` runs them beside recall (and the lookup beside the facts search). The same Jev
# requests are sent and the pack is built the same way; only the waiting changes. Off until measured.
CONTEXT_PARALLEL = os.environ.get("BRAIN_CONTEXT_PARALLEL", "0") == "1"
NOTE_STYLE = os.environ.get("BRAIN_NOTE_STYLE", "full") if os.environ.get("BRAIN_NOTE_STYLE") in NOTE_STYLES else "full"
# exp-stable-pack-v1: Jev's score for one passage moves by about 0.01 between two identical requests and comes
# in steps of 0.01, so the tenth place often sits among passages it cannot tell apart, and the same question
# asked twice got the same ten only 42% of the time. A band cuts the scores into steps of this width before
# sorting; inside a step the fused rank decides, which does not move. 0 (the default) sorts by the raw score.
RERANK_BAND_MAX = 0.5
RERANK_BAND = min(RERANK_BAND_MAX, max(0.0, float(os.environ.get("BRAIN_RERANK_BAND", "0") or 0)))
HNSW_EF_SEARCH = 100    # pgvector's default (40) would cap the dense side below DENSE_TOP
WRITE_PATH_ROLES = EXTRACT_ROLES        # roles whose turns get a write_path job (brain/claims.py)
# A role=user turn from a key without speaks_as_user is stored under this role (migration 014): it is what an
# agent says the user said. It is searchable, takes the strict screen (questions.trusted_source is false for it)
# and carries no claims, so it can never become a current fact or a directive.
AGENT_REPORTED = "agent_reported"


def _utc(ts: datetime | None) -> datetime | None:
    if ts is not None and ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _vec(v: list[float] | None) -> str | None:
    return None if v is None else "[" + ",".join(f"{x:.8g}" for x in v) + "]"


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


# ---------------------------------------------------------------- 7.1 capture

def _offset_min(ts: datetime) -> int:
    off = ts.utcoffset()
    return int(off.total_seconds() // 60) if off is not None else 0


def _index(conn: psycopg.Connection, providers: Providers, ns: str,
           eps: list[tuple[dict, datetime, str, str | None, str]]) -> str | None:
    """Chunk, render, embed and insert the search rows of episodes given as (out, ts in the client's own
    offset, role, speaker, text). Sets out["chunks"] and out["embedded"]. If Voyage is unavailable or the cap
    is reached, the chunks are stored without vectors (keyword search finds them at once), an embed_backfill
    job is queued per episode, and "no_embed" is returned."""
    windows = [(out, piece, render(ts, role, speaker, text_piece))
               for out, ts, role, speaker, text in eps for piece, text_piece in enumerate(split_text(text))]
    vectors: list | None = None
    degraded = None
    if windows:
        try:
            vectors = providers.embedder.embed_documents([w for _, _, w in windows])
        except (CapReached, ProviderError) as e:
            degraded = "no_embed"
            log.warning("indexing without embeddings (%s); queued embed_backfill", type(e).__name__)
    model = providers.embedder.model if vectors is not None else None
    rows = [(ns, out["id"], piece, w, _vec(vectors[i]) if vectors is not None else None, model)
            for i, (out, piece, w) in enumerate(windows)]
    for out, *_ in eps:
        out["chunks"], out["embedded"] = 0, vectors is not None
    for out, _, _ in windows:
        out["chunks"] += 1
    with conn.cursor() as cur:
        if rows:
            cur.executemany("""insert into chunk (namespace, kind, ref_id, piece, text, embedding, embed_model)
                               values (%s, 'episode_line', %s, %s, %s, %s::vector, %s)""", rows)
        if vectors is None and eps:
            cur.executemany("insert into job (kind, ref_id, namespace) values ('embed_backfill', %s, %s)",
                            [(out["id"], ns) for out, *_ in eps])
    return degraded


def _queue_write_path(conn: psycopg.Connection, ns: str, eps: list[tuple[dict, datetime, str, str | None, str]]):
    jobs = [(out["id"], ns) for out, _, role, _, _ in eps if role in WRITE_PATH_ROLES]
    if jobs:
        with conn.cursor() as cur:
            cur.executemany("insert into job (kind, ref_id, namespace) values ('write_path', %s, %s)", jobs)


def _receive_claims(conn: psycopg.Connection, ns: str, client: str, episode_id: int, claims) -> None:
    """Claims the user's model sent with a turn wait as 'received' for the write path's checks (brain/claims.py)."""
    with conn.cursor() as cur:
        cur.executemany(
            """insert into claim_candidate (namespace, episode_id, mention, slot, value, time_phrase, polarity, quote,
                                            status, extractor)
               values (%s, %s, %s, %s, %s, %s, %s, %s, 'received', %s)""",
            [(ns, episode_id, c.mention.strip(), normalize_slot(c.slot), c.value.strip(),
              (c.time_phrase or "").strip() or None, normalize_polarity(c.polarity, c.value), c.quote.strip(), client)
             for c in claims])


def capture(conn: psycopg.Connection, providers: Providers, caller: Caller, namespace: str | None,
            turns: list[TurnIn], session_id: str | None = None, request_id: str | None = None,
            write_path_enabled: bool = False, dates_enabled: bool = False) -> dict:
    """Store each turn as an episode, then split, render, embed and index it (_index). A repeated
    (namespace, idempotency_key) returns the existing episode as deduped; if that episode was forgotten it is
    not brought back (a client re-sending its history must not undo a forget) and is marked forgotten."""
    ns = caller.namespace(namespace, write=True)
    now = datetime.now(timezone.utc)
    episodes: list[dict] = []
    fresh: list[tuple[dict, datetime, str, str | None, str]] = []
    deduped = 0
    for t in turns:
        ts = _utc(t.ts) or now
        role = AGENT_REPORTED if t.role == "user" and not caller.speaks_as_user else t.role
        row = conn.execute(
            """insert into episode (namespace, client, session_id, ts, ts_offset_min, role, speaker, channel,
                                    body, meta, idempotency_key)
               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               on conflict (account_id, namespace, idempotency_key) where idempotency_key is not null do nothing
               returning id""",
            (ns, caller.name, session_id, ts, _offset_min(ts), role, t.speaker, t.channel, t.text,
             Jsonb(t.meta), t.idempotency_key)).fetchone()
        if row is None:  # same (namespace, idempotency_key) seen before
            eid, forgotten = conn.execute("""select id, deleted_at is not null from episode
                                             where namespace = %s and idempotency_key = %s""",
                                          (ns, t.idempotency_key)).fetchone()
            n, n_emb = conn.execute("""select count(*), count(embedding) from chunk
                                       where kind = 'episode_line' and ref_id = %s""", (eid,)).fetchone()
            out = {"id": eid, "chunks": n, "embedded": n > 0 and n == n_emb, "deduped": True}
            if forgotten:
                out["forgotten"] = True
            episodes.append(out)
            deduped += 1
            continue
        out = {"id": row[0], "chunks": 0, "embedded": False, "deduped": False}
        if role != t.role:
            out["stored_role"] = role       # this key does not speak as the user
        episodes.append(out)
        fresh.append((out, ts, role, t.speaker, t.text))
        if t.claims:
            # Claims are ledger facts, so only a key that speaks for the user may file them, whatever the
            # role (an attachment's claims are filed under its speaker, which the client names).
            if write_path_enabled and role in WRITE_PATH_ROLES and caller.speaks_as_user:
                _receive_claims(conn, ns, caller.name, row[0], t.claims)
                out["claims_received"] = len(t.claims)
            else:
                out["claims_ignored"] = len(t.claims)     # write path off, or a role that never carries claims

    degraded = _index(conn, providers, ns, fresh)
    if write_path_enabled and caller.speaks_as_user:
        _queue_write_path(conn, ns, fresh)
    if dates_enabled:
        _dates(conn, providers, fresh)
    return {"namespace": ns, "episodes": episodes, "deduped": deduped, "degraded": degraded,
            "request_id": request_id}


def _dates(conn: psycopg.Connection, providers: Providers,
           eps: list[tuple[dict, datetime, str, str | None, str]]) -> int:
    """Write path step 3 (build step 3): time phrases -> time_mention rows, resolved against each turn's own date
    in the client's offset. A Jev failure or the cap skips it (the turn is still stored and searchable)."""
    turns = [(out["id"], text, ts.date()) for out, ts, _, _, text in eps]
    try:
        return dates.store(conn, dates.mentions_for(providers, turns))
    except (CapReached, ProviderError) as e:
        log.warning("dates skipped (%s)", type(e).__name__)
        return 0


# ---------------------------------------------------------------- 7.2 recall

_FILTERS = """c.kind = 'episode_line' and c.namespace = any(%(ns)s) and e.deleted_at is null
              and (%(sid)s::text is null or e.session_id = %(sid)s)
              and (%(channels)s::text[] is null or e.channel = any(%(channels)s))
              and (%(tf)s::timestamptz is null or e.ts >= %(tf)s)
              and (%(tt)s::timestamptz is null or e.ts <= %(tt)s)"""

# One statement: dense top 60 (cosine), keyword top 60 (ts_rank_cd against an OR query of the question's
# lexemes), reciprocal rank fusion with k = 60, keep 30. A question with no lexemes (only stop words) has no
# keyword side; the spec's literal fallback 'nothing' would have matched chunks containing that word.
RECALL_SQL = f"""
with lex as (
    select x from unnest(tsvector_to_array(to_tsvector('english', %(q)s))) x
),
big as (     -- is the search space large? a bounded count on the (account_id, namespace, kind) index
    select count(*) >= %(kw_common)s as yes from (select 1 from chunk c where c.kind = 'episode_line'
                                                   and c.namespace = any(%(ns)s) limit %(kw_common)s) s
),
common as ( -- lexemes in more than kw_freq of all chunks, from Postgres' column statistics (free to read); through
            -- a function because pg_stats shows nothing to brain_app under RLS (migration 004)
    select e from brain.common_lexemes(%(kw_freq)s) e
),
kept as (   -- in a large search space drop common lexemes (BM25 gives them almost no weight), unless all are
    select x from lex
    where not (select yes from big) or x not in (select e from common)
       or not exists (select 1 from lex l where l.x not in (select e from common))
),
q as (
    select case when count(*) = 0 then null
                else to_tsquery('simple', string_agg(quote_literal(x), ' | ' order by x)) end as tsq
    from kept
),
dense as (
    select id, row_number() over (order by dist, id) as r from (
        select c.id, c.embedding <=> %(qv)s::vector as dist
        from chunk c join episode e on e.id = c.ref_id
        where {_FILTERS} and %(qv)s::vector is not null and c.embedding is not null
          and not c.cold    -- the hot vector index is partial (migration 009); cold rows stay in keyword search
        order by dist limit %(dense_top)s) d
),
kw as (
    select id, row_number() over (order by rank desc, id) as r from (
        select c.id, ts_rank_cd(c.tsv, q.tsq) as rank
        from chunk c join episode e on e.id = c.ref_id cross join q
        where {_FILTERS} and q.tsq is not null and c.tsv @@ q.tsq
        order by rank desc, c.id limit %(kw_top)s) k
),
fused as (
    select id, sum(1.0 / (%(rrf_k)s + r))::float8 as rrf, min(dr) as dense_rank, min(kr) as keyword_rank
    from (select id, r, r as dr, null::bigint as kr from dense
          union all select id, r, null::bigint, r from kw) u
    group by id
    order by (min(dr) <= %(dense_seats)s or min(kr) <= %(kw_seats)s) is true desc, rrf desc, id
    limit %(shortlist)s
)
select f.id, f.rrf, f.dense_rank, f.keyword_rank, c.text, c.piece, c.namespace,
       e.id, e.ts, e.role, e.speaker, e.channel, e.session_id, e.client
from fused f join chunk c on c.id = f.id join episode e on e.id = c.ref_id
order by f.rrf desc, f.id
"""

_ITERATIVE_SCAN: dict[str, bool] = {}


def _tune_hnsw(conn: psycopg.Connection) -> None:
    """ef_search >= DENSE_TOP for this transaction; on pgvector >= 0.8 also let filtered HNSW scans keep
    going until enough rows pass the namespace filter."""
    conn.execute("select set_config('hnsw.ef_search', %s, true)", (str(HNSW_EF_SEARCH),))
    if "ok" not in _ITERATIVE_SCAN:
        v = conn.execute("select extversion from pg_extension where extname = 'vector'").fetchone()
        parts = tuple(int(p) for p in (v[0] if v else "0").split(".")[:2] if p.isdigit())
        _ITERATIVE_SCAN["ok"] = parts >= (0, 8)
    if _ITERATIVE_SCAN["ok"]:
        conn.execute("select set_config('hnsw.iterative_scan', 'relaxed_order', true)")


def _recall_namespaces(caller: Caller, namespace: str | None, namespaces: list[str] | None) -> list[str]:
    if namespaces:
        out: list[str] = []
        for n in namespaces:
            n = caller.namespace(n)
            if n not in out:
                out.append(n)
        return out
    return [caller.namespace(namespace)]


def rerank_order(kept: list[dict], band: float = 0.0) -> list[dict]:
    """The candidates (given in fused order) by Jev's answers_query score, best first. The sort is stable, so
    the fused order breaks ties. With a band the score is first cut into steps of that width: two passages in
    the same step are ordered by the fused rank alone."""
    if band > 0:
        return sorted(kept, key=lambda c: -int(c["scores"]["rerank"] / band + 1e-9))
    return sorted(kept, key=lambda c: -c["scores"]["rerank"])


def reserve_sessions(ranked: list[dict], k: int, floor: float = SESSION_MIN) -> list[dict]:
    """The k candidates to return from `ranked` (already in Jev-score order), one slot per session first.

    Walking down the candidates that score at least `floor`, the best one of each distinct session takes a slot
    (up to k). The remaining slots go to the highest-scoring candidates not yet taken. The result stays in score
    order. A question whose evidence is spread over several sessions then keeps one passage from each, instead
    of losing some to several near-identical passages of one session. A candidate with no session id counts as
    its own session."""
    if k <= 0:
        return []
    picked, seen = [], set()
    for i, c in enumerate(ranked):
        if len(picked) >= k or (c["scores"]["rerank"] or 0.0) < floor:
            break
        key = (c["namespace"], c["session_id"]) if c.get("session_id") else ("episode", c["episode_id"])
        if key not in seen:
            seen.add(key)
            picked.append(i)
    taken = set(picked)
    for i in range(len(ranked)):
        if len(taken) >= k:
            break
        taken.add(i)
    return [ranked[i] for i in sorted(taken)]


def recall(conn: psycopg.Connection, providers: Providers, caller: Caller, namespace: str | None, query: str,
           k: int = 10, namespaces: list[str] | None = None, session_id: str | None = None,
           channels: list[str] | None = None, time_from: datetime | None = None, time_to: datetime | None = None,
           rerank: bool = True, screen: bool = True, request_id: str | None = None,
           band: float | None = None, seats: tuple[int, int] | None = None,
           shortlist: int | None = None) -> dict:
    """The open route. Hybrid candidates in SQL, then one Jev request per candidate with three Nouls
    (answers_query, contains_prompt_injection, subverts_user). The screen drops candidates by the provenance rule
    (questions.screen_drops: the user's own words only if both screen Nouls fire, anything else if either does);
    the rest are sorted by answers_query and the top k returned. If Jev is down or the cap is reached, the RRF order is
    returned with degraded = "no_rerank" and screened = false.

    decision_log, one row per candidate and stage: recall.injection (outcome kept or dropped, threshold 0.70)
    and recall.rerank (outcome kept if returned, else excluded); plus one recall.route row (decider code)."""
    ns_list = _recall_namespaces(caller, namespace, namespaces)

    dense = True
    try:
        qvec = providers.embedder.embed_query(query)
    except (CapReached, ProviderError) as e:
        qvec, dense = None, False
        log.warning("recall without the dense side (%s)", type(e).__name__)

    if shortlist is None:
        shortlist = PACK_WIDE[0] if PACK_WIDE else SHORTLIST
    seats = SHORTLIST_SEATS if seats is None else seats
    seats = seats if seats[0] + seats[1] <= shortlist and min(seats) >= 0 else (0, 0)
    _tune_hnsw(conn)
    params = {"ns": ns_list, "q": query, "qv": _vec(qvec), "sid": session_id, "channels": channels,
              "tf": _utc(time_from), "tt": _utc(time_to), "dense_top": DENSE_TOP, "kw_top": KEYWORD_TOP, "kw_common": KEYWORD_COMMON, "kw_freq": KEYWORD_FREQ,
              "rrf_k": RRF_K, "shortlist": shortlist, "dense_seats": seats[0], "kw_seats": seats[1]}
    cands = [{"chunk_id": r[0], "episode_id": r[7], "namespace": r[6], "text": r[4], "piece": r[5],
              "ts": _iso(r[8]), "role": r[9], "speaker": r[10], "channel": r[11], "session_id": r[12],
              "client": r[13],
              "scores": {"rrf": round(r[1], 6), "dense_rank": r[2], "keyword_rank": r[3],
                         "rerank": None, "injection": None, "subverts": None}}
             for r in conn.execute(RECALL_SQL, params).fetchall()]

    questions = {}
    if rerank:
        questions["answers_query"] = ANSWERS_QUERY
    if screen:
        questions["contains_prompt_injection"] = CONTAINS_PROMPT_INJECTION
        questions["subverts_user"] = SUBVERTS_USER
    results = None
    degraded = None
    if questions and cands:
        try:
            results = providers.jev.ask_many(
                [({"question": query, "candidate_passage": c["text"][:PASSAGE_CHARS]}, questions) for c in cands])
        except (CapReached, ProviderError) as e:
            degraded = "no_rerank" if rerank else "no_screen"
            log.warning("recall without Jev (%s)", type(e).__name__)
    reranked = results is not None and rerank
    screened = results is not None and screen

    if results is not None:
        for c, r in zip(cands, results):
            if rerank:
                c["scores"]["rerank"] = round(r.noul("answers_query"), 6)
            if screen:
                c["scores"]["injection"] = round(r.noul("contains_prompt_injection"), 6)
                c["scores"]["subverts"] = round(r.noul("subverts_user"), 6)
    for c in cands:
        c["_drop"] = screened and screen_drops(c["scores"]["injection"], c["scores"]["subverts"], c["role"],
                                               c["channel"])
    kept = [c for c in cands if not c["_drop"]]
    band = RERANK_BAND if band is None else min(RERANK_BAND_MAX, max(0.0, band))
    if reranked:
        kept = rerank_order(kept, band)
    session_slots = reranked and SESSION_SLOTS
    items = reserve_sessions(kept, k) if session_slots else kept[:k]

    returned = {c["chunk_id"] for c in items}
    logs = []
    if results is not None:
        for c, r in zip(cands, results):
            inp = Jsonb({"query": query, "chunk_id": c["chunk_id"]})
            if screen:
                inj = c["scores"]["injection"]
                out = Jsonb({"subverts_user": c["scores"]["subverts"], "trusted": trusted_source(c["role"], c["channel"]),
                             "rule": SCREEN_RULE})
                logs.append((c["namespace"], "recall.injection", "jev", r.model, inp, out, inj,
                             INJECTION_THRESHOLD, "dropped" if c["_drop"] else "kept", request_id))
            if rerank:
                logs.append((c["namespace"], "recall.rerank", "jev", r.model, inp, None, c["scores"]["rerank"],
                             None, "kept" if c["chunk_id"] in returned else "excluded", request_id))
    logs.append((ns_list[0] if len(ns_list) == 1 else None, "recall.route", "code", None, Jsonb({"query": query}),
                 Jsonb({"namespaces": ns_list, "degraded": degraded, "dense": dense, "candidates": len(cands),
                        "returned": len(items), "session_slots": session_slots, "band": band,
                        "seats": list(seats), "shortlist": shortlist}), None, None, "open",
                 request_id))
    with conn.cursor() as cur:
        cur.executemany("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                     probability, threshold, outcome, request_id)
                           values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", logs)
    for c in cands:
        c.pop("_drop", None)
    return {"route": "open", "namespaces": ns_list, "items": items, "degraded": degraded, "dense": dense,
            "screened": screened, "request_id": request_id}


# ---------------------------------------------------------------- 7.3 answer

FALSE_PREMISE_AT = 0.7     # thresholds to tune on LongMemEval's _abs questions (benchmark phase)
ENOUGH_BELOW = 0.25


def _ledger_notes(conn: psycopg.Connection, chosen: list[dict], texts: list[str], style: str = "full") -> list[str]:
    """Passage classification from the ledger (DESIGN.md read path, 'classify' in code): an excerpt whose turn gave
    a claim that was later superseded or is contested says so, so a stale passage cannot pass as current."""
    ids = [it["episode_id"] for it in chosen]
    rows = conn.execute(
        """select o.episode_id, o.value, o.status, n.value, n.valid_from
           from claim o
           left join claim_link l on l.to_claim = o.id and l.type = 'supersedes' and l.reversed_at is null
           left join claim n on n.id = l.from_claim and n.status = 'active'
           where o.episode_id = any(%s) and o.status in ('superseded', 'contested')""", (ids,)).fetchall()
    notes: dict[int, list[str]] = {}
    for eid, old, st, new, new_from in rows:
        if style == "soft":        # point at the later statement; do not say what it did to this one
            if st == "contested":
                notes.setdefault(eid, []).append(f"another statement may differ on '{old}'")
            elif new is not None:
                notes.setdefault(eid, []).append(f"see also the later statement of {new_from}: '{new}'")
            continue               # a value that ended with nothing in its place gets no note
        if st == "contested":
            notes.setdefault(eid, []).append(f"'{old}' is disputed by another statement")
        elif new is not None:
            notes.setdefault(eid, []).append(f"later replaced: '{old}' -> '{new}' on {new_from}")
        else:
            notes.setdefault(eid, []).append(f"'{old}' no longer holds")
    return [t + (f" [memory: {'; '.join(notes[it['episode_id']][:3])}]" if it["episode_id"] in notes else "")
            for it, t in zip(chosen, texts)]


def _neighbours(conn: psycopg.Connection, providers: Providers, ns_list: list[str], question: str,
                chosen: list[dict], n: int, request_id: str | None) -> list[dict]:
    """The turns up to `n` before and after each excerpt in its own session, which the excerpts do not already
    hold: the other half of an exchange ("Did you go?" / "Yes, last week") is often the evidence, and it is
    rarely similar enough to the question to be a candidate. Each goes through the same screen as a recalled
    passage (the two screen Nouls and the provenance rule) and the decision is logged; if Jev does not answer
    there are no neighbours. Forgotten turns are never returned. Not reranked: a neighbour is context for its
    excerpt, not an answer on its own."""
    ids = [it["episode_id"] for it in chosen]
    if n <= 0 or not ids:
        return []
    rows = conn.execute(
        """with s as (
               select e.id, e.namespace, e.session_id, e.ts, e.role, e.speaker, e.channel, e.body,
                      row_number() over (partition by e.namespace, e.session_id order by e.ts, e.id) as rn
               from episode e
               where e.namespace = any(%(ns)s) and e.deleted_at is null and e.session_id in (
                   select session_id from episode where id = any(%(ids)s) and session_id is not null))
           select distinct b.id, b.namespace, b.ts, b.role, b.speaker, b.channel, b.body
           from s a join s b on b.namespace = a.namespace and b.session_id = a.session_id
                            and b.rn between a.rn - %(n)s and a.rn + %(n)s
           where a.id = any(%(ids)s) and not (b.id = any(%(ids)s)) and b.body <> ''
           order by b.ts, b.id""", {"ns": ns_list, "ids": ids, "n": n}).fetchall()
    near = [{"episode_id": r[0], "namespace": r[1], "ts": _iso(r[2]), "role": r[3], "channel": r[5],
             "text": render(r[2], r[3], r[4], r[6][:NEIGHBOUR_CHARS])} for r in rows]
    if not near:
        return []
    questions = {"contains_prompt_injection": CONTAINS_PROMPT_INJECTION, "subverts_user": SUBVERTS_USER}
    try:
        results = providers.jev.ask_many([({"question": question, "candidate_passage": c["text"]}, questions)
                                          for c in near])
    except (CapReached, ProviderError) as e:
        log.warning("neighbours skipped (%s)", type(e).__name__)
        return []
    kept, logs = [], []
    for c, r in zip(near, results):
        inj, sub = round(r.noul("contains_prompt_injection"), 6), round(r.noul("subverts_user"), 6)
        drop = screen_drops(inj, sub, c["role"], c["channel"])
        logs.append((c["namespace"], "recall.neighbour", "jev", r.model,
                     Jsonb({"query": question, "episode_id": c["episode_id"]}),
                     Jsonb({"subverts_user": sub, "trusted": trusted_source(c["role"], c["channel"]),
                            "rule": SCREEN_RULE}), inj, INJECTION_THRESHOLD, "dropped" if drop else "kept",
                     request_id))
        if not drop:
            kept.append(c)
    with conn.cursor() as cur:
        cur.executemany("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                     probability, threshold, outcome, request_id)
                           values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", logs)
    return kept


def _sufficiency(providers: Providers, question: str, question_date: str, texts: list[str], blocks: list[str]) -> dict:
    """DESIGN.md principle 7, abstain on evidence (the booster's R7): one Jev request over the packed evidence."""
    state = {"question": question, "question_date": question_date,
             "evidence": [t[:400] for t in texts] + [b[:1200] for b in blocks]}
    r = providers.jev.ask(state, {
        "enough": {"type": "noul", "instructions": "Does `evidence` contain what is needed to answer `question`?"},
        "false_premise": {"type": "noul", "instructions": (
            "Does `question` assume something about the user or their world that `evidence` never states (a "
            "person, event or detail that is not there)?")}})
    return {"enough": round(r.noul("enough"), 4), "false_premise": round(r.noul("false_premise"), 4)}


# count-route-reading-v1: the count route's code-counted number equalled the gold on 2 of 61 LongMemEval count
# questions. "0" leaves the gathered list out of the pack on count questions (the excerpts answer alone), so that
# what the block contributes can be measured. Default "1": the block as built.
COUNT_BLOCK = os.environ.get("BRAIN_COUNT_BLOCK", "1") == "1"


def _route_and_ledger(conn: psycopg.Connection, providers: Providers, ns_list: list[str], question: str,
                      qdate: str, ledger_route: bool, open_mode: str, ex: ThreadPoolExecutor | None = None,
                      count_block: bool = True, count_style: str | None = None,
                      count_top: int | None = None) -> tuple:
    """The part of `context` that needs the question and not the excerpts: the router, then the ledger's reads
    for the routes that read the ledger by the question alone (count, current, history, and the side lookup on
    open and when). Returns (decision, found, plain_open, error): a provider error or the cap stops it where it
    is and comes back as `error`, with what was decided before it. The preference and assistant routes read the
    excerpts, so `context` runs them itself.

    With an executor the exact lookup and the facts search run side by side. The connection is shared with the
    caller's thread; psycopg serialises its use."""
    d, found, plain_open = None, [], False
    try:
        d = router.route(providers, question, qdate)
        # The ledger has its own gates (does the question name a known entity, and one of its facts?), so it
        # also runs for open and when questions: the router often sends "what day is my class" to open.
        if ledger_route and d.route not in ("preference", "assistant"):
            qday = dates.parse_day(qdate)
            if d.route == "count" or d.nouls.get("gathers_many", 0) > 0.5:      # build step 6
                found = [ledger.gather(conn, providers, ns_list, question, qday, count_style, count_top)] if count_block else []
            elif d.route == "open" and open_mode != "full":   # the ledger speaks only when the router asks it
                plain_open = open_mode == "plain"
            else:     # the exact lookup, and the facts search as its hedge against a mis-filed fact
                look = lambda: ledger.lookup(conn, providers, ns_list, question, qday,      # noqa: E731
                                             "history" if d.route == "history" else "current",
                                             decisive=d.route in ("current", "history"))
                if ex is None:
                    found = [look(), ledger.facts(conn, providers, ns_list, question, qday)]
                else:
                    f = ex.submit(contextvars.copy_context().run, ledger.facts, conn, providers, ns_list, question, qday)
                    try:
                        first = look()
                    finally:
                        f.exception()
                    found = [first, f.result()]
    except (CapReached, ProviderError) as e:
        return d, [], plain_open, e
    return d, found, plain_open, None


def context(conn: psycopg.Connection, providers: Providers, caller: Caller, namespace: str | None, question: str,
            question_date: str | None = None, style: str = "auto", request_id: str | None = None,
            when_route: bool = True, ledger_route: bool = True, memory_check: bool = False,
            ledger_notes: bool = True, open_route: str | None = None, neighbours: int | None = None,
            note_style: str | None = None, date_notes: bool | None = None, parallel: bool | None = None,
            rerank_band: float | None = None, seats: tuple[int, int] | None = None,
            count_block: bool | None = None, count_style: str | None = None,
            wide: tuple[int, int] | None = None, count_top: int | None = None) -> dict:
    """The context pack (DESIGN.md read path, amended 2026-11-09): what the brain hands the user's model to answer
    from. Recall k = 10 packed within the measured budget and ordered by date (the open route, always there as the
    hedge); then the router, and per route the brain's own blocks: computed dates (when, build step 3), the
    ledger's current value or history (build step 5), a gathered and code-counted list (count, step 6), or the
    preferences to apply (preference, step 7). No generative model; `prompt` is ready to give any model."""
    today = datetime.now(timezone.utc).date().isoformat()
    open_mode = open_route if open_route in OPEN_ROUTE_MODES else OPEN_ROUTE
    parallel = CONTEXT_PARALLEL if parallel is None else parallel
    count_block = COUNT_BLOCK if count_block is None else count_block
    shortlist, answer_k = (PACK_WIDE if wide is None else _wide(wide)) or (SHORTLIST, ANSWER_K)
    side = None
    if parallel and (when_route or ledger_route):
        try:        # the facts search and recall embed the same question: once, before they run side by side
            providers.embedder.embed_query(question)
        except (CapReached, ProviderError):
            pass    # each of them handles the missing dense side itself
        _tune_hnsw(conn)    # recall sets this for the transaction; the facts search may now run first
        ex = ThreadPoolExecutor(max_workers=3)
        side = ex.submit(contextvars.copy_context().run, _route_and_ledger, conn, providers, _recall_namespaces(caller, namespace, None), question,
                         question_date or today, ledger_route, open_mode, ex, count_block, count_style, count_top)
    try:
        r = recall(conn, providers, caller, namespace, question, k=answer_k, request_id=request_id,
                   band=rerank_band, seats=seats, shortlist=shortlist)
    finally:
        if side is not None:
            side.exception()        # wait for the side reads before the connection is used again or released
            ex.shutdown()
    items = r["items"]
    picked = [items[i] for i in pack([it["text"] for it in items], answer_k, ANSWER_BUDGET_TOKENS)]
    chosen = sorted(picked, key=lambda it: (datetime.fromisoformat(it["ts"]), it["episode_id"], it["piece"]))
    style = answer_style(question) if style == "auto" else style
    blocks: list[str] = []
    route, ledger_info = None, None
    plain_open = False
    if when_route or ledger_route:
        d, found, plain_open, err = side.result() if side is not None else _route_and_ledger(
            conn, providers, r["namespaces"], question, question_date or today, ledger_route, open_mode,
            count_block=count_block, count_style=count_style, count_top=count_top)
        try:
            if d is not None:
                route = d.route
                if when_route and chosen and (d.route == "when" or d.nouls["needs_date_math"] > 0.5):
                    route = "when"
                    blocks.append(dates.block(conn, picked, question_date or today))
            if err is not None:
                raise err
            if ledger_route and d.route == "preference":                             # build step 7
                p = preferences.pack(conn, providers, r["namespaces"], caller.name, question,
                                     [it["text"] for it in chosen])
                found = [p] if p else []
                if p and style == "default":
                    style = "preference"
            elif ledger_route and d.route == "assistant":
                found = [ledger.assistant_lines(conn, providers, r["namespaces"], question, chosen)]
            found = [f for f in found if f]
            if found:
                ledger_info = {k: v for f in found for k, v in f.items() if k != "block"}
                blocks.extend(f["block"] for f in found)
        except (CapReached, ProviderError) as e:      # no router or ledger: the open route's pack, as before
            log.warning("router skipped (%s)", type(e).__name__)
    texts = [it["text"] for it in chosen]
    date_notes = DATE_NOTES if date_notes is None else date_notes
    if date_notes and chosen:
        texts = dates.annotate(conn, [(it["episode_id"], it["text"], it["ts"]) for it in chosen])
    sufficiency = None
    note_style = note_style if note_style in NOTE_STYLES else NOTE_STYLE
    if ledger_route and ledger_notes and note_style != "off" and chosen and not plain_open:
        texts = _ledger_notes(conn, chosen, texts, note_style)
    if ledger_route and memory_check and chosen:      # off by default: exp-memory-check-v1
        try:
            sufficiency = _sufficiency(providers, question, question_date or today, texts, blocks)
            if sufficiency["false_premise"] >= FALSE_PREMISE_AT:
                # exp-locomo-brain-v1: "say so instead of guessing" made the reader refuse questions that only had
                # a name or date slightly off; a close match is answered, with the difference named
                blocks.append("Memory check: the question may assume something the user's memory never states. If "
                              "the excerpts hold something close (a similar name, a nearby date), answer from it and "
                              "name the difference; if nothing is close, say memory does not hold it.")
            elif sufficiency["enough"] < ENOUGH_BELOW:
                blocks.append("Memory check: memory may not hold the whole answer; answer from what the excerpts do "
                              "say and name what is missing.")
        except (CapReached, ProviderError) as e:
            log.warning("sufficiency skipped (%s)", type(e).__name__)
    near_n = NEIGHBOURS if neighbours is None else min(NEIGHBOURS_MAX, max(0, neighbours))
    nearby = _neighbours(conn, providers, r["namespaces"], question, chosen, near_n, request_id) if chosen else []
    near_texts = (dates.annotate(conn, [(c["episode_id"], c["text"], c["ts"]) for c in nearby])
                  if date_notes and nearby else [c["text"] for c in nearby])
    if nearby:      # one list in the order of the conversation; a neighbour is marked so it is not read as a hit
        lines = [(datetime.fromisoformat(it["ts"]), it["episode_id"], it["piece"], t) for it, t in zip(chosen, texts)]
        lines += [(datetime.fromisoformat(c["ts"]), c["episode_id"], 0, "(nearby turn) " + t)
                  for c, t in zip(nearby, near_texts)]
        texts = [t for *_, t in sorted(lines, key=lambda x: x[:3])]
    user = build_answer_user(question, question_date or today, texts)
    for b in blocks:
        user = user.replace("\n\nToday's date:", f"\n\n{b}\n\nToday's date:", 1)
    return {"namespace": r["namespaces"][0], "namespaces": r["namespaces"], "route": route, "style": style,
            "excerpts": [{"chunk_id": it["chunk_id"], "text": it["text"], "ts": it["ts"]} for it in chosen],
            "nearby": [{"episode_id": c["episode_id"], "text": c["text"], "ts": c["ts"]} for c in nearby],
            "blocks": blocks, "ledger": ledger_info, "sufficiency": sufficiency, "prompt": user, "degraded": r["degraded"],
            "screened": r["screened"], "request_id": request_id}


def answer(conn: psycopg.Connection, providers: Providers, caller: Caller, namespace: str | None, question: str,
           question_date: str | None = None, style: str = "auto", request_id: str | None = None,
           when_route: bool = False, ledger_route: bool = False, memory_check: bool = False,
           ledger_notes: bool = True, open_route: str | None = None) -> dict:
    """The reader adapter (outside the brain; DESIGN.md): the context pack, then Haiku writes
    the answer with the measured prompts, for benchmarks and clients with no model. Cap reached is 503."""
    ctx = context(conn, providers, caller, namespace, question, question_date, style, request_id,
                  when_route=when_route, ledger_route=ledger_route, memory_check=memory_check,
                  ledger_notes=ledger_notes, open_route=open_route)
    style = ctx["style"]
    system = ANSWER_PREFERENCE if style == "preference" else ANSWER_DEFAULT
    user = ctx["prompt"]
    r = {"namespaces": ctx["namespaces"], "degraded": ctx["degraded"], "screened": ctx["screened"]}
    chosen, route, ledger_info = ctx["excerpts"], ctx["route"], ctx["ledger"]
    try:
        c = providers.haiku.complete(system, user, ANSWER_MAX_TOKENS)
    except CapReached:
        raise ApiError(503, "cap_reached", "today's provider spend cap is reached; try again after 00:00 UTC") from None
    except ProviderUnavailable:
        raise ApiError(503, "unavailable", "the answer model is not configured") from None
    except ProviderError as e:
        raise ApiError(502, "provider_error", f"the answer model failed (status {e.status})") from None
    return {"namespace": r["namespaces"][0], "answer": c.text, "citations": [it["chunk_id"] for it in chosen],
            "model": c.model, "style": style, "route": route, "ledger": ledger_info, "degraded": r["degraded"],
            "screened": r["screened"], "request_id": request_id}


# ---------------------------------------------------------------- 7.4 forget, purge and restore

_MATCH = "namespace = %s and (id = any(%s::bigint[]) or session_id = %s::text)"


def _one_of(episode_ids, session_id) -> None:
    if bool(episode_ids) == bool(session_id):
        raise ApiError(400, "bad_request", "give exactly one of episode_ids or session_id")


def _purge(conn: psycopg.Connection, ids: list[int]) -> int:
    """Erase the text of forgotten episodes; keep the skeleton (id, namespace, client, session_id, ts, role,
    channel, idempotency_key, deleted_at). Purged episodes cannot be restored."""
    if not ids:
        return 0
    conn.execute("delete from chunk where kind = 'episode_line' and ref_id = any(%s)", (ids,))   # stragglers
    return conn.execute("""update episode set body = '', meta = '{}'::jsonb, speaker = null, purged_at = now()
                           where id = any(%s) and deleted_at is not null and purged_at is null""", (ids,)).rowcount


def _forget_entity(conn: psycopg.Connection, ns: str, entity: str, request_id: str | None) -> dict:
    """Archive what the brain knows about one person or thing: the entity, its claims and page leave every read
    (context, current, history, page, session start, counts, preferences) and its learned directives retire. The
    turns stay (they may hold other facts) and keep being searchable. restore(entity=...) undoes it."""
    eid = ledger._find_exact(conn, ns, entity)
    if eid is None:
        raise ApiError(404, "not_found", "no entity with that name in this namespace")
    conn.execute("update entity set archived_at = now() where id = %s", (eid,))
    retired = conn.execute(
        """update directive d set retired_at = now(), retired_reason = 'entity_forgotten'
           where d.retired_at is null and d.source_claim in (select id from claim where entity_id = %s)""",
        (eid,)).rowcount
    claims = conn.execute("select count(*) from claim where entity_id = %s and status <> 'retracted'",
                          (eid,)).fetchone()[0]
    return {"namespace": ns, "forgotten": {"entity": entity, "claims": claims, "directives": retired},
            "archived": True, "restorable": True, "request_id": request_id}


def forget(conn: psycopg.Connection, caller: Caller, namespace: str | None, episode_ids: list[int] | None = None,
           session_id: str | None = None, entity: str | None = None, purge: bool = False, grace_days: int = 0,
           request_id: str | None = None, confirm_erase: bool = False) -> dict:
    """Forget archives (the owner, 2026-11-09: keep the means to undo anything the user forgets). The episodes are
    tombstoned and what was derived from them leaves every read at once: chunks (and their vectors), time
    mentions, decision_log rows that carry their text, pending jobs; their claims are retracted (a value they had
    replaced holds again) and their pages re-rendered. The user can restore them; nothing is erased unless the
    user asks: purge=True with confirm_erase=True erases the text now, for good (existing backups keep it until
    they rotate out). With grace_days > 0 an operator can also have the nightly worker purge archived text after
    that many days. entity=... archives what is known about one person or thing instead (_forget_entity).
    Unknown ids, or ids in another namespace, are not counted (and not revealed)."""
    ns = caller.namespace(namespace, write=True)
    if entity is not None:
        if episode_ids or session_id or purge:
            raise ApiError(400, "bad_request", "forget an entity on its own (no episode_ids, session_id or purge)")
        return _forget_entity(conn, ns, entity, request_id)
    if purge and not confirm_erase:
        raise ApiError(400, "confirm_erase", "erasing cannot be undone: send confirm_erase=true only when the user "
                                             "asked to delete permanently; plain forget can be restored")
    if purge and not caller.is_admin:
        # A confirm_erase flag can be filled in by a model. Erasing is final, so it takes an admin key, which is
        # not handed to a model: the CLI or REST with the account's admin key (room review v2, group 2).
        raise ApiError(403, "erase_needs_admin", "permanent erase needs the account's admin key (CLI or REST); "
                                                 "this key can forget, which archives and can be restored")
    _one_of(episode_ids, session_id)
    match = (ns, episode_ids or [], session_id)
    eps = [r[0] for r in conn.execute(
        f"update episode set deleted_at = now() where deleted_at is null and {_MATCH} returning id",
        match).fetchall()]
    counts = {"episodes": len(eps), "chunks": 0, "time_mentions": 0, "claims": 0, "claim_candidates": 0, "pages": 0,
              "decision_log": 0, "jobs": 0, "purged": 0}
    if eps:
        chunk_ids = [r[0] for r in conn.execute(
            "delete from chunk where kind = 'episode_line' and ref_id = any(%s) returning id", (eps,)).fetchall()]
        counts["chunks"] = len(chunk_ids)
        counts["time_mentions"] = conn.execute("delete from time_mention where episode_id = any(%s)",
                                               (eps,)).rowcount
        retracted = conn.execute(
            """update claim set status = 'retracted' where episode_id = any(%s) and status <> 'retracted'
               returning entity_id, id""", (eps,)).fetchall()
        counts["claims"] = len(retracted)
        if retracted:     # a retracted claim's search text goes with it
            gone = [r[1] for r in retracted]
            counts["chunks"] += conn.execute("delete from chunk where kind = 'claim' and ref_id = any(%s)",
                                             (gone,)).rowcount
            # a claim superseded only by forgotten claims holds again
            counts["claims_restored"] = conn.execute(
                """update claim o set status = 'active', valid_to = null
                   where o.status = 'superseded'
                     and exists (select 1 from claim_link l where l.to_claim = o.id and l.type = 'supersedes'
                                 and l.reversed_at is null and l.from_claim = any(%s))
                     and not exists (select 1 from claim_link l join claim n on n.id = l.from_claim
                                     where l.to_claim = o.id and l.type = 'supersedes' and l.reversed_at is null
                                       and n.status <> 'retracted')""",
                (gone,)).rowcount
        entities = sorted({r[0] for r in retracted})
        if entities:
            counts["pages"] = conn.execute("""update page set dirty = true
                                              where namespace = %s and entity_id = any(%s) and not dirty""",
                                           (ns, entities)).rowcount
        counts["claim_candidates"] = conn.execute("delete from claim_candidate where episode_id = any(%s)",
                                                  (eps,)).rowcount
        # decision_log rows about these chunks (recall) or these episodes (write path: they carry the turn text)
        counts["decision_log"] = conn.execute(
            """delete from decision_log where input->>'chunk_id' = any(%s::text[])
                                           or input->>'episode_id' = any(%s::text[])""",
            ([str(c) for c in chunk_ids], [str(e) for e in eps])).rowcount
        counts["jobs"] = conn.execute(
            """update job set status = 'skipped', last_error = 'episode forgotten', updated_at = now()
               where status = 'pending' and kind in ('embed_backfill', 'write_path') and ref_id = any(%s)""",
            (eps,)).rowcount
    restorable_until = None
    out = {"namespace": ns, "forgotten": counts, "archived": not purge, "restorable": not purge}
    if purge:
        due = [r[0] for r in conn.execute(
            f"select id from episode where deleted_at is not null and purged_at is null and {_MATCH}",
            match).fetchall()]
        counts["purged"] = _purge(conn, due)
        out["backups"] = ("erased here now; the text stays in existing backups until they rotate out "
                          "(the newest BRAIN_BACKUP_KEEP nightly backups are kept) and is in no new backup")
    elif eps and grace_days > 0:
        restorable_until = conn.execute("select now() + make_interval(days => %s)", (grace_days,)).fetchone()[0]
    return {**out, "restorable_until": _iso(restorable_until), "request_id": request_id}


def purge_forgotten(conn: psycopg.Connection, grace_days: int) -> int:
    """Nightly worker: with grace_days > 0 (an operator setting), erase the text of every episode archived more
    than grace_days ago. At 0 (the default) archived text is kept until the user asks to erase it."""
    if grace_days <= 0:
        return 0
    ids = [r[0] for r in conn.execute(
        """select id from episode where deleted_at is not null and purged_at is null
           and deleted_at < now() - make_interval(days => %s)""", (grace_days,)).fetchall()]
    return _purge(conn, ids)


def _restore_entity(conn: psycopg.Connection, ns: str, entity: str, request_id: str | None) -> dict:
    row = conn.execute("""select id from entity where namespace = %s and archived_at is not null
                          and (lower(name) = %s or %s = any(aliases)) order by archived_at desc limit 1""",
                       (ns, ledger._key(entity), ledger._key(entity))).fetchone()
    if row is None:
        raise ApiError(404, "not_found", "no forgotten entity with that name in this namespace")
    conn.execute("update entity set archived_at = null where id = %s", (row[0],))
    back = conn.execute(
        """update directive d set retired_at = null, retired_reason = null
           where d.retired_reason = 'entity_forgotten' and d.source_claim in (select id from claim where entity_id = %s)""",
        (row[0],)).rowcount
    pages.mark_dirty(conn, ns, row[0])
    return {"namespace": ns, "restored": {"entity": entity, "directives": back}, "request_id": request_id}


def restore(conn: psycopg.Connection, providers: Providers, caller: Caller, namespace: str | None,
            episode_ids: list[int] | None = None, session_id: str | None = None, write_path_enabled: bool = False,
            request_id: str | None = None, dates_enabled: bool = False, entity: str | None = None) -> dict:
    """Undo a forget (the user, or an operator): bring back archived episodes that were not erased. Clears
    deleted_at, rebuilds their chunks (re-embedded, or a backfill job), and brings their claims back: retracted
    claims are active again, what they had superseded is superseded again, their search lines and pages are
    rebuilt. Not restored: decision_log rows forget deleted (they carried the text). entity=... undoes an entity
    forget."""
    ns = caller.namespace(namespace, write=True)
    if entity is not None:
        if episode_ids or session_id:
            raise ApiError(400, "bad_request", "restore an entity on its own (no episode_ids or session_id)")
        return _restore_entity(conn, ns, entity, request_id)
    _one_of(episode_ids, session_id)
    match = (ns, episode_ids or [], session_id)
    rows = conn.execute(
        f"""update episode set deleted_at = null
            where deleted_at is not null and purged_at is null and {_MATCH}
            returning id, ts, ts_offset_min, role, speaker, body""", match).fetchall()
    eps = [({"id": r[0]}, r[1].astimezone(timezone(timedelta(minutes=r[2] or 0))), r[3], r[4], r[5])
           for r in sorted(rows)]
    degraded = _index(conn, providers, ns, eps)
    ids = [out["id"] for out, *_ in eps]
    claims_back = 0
    if ids:
        back = conn.execute(
            """update claim set status = 'active' where episode_id = any(%s) and status = 'retracted'
               returning id, entity_id, valid_from, value, quote""", (ids,)).fetchall()
        claims_back = len(back)
        if back:
            back_ids = [r[0] for r in back]
            conn.execute(   # what the restored claims had superseded is superseded again
                """update claim o set status = 'superseded', valid_to = n.valid_from
                   from claim_link l join claim n on n.id = l.from_claim
                   where l.type = 'supersedes' and l.reversed_at is null and l.from_claim = any(%s)
                     and l.to_claim = o.id and o.status = 'active'""", (back_ids,))
            for cid, eid, vf, value, quote in back:
                slot, ent = conn.execute("""select s.name, n.name from claim c join slot s on s.id = c.slot_id
                                            join entity n on n.id = c.entity_id where c.id = %s""", (cid,)).fetchone() \
                    or (None, None)
                if slot is not None:
                    conn.execute("insert into chunk (namespace, kind, ref_id, text) values (%s, 'claim', %s, %s)",
                                 (ns, cid, f"[{vf}] {ent}: {slot.replace('_', ' ')} = {value} (\"{quote[:300]}\")"))
                pages.mark_dirty(conn, ns, eid)
        # claims that never existed for these turns (write path off at the time) come from a new write_path job
    if write_path_enabled:
        _queue_write_path(conn, ns, eps)
    if dates_enabled:
        _dates(conn, providers, eps)
    purged = conn.execute(f"select count(*) from episode where purged_at is not null and {_MATCH}",
                          match).fetchone()[0]
    return {"namespace": ns, "restored": [out for out, *_ in eps], "claims_restored": claims_back,
            "not_restorable_erased": purged, "degraded": degraded, "request_id": request_id}


# ---------------------------------------------------------------- 7.5 directives and session start

def _directive(r) -> dict:
    return {"id": r[0], "scope": r[1], "text": r[2], "pinned": r[3], "created_by": r[4], "created_at": _iso(r[5]),
            "status": r[6], "screen": r[7]}


_DCOLS = "id, scope, text, pinned, created_by, created_at, status, screen"
PENDING_NOTE = ("Saved as pending. It takes effect only after the user confirms it outside this conversation "
                "(python -m brain.cli admin confirm-directive ID).")


def list_directives(conn: psycopg.Connection, caller: Caller, namespace: str | None,
                    request_id: str | None = None) -> dict:
    ns = caller.namespace(namespace)
    # A pending directive is a proposal, not one of the user's rules: only its proposer and an admin see it.
    # Every other client would read it, under "the user's standing directives", as something to follow.
    rows = conn.execute(f"""select {_DCOLS} from directive
                            where namespace = %s and retired_at is null
                              and (status = 'confirmed' or %s or created_by = %s)
                            order by created_at, id""", (ns, caller.is_admin, caller.name)).fetchall()
    return {"namespace": ns, "directives": [_directive(r) for r in rows], "request_id": request_id}


def add_directive(conn: psycopg.Connection, caller: Caller, namespace: str | None, text: str, scope: str = "all",
                  request_id: str | None = None, providers: Providers | None = None) -> dict:
    """Propose a directive (room review v2, group 2). Whoever calls, it is stored PENDING: unpinned, scoped to the
    calling client, and served to nobody. Only confirm_directive (an admin call, never an MCP tool) pins it or
    widens it to every client; the scope asked for is returned as `requested_scope` for the user to decide on.
    The text is screened first (preferences.screen_texts); a text that works against the user is filed as
    rejected and never becomes confirmable."""
    ns = caller.namespace(namespace, write=True)
    if scope != "all" and not valid_ns(scope):
        raise ApiError(400, "bad_scope", "scope must be 'all' or a client name")
    text = text.strip()
    screen = preferences.screen_texts(providers.jev, [text])[0] if providers is not None else None
    bad = bool(screen and screen["flagged"])
    r = conn.execute(f"""insert into directive (namespace, scope, text, created_by, pinned, status, screen,
                                                retired_at, retired_reason)
                         values (%s, %s, %s, %s, false, %s, %s, %s, %s) returning {_DCOLS}""",
                     (ns, caller.name, text, caller.name, "rejected" if bad else "pending",
                      Jsonb(screen) if screen else None, datetime.now(timezone.utc) if bad else None,
                      "screened_out" if bad else None)).fetchone()
    if screen:
        conn.execute("""insert into decision_log (namespace, stage, decider, model_version, input, output,
                                                  probability, threshold, outcome, request_id)
                        values (%s, 'directive.screen', 'jev', %s, %s, %s, %s, %s, %s, %s)""",
                     (ns, screen["model"], Jsonb({"directive_id": r[0]}),
                      Jsonb({"injection": screen["injection"], "rule": preferences.DIRECTIVE_RULE}),
                      screen["subverts_user"], preferences.SUBVERT_THRESHOLD, "rejected" if bad else "pending",
                      request_id))
    out = {"namespace": ns, "directive": _directive(r), "pending": not bad, "rejected": bad,
           "requested_scope": scope, "request_id": request_id}
    out["note"] = ("Not saved as a directive: the screen judged that it works against the user." if bad
                   else PENDING_NOTE)
    return out


def confirm_directive(conn: psycopg.Connection, caller: Caller, namespace: str | None, directive_id: int,
                      all_clients: bool = False, request_id: str | None = None) -> dict:
    """The user's confirmation, outside any model: pin a pending directive so it is served, and with all_clients
    widen it from the client that proposed it to every client. Admin keys only (the CLI or REST); there is no MCP
    tool for this. Confirming a confirmed directive again can widen it."""
    if not caller.is_admin:
        raise ApiError(403, "forbidden", "confirming a directive needs the account's admin key")
    ns = caller.namespace(namespace, write=True)
    r = conn.execute(f"""update directive set status = 'confirmed', pinned = true,
                                confirmed_at = now(), confirmed_by = %s,
                                scope = case when %s then 'all' else scope end
                         where id = %s and namespace = %s and retired_at is null and status in ('pending', 'confirmed')
                         returning {_DCOLS}""", (caller.label, bool(all_clients), directive_id, ns)).fetchone()
    if r is None:
        raise ApiError(404, "not_found", "no pending or live directive with that id in this namespace")
    conn.execute("""insert into decision_log (namespace, stage, decider, input, output, outcome, request_id)
                    values (%s, 'directive.confirm', 'code', %s, %s, 'confirmed', %s)""",
                 (ns, Jsonb({"directive_id": directive_id}),
                  Jsonb({"by": caller.label, "human": True, "all_clients": bool(all_clients)}), request_id))
    return {"namespace": ns, "directive": _directive(r), "confirmed": True, "request_id": request_id}


def retire_directive(conn: psycopg.Connection, caller: Caller, namespace: str | None, directive_id: int,
                     request_id: str | None = None) -> dict:
    """Retire a directive. An admin can retire any; any other key only a pending one it proposed itself, so a
    client cannot silently remove one of the user's confirmed rules."""
    ns = caller.namespace(namespace, write=True)
    r = conn.execute(f"""update directive set retired_at = now()
                         where id = %s and namespace = %s and retired_at is null
                           and (%s or (status = 'pending' and created_by = %s)) returning {_DCOLS}""",
                     (directive_id, ns, caller.is_admin, caller.name)).fetchone()
    if r is None:
        raise ApiError(404, "not_found", "no live directive with that id in this namespace that this key may retire")
    return {"namespace": ns, "directive": _directive(r), "retired": True, "request_id": request_id}


def session_start(conn: psycopg.Connection, caller: Caller, namespace: str | None,
                  request_id: str | None = None) -> dict:
    """The prompt index a client loads at the start of a session (DESIGN.md interop): pinned directives for all
    clients or for this one, plus the summaries of the most-mentioned entity pages (build step 8)."""
    ns = caller.namespace(namespace)
    rows = conn.execute(f"""select {_DCOLS} from directive
                            where namespace = %s and retired_at is null and pinned and status = 'confirmed'
                              and scope in ('all', %s)
                            order by created_at, id""", (ns, caller.name)).fetchall()
    directives = [_directive(r) for r in rows]
    summaries = pages.session_summaries(conn, ns)
    parts = []
    if directives:
        parts.append("Standing directives from the user's memory:\n" + "\n".join(f"- {d['text']}" for d in directives))
    if summaries:
        parts.append("People and things in the user's memory (ask memory_page for more):\n"
                     + "\n".join(f"- {p['summary']}" for p in summaries))
    return {"namespace": ns, "directives": directives, "pages": summaries, "prompt": "\n\n".join(parts),
            "request_id": request_id}


def _ledger_rows(conn, ns: str, entity: str, slot: str | None):
    eid = ledger._find_exact(conn, ns, entity)
    if eid is None:
        raise ApiError(404, "not_found", "no entity with that name in this namespace")
    rows = conn.execute(
        """select s.name, s.cardinality, c.id, c.value, c.valid_from, c.valid_to, c.status, c.polarity, c.kind, c.quote,
                  c.episode_id
           from claim c join slot s on s.id = c.slot_id join episode e on e.id = c.episode_id
           where c.entity_id = %s and c.status <> 'retracted' and (%s::text is null or s.name = %s)
           order by s.name, coalesce(c.valid_from, e.ts::date), e.ts, c.id""", (eid, slot, slot)).fetchall()
    if slot is not None and not rows:
        raise ApiError(404, "not_found", "no facts in that slot for that entity")
    return [{"slot": r[0], "cardinality": r[1], "claim_id": r[2], "value": r[3],
             "valid_from": r[4].isoformat() if r[4] else None, "valid_to": r[5].isoformat() if r[5] else None,
             "status": r[6], "polarity": r[7], "kind": r[8], "quote": r[9], "episode_id": r[10]} for r in rows]


def get_current(conn: psycopg.Connection, caller: Caller, namespace: str | None, entity: str,
                slot: str | None = None, request_id: str | None = None) -> dict:
    """What holds now for an entity (build step 5): per slot, the active asserted values (one-value slots: the
    latest), plus contested values flagged. Exact names; no model call."""
    ns = caller.namespace(namespace)
    rows = _ledger_rows(conn, ns, entity, ledger._slug(slot) if slot else None)
    out: dict[str, list] = {}
    for r in rows:
        if r["status"] in ("active", "contested") and r["polarity"] in ("asserted", "corrected", "reported"):
            out.setdefault(r["slot"], []).append(r)
    facts = {s: (v[-1:] if v[-1]["cardinality"] == "one" and not any(x["status"] == "contested" for x in v) else v)
             for s, v in out.items()}
    return {"namespace": ns, "entity": entity, "current": facts, "request_id": request_id}


def get_history(conn: psycopg.Connection, caller: Caller, namespace: str | None, entity: str, slot: str,
                request_id: str | None = None) -> dict:
    """Every claim on one slot of an entity, oldest first, with status (build step 5)."""
    ns = caller.namespace(namespace)
    return {"namespace": ns, "entity": entity, "slot": ledger._slug(slot),
            "history": _ledger_rows(conn, ns, entity, ledger._slug(slot)), "request_id": request_id}


def resolve_conflict(conn: psycopg.Connection, caller: Caller, namespace: str | None, keep_claim_id: int,
                     request_id: str | None = None) -> dict:
    """The user settles a contested claim (DESIGN.md interop): the kept claim becomes active and settled; the
    claims it was contested with are superseded from its date."""
    ns = caller.namespace(namespace, write=True)
    row = conn.execute("select valid_from, status from claim where id = %s and namespace = %s",
                       (keep_claim_id, ns)).fetchone()
    if row is None or row[1] != "contested":
        raise ApiError(404, "not_found", "no contested claim with that id in this namespace")
    others = [r[0] for r in conn.execute(
        """select case when l.from_claim = %s then l.to_claim else l.from_claim end from claim_link l
           where l.type = 'contradicts' and l.reversed_at is null and %s in (l.from_claim, l.to_claim)""",
        (keep_claim_id, keep_claim_id))]
    conn.execute("update claim set status = 'active', settled_at = now() where id = %s", (keep_claim_id,))
    superseded = conn.execute("""update claim set status = 'superseded', valid_to = %s
                                 where id = any(%s) and status = 'contested'""", (row[0], others)).rowcount
    for o in others:
        ledger._link(conn, keep_claim_id, o, "supersedes", "user.resolve", request_id)
    ent = conn.execute("select entity_id from claim where id = %s", (keep_claim_id,)).fetchone()[0]
    pages.mark_dirty(conn, ns, ent)
    return {"namespace": ns, "kept": keep_claim_id, "superseded": superseded, "request_id": request_id}


def get_page(conn: psycopg.Connection, caller: Caller, namespace: str | None, entity: str,
             request_id: str | None = None) -> dict:
    """An entity's page (build step 8): code-rendered from its claims, every line citing its claim."""
    ns = caller.namespace(namespace)
    p = pages.get(conn, ns, entity)
    if p is None:
        raise ApiError(404, "not_found", "no page for that name in this namespace")
    return {"namespace": ns, **p, "request_id": request_id}


# ---------------------------------------------------------------- 7.6 export

EXPORT_SQL = """select id, session_id, ts, recorded_at, role, speaker, channel, body, meta, client, idempotency_key
                from episode where namespace = %s and deleted_at is null
                order by min(ts) over (partition by session_id), session_id nulls first, ts, id"""


def _ndjson(rows: Iterator) -> Iterator[str]:
    for r in rows:
        yield json.dumps({"id": r[0], "session_id": r[1], "ts": _iso(r[2]), "recorded_at": _iso(r[3]),
                          "role": r[4], "speaker": r[5], "channel": r[6], "text": r[7], "meta": r[8],
                          "client": r[9], "idempotency_key": r[10]}, ensure_ascii=False) + "\n"


def _markdown(ns: str, rows: Iterator) -> Iterator[str]:
    yield (f"# Brain-Memory export: namespace {ns}\n\nExported {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
           ". Live episodes only, grouped by session, oldest first. Times are UTC.\n")
    session, day = object(), None
    for r in rows:
        ts = r[2].astimezone(timezone.utc)
        if r[1] != session:
            session, day = r[1], None
            yield f"\n## Session {r[1]}\n" if r[1] is not None else "\n## No session\n"
        if ts.date() != day:
            day = ts.date()
            yield f"\n### {day.isoformat()}\n\n"
        who = r[4] + (f" ({r[5]})" if r[5] and r[5] != r[4] else "")
        src = f", {r[8]['filename']}" if isinstance(r[8], dict) and r[8].get("filename") else ""
        body = r[7].replace("\n", "\n  ")
        yield f"- {ts.strftime('%H:%M')} {who}, {r[6]}{src}, episode {r[0]}: {body}\n"


def export(pool, caller: Caller, namespace: str | None, fmt: str = "markdown") -> tuple[str, str, Iterator[str]]:
    """Returns (namespace, media type, line iterator). The scope check happens now; the rows stream from a
    server-side cursor while the response is written."""
    ns = caller.namespace(namespace)
    if fmt not in ("markdown", "jsonl"):
        raise ApiError(400, "bad_format", "format must be markdown or jsonl")

    def rows() -> Iterator:
        with tenant(pool, caller.account_id) as conn, conn.cursor(name="brain_export") as cur:
            cur.itersize = 500
            cur.execute(EXPORT_SQL, (ns,))
            yield from cur

    if fmt == "jsonl":
        return ns, "application/x-ndjson", _ndjson(rows())
    return ns, "text/markdown; charset=utf-8", _markdown(ns, rows())


# Everything the brain holds for one account, as it stands: the user's own copy of their data (release list B5).
# One JSON object per line, each with "t": the turns of every namespace, the claims clients sent with them, the
# ledger's facts with their status and dates, and the directives. Archived turns and what was derived from them
# are not in it (restore brings those back first); retracted facts are left out.
ACCOUNT_EXPORT = (
    ("episode", """select jsonb_build_object('namespace', namespace, 'id', id, 'session_id', session_id, 'ts', ts,
                       'recorded_at', recorded_at, 'role', role, 'speaker', speaker, 'channel', channel, 'text', body,
                       'meta', meta, 'client', client)
                   from episode where deleted_at is null order by namespace, ts, id"""),
    ("claim_sent", """select jsonb_build_object('namespace', k.namespace, 'episode_id', k.episode_id,
                          'mention', k.mention, 'slot', k.slot, 'value', k.value, 'time_phrase', k.time_phrase,
                          'polarity', k.polarity, 'quote', k.quote, 'status', k.status, 'drop_reason', k.drop_reason,
                          'sent_by', k.extractor, 'created_at', k.created_at)
                      from claim_candidate k join episode e on e.id = k.episode_id
                      where e.deleted_at is null order by k.id"""),
    ("fact", """select jsonb_build_object('namespace', c.namespace, 'about', n.name, 'attribute', s.name,
                    'value', c.value, 'polarity', c.polarity, 'kind', c.kind, 'status', c.status,
                    'valid_from', c.valid_from, 'valid_to', c.valid_to, 'quote', c.quote, 'episode_id', c.episode_id)
                from claim c join entity n on n.id = c.entity_id left join slot s on s.id = c.slot_id
                where c.status <> 'retracted' order by c.id"""),
    ("directive", "select to_jsonb(d) - 'account_id' from directive d order by d.id"),
)


def export_account(pool, caller: Caller) -> Iterator[str]:
    """NDJSON lines of the caller's whole account. Needs the account's admin key: it crosses every namespace."""
    caller.require_admin()

    def lines() -> Iterator[str]:
        yield json.dumps({"t": "export", "account": caller.account,
                          "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "holds": [t for t, _ in ACCOUNT_EXPORT]}) + "\n"
        with tenant(pool, caller.account_id) as conn:
            for kind, sql in ACCOUNT_EXPORT:
                with conn.cursor(name=f"brain_export_{kind}") as cur:
                    cur.itersize = 500
                    cur.execute(sql)
                    for (row,) in cur:
                        yield json.dumps({"t": kind, **row}, ensure_ascii=False, default=str) + "\n"
    return lines()


# ---------------------------------------------------------------- 7.7 features of later build steps (none left)

NOT_BUILT: dict[str, int] = {}     # every DESIGN.md build step is built (2026-11-09)


def not_built(feature: str) -> ApiError:
    step = NOT_BUILT[feature]
    return ApiError(501, "not_built", f"{feature} comes with DESIGN.md build step {step}", extra={"build_step": step})
