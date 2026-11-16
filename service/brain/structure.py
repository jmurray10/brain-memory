"""Structural scoring and the cold tier (DESIGN.md nightly step 5; build step 9).

Nightly, per account, each episode line gets a heat score from the brain's derivation graph and its use:
  - recall traffic: how often recall kept the line for an answer (decision_log recall.rerank, outcome kept);
  - provenance: how many claims its turn produced, whether one is still active, whether one has settled;
  - worth: triage's keep probability for its turn (write.triage);
  - age: days since the turn.
A line whose turn backs a live directive or an active settled claim is pinned (never cold).

Policy "structural" (the default) combines these; policy "recency" (the step 9 baseline, DESIGN.md: "beat
recency-only retention at 25% cold first") uses age alone. Then, per namespace, the coldest COLD_FRACTION of lines
older than COLD_MIN_AGE_DAYS go cold: out of the partial hot vector index (migration 009), still in keyword search
and SQL; a line whose score rises warms again. Nothing is deleted. COLD_FRACTION 0 (the default) keeps all hot.

The weights are a hand-set prior until logged recall outcomes exist to fit them (DESIGN.md: CatBoost first, then
the structural encode's features).
"""
from __future__ import annotations

PINNED_HEAT = 1e6
W_TRAFFIC, W_ACTIVE, W_SETTLED, W_CLAIMS, W_WORTH, W_AGE_PER_YEAR = 2.0, 1.5, 1.0, 0.5, 1.0, 1.0

_SCORE_SQL = """
with traffic as (
    select (input->>'chunk_id')::bigint as cid, count(*) as n from decision_log
    where stage = 'recall.rerank' and outcome = 'kept' and input ? 'chunk_id' group by 1
),
prov as (
    select episode_id, count(*) as n, bool_or(status = 'active') as active,
           bool_or(status = 'active' and settled_at is not null) as settled
    from claim where status <> 'retracted' group by episode_id
),
pinned as (
    select c.episode_id from directive d join claim c on c.id = d.source_claim where d.retired_at is null
),
worth as (
    select (input->>'episode_id')::bigint as eid, max(probability) as p from decision_log
    where stage = 'write.triage' and input ? 'episode_id' group by 1
),
scored as (
    select k.id,
           case when exists (select 1 from pinned p where p.episode_id = e.id) or coalesce(pv.settled, false)
                then %(pinned)s
                when %(policy)s = 'recency'
                then -extract(epoch from (now() - e.ts)) / 86400.0
                else %(w_traffic)s * ln(1 + coalesce(t.n, 0))
                   + %(w_active)s * coalesce(pv.active, false)::int
                   + %(w_settled)s * coalesce(pv.settled, false)::int
                   + %(w_claims)s * ln(1 + coalesce(pv.n, 0))
                   + %(w_worth)s * coalesce(w.p, 0)
                   - %(w_age)s * extract(epoch from (now() - e.ts)) / (365.0 * 86400.0)
           end as heat
    from chunk k join episode e on e.id = k.ref_id
    left join traffic t on t.cid = k.id
    left join prov pv on pv.episode_id = e.id
    left join worth w on w.eid = e.id
    where k.kind = 'episode_line' and e.deleted_at is null
)
update chunk k set heat = s.heat from scored s where k.id = s.id
"""


def score(conn, policy: str = "structural") -> int:
    if policy not in ("structural", "recency"):
        raise ValueError(f"unknown cold policy {policy!r}")
    return conn.execute(_SCORE_SQL, {"policy": policy, "pinned": PINNED_HEAT, "w_traffic": W_TRAFFIC,
                                     "w_active": W_ACTIVE, "w_settled": W_SETTLED, "w_claims": W_CLAIMS,
                                     "w_worth": W_WORTH, "w_age": W_AGE_PER_YEAR}).rowcount


def apply_cold(conn, fraction: float, min_age_days: int) -> dict:
    """Per namespace: the coldest `fraction` of all episode lines, among those older than min_age_days and not
    pinned, go cold; every other line is hot."""
    fraction = max(0.0, min(fraction, 0.95))
    cold = conn.execute(
        """with ranked as (
               select k.id, k.namespace,
                      row_number() over (partition by k.namespace order by k.heat nulls first, k.id) as r,
                      count(*) over (partition by k.namespace) as n_eligible
               from chunk k join episode e on e.id = k.ref_id
               where k.kind = 'episode_line' and e.ts < now() - make_interval(days => %(age)s)
                 and coalesce(k.heat, 0) < %(pinned)s
           ),
           totals as (select namespace, count(*) as n from chunk where kind = 'episode_line' group by namespace),
           chosen as (
               select r.id from ranked r join totals t on t.namespace = r.namespace
               where r.r <= floor(%(f)s * t.n)
           )
           update chunk k set cold = (k.id in (select id from chosen))
           where k.kind = 'episode_line' and k.cold is distinct from (k.id in (select id from chosen))
           returning k.cold""", {"age": min_age_days, "pinned": PINNED_HEAT, "f": fraction}).fetchall()
    went_cold = sum(1 for (c,) in cold if c)
    total_cold = conn.execute("select count(*) from chunk where kind = 'episode_line' and cold").fetchone()[0]
    return {"went_cold": went_cold, "warmed": len(cold) - went_cold, "cold": total_cold}


def run(conn, policy: str, fraction: float, min_age_days: int) -> dict:
    scored = score(conn, policy)
    return {"policy": policy, "scored": scored, **apply_cold(conn, fraction, min_age_days)}
