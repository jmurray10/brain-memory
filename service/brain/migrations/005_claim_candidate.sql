-- Build step 4 (DESIGN.md write path steps 2, 4 and 5): checked claim candidates. A candidate is a claim with an
-- exact quote that passed the quote check and Jev's verify, but whose entity and slot are not resolved yet; build
-- step 5 (entity resolution, slot placement, relate) turns candidates into rows of `claim`. Derived from episodes,
-- so forget removes them and a backup does not carry them.
set search_path = brain, extensions, public;

create table if not exists claim_candidate (
    id              bigint generated always as identity primary key,
    account_id      bigint not null default nullif(current_setting('brain.tenant', true), '')::bigint
                    references account (id),
    namespace       text not null,
    episode_id      bigint not null references episode (id) on delete cascade,
    mention         text not null default '',          -- as written in the turn; '' = the speaker
    slot            text not null,                      -- a listed slot, or 'new:<name>' (no slot tree until step 5)
    value           text not null,
    time_phrase     text,
    polarity        text not null check (polarity in ('asserted', 'negated', 'planned', 'hypothetical',
                                                       'corrected', 'reported')),
    quote           text not null,
    p_hallucinated  real,
    p_off_target    real,
    p_incomplete    real,
    status          text not null default 'pending' check (status in ('pending', 'dropped', 'placed')),
    drop_reason     text,
    extractor       text not null,                      -- model version that extracted it
    created_at      timestamptz not null default now()
);
create index if not exists claim_candidate_episode on claim_candidate (account_id, episode_id);
create index if not exists claim_candidate_pending on claim_candidate (account_id, namespace) where status = 'pending';

alter table claim_candidate enable row level security;
grant select, insert, update, delete on claim_candidate to brain_app;
drop policy if exists tenant_isolation on claim_candidate;
create policy tenant_isolation on claim_candidate for all to brain_app
    using (account_id = nullif(current_setting('brain.tenant', true), '')::bigint)
    with check (account_id = nullif(current_setting('brain.tenant', true), '')::bigint);

-- Supabase's API roles get nothing here, as on every other brain table (no-op where the roles do not exist).
do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        execute 'revoke all on brain.claim_candidate from anon';
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        execute 'revoke all on brain.claim_candidate from authenticated';
    end if;
end $$;
