-- Room review v2, item 1 (2026-11-09): every ledger change says where it came from and can be undone.
-- claim_link: which step ended a value (write.relate, nightly.merge_slots, nightly.plans, recheck) and in which run;
--             reversed_at marks a supersession that a later re-check undid (never a booster label).
-- claim:      orig_slot_id and merged_run for a claim moved by merge_slots, so a night's merges can be undone.
-- slot_merge: one row per slot merged away in a run, enough to recreate the slot on undo.
-- decision_log.reversed: a decision a later re-check reversed; excluded from booster training (the booster plan).
set search_path = brain, extensions, public;

alter table claim_link add column if not exists source text;
alter table claim_link add column if not exists run_id text;
alter table claim_link add column if not exists reversed_at timestamptz;

alter table claim add column if not exists orig_slot_id bigint;
alter table claim add column if not exists merged_run text;
create index if not exists claim_merged_run on claim (account_id, merged_run) where merged_run is not null;

alter table decision_log add column if not exists reversed boolean not null default false;

create table if not exists slot_merge (
    id           bigint generated always as identity primary key,
    account_id   bigint not null default nullif(current_setting('brain.tenant', true), '')::bigint
                 references account (id),
    namespace    text not null,
    run_id       text not null,
    entity_id    bigint not null references entity (id) on delete cascade,
    gone_slot_id bigint not null,                      -- the merged-away slot's id (the row itself is deleted)
    gone_name    text not null,
    gone_cardinality text not null,
    keep_slot_id bigint not null,                      -- no FK: the row must outlive a later merge of this slot
    keep_cardinality_before text not null,             -- the kept slot's cardinality before the merge
    claim_ids    bigint[] not null,
    at           timestamptz not null default now(),
    undone_at    timestamptz
);
create index if not exists slot_merge_run on slot_merge (account_id, run_id);

alter table slot_merge enable row level security;
grant select, insert, update, delete on slot_merge to brain_app;
drop policy if exists tenant_isolation on slot_merge;
create policy tenant_isolation on slot_merge for all to brain_app
    using (account_id = nullif(current_setting('brain.tenant', true), '')::bigint)
    with check (account_id = nullif(current_setting('brain.tenant', true), '')::bigint);

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        execute 'revoke all on brain.slot_merge from anon';
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        execute 'revoke all on brain.slot_merge from authenticated';
    end if;
end $$;
