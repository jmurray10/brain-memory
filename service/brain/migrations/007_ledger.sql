-- Build step 5 (DESIGN.md write steps 6-9): entities, slots and supersession.
-- Slots grow from use, per account and per entity (DESIGN.md to step 5): the global
-- `attribute` registry is readable by every account, so slot names built from a user's own words must not live
-- there; it stays for a curated shared taxonomy later. A claim now points at a slot (claim.slot_id) and
-- attribute_path becomes optional.
set search_path = brain, extensions, public;

create table if not exists slot (
    id           bigint generated always as identity primary key,
    account_id   bigint not null default nullif(current_setting('brain.tenant', true), '')::bigint
                 references account (id),
    namespace    text not null,
    entity_id    bigint not null references entity (id) on delete cascade,
    name         text not null,                         -- snake_case, from the client's proposal or merged by Jev
    cardinality  text not null check (cardinality in ('one', 'many', 'event')),
    created_at   timestamptz not null default now(),
    unique (account_id, namespace, entity_id, name)
);
create index if not exists slot_entity on slot (account_id, namespace, entity_id);

alter table entity add column if not exists summary text;            -- short code-built description for alignment
alter table entity add column if not exists mentions int not null default 0;
alter table entity add column if not exists last_seen timestamptz;
create index if not exists entity_recent on entity (account_id, namespace, last_seen desc) where merged_into is null;

alter table claim alter column attribute_path drop not null;
alter table claim add column if not exists slot_id bigint references slot (id) on delete cascade;
alter table claim add column if not exists candidate_id bigint references claim_candidate (id) on delete set null;
alter table claim add column if not exists polarity text not null default 'asserted'
    check (polarity in ('asserted', 'negated', 'planned', 'hypothetical', 'corrected', 'reported'));
alter table claim add column if not exists time_phrase text;
alter table claim add column if not exists tentative boolean not null default false;   -- entity link to review
create index if not exists claim_slot_active on claim (account_id, namespace, slot_id) where status = 'active';
create index if not exists claim_episode on claim (account_id, episode_id);

alter table slot enable row level security;
grant select, insert, update, delete on slot to brain_app;
drop policy if exists tenant_isolation on slot;
create policy tenant_isolation on slot for all to brain_app
    using (account_id = nullif(current_setting('brain.tenant', true), '')::bigint)
    with check (account_id = nullif(current_setting('brain.tenant', true), '')::bigint);

do $$
begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        execute 'revoke all on brain.slot from anon';
    end if;
    if exists (select 1 from pg_roles where rolname = 'authenticated') then
        execute 'revoke all on brain.slot from authenticated';
    end if;
end $$;
