-- Migration 003: accounts and row-level security (REFERENCE.md, decided 2026-11-08).
--
-- Every customer is an account: an individual is an account with one member; a business (several members) comes
-- later. Namespaces and client keys belong to an account. Customers reach the brain only through the service API
-- with their key; only the service connects to Postgres.
--
-- How isolation works: the service connects as the schema owner. For every call it opens a transaction, runs
-- `set local role brain_app` (a role that is not the owner and has no BYPASSRLS) and
-- `set_config('brain.tenant', <account id from the key>, true)`. The policies below then hide every other
-- account's rows, even from a query that forgets a filter, and new rows take account_id from brain.tenant.
-- Without a tenant set, brain_app sees nothing and cannot insert (fail closed). Key lookup and maintenance run as
-- the owner, outside RLS.
--
-- Not per account: attribute (a shared, read-only vocabulary), spend (the operator's provider bill and daily cap)
-- and schema_migrations.
set search_path = brain, extensions, public;

create table if not exists account (
    id           bigint generated always as identity primary key,
    name         text not null unique check (name ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
    kind         text not null default 'individual' check (kind in ('individual', 'business')),
    created_at   timestamptz not null default now(),
    disabled_at  timestamptz
);
alter table account enable row level security;

-- The operator's own account. BRAIN_ADMIN_KEY acts in it, and rows that exist before this migration move into it.
insert into account (name) values ('owner') on conflict (name) do nothing;

-- account_id on every per-account table: backfill existing rows to the owner account, then default to the
-- transaction's tenant (null without one, so the not-null constraint refuses the insert).
do $$
declare
    t text;
    owner_id bigint := (select id from brain.account where name = 'owner');
begin
    foreach t in array array['api_client', 'job', 'episode', 'time_mention', 'entity', 'claim', 'claim_link',
                             'directive', 'page', 'chunk', 'decision_log'] loop
        execute format('alter table brain.%I add column if not exists account_id bigint references brain.account (id)', t);
        execute format('update brain.%I set account_id = %s where account_id is null', t, owner_id);
        execute format($f$alter table brain.%I alter column account_id
                           set default nullif(current_setting('brain.tenant', true), '')::bigint$f$, t);
        execute format('alter table brain.%I alter column account_id set not null', t);
    end loop;
end $$;

-- Uniqueness is per account now.
alter table api_client drop constraint if exists api_client_name_key;
create unique index if not exists api_client_account_name on api_client (account_id, name);

drop index if exists episode_idem;
create unique index if not exists episode_idem on episode (account_id, namespace, idempotency_key)
    where idempotency_key is not null;

-- Lead the namespace indexes with account_id (the policies add account_id = tenant to every query).
drop index if exists episode_ns_ts;
create index if not exists episode_ns_ts on episode (account_id, namespace, ts);
drop index if exists episode_session;
create index if not exists episode_session on episode (account_id, namespace, session_id, ts);
drop index if exists chunk_ns;
create index if not exists chunk_ns on chunk (account_id, namespace, kind);
drop index if exists directive_ns;
create index if not exists directive_ns on directive (account_id, namespace) where retired_at is null;
drop index if exists entity_ns_name;
create index if not exists entity_ns_name on entity (account_id, namespace, lower(name));
drop index if exists claim_slot;
create index if not exists claim_slot on claim (account_id, namespace, entity_id, attribute_path) where status = 'active';
create index if not exists decision_log_account on decision_log (account_id, ts);

-- The role the service switches to for every call. Roles are cluster-wide, so create it only once.
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'brain_app') then
        create role brain_app nologin noinherit nobypassrls;
    end if;
    -- The connecting user (the schema owner) must be able to `set role brain_app`.
    execute format('grant brain_app to %I', current_user);
end $$;

grant usage on schema brain to brain_app;
grant select, insert, update, delete on api_client, job, episode, time_mention, entity, claim, claim_link,
    directive, page, chunk, decision_log to brain_app;
grant select on account, attribute to brain_app;
grant usage, select on all sequences in schema brain to brain_app;

-- pgvector's type and operators live in schema extensions. On Supabase the owner may lack the grant option;
-- there the extensions schema is already usable by every role.
do $$
begin
    execute 'grant usage on schema extensions to brain_app';
exception when insufficient_privilege then
    raise notice 'could not grant usage on schema extensions to brain_app; relying on existing grants';
end $$;

-- One policy per table: rows of the transaction's tenant only, for reads and writes alike.
do $$
declare t text;
begin
    foreach t in array array['api_client', 'job', 'episode', 'time_mention', 'entity', 'claim', 'claim_link',
                             'directive', 'page', 'chunk', 'decision_log'] loop
        execute format('drop policy if exists tenant_isolation on brain.%I', t);
        execute format($f$create policy tenant_isolation on brain.%I for all to brain_app
                           using (account_id = nullif(current_setting('brain.tenant', true), '')::bigint)
                           with check (account_id = nullif(current_setting('brain.tenant', true), '')::bigint)$f$, t);
    end loop;
    drop policy if exists tenant_read on brain.account;
    create policy tenant_read on brain.account for select to brain_app
        using (id = nullif(current_setting('brain.tenant', true), '')::bigint);
    drop policy if exists shared_read on brain.attribute;
    create policy shared_read on brain.attribute for select to brain_app using (true);
end $$;
