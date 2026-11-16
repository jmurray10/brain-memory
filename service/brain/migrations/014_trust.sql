-- Room review consensus v2, group 2 (trust): one brain is shared by every client, so what a client may do to it
-- is decided per key and by the user, not by whatever model holds the key.

set search_path = brain, extensions, public;

-- 1. Provenance. A key speaks in the user's own words only when the user has said so (speaks_as_user). A
--    role=user turn from any other key is stored as role 'agent_reported': searchable, screened by the strict
--    rule, and it carries no claims.
alter table api_client add column if not exists speaks_as_user boolean not null default false;
-- An account's admin key is its owner's own key (accounts.create gives a new account's first key the permission).
update api_client set speaks_as_user = true where is_admin and revoked_at is null;
alter table episode drop constraint if exists episode_role_check;
alter table episode add constraint episode_role_check
    check (role in ('user', 'assistant', 'system', 'tool', 'attachment', 'agent_reported'));

-- 2. Directives. A model can propose one; only the user's confirmation, outside any model, makes it take effect
--    (pinned) or widens it to every client. New rows are pending, unpinned and scoped to the client that made
--    them. Rows that existed before this migration keep working and are marked for review (confirmed_by).
alter table directive add column if not exists status text not null default 'confirmed'
    check (status in ('pending', 'confirmed', 'rejected'));
alter table directive add column if not exists confirmed_at timestamptz;
alter table directive add column if not exists confirmed_by text;
alter table directive add column if not exists screen jsonb;
update directive set confirmed_by = 'migration-014' where status = 'confirmed' and confirmed_by is null;
alter table directive alter column status set default 'pending';
alter table directive alter column pinned set default false;

-- 3. decision_log: where a row came from. Rows written before this migration are 'unknown'. After it, a row is
--    'client' unless the process says otherwise (BRAIN_LOG_SOURCE=benchmark in the eval scripts; db.tenant sets
--    brain.source for the transaction). The account is already on every row (migration 003).
alter table decision_log add column if not exists source text not null default 'unknown';
alter table decision_log alter column source
    set default coalesce(nullif(current_setting('brain.source', true), ''), 'client');
create index if not exists decision_log_source on decision_log (account_id, source, ts);
