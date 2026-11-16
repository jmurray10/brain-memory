-- Forget becomes archive (2026-11-09, the owner: keep the means to undo anything the user forgets). Forgetting hides
-- and retracts; nothing is erased unless the user asks to erase. An entity can be archived as a whole; a learned
-- directive retired because its entity was forgotten says so, so a restore can bring it back.
set search_path = brain, extensions, public;

alter table entity add column if not exists archived_at timestamptz;
alter table directive add column if not exists retired_reason text;
create index if not exists entity_archived on entity (account_id, namespace) where archived_at is not null;
