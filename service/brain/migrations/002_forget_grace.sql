-- Migration 002: forget with a grace period (REFERENCE.md as amended 2026-11-07).
-- forget sets deleted_at at once (the turn leaves recall, answer and export, and its chunks, vectors and
-- decision_log rows are deleted). The body stays restorable by an admin until purged_at is set: by the nightly
-- worker after BRAIN_FORGET_GRACE_DAYS, or at once with forget(purge=true). A purged episode keeps only its
-- skeleton (id, namespace, client, session_id, ts, role, channel, idempotency_key, deleted_at, purged_at).
set search_path = brain, extensions, public;

alter table episode add column if not exists purged_at timestamptz;

-- The client's UTC offset in minutes when the turn was captured, so a restored turn renders with the same
-- calendar date as at capture ("[YYYY-MM-DD] speaker: text" uses the client's own date).
alter table episode add column if not exists ts_offset_min smallint;

create index if not exists episode_purge_due on episode (deleted_at) where deleted_at is not null and purged_at is null;
