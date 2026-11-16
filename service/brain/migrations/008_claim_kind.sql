-- Build step 7 (preferences and directives): what kind of belief a claim is. Preferences (likes, habits,
-- standing instructions) are promoted nightly to directives; events are counted; plans never change what is current.
set search_path = brain, extensions, public;

alter table claim add column if not exists kind text not null default 'fact'
    check (kind in ('fact', 'preference', 'event', 'plan'));
create index if not exists claim_preference on claim (account_id, namespace) where kind = 'preference' and status = 'active';
create index if not exists directive_source on directive (account_id, source_claim) where source_claim is not null;
