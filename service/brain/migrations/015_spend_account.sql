-- Spend per account (release list B2). Until now the daily cap and the spend table were global: the use of one
-- account could exhaust the cap for every account. A spend row now says which account it was for. 0 means "no
-- account": the operator's own calls and work done for all accounts at once (the embedding backfill).
-- Not row-level-secured: spend is the operator's bill, read only by the operator's report.

set search_path = brain, extensions, public;

alter table brain.spend add column if not exists account_id bigint not null default 0;
alter table brain.spend drop constraint if exists spend_pkey;
alter table brain.spend add primary key (day, provider, account_id);
