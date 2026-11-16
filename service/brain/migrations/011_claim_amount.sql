-- The fine value of a claim (DESIGN.md: pre-parsed values; code finds amounts, Jev picks one, code normalizes):
-- the number a value states and its unit, so code can add and compare (totals in the count route).
set search_path = brain, extensions, public;

alter table claim add column if not exists amount numeric;
alter table claim add column if not exists unit text;
