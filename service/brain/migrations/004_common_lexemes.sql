-- Keyword-query pruning (core.RECALL_SQL) drops lexemes that are in more than 5% of chunks, read from Postgres'
-- column statistics. pg_stats hides every row for a table under row-level security for the current role, so as
-- brain_app (RLS active on chunk) the common list was always empty and nothing was pruned. This function runs as
-- the schema owner (RLS not active) and returns only those lexemes: never row data, never a frequency, and never
-- below the 5% floor, so a caller cannot lower the threshold to probe for rare words across accounts.
set search_path = brain, extensions, public;

create or replace function brain.common_lexemes(min_freq float8)
returns setof text
language sql stable security definer
set search_path = pg_catalog
as $$
    select m.e
    from pg_catalog.pg_stats s,
         unnest(s.most_common_elems::text::text[], s.most_common_elem_freqs) as m(e, f)
    where s.schemaname = 'brain' and s.tablename = 'chunk' and s.attname = 'tsv'
      and m.e is not null    -- the freqs array has 3 trailing summary values; unnest pads the elements with NULL
      and m.f > greatest(coalesce(min_freq, 0.05), 0.05)
$$;

revoke all on function brain.common_lexemes(float8) from public;
grant execute on function brain.common_lexemes(float8) to brain_app;
