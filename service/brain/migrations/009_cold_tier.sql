-- Build step 9 (DESIGN.md nightly step 5, memory tiers): structural scoring and the cold tier.
-- Cold means out of the hot vector index but still in SQL and keyword search; nothing is deleted. The vector index
-- becomes partial (hot rows only), so its size follows the hot set, not the whole store: the HNSW index has to fit
-- in memory (the experiment log stress-c capacity finding). Dense queries filter on `not cold` to use it.
set search_path = brain, extensions, public;

alter table chunk add column if not exists cold boolean not null default false;
alter table chunk add column if not exists heat real;                  -- nightly structural score, higher = hotter

drop index if exists chunk_vec;
create index if not exists chunk_vec_hot on chunk using hnsw (embedding vector_cosine_ops) where not cold;
create index if not exists chunk_cold on chunk (account_id, namespace) where cold;
