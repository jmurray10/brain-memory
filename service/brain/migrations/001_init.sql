-- Brain-Memory v2 service, migration 001: the ten DESIGN.md tables plus service tables.
-- Runs on Supabase Postgres and on plain Postgres 16 with pgvector (>= 0.5 for HNSW).
-- Everything lives in schema "brain", which Supabase's REST API (PostgREST) does not expose by default.
-- The service connects with the database connection string; there is no client access through Supabase APIs.

create schema if not exists extensions;
create extension if not exists vector with schema extensions;
create schema if not exists brain;
set search_path = brain, extensions, public;

-- ---------------------------------------------------------------- service tables

-- One row per API key. Keys are random; only their sha256 is stored. '*' in a scope list means every namespace.
create table if not exists api_client (
    id           bigint generated always as identity primary key,
    name         text not null unique,
    key_hash     text not null unique,
    read_ns      text[] not null default '{}',
    write_ns     text[] not null default '{}',
    default_ns   text,                              -- namespace used when a call names none (MCP tools)
    is_admin     boolean not null default false,
    created_at   timestamptz not null default now(),
    revoked_at   timestamptz
);

-- Spend per UTC day and provider, so the daily cap survives restarts and is shared by every replica.
create table if not exists spend (
    day          date not null,
    provider     text not null check (provider in ('voyage', 'jev', 'haiku')),
    usd          numeric(12, 6) not null default 0,
    calls        bigint not null default 0,
    tokens_in    bigint not null default 0,
    tokens_out   bigint not null default 0,
    primary key (day, provider)
);

-- Work queue for the write path (DESIGN.md steps 2-9 plug in here) and maintenance jobs.
create table if not exists job (
    id           bigint generated always as identity primary key,
    kind         text not null,                 -- e.g. 'embed_backfill', 'write_path'
    ref_id       bigint,
    namespace    text,
    status       text not null default 'pending' check (status in ('pending', 'running', 'done', 'failed', 'skipped')),
    attempts     int not null default 0,
    last_error   text,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);
create index if not exists job_pending on job (kind, created_at) where status = 'pending';

-- ---------------------------------------------------------------- DESIGN.md tables

-- Raw turns. Immutable except deleted_at (forget). The only ground truth.
create table if not exists episode (
    id              bigint generated always as identity primary key,
    namespace       text not null,
    client          text not null,
    session_id      text,
    ts              timestamptz not null,               -- when the turn happened (client-supplied)
    recorded_at     timestamptz not null default now(),
    role            text not null check (role in ('user', 'assistant', 'system', 'tool', 'attachment')),
    speaker         text,
    channel         text not null default 'chat'
                    check (channel in ('chat', 'voice', 'email', 'meeting', 'agent_log', 'notes', 'document', 'image')),
    body            text not null,
    meta            jsonb not null default '{}'::jsonb, -- filename, page, url, client message id ...
    idempotency_key text,
    deleted_at      timestamptz
);
create unique index if not exists episode_idem on episode (namespace, idempotency_key) where idempotency_key is not null;
create index if not exists episode_ns_ts on episode (namespace, ts);
create index if not exists episode_session on episode (namespace, session_id, ts);

create table if not exists time_mention (
    id           bigint generated always as identity primary key,
    episode_id   bigint not null references episode (id) on delete cascade,
    span         text not null,
    date_from    date,
    date_to      date,
    granularity  text check (granularity in ('day', 'week', 'month', 'year', 'season', 'range')),
    confidence   real
);
create index if not exists time_mention_ep on time_mention (episode_id);

create table if not exists entity (
    id           bigint generated always as identity primary key,
    namespace    text not null,
    name         text not null,
    aliases      text[] not null default '{}',
    kind         text,
    merged_into  bigint references entity (id),
    created_at   timestamptz not null default now()
);
create index if not exists entity_ns_name on entity (namespace, lower(name));
create index if not exists entity_aliases on entity using gin (aliases);

create table if not exists attribute (
    path         text primary key,                       -- e.g. person.home.city
    cardinality  text not null check (cardinality in ('one', 'many', 'event')),
    parent       text references attribute (path),
    description  text
);

create table if not exists claim (
    id              bigint generated always as identity primary key,
    namespace       text not null,
    entity_id       bigint not null references entity (id),
    attribute_path  text not null references attribute (path),
    value           text not null,
    value_norm      text,
    valid_from      date,
    valid_to        date,
    recorded_at     timestamptz not null default now(),
    status          text not null default 'active' check (status in ('active', 'superseded', 'contested', 'retracted')),
    settled_at      timestamptz,
    confidence      real,
    quote           text not null,
    episode_id      bigint not null references episode (id)
);
create index if not exists claim_slot on claim (namespace, entity_id, attribute_path) where status = 'active';
create index if not exists claim_episode on claim (episode_id);

create table if not exists claim_link (
    from_claim   bigint not null references claim (id) on delete cascade,
    to_claim     bigint not null references claim (id) on delete cascade,
    type         text not null check (type in ('supersedes', 'contradicts', 'supports', 'same_event')),
    created_at   timestamptz not null default now(),
    primary key (from_claim, to_claim, type)
);

create table if not exists directive (
    id            bigint generated always as identity primary key,
    namespace     text not null,
    scope         text not null default 'all',           -- 'all' or a client name
    text          text not null,
    source_claim  bigint references claim (id),
    pinned        boolean not null default true,
    created_by    text not null,
    created_at    timestamptz not null default now(),
    retired_at    timestamptz
);
create index if not exists directive_ns on directive (namespace) where retired_at is null;

create table if not exists page (
    id               bigint generated always as identity primary key,
    namespace        text not null,
    entity_id        bigint not null references entity (id),
    summary          text check (char_length(summary) <= 160),
    body_md          text not null,
    rendered_at_rev  bigint not null default 0,
    sentence_claims  jsonb not null default '[]'::jsonb,   -- [{sentence, claim_ids:[...]}]
    rendered_at      timestamptz not null default now(),
    dirty            boolean not null default false,
    unique (namespace, entity_id)
);

-- Search units. Episode lines now; claim chunks (step 4+) and page chunks (step 8) later.
create table if not exists chunk (
    id          bigint generated always as identity primary key,
    namespace   text not null,
    kind        text not null check (kind in ('episode_line', 'claim', 'page')),
    ref_id      bigint not null,                         -- episode.id / claim.id / page.id
    piece       int not null default 0,                  -- window index within a long turn
    text        text not null,                           -- rendered: "[YYYY-MM-DD] speaker: text"
    embedding   vector(1024),                            -- voyage-4-lite; null until embedded (backfill job)
    embed_model text,
    tsv         tsvector generated always as (to_tsvector('english', text)) stored,
    created_at  timestamptz not null default now()
);
create index if not exists chunk_ref on chunk (kind, ref_id);
create index if not exists chunk_ns on chunk (namespace, kind);
create index if not exists chunk_tsv on chunk using gin (tsv);
create index if not exists chunk_vec on chunk using hnsw (embedding vector_cosine_ops);
create index if not exists chunk_unembedded on chunk (id) where embedding is null;

-- Every gate decision, with who decided. The booster's future training data.
create table if not exists decision_log (
    id             bigint generated always as identity primary key,
    ts             timestamptz not null default now(),
    namespace      text,
    stage          text not null,                        -- e.g. 'recall.rerank', 'recall.injection'
    decider        text not null check (decider in ('booster', 'jev', 'code', 'haiku')),
    model_version  text,
    input          jsonb not null,                       -- question, passage ref, ...
    output         jsonb,
    probability    real,
    threshold      real,
    outcome        text,
    request_id     text
);
create index if not exists decision_log_stage on decision_log (stage, ts);

-- Lock the tables down for Supabase's API roles (no-op on plain Postgres where the roles do not exist).
do $$
declare t text;
begin
    for t in select tablename from pg_tables where schemaname = 'brain' loop
        execute format('alter table brain.%I enable row level security', t);
    end loop;
    if exists (select 1 from pg_roles where rolname = 'anon') then
        execute 'revoke all on schema brain from anon, authenticated';
        execute 'revoke all on all tables in schema brain from anon, authenticated';
    end if;
end $$;
