# Brain-Memory service

One memory brain behind REST, MCP and a CLI, on Postgres with pgvector. The API, tools and settings are in
[../REFERENCE.md](../REFERENCE.md); the design of the brain is [../DESIGN.md](../DESIGN.md).

## Set it up in one go

With Docker, nothing else needs installing ([../SETUP.md](../SETUP.md) walks through it):

```bash
docker compose run --rm setup    # first run: writes .env (an operator key; trial mode with no provider keys)
docker compose up -d --build     # the database, the API, and the two workers
docker compose run --rm setup    # with the service up: a client key, and how to connect Claude Code and Claude Desktop
```

Without Docker, the same command is `python -m brain.setup`, run from this folder inside a virtual environment
(`python -m venv .venv && .venv/bin/pip install -r requirements.txt`); add `--desktop` and it puts the server
into Claude Desktop's config itself, keeping a copy of the file as it was.

It is safe to run again: it never replaces a value in `.env`, never reuses a client name. `--client NAME` and
`--namespace NAME` choose the key's name and its namespace (defaults `claude` and `me`).

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # BRAIN_OFFLINE=1 needs no provider keys (stub providers)
python -m brain.serve           # applies migrations, serves on :8000; OpenAPI at /docs
python -m brain.worker --once   # the write path, embedding backfill, clean-up, backup and the nightly pass
```

`DATABASE_URL` must point at a Postgres with the `vector` extension. The service keeps everything in schema
`brain`. In production the API is one always-on process and the worker runs from a scheduler: every ten minutes
with `--steps write_path,embed_backfill`, and once a night with every step. The `Dockerfile` builds the image
(`python -m brain.serve` is its command).

Settings are environment variables, listed with their defaults in the reference. The ones that turn
the upper layers on: `WRITE_PATH_ENABLED=1` (keep client-sent claims and build the ledger) and
`BRAIN_LEDGER_ROUTE=1` (use the ledger and routes in the context pack). `BRAIN_DAILY_CAP_USD` caps provider spend
per day; `BRAIN_ACCOUNT_CAP_USD` caps each account inside it.

## Accounts and keys

`BRAIN_ADMIN_KEY` is the operator's key and works in the `owner` account. Give every client its own key.

```bash
python -m brain.cli admin create-account alice     # operator: prints alice's account-admin key once
BRAIN_KEY=<alice admin key> python -m brain.cli admin create-client claude-code --read alice --write alice --default alice
python -m brain.cli admin speaks-as-user claude-code      # this client's user turns are the user's own words
python -m brain.cli admin accounts                 # operator
python -m brain.cli admin disable-account alice    # her keys stop; enable-account undoes it
```

A client's `role: user` turns count as the user's own words only if its key has `speaks_as_user`; otherwise they
are stored as reported by an agent (searchable, strictly screened, no claims). A directive a client adds, or one
the nightly pass learns from a stated preference, stays pending until it is confirmed with an admin key.

## Connect a client

```bash
claude mcp add --transport http brain http://localhost:8000/mcp --header "Authorization: Bearer $KEY"
```

Clients that only speak stdio run the proxy, from this folder:

```json
{"mcpServers": {"brain": {"command": "python", "args": ["-m", "brain.mcp_stdio"],
  "env": {"BRAIN_URL": "http://localhost:8000", "BRAIN_KEY": "bk_..."}}}}
```

The capture tool's description carries the contract for the claims a client model sends with each turn
(`brain/claims.py`, `CLAIMS_CONTRACT`). A client that sends none still gets storage and retrieval.

## CLI

`BRAIN_URL` (default `http://localhost:8000`) and `BRAIN_KEY`; `--json` for scripts.

```bash
python -m brain.cli capture "My sister lives in Austin." -n me
python -m brain.cli capture --file notes.txt -n me
python -m brain.cli recall "Where does my sister live?" -n me
python -m brain.cli answer "Where does my sister live?" -n me
python -m brain.cli forget --episode 12 -n me             # archives; --purge --confirm-erase erases for good
python -m brain.cli admin restore --episode 12 -n me
python -m brain.cli directives add "Use metric units." -n me
python -m brain.cli admin confirm-directive 1 -n me
python -m brain.cli session-start -n me
python -m brain.cli export --format jsonl --out me.jsonl -n me
python -m brain.cli status
```

## REST

Every `/v1/*` and `/mcp` call needs `Authorization: Bearer <key>`.

```bash
curl -H "Authorization: Bearer $KEY" localhost:8000/v1/capture -H 'content-type: application/json' \
  -d '{"namespace": "me", "turns": [{"role": "user", "text": "My sister lives in Austin."}]}'
curl -H "Authorization: Bearer $KEY" localhost:8000/v1/context -H 'content-type: application/json' \
  -d '{"namespace": "me", "question": "Where does my sister live?"}'
```

Recall returns `degraded: "no_rerank"` when Jev is down or a spend cap is reached, and capture returns
`degraded: "no_embed"` when the embedder is; nothing is lost, and the worker fills in vectors later.

`GET /v1/admin/export` (an admin key; CLI `export --account`) streams everything the account holds as JSON lines:
every namespace's turns, the claims sent, the ledger's facts and the directives.

Forget archives: a turn leaves every read at once and what was derived from it is retracted, and restore brings
it back. Only `"purge": true` with `"confirm_erase": true` and an admin key erases the text, and there is no MCP
tool for it.

## Backups

The nightly worker writes the original data (accounts, key hashes, directives, episodes and the claims clients
sent with them) as a compressed file to
`BRAIN_BACKUP_DIR` or a storage bucket and keeps the newest `BRAIN_BACKUP_KEEP`. `python -m brain.backup
dump|list|fetch|load` does the same by hand; `load` only restores into an empty database, and the next worker runs
embed the text and file the claims into the ledger again.

## Tests

```bash
export TEST_DATABASE_URL=postgresql://...     # a throwaway database: the tests drop and recreate schema `brain`
BRAIN_OFFLINE=1 python -m pytest -q
```

## Evaluations

`scripts/` holds the evaluation and replay scripts used for [../RESULTS.md](../RESULTS.md). They need provider
keys and the benchmark data described in [../DATA.md](../DATA.md), and each takes a spend cap.
