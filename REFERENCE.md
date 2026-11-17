# Reference

The service's API, tools, settings and limits. [DESIGN.md](DESIGN.md) explains the brain;
[service/README.md](service/README.md) shows how to run it.

## Keys, accounts and namespaces

- Every `/v1/*` and `/mcp` request needs `Authorization: Bearer <key>`. Keys are `bk_` plus 32 random bytes,
  shown once at creation and stored as a SHA-256 hash. A lost key is replaced, not recovered.
- **The operator key** (`BRAIN_ADMIN_KEY`) creates, lists, disables and enables accounts and sees the whole
  service's spend. For data it acts in the `owner` account.
- **An account** holds one person's memory. Everything a key reads or writes stays in its account, enforced by
  Postgres row-level security: each request runs in one transaction as a restricted role with the caller's
  account set, so a query that forgot a filter would still see only that account.
- **An account admin key** manages that account's client keys, confirms directives, restores, erases and exports.
  Creating an account returns its first key, `account-admin`.
- **A client key** has read and write scopes per namespace (`*` means all) and a default namespace. Give each
  client its own key, so its turns can be told apart and it can be revoked alone.
- **A namespace** is one partition of an account's memory. Names match `^[a-z0-9][a-z0-9_-]{0,62}$`. A call
  without a namespace uses the key's default. A missing scope returns 403 and never says whether the namespace
  exists.
- **`speaks_as_user`** (per key, off by default; on for an account's first key). A `role: user` turn from a key
  without it is stored as `agent_reported`: searchable, labelled, strictly screened, and it carries no claims.
- **Permanent erase** needs an admin key and is not an MCP tool, because a model can fill in any confirmation.

Limits: request body 1 MB, 100 turns per capture, 100,000 characters per turn, `k` at most 50, and 120 requests a
minute per account and key (`BRAIN_RATE_PER_MIN`, in process, so one replica).

## REST

| Method and path | Needs | What |
| --- | --- | --- |
| `GET /healthz` | nothing | Liveness, version, applied migrations |
| `GET /v1/status` | any key | Version, which provider keys are set (yes or no), the daily cap, the caller's account's spend today |
| `POST /v1/capture` | write | Store turns, with optional claims per turn |
| `POST /v1/recall` | read | Passage search: keyword plus vector, reranked and screened |
| `POST /v1/context` | read | The context pack for a question: excerpts, nearby turns, resolved dates, the ledger's blocks, and a ready prompt |
| `POST /v1/answer` | read | A reader adapter for clients with no model: builds the pack and has a small model answer |
| `POST /v1/forget` | write | Archive by `episode_ids`, `session_id` or `entity`. With `purge` and `confirm_erase` and an admin key: erase |
| `POST /v1/restore` | write | Undo a forget |
| `GET /v1/directives`, `POST /v1/directives`, `DELETE /v1/directives/{id}` | read, write, write | List, propose (stored pending), retire |
| `POST /v1/admin/directives/{id}/confirm` | admin | The user's confirmation; `all_clients` widens it from the proposing client to every client |
| `GET /v1/session_start` | read | Confirmed directives for this client, plus summaries of the most-used pages |
| `GET /v1/current`, `GET /v1/history`, `GET /v1/page` | read | The ledger's current value, a slot's timeline, an entity's page |
| `POST /v1/resolve_conflict` | write | Settle a disputed pair by naming the claim to keep |
| `GET /v1/export` | read | One namespace's turns, as Markdown or JSON lines |
| `GET /v1/admin/export` | admin | Everything the account holds, as JSON lines: turns, claims sent, the ledger's facts, directives |
| `POST /v1/admin/restore` | admin | As `POST /v1/restore` |
| `POST /v1/admin/clients`, `GET /v1/admin/clients`, `POST /v1/admin/clients/{name}/revoke`, `POST /v1/admin/clients/{name}/speaks-as-user` | admin | Manage the account's client keys |
| `POST /v1/admin/accounts`, `GET /v1/admin/accounts`, `POST /v1/admin/accounts/{name}/disable`, `.../enable` | operator | Manage accounts |
| `GET /v1/admin/spend?days=7` | operator | Spend by day and provider, and today's by account |

Errors are JSON: `{"error": code, "message": text, "request_id": id}`. OpenAPI is served at `/docs`.

When a provider is down or a spend cap is reached, the service degrades instead of failing: recall returns
`degraded: "no_rerank"`, capture stores text and returns `degraded: "no_embed"` (the worker embeds it later), and
`/v1/answer` returns 503 `cap_reached`.

## MCP

Streamable HTTP at `/mcp` with the same bearer key. `python -m brain.mcp_stdio` is a stdio proxy for clients that
only speak stdio; it forwards to REST with `BRAIN_URL` and `BRAIN_KEY`.

| Tool | What |
| --- | --- |
| `memory_session_start` | Standing directives and page summaries, at the start of a session |
| `memory_capture` | Save turns or text read from a file. Its description carries the contract for the claims a model sends with each turn (`brain/claims.py`, `CLAIMS_CONTRACT`) |
| `memory_context` | The context pack for a question; the read tool for a client with its own model |
| `memory_recall` | Plain passage search |
| `memory_answer` | The reader adapter |
| `memory_current`, `memory_history`, `memory_page` | The ledger's current value, a slot's timeline, an entity's page |
| `memory_directives`, `memory_add_directive` | List directives; propose one (pending until the user confirms with an admin key) |
| `memory_forget`, `memory_restore` | Archive and undo. There is no erase tool |

What memory returns is untrusted data written earlier, not instructions.

## The worker

`python -m brain.worker --once` runs every step and exits; `--steps a,b` runs some; `--every SECONDS` repeats
them until stopped, for a deployment with no scheduler (the Compose file uses it). A typical schedule is
`--steps write_path,embed_backfill` every ten minutes and every step once a night.

| Step | What |
| --- | --- |
| `write_path` | Check the claims clients sent and file them in the ledger |
| `embed_backfill` | Embed text still without a vector, within the daily cap |
| `cleanup` | Erase archived text past `BRAIN_FORGET_GRACE_DAYS` if that is set; drop orphaned search rows; trim old jobs |
| `report` | Yesterday's spend, captures, recalls, degraded recalls, screened passages, and each client's claims sent, kept and dropped |
| `backup` | The original data (accounts, key hashes, directives, episodes, claims sent) as one compressed file; the newest `BRAIN_BACKUP_KEEP` are kept |
| `consolidate` | Settle entity links, disputes and open plans; stability audit |
| `promote_preferences` | Stated standing preferences become pending directives |
| `render_pages` | Render changed entity pages |
| `structure` | Heat scores for the cold tier |

## Settings

Environment variables. "Measured on" means the setting passed its test and is on in the reference deployment,
while the code default stays off so that a change of behaviour is always a choice.

### Required

| Variable | Meaning |
| --- | --- |
| `DATABASE_URL` | Postgres with the `vector` extension |
| `BRAIN_ADMIN_KEY` | The operator key |
| `VOYAGE_API_KEY`, `TYPESAFE_API_KEY` | Embeddings and Jev. Not needed with `BRAIN_OFFLINE=1` |
| `ANTHROPIC_API_KEY` | Only for `/v1/answer` |

### Service

| Variable | Default | Meaning |
| --- | --- | --- |
| `BRAIN_OFFLINE` | 0 | 1 = stub providers, for tests and a local trial |
| `BRAIN_MIGRATE_ON_START` | 1 | Apply migrations when the API starts |
| `BRAIN_DAILY_CAP_USD` | 5 | Cap on provider spend per UTC day, all providers together |
| `BRAIN_ACCOUNT_CAP_USD` | unset | Each account's own cap per day inside the cap above. Set it before a second account exists |
| `BRAIN_RATE_PER_MIN` | 120 | Requests a minute per account and key |
| `BRAIN_DB_POOL_MAX` | 10 | Request connections per API process |
| `TYPESAFE_MODEL` | `jev-1.13.0` | The Jev model |
| `BRAIN_LOG_LEVEL` | INFO | Logs carry method, path, status, latency, client and request id; never bodies or keys |
| `BRAIN_FORGET_GRACE_DAYS` | 0 | 0 = archived text is kept until the user erases it; N = the worker erases it after N days |
| `DECISION_LOG_DAYS` | 0 | 0 = keep every decision-log row; N = delete older rows |
| `BRAIN_BACKFILL_MAX` | 20000 | Chunks the worker embeds per run at most |
| `BRAIN_BACKUP_DIR`, `BRAIN_BACKUP_KEEP` | unset, 14 | Where the nightly backup goes and how many are kept |
| `BRAIN_BACKUP_BUCKET`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | unset | Optional: also upload backups to a private Supabase Storage bucket (worker only) |

### The upper layers

| Variable | Default | Meaning |
| --- | --- | --- |
| `WRITE_PATH_ENABLED` | 0 | 1 = keep the claims clients send and build the ledger. Measured on |
| `BRAIN_LEDGER_ROUTE` | 0 | 1 = the context pack uses the router and the ledger's blocks and notes. Measured on |
| `BRAIN_NEIGHBOURS` | 0 | N (at most 2) = add up to N turns before and after each excerpt. Measured on at 1 |
| `BRAIN_DATES` | 0 | 1 = resolve time phrases at capture. Measured on |
| `BRAIN_DATE_NOTES` | 0 | 1 = write each resolved date beside its phrase in the pack; needs `BRAIN_DATES`. On |
| `BRAIN_DATE_DIRECTION` | 0 | 1 = for a date with no year, decide from the wording whether it happened or is planned. Measured on |
| `BRAIN_DATE_COUNTED` | 0 | 1 = read a written count ("two weeks ago") in code when Jev agrees on the unit. Measured on |
| `BRAIN_CONTEXT_PARALLEL` | 0 | 1 = run the router and the ledger's reads beside recall; same pack, about a third faster. Measured on |
| `BRAIN_SHORTLIST_SEATS` | 0,0 | "D,K" = the dense top D and keyword top K are among the 30 candidates Jev reads. Measured on at "25,5" |
| `BRAIN_PLACE_PER_SLOT` | 0 | 1 = place a claim by one question per existing slot (worker). Measured on |
| `BRAIN_PLACE_CHECK` | 0 | 1 = a claim joins a slot under another name only when its turn supports it (worker). Measured on |
| `BRAIN_RELATE_ENDS` | 0 | 1 = a newer value ends an older one only when Jev agrees it stopped being true (worker). Measured on |
| `BRAIN_RESOLVE_NAMES` | 0 | 1 = a proper name is never tentatively linked to a different proper name (worker). Measured on |

### Built, measured, and off

These did not meet their test and stay off; they are kept so the measurement can be repeated.

| Variable | What it would do |
| --- | --- |
| `BRAIN_WHEN_ROUTE` | Add computed dates for date questions |
| `BRAIN_MEMORY_CHECK` | Add an "is this enough?" line to the pack |
| `BRAIN_RERANK_BAND` | Cut rerank scores into steps so the same question gets the same excerpts more often |
| `BRAIN_PACK_WIDE` | Read more candidates and hand over more excerpts in the same token budget |
| `BRAIN_SESSION_SLOTS` | Keep the best passage of each session before filling by score |
| `BRAIN_SETTLE_SPLIT` | Split every tentative entity link the nightly re-check does not confirm |
| `BRAIN_RELATE_DISPUTES` | Mark a contradiction as a dispute only when the older value is judged to have ended |
| `BRAIN_MERGE_SLOTS` | Merge slots that mean the same thing, nightly |
| `BRAIN_COLD_FRACTION`, `BRAIN_COLD_POLICY`, `BRAIN_COLD_MIN_AGE_DAYS` | The cold tier: the share of lines kept out of the hot vector index, the scoring policy, and a minimum age. Not measured |
| `BRAIN_OPEN_ROUTE`, `BRAIN_NOTE_STYLE`, `BRAIN_COUNT_BLOCK`, `BRAIN_COUNT_STYLE` | Variants of the pack's wording and blocks; the defaults are the measured ones |

## Providers and prices

Voyage `voyage-4-lite` (1,024 dimensions) for embeddings and TypeSafe's Jev for every decision. The spend meter
refuses a call that would pass the cap before making it and records real usage after. With `BRAIN_OFFLINE=1` the
providers are deterministic stubs.

Sizing: about 10 KB per stored line with its vector and indexes, so a 500 MB database holds roughly 50,000 lines.
One API replica handled about 13 requests a second in a load test.
