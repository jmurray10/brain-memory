# Security

## Reporting

Please report a vulnerability privately through GitHub's "Report a vulnerability" on this repository (Security
tab), not in a public issue. Include what you did, what happened and what you expected.

## What the service does to protect a memory

- Every call needs a bearer key. Keys are stored hashed and shown once. One key per client; a key carries the
  namespaces it may read and write.
- Each account's rows are kept apart by Postgres row-level security: every request runs as a role that can see
  only its account, even if a query forgets a filter.
- What memory returns is treated as untrusted text. Retrieved passages are screened for prompt injection before
  they are handed to a model, and standing directives a client proposes stay pending until the user confirms
  them with an admin key.
- A turn written by a key that has not been allowed to speak as the user is stored as reported by an agent and
  files no claims.
- Forget archives and can be undone. Erasing for good needs an admin key and an explicit confirmation, and is
  not available as an MCP tool, so a model cannot erase by mistake.
- Rate limits and spend are kept per account; an optional per-account daily spend cap sits inside the service's.

## What you must do when you run it

- Keep `BRAIN_ADMIN_KEY`, the database URL and provider keys out of the repository and out of any model's
  context. The pre-commit hook refuses commits that look like they contain one.
- Text you store is sent to the providers you configure (embeddings and Jev's typed questions) to be processed.
- Backups hold original text. An erase removes text from the database at once; existing backups keep it until
  they rotate out.
