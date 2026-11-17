# Changelog

## 0.1.0 (first public release)

The brain as measured in [RESULTS.md](RESULTS.md).

- REST, MCP (streamable HTTP and a stdio proxy) and a CLI over one core.
- Capture with client-sent claims, checked in code and by Jev; the ledger (entities, slots, supersession,
  corrections, disputes); current, history, count, when, preference and assistant routes; the context pack.
- Nightly pass: settle links, resolve disputes, stability audit, hardening, preference promotion, pages, backup.
- Accounts with row-level security, one key per client, rate limit and spend recorded per account, an optional
  per-account daily spend cap.
- Forget (archive), restore, and confirmed erase; export.
- Replies that explain themselves: each recalled passage names the client that wrote it, and the tool
  descriptions say what the scores, `screened` and `claims_received` mean.
- A daily report, including each client's claims sent, kept and dropped (which connected model extracts
  poorly, with no model inside the brain).
- Built and off by default, each behind a setting and with its measurement in the design documents: the when
  route, the memory check, the cold tier, slot merging, wider packs, cross-name comparison at write time.
- The booster (a small trained model in front of Jev) is not wired in.
