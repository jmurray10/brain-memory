# Contributing

Issues and pull requests are welcome. A few rules keep the brain honest; they are the ones it was built under.

1. **Tests ship with every change**, and they run offline with no keys: `cd service && BRAIN_OFFLINE=1 python -m
   pytest -q` against a throwaway Postgres (`TEST_DATABASE_URL`; the tests recreate schema `brain`).
2. **A change to how the brain behaves goes behind a setting, off by default, with a measurement.** Say before
   the run what result would count as a gain, then report what happened, including when it is nothing.
3. **Fixes are general.** Nothing may be keyed to a benchmark's content, a domain or a test set. Benchmarks are
   tests, never training data and never the target (see [DATA.md](DATA.md)).
4. **No generative model inside the brain.** The user's model extracts and answers through the tools; the brain
   checks, stores and computes. Decisions are typed questions with a probability, logged with their input.
5. **Never edit an applied migration.** Add the next numbered file under `service/brain/migrations/`, and start
   it with `set search_path = brain, extensions, public;`.
6. **Never commit a key or a database URL.** `.githooks/pre-commit` runs `service/scripts/secret_scan.py`;
   enable it with `git config core.hooksPath .githooks`.
7. **Forget must stay undoable and erase must stay final.** Anything that stores a user's text has to be reached
   by both.

Paid evaluation runs need your own provider keys and cost real money; every script takes a `--cap` in dollars.
