**What changed and why**

**Evidence** (quote the output, do not paraphrase it)

- [ ] `cd service && BRAIN_OFFLINE=1 python -m pytest -q` passes against a throwaway database
- [ ] New behaviour has a test that failed before the change
- [ ] A change to how the brain behaves is behind a setting, off by default, with a measurement
- [ ] Nothing is keyed to a benchmark's content, a domain or a test set
- [ ] No applied migration edited; a new numbered file starts with `set search_path = brain, extensions, public;`
- [ ] No key, database URL or real person's text added
