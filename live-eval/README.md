# live-eval — live benchmark harness for Brain-Memory v2

Build step 1 of `../DESIGN.md` (replaces the buggy `longmemeval-park/` harness; see the harness notes).
Runs **oracle** (evidence sessions only) against **iso** (flat hybrid retrieval over the full haystack)
on LongMemEval (cleaned) or LoCoMo10, with Haiku 5.5 answering and judging.

## Setup

- Python 3.10+, `numpy`. No Postgres: vectors are searched in memory.
- Keys come from the repo-root `.env` (git-ignored). Names read:
  - Anthropic: `ANTHROPIC_API_KEY`
  - Voyage: `VOYAGE_API_KEY_NEW`, `VOYAGE_API_KEY_TEST`, `VOYAGE_API_KEY`
  - Jev: `TYPESAFE_API_KEY` or `JEV_API_KEY`
- LongMemEval files download on first use into `data/` and are checked against pinned sha256 hashes.

## Commands (run from this folder)

```bash
python3 -m unittest tests/test_offline.py -v                                   # offline tests
python3 run.py --bench lme --sample samples/lme_s_20261106_100.txt --dry-run --limit 5   # no API calls
python3 run.py --bench lme --sample samples/lme_s_20261106_100.txt --preflight           # cost estimate
python3 run.py --bench lme --sample samples/lme_s_20261106_100.txt --cap 5 --run-id lme100_v1
python3 run.py --bench locomo --convs conv-26 --cap 2 --run-id locomo_conv26
python3 run.py --summarize runs/lme100_v1
```

Runs are resumable: rerun the same `--run-id` and finished rows are skipped. The spend cap counts
everything already spent on that run.

## What it fixes versus the old park

| Old bug | Here |
| --- | --- |
| Oracle arm packed only 10 turns | Oracle gets every evidence-session turn (30K-token budget; truncation logged) |
| No `question_date` in the answer prompt | Today's date is always in the prompt |
| One generic judge; preference golds are rubrics | Official LongMemEval judge prompt per question type; LoCoMo keeps the ledger's judge |
| `CORRECT: no` parsed as correct | Strict yes/no parser; anything else is a parse failure |
| Recall counted any one gold session | Reports recall_any, recall_all and turn-level recall |
| Turns cut to 1,200 characters | Long turns split into windows; no text dropped |
| Packing stopped at the first oversized turn | Oversized chunks are skipped, packing continues |
| Jev reranked 12 of 30 | Jev reranks the full shortlist of 30 |
| Imports only resolved on the bot's box | Runs from this repo; reuses `typesafe_cookbooks/` for Jev |

## Run invariants (enforced in `invariants.py`)

A run is marked **INVALID** in `summary.md` if: iso beats oracle overall by more than 3 points or on a
type (n >= 10) by more than 15; any arm scores above 95% without `--manual-checked`; judge parse
failures or errors exceed 2%; or the spend cap stopped the run.

## Outputs

`runs/<run-id>/config.json` (code and sample hashes), `rows.jsonl` (one row per arm and question),
`spend.json`, `summary.json`, `summary.md`.

## Answer styles

The answer prompt is picked from the question text only (`router.py`): a preference prompt for
requests like "can you recommend...", the default prompt otherwise. Gold question types are used only
by the judge and the per-type report. Build step 2 replaces the keyword router with a Jev router.
