# Data

This repository's licence (MIT) covers its code and documents. It does not cover the public benchmarks the
evaluation scripts read, and **their text is not in this repository.** Download them yourself and check their
terms before use.

| Benchmark | Used for | Where the scripts look | Terms |
| --- | --- | --- | --- |
| LoCoMo (`locomo10.json`) | `service/scripts/locomo_eval.py`, `count_route_set.py`, `over_time_set.py` | one folder above the repository root | CC BY-NC 4.0: non-commercial only. See the LoCoMo repository |
| LongMemEval (cleaned S and oracle files) | `live-eval/`, `service/scripts/brain_eval.py` | `live-eval/data/` | see the LongMemEval repository |

Files derived from those benchmarks (run outputs, question samples, route labels built on their questions) are
left out of this repository for the same reason. The scripts that build them are here; run them after you have
the data.

## What is in the repository and is ours

- `service/evals/worlds/`: six generated professional worlds (mathematics, genomics, a clinical trial, patent
  litigation, a mass-timber building, a fund's risk audit) with exact answers and evidence turns.
- `service/evalsets/ledger_set_v1_*.json`: generated conversations about facts that change, with computed
  answers (`service/scripts/ledger_set.py` scripts the timelines; a model writes the turns; code checks them).
- `service/evalsets/worlds_v1.json`: the six worlds in the layout `locomo_eval.py --data` reads.
- `service/evals/injection_heldout_*.jsonl`, `route_domain_v1.jsonl`: generated test items.

These are synthetic. They are useful for finding faults and say little about real use by themselves.

## A rule this project keeps

Benchmarks are tests, never training data and never a target. Nothing in the brain is keyed to a benchmark's
content, and the booster (a small trained model, off) may not be trained on any of them.
