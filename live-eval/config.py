"""Frozen settings for the live eval. Change one thing per experiment and log it."""
from __future__ import annotations

from env import LIVE_EVAL, REPO_ROOT

DATA = LIVE_EVAL / "data"
CACHE = LIVE_EVAL / "cache"
EMBED_CACHE = CACHE / "embed_cache"   # matches repo .gitignore
JEV_CACHE = CACHE / "jev_cache"       # matches repo .gitignore
RUNS = LIVE_EVAL / "runs"
SAMPLES = LIVE_EVAL / "samples"
LOCOMO_PATH = REPO_ROOT / "locomo10.json"

# LongMemEval cleaned (MIT). Same files and hashes pinned on 2026-11-06.
LME_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
LME_FILES = {
    "oracle": ("longmemeval_oracle.json",
               "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"),
    "s": ("longmemeval_s_cleaned.json",
          "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"),
}

# Models
ANSWER_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-haiku-5-5"
VOYAGE_MODEL = "voyage-4-lite"
VOYAGE_DIM = 1024

# Retrieval (iso arm)
BM25_TOP = 60
DENSE_TOP = 60
SHORTLIST = 30          # all of these go to Jev rerank (old harness only reranked 12)
KEEP = 10               # chunks kept after rerank
ISO_BUDGET_TOKENS = 6000
CHUNK_MAX_CHARS = 1500  # long turns split into windows; nothing is truncated

# Oracle arm: all evidence-session turns, generous budget; truncation is logged
ORACLE_MAX_ITEMS = 600
ORACLE_BUDGET_TOKENS = 30000

# Generation
ANSWER_MAX_TOKENS = 400
JUDGE_MAX_TOKENS = 10
JUDGE_VOTES = 3          # majority vote; Haiku's rubric verdicts flip between calls (measured 2026-11-06)
JUDGE_MAX_CALLS = 6      # retries when the judge does not answer yes/no

# Spend
DEFAULT_CAP_USD = 5.00

# Invariant tolerances
INV_OVERALL_TOL = 0.03      # iso may not beat oracle overall by more than this
INV_TYPE_TOL = 0.15         # per type, only for types with n >= INV_TYPE_MIN_N
INV_TYPE_MIN_N = 10
INV_TOO_GOOD = 0.95         # any arm above this needs a manual row check
INV_MAX_PARSE_FAIL = 0.02
INV_MAX_ERRORS = 0.02
