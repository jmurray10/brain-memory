"""Benchmark phase: the whole brain on LoCoMo10, with Haiku 5.5 (thinking off) as the user's model.

    python scripts/locomo_eval.py 60        # up to 60 questions per conversation (categories 1-4), 10 conversations
    python scripts/locomo_eval.py 40 40     # the next 40 per conversation by the same hash order (a fresh slice)
    python scripts/locomo_eval.py 40 80 --check   # adds an arm: the brain with its memory check on
    python scripts/locomo_eval.py 40 120 --notes  # adds an arm: the brain without the ledger's notes
    python scripts/locomo_eval.py 40 160 --twin --reads 3   # the brain twice (the noise), each answer written 3 times
    python scripts/locomo_eval.py 40 160 --twin --memo      # one Jev answer and one Haiku answer per distinct request

Each conversation is captured into a database of its own on the Supabase test branch: both speakers' turns (role
user, speaker by name, shared images as their captions), each with the claims Haiku extracts under the capture
tool's contract (cached with brain_eval's client cache). The worker's brain steps run as at night. Two arms answer
from the brain's context pack (core.context) with the measured LoCoMo answer prompt (live-eval ANSWER_LOCOMO):
flat (routes off) and brain (ledger, counts, preferences, evidence check on; dates off). Judge: the measured
LoCoMo judge prompt, Haiku, majority of 3. Questions per conversation: the first N by a hash of the question
index. Writes runs/locomo_eval_<stamp>/. Cap $5.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import hashlib
import json
import random
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from testdb import admin_url, db_url  # noqa: E402

DB = "brain_locomo"
CAP = 5.0
CACHE = SERVICE / "runs" / "cache" / "client_claims.jsonl"
DECISIONS = SERVICE / "runs" / "cache" / "decisions.jsonl"
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def category(qa: dict, named: bool = False) -> str | None:
    """A question's category: LoCoMo's numbered ones, or (a --data set, `named`) the name the set gives it."""
    c = qa.get("category")
    return str(c) if named and isinstance(c, str) and c else CATS.get(c) if not isinstance(c, str) else None
_lock = threading.Lock()
# (name, ledger_route, memory_check, ledger_notes); brain is the shipped default (memory check off since
# exp-memory-check-v1); --check and --notes add an arm that differs from brain in that one switch
ARMS = (("flat", False, False, False), ("brain", True, False, True))
# What each arm name means (the keys are kept; older runs use them). Both arms answer from the brain service:
# "flat" is its retrieval layer on its own, not a no-memory baseline.
ARM_LABELS = {
    "near1seats": "near1dates (the shipped pack since 2026-11-12) with the 30 candidates Jev reads chosen as "
                  "the dense top 25 and the keyword top 5, the rest by the fused rank (core.SHORTLIST_SEATS)",
    "near1band": "near1dates (the shipped pack since 2026-11-12) with Jev's rerank scores cut into steps of 0.1 "
                 "before the top ten are taken, the fused rank deciding inside a step (core.RERANK_BAND)",
    "near1dates": "near1 (the shipped pack) with each resolved date written beside its time phrase in the excerpts "
                  "and nearby turns (core.DATE_NOTES; dates stored at capture)",
    "near1gather": "near1seats (the shipped pack) with the count route reading more of the ledger: the number "
                   "of claims a side given to --gather, where the shipped route reads 40 (exp-gather-top-v1)",
    "near1wide": "near1seats (the shipped pack) with 45 candidates read by Jev (the dense top 35 and the keyword "
                 "top 10 seated) and up to 20 excerpts handed to the reader (core.PACK_WIDE)",
    "near1disputes": "near1seats (the shipped pack) over a second ingest made with BRAIN_RELATE_DISPUTES: a "
                     "\"contradicts\" marks two claims disputed only when the older one was judged to have stopped being true",
    "near1ends": "near1 (the shipped pack) over a second ingest in which an update ends the older value only when "
                 "Jev also judges that the older value stopped being true (ledger.RELATE_ENDS)",
    "near1soft": "near1 (the shipped pack) with the ledger notes worded softly: they point at the later statement "
                 "and do not say it replaced anything (core.NOTE_STYLE soft)",
    "near1off": "near1 (the shipped pack) without the ledger notes on excerpts (core.NOTE_STYLE off)",
    "perslot": "full brain over a second ingest of the same conversation in which a claim is placed by one Jev "
               "question per existing slot (ledger.PLACE_PER_SLOT) instead of fits + which",
    "flat": "retrieval only: the brain service's own recall (stored turns, keyword and vector search, Jev rerank); "
            "ledger, counts, preferences and notes off. Not a no-memory baseline",
    "brain": "full brain: retrieval plus the ledger's facts, counts, preferences and the notes on excerpts",
    "near1": "full brain, plus the turn before and after each excerpt in its session, screened, as context",
    "near2": "full brain, plus the two turns before and after each excerpt in its session, screened, as context",
    "twin": "full brain a second time, same settings: brain minus twin is the noise between two identical arms",
    "nonotes": "full brain without the ledger notes on excerpts",
    "quiet": "full brain, but on open-route questions no ledger blocks (notes kept)",
    "plain": "full brain, but on open-route questions retrieval only (no ledger blocks, no notes)",
    "flat_norerank": "flat with Jev unavailable to recall: the fused keyword and vector order, no rerank, no screen",
    "flat_kw": "flat with no query vector: keyword search only, then Jev's rerank and screen",
    "flat_kw_norerank": "flat with neither: keyword search only, in keyword order",
}
# exp-ablate-providers-v1: what each provider adds to retrieval. An arm named here answers from the same stored
# turns as flat, with the named provider failing the way an outage does, so the service's own fallback is measured.
ABLATE = {"flat_norerank": ("jev",), "flat_kw": ("embed",), "flat_kw_norerank": ("jev", "embed")}


class _Down:
    """A provider that is down: every call raises ProviderError, which recall answers by degrading."""

    def __getattr__(self, name):
        def call(*a, **k):
            from brain.providers import ProviderError
            raise ProviderError("ablated", name)
        return call


class _NoQueryVector:
    """The embedder with no vector for a question (stored lines keep theirs): recall runs its keyword side only."""

    def __init__(self, inner):
        self._inner = inner

    def embed_query(self, *a, **k):
        from brain.providers import ProviderError
        raise ProviderError("ablated", "embed_query")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def ablated(prov, what: tuple):
    """`prov` with the named providers down for one arm; `prov` itself when nothing is named."""
    if not what:
        return prov
    import dataclasses
    return dataclasses.replace(prov, **({"jev": _Down()} if "jev" in what else {}),
                               **({"embedder": _NoQueryVector(prov.embedder)} if "embed" in what else {}))


def session_dt(s: str) -> datetime:
    """'1:56 pm on 8 May, 2023' -> a UTC datetime."""
    t, d = s.split(" on ")
    return datetime.strptime(f"{d.strip()} {t.strip()}", "%d %B, %Y %I:%M %p").replace(tzinfo=timezone.utc)


_GOLD_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december"]
_GOLD_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_GOLD_REL = re.compile(
    r"\b(two weeks|week|weekend|few days|" + "|".join(_GOLD_DAYS) + r")\s+before\s+"
    r"(?:(\d{1,2})\s*([a-z]+)|([a-z]+)\s+(\d{1,2})),?\s*(\d{4})", re.I)


def gold_dates(gold: str):
    """exp-judge-dates-v1: a gold answer written as a phrase before a date ("The Friday before 24June, 2022", "the
    weekend before March 26, 2023") as the dates it means, computed by code: (first day, last day), or None for
    any other gold answer. A weekday is the latest such day before the date; a weekend the latest Saturday and
    Sunday before it; "the week before" covers both readings in use in the data (the calendar week before, and
    the seven days before); "two weeks before" is the calendar week two back; "few days before" the seven days
    before."""
    m = _GOLD_REL.search(str(gold))
    if not m:
        return None
    unit, month = m.group(1).lower(), (m.group(3) or m.group(4)).lower()
    if month not in _GOLD_MONTHS:
        return None
    try:
        d = datetime(int(m.group(6)), _GOLD_MONTHS.index(month) + 1, int(m.group(2) or m.group(5))).date()
    except ValueError:
        return None
    monday = d - timedelta(days=d.weekday())
    if unit in _GOLD_DAYS:
        day = d - timedelta(days=((d.weekday() - _GOLD_DAYS.index(unit)) % 7) or 7)
        return day, day
    if unit == "weekend":
        sunday = d - timedelta(days=((d.weekday() - 6) % 7) or 7)
        return sunday - timedelta(days=1), sunday
    if unit == "week":
        return monday - timedelta(days=7), d - timedelta(days=1)
    if unit == "two weeks":
        return monday - timedelta(days=14), monday - timedelta(days=8)
    return d - timedelta(days=7), d - timedelta(days=1)          # few days


DATE_PICK_SYSTEM = (
    "You read an answer to a question about when something happened. Reply with only the calendar date the answer "
    "gives for the event itself: 'YYYY-MM-DD', or 'YYYY-MM-DD to YYYY-MM-DD' when it gives a range. Do not reply "
    "with the date of the conversation or message in which the event was mentioned. If the answer gives no "
    "calendar date for the event, gives several different dates without choosing one, or says it does not know, "
    "reply NONE.")


def date_pick_user(question: str, pred: str) -> str:
    return f"Question: {question}\nAnswer: {pred}"


def parse_span(raw: str | None):
    """The date picker's reply as (first day, last day); None for NONE or anything that is not one date or range."""
    found = []
    for y, m, d in re.findall(r"(\d{4})-(\d{2})-(\d{2})", raw or ""):
        try:
            found.append(datetime(int(y), int(m), int(d)).date())
        except ValueError:
            return None
    if len(found) not in (1, 2) or re.search(r"\bnone\b", raw or "", re.I):
        return None
    return (found[0], found[-1]) if found[0] <= found[-1] else None


def dated_verdict(gold: str, picks: list[str | None]):
    """exp-judge-dates-v2: the model only picks the date out of the answer; code compares. True when the answer's
    date or range lies inside the dates the gold phrase means, False when it does not, None when the gold answer
    is not such a phrase or most of the picks hold no date (the usual judge then decides). picks: the picker's
    replies, an odd number."""
    span = gold_dates(gold)
    if not span:
        return None
    got = [parse_span(p) for p in picks]
    dated = [g for g in got if g]
    if len(dated) * 2 <= len(got):
        return None
    return sum(span[0] <= lo and hi <= span[1] for lo, hi in dated) * 2 > len(dated)


def gold_turns(evidence) -> list[str]:
    """A LoCoMo question's evidence as turn ids ("D3:11"); the data has a few written as "D8:6; D9:17"."""
    return sorted(set(re.findall(r"D\d+:\d+", " ".join(str(e) for e in evidence or []))))


def evidence_reached(excerpts: list[dict], turn_at: dict, gold: list[str]) -> list[str]:
    """Which of the question's evidence turns are among the excerpts the reader was handed. `turn_at` maps a
    captured turn's timestamp to its LoCoMo turn id. Without this a wrong answer cannot be told apart: memory
    did not reach the evidence, or the reader had it and still missed."""
    seen = {turn_at.get(datetime.fromisoformat(it["ts"]).astimezone(timezone.utc)) for it in excerpts}
    return sorted(set(gold) & seen)


def evidence_ranks(items: list[dict], turn_at: dict, gold: list[str]) -> dict:
    """Where each evidence turn stands in the reranked candidates (1 is the top; None: not a candidate at all).
    Says whether a missing turn was just below the cut of ten or never retrieved."""
    order = [turn_at.get(datetime.fromisoformat(it["ts"]).astimezone(timezone.utc)) for it in items]
    return {g: (order.index(g) + 1 if g in order else None) for g in gold}


def wide_ranks(wide: list[tuple], turn_at: dict, gold: list[str]) -> dict:
    """Each evidence turn's place in a wide candidate list given as (ts, dense rank, keyword rank) in fused
    order: {"rrf": position, "dense": rank or None, "keyword": rank or None}, or None if it is not in the list."""
    out = {g: None for g in gold}
    for pos, (ts, dense, kw) in enumerate(wide, 1):
        g = turn_at.get(ts.astimezone(timezone.utc))
        if g in out and out[g] is None:
            out[g] = {"rrf": pos, "dense": dense, "keyword": kw}
    return out


def partial_load(path: Path) -> tuple[dict[int, list], float]:
    """What an earlier, stopped start of the same run already finished: rows per conversation index, and what
    those starts had spent when they last saved (each start has its own meter)."""
    done, spent = {}, {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["conv"]] = r["rows"]
                spent[r["start"]] = max(spent.get(r["start"], 0.0), r["spent"])
    return done, round(sum(spent.values()), 4)


def partial_add(path: Path, start: str, conv: int, rows: list, spent: float, packs: dict | None = None) -> None:
    """Save one finished conversation, so a stopped run loses only the conversations still in work. `packs` are
    the context packs its rows name (sha to text): without them a resumed run has rows whose packs are gone, and
    reader_swap.py cannot read them (it happened: locomo_eval_20261113T235606Z kept one conversation's packs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"start": start, "conv": conv, "spent": round(spent, 4), "rows": rows}
    if packs:
        rec["packs"] = packs
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + chr(10))


def partial_packs(path: Path) -> dict[str, str]:
    """The packs saved with the conversations a stopped start finished."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.update(json.loads(line).get("packs") or {})
    return out


def packs_of(rows: list, packs: dict[str, str]) -> dict[str, str]:
    """The packs a conversation's rows name: every arm's context_sha that is held."""
    shas = {v["context_sha"] for r in rows for v in r.values() if isinstance(v, dict) and v.get("context_sha")}
    return {s: packs[s] for s in sorted(shas) if s in packs}


NOTE_KINDS = {"disputed": "is disputed by another statement", "replaced": "later replaced:",
              "ended": "no longer holds"}


def note_counts(prompt: str) -> dict:
    """How many ledger notes of each kind a context pack carries (core._ledger_notes's wordings), and how many
    times a ledger block calls a value disputed. Kept per row so answers can be read against what the pack said."""
    out = {k: prompt.count(v) for k, v in NOTE_KINDS.items()}
    out["disputed_in_blocks"] = prompt.count("disputed") - out["disputed"]
    return out


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    per_conv = int(sys.argv[1])
    global ARMS
    if "--check" in sys.argv:                  # the brain with its memory check on
        sys.argv.remove("--check")
        ARMS = ARMS + (("withcheck", True, True, True),)
    if "--notes" in sys.argv:                  # the brain without the ledger's notes on excerpts
        sys.argv.remove("--notes")
        ARMS = ARMS + (("nonotes", True, False, False),)
    if "--open" in sys.argv:                   # exp-open-route-v1: the open route with less of the ledger
        sys.argv.remove("--open")
        ARMS = ARMS + (("quiet", True, False, True, "quiet"), ("plain", True, False, True, "plain"))
    ranks = "--ranks" in sys.argv              # exp-more-excerpts-v1 spike: each evidence turn's rerank position
    if ranks:
        sys.argv.remove("--ranks")
    if "--near" in sys.argv:                   # exp-more-excerpts-v1: the turns next to each excerpt as context
        sys.argv.remove("--near")
        ARMS = ARMS + (("near1", True, False, True, "full", 1), ("near2", True, False, True, "full", 2))
    if "--twin" in sys.argv:                   # exp-reads-v1: the full brain a second time, to measure the noise
        sys.argv.remove("--twin")
        ARMS = ARMS + (("twin", True, False, True),)
    if "--wording" in sys.argv:                # exp-ledger-wording-v1: the shipped pack with other note wordings
        sys.argv.remove("--wording")
        ARMS = ARMS + (("near1", True, False, True, "full", 1), ("near1soft", True, False, True, "full", 1, "soft"),
                       ("near1off", True, False, True, "full", 1, "off"))
    judge_dates = "--judge-dates" in sys.argv  # exp-judge-dates-v1: relative gold answers also given as dates
    if judge_dates:
        sys.argv.remove("--judge-dates")
    with_dates = "--dates" in sys.argv         # exp-when-locomo-v1: dates stored at capture, written beside the phrase
    if with_dates:
        sys.argv.remove("--dates")
        ARMS = ARMS + (("near1", True, False, True, "full", 1), ("near1dates", True, False, True, "full", 1, None, True))
    if "--band" in sys.argv:                  # exp-stable-pack-v1: the shipped pack with a steadier top ten
        sys.argv.remove("--band")
        with_dates = True
        ARMS = ARMS + (("near1dates", True, False, True, "full", 1, None, True),
                       ("near1band", True, False, True, "full", 1, None, True, 0.1))
    if "--seats" in sys.argv:                 # exp-shortlist-v1: seats for the dense and keyword lists
        sys.argv.remove("--seats")
        with_dates = True
        ARMS = ARMS + (("near1dates", True, False, True, "full", 1, None, True),
                       ("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)))
    if "--ablate" in sys.argv:                # exp-ablate-providers-v1: recall without the rerank, without vectors
        sys.argv.remove("--ablate")
        ARMS = ARMS + tuple((name, False, False, False) for name in ABLATE)
    if "--wide" in sys.argv:                  # exp-wide-pack-v1: 45 candidates for Jev, 20 excerpts for the reader
        sys.argv.remove("--wide")
        with_dates = True
        ARMS = ARMS + (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                       ("near1wide", True, False, True, "full", 1, None, True, 0.0, (35, 10), (45, 20)))
    if "--gather" in sys.argv:                # exp-gather-top-v1: the count route's search reads this many a side
        k = sys.argv.index("--gather")
        gather_top = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
        with_dates = True
        ARMS = ARMS + (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                       ("near1gather", True, False, True, "full", 1, None, True, 0.0, (25, 5), None, gather_top))
    ends = "--ends" in sys.argv                # exp-relation-ends-v1: a second ingest with the ends check
    if ends:
        sys.argv.remove("--ends")
        ARMS = ARMS + (("near1", True, False, True, "full", 1), ("near1ends", True, False, True, "full", 1))
    disputes = "--disputes" in sys.argv        # exp-relate-disputes-v1: a second ingest with BRAIN_RELATE_DISPUTES
    if disputes:
        sys.argv.remove("--disputes")
        with_dates = True
        ARMS = ARMS + (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                       ("near1disputes", True, False, True, "full", 1, None, True, 0.0, (25, 5)))
    extractor = None                           # exp-full-client-v1: the model that reads the claims at capture
    if "--extractor" in sys.argv:              # (scripts/readers.py: haiku, sonnet, opus); default Haiku as before
        k = sys.argv.index("--extractor")
        extractor = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    only_conv = None                           # one conversation by its sample id (a pilot)
    if "--only-conv" in sys.argv:
        k = sys.argv.index("--only-conv")
        only_conv = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    place = "--place" in sys.argv              # exp-ledger-stable-v1: a second ingest with per-slot placement
    if place:
        sys.argv.remove("--place")
        ARMS = ARMS + (("perslot", True, False, True),)
    from brain_eval import MemoHaiku, MemoJev, ProbeJev, unrouted, flag_int, majority_read, read_splits
    memo = "--memo" in sys.argv                # one Jev answer per distinct request for the whole run
    if memo:
        sys.argv.remove("--memo")
    reads = flag_int(sys.argv, "--reads", 1)   # each answer written this many times; the majority verdict counts
    probe = flag_int(sys.argv, "--probe", 0)   # exp-jev-probe-v1: sample this many requests per kind, re-ask them
    if memo and reads != 1:
        sys.exit("--memo gives one answer per prompt, so it cannot be used with --reads")
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    import env
    env.load_env()
    from prompts import ANSWER_LOCOMO, LOCOMO_JUDGE_SYSTEM, locomo_judge_user, parse_yes_no

    from brain import claims, core, ledger, worker
    from brain.auth import Caller
    from brain.db import make_pool, migrate, tenant
    from brain.providers import ProviderError, make_providers
    from brain.schemas import ClaimIn, TurnIn
    from brain.settings import load
    from brain.spend import CapReached, PgSpendStore, SpendMeter

    dream = "end"                              # exp-ledger-dream-v1: when the nightly consolidation runs. end (as
    if "--dream" in sys.argv:                  # built): once, after the last session; none: never; nightly: after
        dream = sys.argv[sys.argv.index("--dream") + 1]     # every session, as the service's own nights do
        if dream not in ("end", "none", "nightly", "thrice"):      # thrice: three passes after the last session
            sys.exit("--dream takes end, none, nightly or thrice")
    data = None                                # another set in LoCoMo's layout (scripts/ledger_set.py writes one)
    if "--data" in sys.argv:
        data = Path(sys.argv[sys.argv.index("--data") + 1]).resolve()
    data_raw = (data or SERVICE.parents[1] / "locomo10.json").read_bytes()
    convs = json.loads(data_raw.decode("utf-8"))
    data_id = [data.name, hashlib.sha256(data_raw).hexdigest()[:16]] if data else None
    if only_conv:
        convs = [c for c in convs if c["sample_id"] == only_conv]
    code = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            for f in [SERVICE / "brain/core.py", SERVICE / "brain/ledger.py", Path(__file__).resolve()]}
    # a stopped start is resumed only by the same code: every module of the brain (see brain_eval.py)
    code_all = hashlib.sha256(b"".join(f.read_bytes() for f in sorted((SERVICE / "brain").glob("*.py")))).hexdigest()[:16]
    run_key = hashlib.sha256(json.dumps([per_conv, skip, [list(a) for a in ARMS], reads, memo, ranks, probe, place, ends, with_dates, judge_dates, code, disputes, code_all, extractor, only_conv] + ([data_id] if data_id else []) + ([dream] if dream != "end" else []) + (["across"] if ledger.RELATE_ACROSS else [])).encode()).hexdigest()[:12]
    partial = CACHE.parent / f"partial_locomo_eval_{run_key}.jsonl"    # same arguments and code: resumed
    start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    done, spent_before = partial_load(partial)
    if done:
        print(f"resuming: {len(done)} conversations already saved by a stopped start (${spent_before} spent then)",
              flush=True)

    def fresh(name: str) -> str:
        with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
            c.execute(f"drop database if exists {name} with (force)")
            c.execute(f"create database {name}")
        u = db_url(name)
        migrate(u)
        return u

    meter_pool = make_pool(fresh(DB), max_size=2)
    cap = flag_int(sys.argv, "--cap", int(CAP))   # dollars; above 5 only on the owner's explicit go for that run
    meter = SpendMeter(PgSpendStore(meter_pool), float(cap))
    st = load()
    prov = make_providers(st, meter)
    extract = prov.haiku
    if extractor:
        from readers import Reader
        extract = Reader(extractor, meter, st.anthropic_key)
    focus = tuple(a.split("=", 1)[1].split(",")[0:20] for a in sys.argv if a.startswith("--probe-focus="))
    prober = ProbeJev(prov.jev, probe, focus=focus[0] if focus else ()) if probe else None   # under the memo:
    # it sees each distinct request once
    if prober:
        prov.jev = prober
    if memo:
        prov.jev, prov.haiku = MemoJev(prov.jev), MemoHaiku(prov.haiku)
    try:
        prov.jev.ask({"text": "We moved last spring."}, {"q": {"type": "noul", "instructions": "Does the text mention a move?"}})
    except ProviderError as e:
        sys.exit(f"Jev preflight failed, nothing run: {e}")
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["k"]] = r["claims"]
    new_cache: list[dict] = []
    system = claims.CLAIMS_CONTRACT + "\nReply with a JSON list of claims only; [] if the turn states none."

    def client_claims(text: str, speaker: str, prev: list[str], date: str) -> list:
        key = hashlib.sha256(json.dumps([system, text, speaker, prev, date] + ([extractor] if extractor else [])).encode()).hexdigest()[:24]
        with _lock:
            hit = cache.get(key)
        if hit is None:
            seen = {"speaker": speaker, "role": "user", "channel": "chat", "session_date": date,
                    "previous_turns": prev, "known_slots": [], "turn": text}
            try:
                raw = extract.complete(system, json.dumps(seen, ensure_ascii=False), 2000).text
            except CapReached:
                raise                # a capture without its claims is not the thing measured: stop the run
            except Exception:
                return []
            hit, _ = claims.parse_claims(raw)
            with _lock:
                cache[key] = hit
                new_cache.append({"k": key, "claims": hit})
        out = []
        for c in hit:
            try:
                out.append(ClaimIn(**c))
            except ValueError:
                pass
        return out[:25]

    def judge(question, gold, pred):
        if judge_dates and gold_dates(gold):       # the model picks the answer's date, code compares
            picks = [prov.haiku.complete(DATE_PICK_SYSTEM, date_pick_user(question, pred), 30).text for _ in range(3)]
            v = dated_verdict(gold, picks)
            if v is not None:
                return v, [v, "dates", *picks]
        votes = []
        for _ in range(6):
            v = parse_yes_no(prov.haiku.complete(LOCOMO_JUDGE_SYSTEM, locomo_judge_user(question, gold, pred), 10).text)
            if v is not None:
                votes.append(v)
            if len(votes) == 3:
                break
        return (sum(votes) >= 2) if len(votes) == 3 else None, votes

    packs: dict[str, str] = partial_packs(partial)

    def one(item):
        if item[0] in done:
            return done[item[0]]
        if disputes:
            rows = one_db(item, ARMS[:-1], None, "", None, False)
            for row, second in zip(rows, one_db(item, ARMS[-1:], None, "b", None, True)):
                assert row["qid"] == second["qid"]
                row["near1disputes"] = second["near1disputes"]
                row["ledger_totalsb"] = second["ledger_totalsb"]
            return rows
        if ends:
            rows = one_db(item, ARMS[:-1], None, "", False)
            for row, second in zip(rows, one_db(item, ARMS[-1:], None, "b", True)):
                assert row["qid"] == second["qid"]
                row["near1ends"] = second["near1ends"]
                row["ledger_totalsb"] = second["ledger_totalsb"]
            return rows
        if not place:
            return one_db(item, ARMS, None, "")
        rows = one_db(item, ARMS[:-1], False, "")
        for row, second in zip(rows, one_db(item, ARMS[-1:], True, "b")):
            assert row["qid"] == second["qid"]
            row["perslot"] = second["perslot"]
        return rows

    def one_db(item, arm_set, per_slot, suffix, check_ends=None, check_disputes=None):
        i, conv = item
        qas = [(k, qa) for k, qa in enumerate(conv["qa"]) if category(qa, bool(data))]
        qas = sorted(qas, key=lambda x: hashlib.sha256(f"{conv['sample_id']}:{x[0]}".encode()).hexdigest())[skip:skip + per_conv]
        if not qas:            # nothing asked of this conversation in this slice: do not pay to load it
            return []
        c = conv["conversation"]
        name = f"{DB}_{i:02d}{suffix}"
        pool = make_pool(fresh(name), max_size=2)
        rows = []
        token = ledger.PLACE_PER_SLOT_OVERRIDE.set(per_slot)     # this thread's ingest only
        token_ends = ledger.RELATE_ENDS_OVERRIDE.set(check_ends)
        token_disputes = ledger.RELATE_DISPUTES_OVERRIDE.set(check_disputes)
        try:
            with pool.connection() as cn:
                owner = cn.execute("select id from brain.account where name = 'owner'").fetchone()[0]
            caller = Caller("eval", ("*",), ("*",), None, True, owner, "owner", False, True)
            ns = "u"
            n = 1
            last_date = ""
            turn_at = {}
            stored_nightly = 0
            while f"session_{n}" in c:
                date_s = c[f"session_{n}_date_time"]
                base = session_dt(date_s)
                last_date = date_s
                turns, prev = [], []
                for k, t in enumerate(c[f"session_{n}"]):
                    text = str(t.get("text") or "")
                    if t.get("blip_caption"):
                        text += f" [shares an image: {t['blip_caption']}]"
                    if not text.strip():
                        continue
                    sp = str(t.get("speaker") or "user")
                    cl = client_claims(text, sp, prev[-2:], base.date().isoformat())
                    turns.append(TurnIn(role="user", speaker=sp, text=text, ts=base + timedelta(seconds=k), claims=cl))
                    turn_at[base + timedelta(seconds=k)] = t.get("dia_id")
                    prev.append(f"{sp}: {text[:400]}")
                for b in range(0, len(turns), 100):
                    with tenant(pool, owner) as conn:
                        core.capture(conn, prov, caller, ns, turns[b:b + 100], f"session_{n}", None,
                                     write_path_enabled=True, dates_enabled=with_dates)
                if dream == "nightly":         # the night after this session: its claims placed, then consolidated
                    stored_nightly += (worker.write_path(pool, prov, max_episodes=100_000).get("stored") or 0)
                    worker.embed_backfill(pool, prov.embedder, 100_000)
                    worker.consolidate(pool, prov)
                n += 1
            wp = worker.write_path(pool, prov, max_episodes=100_000)
            if dream == "nightly":
                wp = {**wp, "stored": (wp.get("stored") or 0) + stored_nightly}
            worker.embed_backfill(pool, prov.embedder, 100_000)
            for _ in range({"end": 1, "thrice": 3}.get(dream, 0)):
                worker.consolidate(pool, prov)
            worker.promote_preferences(pool, prov)
            worker.render_pages(pool)
            qdate = session_dt(last_date).date().isoformat() if last_date else None
            for k, qa in qas:
                row = {"qid": f"{conv['sample_id']}:{k}", "cat": category(qa, bool(data)), "question": qa["question"],
                       "gold": str(qa.get("answer")), "evidence": gold_turns(qa.get("evidence"))}
                if qa.get("must") is not None:     # a set with computed answers: the literal values a right answer holds
                    row["must"] = qa["must"]
                    row["now"], row["old"] = qa.get("now") or [], qa.get("old") or []
                if ranks:      # the same candidates and (with --memo) the same Jev answers as the arms' recall
                    with tenant(pool, owner) as conn:
                        every = core.recall(conn, prov, caller, ns, qa["question"], k=10_000)
                    row["candidates"] = len(every["items"])
                    row["evidence_ranks"] = evidence_ranks(every["items"], turn_at, row["evidence"])
                    with tenant(pool, owner) as conn:      # the service's candidate statement, wide, no Jev
                        conn.execute("select set_config('hnsw.ef_search', '400', true)")
                        wide = conn.execute(core.RECALL_SQL, {
                            "ns": [ns], "q": qa["question"], "qv": core._vec(prov.embedder.embed_query(qa["question"])),
                            "sid": None, "channels": None, "tf": None, "tt": None, "dense_top": 300, "kw_top": 300,
                            "kw_common": core.KEYWORD_COMMON, "kw_freq": core.KEYWORD_FREQ, "rrf_k": core.RRF_K,
                            "shortlist": 600, "dense_seats": 0, "kw_seats": 0}).fetchall()
                    row["wide_candidates"] = len(wide)
                    row["wide_ranks"] = wide_ranks([(w[8], w[2], w[3]) for w in wide], turn_at, row["evidence"])
                for arm, routed, check, notes, *mode in arm_set:
                    with tenant(pool, owner) as conn:
                        ctx = core.context(conn, ablated(prov, ABLATE.get(arm, ())), caller, ns, qa["question"], qa.get("question_date") or qdate, "default", None,
                                           when_route=False, ledger_route=routed, memory_check=check,
                                           ledger_notes=notes, open_route=mode[0] if mode else "full",
                                           neighbours=mode[1] if len(mode) > 1 else 0,
                                           note_style=mode[2] if len(mode) > 2 else None,
                                           date_notes=mode[3] if len(mode) > 3 else False,
                                           rerank_band=mode[4] if len(mode) > 4 else 0.0,
                                           seats=mode[5] if len(mode) > 5 else (0, 0),
                                           wide=mode[6] if len(mode) > 6 else None,
                                           count_top=mode[7] if len(mode) > 7 else None)
                    with _lock:        # the pack itself, by its hash, for reading answers against (git-ignored)
                        packs[hashlib.sha256(ctx["prompt"].encode()).hexdigest()[:12]] = ctx["prompt"]
                    row[arm] = {**majority_read(
                                    lambda: prov.haiku.complete(ANSWER_LOCOMO, ctx["prompt"], 400).text,
                                    lambda pred: judge(qa["question"], row["gold"], pred), reads),
                                "route": ctx["route"], "ledger": ctx["ledger"], "degraded": ctx.get("degraded"),
                                "evidence_reached": evidence_reached(ctx["excerpts"] + ctx["nearby"], turn_at, row["evidence"]),
                                "nearby": len(ctx["nearby"]), "notes": note_counts(ctx["prompt"]),
                                "excerpt_turns": [turn_at.get(datetime.fromisoformat(it["ts"]).astimezone(timezone.utc))
                                                  for it in ctx["excerpts"]],
                                "context_sha": hashlib.sha256(ctx["prompt"].encode()).hexdigest()[:12],
                                "excerpts": [it["chunk_id"] for it in ctx["excerpts"]]}
                rows.append(row)
            if unrouted(rows, [a[0] for a in arm_set if a[1]]):
                raise RuntimeError("Jev did not answer (the router was skipped): the run is void, stopping")
            with pool.connection() as cn:
                decs = cn.execute("""select stage, decider, model_version, input, output, probability, threshold, outcome
                                     from brain.decision_log where stage like 'write.%' or stage like 'nightly.%'
                                     order by id""").fetchall()
            with _lock:
                DECISIONS.parent.mkdir(parents=True, exist_ok=True)
                with DECISIONS.open("a", encoding="utf-8") as f:
                    f.write("".join(json.dumps({"qid": conv["sample_id"], "stage": s, "decider": dc, "model": m,
                                                "input": inp, "output": o, "p": p, "threshold": th, "outcome": oc},
                                               default=str) + "\n" for s, dc, m, inp, o, p, th, oc in decs))
            led = wp.get("ledger") or {}
            if data and rows:      # the ledger as it stands when the questions are asked (scripts/ledger_report.py)
                with tenant(pool, owner) as conn:
                    rows[0][f"ledger_state{suffix}"] = [
                        {"entity": r[0], "slot": r[1], "value": r[2], "status": r[3], "polarity": r[4],
                         "valid_from": r[5] and str(r[5]), "valid_to": r[6] and str(r[6]), "quote": r[7], "said": str(r[8])[:10]}
                        for r in conn.execute("""select e.name, s.name, c.value, c.status, c.polarity, c.valid_from, c.valid_to,
                                                        c.quote, ep.ts
                                                 from claim c join entity e on e.id = c.entity_id
                                                 left join slot s on s.id = c.slot_id join episode ep on ep.id = c.episode_id
                                                 order by c.id""").fetchall()]
            for row in rows:       # what this ingest's ledger did, per arm set (the same for every question of it)
                row[f"ledger_totals{suffix}"] = {k: led.get(k) for k in ("placed", "slots_created", "superseded",
                                                                         "contested")}
            print(f"{conv['sample_id']}{suffix}: {len(rows)} questions, write path {wp.get('stored')} claims; "
                  f"spent ${meter.spent_today():.3f}", flush=True)
            return rows
        finally:
            ledger.RELATE_DISPUTES_OVERRIDE.reset(token_disputes)
            ledger.RELATE_ENDS_OVERRIDE.reset(token_ends)
            ledger.PLACE_PER_SLOT_OVERRIDE.reset(token)
            pool.close()
            with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as cn:
                cn.execute(f"drop database if exists {name} with (force)")

    def save_claims():
        with _lock:
            if new_cache:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                with CACHE.open("a", encoding="utf-8") as f:
                    f.write("".join(json.dumps(r) + chr(10) for r in new_cache))
                new_cache.clear()

    def one_saved(item):
        part = one(item)
        if item[0] not in done:
            save_claims()
            with _lock:
                partial_add(partial, start, item[0], part, meter.spent_today(), packs_of(part, packs))
        return part

    rows = []
    with cf.ThreadPoolExecutor(3) as ex:
        for part in ex.map(one_saved, enumerate(convs)):
            rows.extend(part)
    if new_cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in new_cache))

    def acc(rs, arm):
        v = [r[arm]["correct"] for r in rs if r[arm]["correct"] is not None]
        return round(sum(v) / len(v), 4) if v else None
    summary = {"n": len(rows), "per_conversation": per_conv, "skip": skip, "reads": reads,
               "read_splits": read_splits(rows, [a[0] for a in ARMS]),
               "jev_memo": {"hits": prov.jev.hits, "misses": prov.jev.misses} if memo else None,
               "arm_labels": {a[0]: ARM_LABELS[a[0]] for a in ARMS}, "by_category": {}}
    for cat in list(CATS.values()) + sorted({r["cat"] for r in rows} - set(CATS.values())) + ["all"]:
        rs = [r for r in rows if cat == "all" or r["cat"] == cat]
        if not rs:
            continue
        diffs = [int(bool(r["brain"]["correct"])) - int(bool(r["flat"]["correct"])) for r in rs]
        rnd = random.Random(7)
        boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
        summary["by_category"][cat] = {"n": len(rs), "flat": acc(rs, "flat"), "brain": acc(rs, "brain"),
                                       "diff": round(sum(diffs) / len(diffs), 4), "ci95": [boots[50], boots[1949]],
                                       "flips": {"brain_only": diffs.count(1), "flat_only": diffs.count(-1)}}
        for x, *_ in ARMS[2:]:                 # one switch's own effect: brain - x
            d2 = [int(bool(r["brain"]["correct"])) - int(bool(r[x]["correct"])) for r in rs]
            b2 = sorted(sum(rnd.choice(d2) for _ in d2) / len(d2) for _ in range(2000))
            summary["by_category"][cat][f"vs_{x}"] = {x: acc(rs, x), "diff": round(sum(d2) / len(d2), 4),
                                                       "ci95": [b2[50], b2[1949]],
                                                       "flips": {"brain_only": d2.count(1), f"{x}_only": d2.count(-1)}}
    summary["spend_usd"] = round(meter.spent_today() + spent_before, 4)
    if done:
        summary["resumed"] = {"conversations_from_stopped_starts": len(done), "their_spend_usd": spent_before}
    out = SERVICE / "runs" / f"locomo_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (SERVICE / "runs" / "cache" / f"packs_{out.name}.json").write_text(json.dumps(packs), encoding="utf-8")
    shipped = {k: _os.environ.get(k) for k in ("BRAIN_PLACE_PER_SLOT", "BRAIN_PLACE_CHECK", "BRAIN_NEIGHBOURS",
                                               "BRAIN_OPEN_ROUTE", "BRAIN_RELATE_ENDS", "BRAIN_NOTE_STYLE", "BRAIN_DATES",
                                               "BRAIN_DATE_NOTES", "BRAIN_RELATE_ACROSS")}       # the settings this run's process had
    (out / "settings.json").write_text(json.dumps(shipped, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"per_conversation": per_conv, "skip": skip, "reads": reads, "cap": cap, "hashes": code, "extractor": extractor or "haiku", "only_conv": only_conv, "data": data_id, "dream": dream}, indent=2))
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                                    encoding="utf-8")
    if prober:
        probe_sum, probe_rows = prober.replay()
        (out / "jev_probe.json").write_text(json.dumps({"summary": probe_sum, "requests": probe_rows}, indent=1),
                                            encoding="utf-8")
        summary["jev_probe"] = probe_sum
        payloads = SERVICE / "runs" / "cache" / f"probe_payloads_{out.name}.json"   # git-ignored: benchmark text
        payloads.parent.mkdir(parents=True, exist_ok=True)
        payloads.write_text(json.dumps(prober.payloads), encoding="utf-8")
        print(f"probe payloads: {payloads}")
        summary["spend_usd"] = round(meter.spent_today() + spent_before, 4)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)
    partial.unlink(missing_ok=True)            # the run is whole now; its rows are in the run dir
    prov.close()
    meter_pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
