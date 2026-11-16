"""Benchmark phase: the whole brain on LongMemEval S, with Haiku 5.5 (thinking off) as the user's model.

    python scripts/brain_eval.py noise test 25          # flat vs flat on the same data: the noise floor
    python scripts/brain_eval.py brain test 20 knowledge-update multi-session single-session-preference
    python scripts/brain_eval.py check test 30 abstention   # the brain with and without its memory check
    python scripts/brain_eval.py brain test 20 knowledge-update --skip 20   # the next 20 by the same hash order
    python scripts/brain_eval.py brain test 20 knowledge-update --reads 3   # each answer written 3 times, majority

For each question (N per type from the split, chosen by a hash of the question id), its whole S haystack is
captured into a database of its own on the Supabase test branch. In brain mode Haiku plays the user's model: it
reads every user turn under the capture tool's claims contract (claims.CLAIMS_CONTRACT) and sends its claims with
the capture call; then the worker's steps run as at night (write path: checks and ledger; embedding backfill;
consolidation; preference promotion; page rendering). Two arms answer from the same data, Haiku writing the
answer (the reader adapter) and graded with the official LongMemEval judge prompt per type (majority of 3):
  noise: flat_a vs flat_b (the measured open route twice): the run-to-run noise any gate has to clear;
  brain: flat (routes off) vs brain (ledger, counts and preferences on; dates off, exp-when-v1 failed);
  check: brain vs withcheck (the memory check on); notes: nonotes (no ledger notes on excerpts) vs brain.
  Type `abstention`: the _abs questions, any type. (exp-memory-check-v1 ran check mode as nocheck, with the notes
  also off, vs brain with both on.)
Client extractions are cached on disk (runs/cache/client_claims.jsonl) so a turn is paid for once across runs.
Writes runs/brain_eval_<mode>_<split>_<stamp>/ (config, rows, summary).
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import hashlib
import json
import os
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

DB = "brain_eval"


def sessions_of(conn, chunk_ids: list[int]) -> list[str]:
    """The sessions behind the excerpts an answer was given (its citations), for reading a result afterwards:
    which evidence sessions reached the reader (exp-session-slots-v1 could not be read without this)."""
    if not chunk_ids:
        return []
    rows = conn.execute("""select distinct coalesce(e.session_id, ce.session_id)
                           from brain.chunk k
                           left join brain.episode e on k.kind = 'episode_line' and e.id = k.ref_id
                           left join brain.claim c on k.kind = 'claim' and c.id = k.ref_id
                           left join brain.episode ce on ce.id = c.episode_id
                           where k.id = any(%s)""", (list(chunk_ids),)).fetchall()
    return sorted(r[0] for r in rows if r[0])
def majority_read(read, judge, n: int = 1) -> dict:
    """One arm's answer to one question, written `n` times from the same context (n odd) and each time judged:
    `correct` is the majority of the n verdicts, None when the unjudged reads leave it open. Haiku 5.5 takes no
    temperature, and two answers from the same context differ often enough to flip about one verdict in ten
    (exp-open-route-v1), so a single read cannot resolve small differences between arms. `answer` and `votes`
    are the first read's; with n > 1 every read is kept under `reads`."""
    if n < 1 or n % 2 == 0:
        raise ValueError("reads must be odd")
    reads = []
    for _ in range(n):
        answer = read()
        ok, votes = judge(answer)
        reads.append({"answer": answer, "correct": ok, "votes": votes})
    yes = sum(r["correct"] is True for r in reads)
    no = sum(r["correct"] is False for r in reads)
    out = dict(reads[0], correct=True if yes * 2 > n else False if no * 2 > n else None)
    if n > 1:
        out["reads"] = reads
    return out


def read_splits(rows: list[dict], arms) -> dict:
    """Per arm: on how many questions the reads did not all get the same verdict (0 with one read)."""
    return {a: sum(len({x["correct"] for x in r[a].get("reads", [])}) > 1 for r in rows) for a in arms}


class MemoJev:
    """Jev for one eval run, answering a request it has seen before with the answer it gave then. Jev's
    probabilities move by up to 0.05 between two identical requests (exp-reads-v1), which gave two arms with
    the same settings a different context on 61% of questions. With the memo, arms differ only where their
    requests differ. It measures nothing new about Jev and is for the eval scripts only, never the service."""

    def __init__(self, jev):
        self._jev, self._seen, self._lock = jev, {}, threading.Lock()
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self._jev, name)

    @staticmethod
    def _key(state: dict, questions: dict) -> str:
        return hashlib.sha256(json.dumps([state, questions], sort_keys=True, default=str).encode()).hexdigest()

    def ask(self, state: dict, questions: dict):
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, requests: list) -> list:
        keys = [self._key(s, q) for s, q in requests]
        with self._lock:
            new = {k: r for k, r in zip(keys, requests) if k not in self._seen}
            self.hits += len(keys) - len(new)
            self.misses += len(new)
        if new:
            got = self._jev.ask_many(list(new.values()))
            with self._lock:
                for k, r in zip(new, got):
                    self._seen.setdefault(k, r)
        with self._lock:
            return [self._seen[k] for k in keys]


class MemoHaiku:
    """The eval's Haiku (reader and judge) with the same memo: one completion per distinct (system, user,
    max_tokens). Two arms that hand the reader the same prompt get the same answer and the same verdict, so
    they can differ only on questions where the brain gave them something different. The judge's three votes
    on one answer become one vote counted three times (its votes split on under 2% of answers)."""

    def __init__(self, haiku):
        self._haiku, self._seen, self._lock = haiku, {}, threading.Lock()
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self._haiku, name)

    def complete(self, system: str, user: str, max_tokens: int = 1024):
        key = hashlib.sha256(json.dumps([system, user, max_tokens]).encode()).hexdigest()
        with self._lock:
            hit = self._seen.get(key)
            self.hits += hit is not None
            self.misses += hit is None
        if hit is None:
            got = self._haiku.complete(system, user, max_tokens)
            with self._lock:
                hit = self._seen.setdefault(key, got)
        return hit


def unrouted(rows: list[dict], routed_arms) -> int:
    """Rows where an arm that asks the router got no route: Jev did not answer (no credits, an outage), the
    service fell back to recall without it, and the run measures something else. A run with any is void."""
    return sum(r[a]["route"] is None for r in rows for a in routed_arms if a in r)


PROBE_LINES = (0.3, 0.5, 0.7, 0.8) # the cut-offs the brain's code uses most; a Noul "crosses" one if its
                                   # repeats fall on both sides of it
SCORE_LINES = (0.5, 1.5)           # the entity-alignment Score's cut-offs (DESIGN.md write path step 6)
PICK_LINES = (0.4, 0.5, 0.7, 0.75)  # gates on a Choice's pick probability: LOOKUP_PICK, SLOT_PICK_AT,
                                    # LOOKUP_PICK_SIDE, ROUTE_HEDGE


def probe_stats(questions: dict, repeats: list[dict]) -> dict:
    """How much one request's answers moved over its repeats. `repeats` is the raw answers of each repeat
    ({question name: answer}). Per question: a Noul's lowest and highest probability and the lines between
    them; the same for a Score (a position on its levels, a number); for a Choice, whether the pick changed."""
    out = {}
    for name, q in questions.items():
        kind = q.get("type")
        if kind == "noul":
            ps = [float(r[name]["noul"]) for r in repeats]
            out[name] = {"type": "noul", "min": min(ps), "max": max(ps),
                         "crosses": [t for t in PROBE_LINES if min(ps) < t <= max(ps)]}
        elif kind == "score":      # exp-jev-probe-v1 compared the positions as text: every small move was a "change"
            ps = [float(r[name]["score"]) for r in repeats]
            out[name] = {"type": "score", "min": min(ps), "max": max(ps),
                         "crosses": [t for t in SCORE_LINES if min(ps) < t <= max(ps)]}
        else:
            picks = [str(r[name]["choice"]) for r in repeats]
            out[name] = {"type": kind, "changed": len(set(picks)) > 1}
            first = picks[0]       # the probability the first repeat's pick got in every repeat
            ps = [float((r[name].get("probabilities") or {}).get(first, 0.0)) for r in repeats]
            if any(r[name].get("probabilities") for r in repeats):
                out[name].update(min=min(ps), max=max(ps),
                                 crosses=[t for t in PICK_LINES if min(ps) < t <= max(ps)])
    return out


def probe_summary(per_request: dict) -> dict:
    """Per kind of request (its question names) and question: how many requests were probed, the mean and
    largest spread of a Noul, the share of requests crossing each line, the share of picks that changed."""
    summary = {}
    for kind, stats in per_request.items():
        names = sorted({n for s in stats for n in s})
        summary[kind] = {"requests": len(stats)}
        for n in names:
            rows = [s[n] for s in stats if n in s]
            if rows[0]["type"] in ("noul", "score"):
                spreads = [r["max"] - r["min"] for r in rows]
                lines = PROBE_LINES if rows[0]["type"] == "noul" else SCORE_LINES
                summary[kind][n] = {"mean_spread": round(sum(spreads) / len(rows), 4),
                                    "max_spread": round(max(spreads), 4),
                                    **{f"crosses_{t}": round(sum(t in r["crosses"] for r in rows) / len(rows), 4)
                                       for t in lines}}
            else:
                summary[kind][n] = {"pick_changed": round(sum(r["changed"] for r in rows) / len(rows), 4)}
                if all("crosses" in r for r in rows):
                    summary[kind][n].update({f"pick_crosses_{t}": round(sum(t in r["crosses"] for r in rows)
                                                                        / len(rows), 4) for t in PICK_LINES})
    return summary


class ProbeJev:
    """Jev for one eval run that keeps a random sample of the requests the brain really makes, per kind of
    request (the set of question names: the rerank, the router, a ledger pick and so on), and can ask each
    sampled request again several times (`replay`). It answers what the design assumed and exp-reads-v1 put in
    doubt: which of the brain's decisions fall on both sides of their threshold between two identical calls."""

    def __init__(self, jev, per_kind: int = 60, seed: int = 7, focus: tuple = (), focus_n: int = 300):
        self._jev, self._per, self._rng, self._lock = jev, per_kind, random.Random(seed), threading.Lock()
        self._focus, self._focus_n = set(focus), focus_n     # question names whose requests keep a larger sample
        self.samples: dict[str, list] = {}
        self.seen: dict[str, int] = {}
        self.payloads: dict[str, list] = {}    # after replay: every sampled request with its raw repeated answers

    def __getattr__(self, name):
        return getattr(self._jev, name)

    def ask(self, state: dict, questions: dict):
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, requests: list) -> list:
        with self._lock:
            for state, questions in requests:          # reservoir sampling: every request equally likely kept
                kind = ",".join(sorted(questions))
                n = self.seen[kind] = self.seen.get(kind, 0) + 1
                kept = self.samples.setdefault(kind, [])
                per = self._focus_n if self._focus & set(questions) else self._per
                if len(kept) < per:
                    kept.append((state, questions))
                else:
                    j = self._rng.randrange(n)
                    if j < per:
                        kept[j] = (state, questions)
        return self._jev.ask_many(requests)

    def replay(self, times: int = 5) -> tuple[dict, dict]:
        """Ask every sampled request `times` times. Returns (summary, the per-request numbers per kind)."""
        per_request = {}
        for kind, reqs in self.samples.items():
            runs = [self._jev.ask_many(reqs) for _ in range(times)]
            per_request[kind] = [probe_stats(q, [run[i].answers for run in runs]) for i, (_, q) in enumerate(reqs)]
            self.payloads[kind] = [{"state": s, "questions": q, "answers": [run[i].answers for run in runs]}
                                   for i, (s, q) in enumerate(reqs)]
        summary = probe_summary(per_request)
        for kind in summary:
            summary[kind]["seen_in_run"] = self.seen[kind]
        return summary, per_request


def arm_diffs(rows: list, a1: str, a2: str) -> list[int]:
    """Per question judged in both arms: +1 if only `a2` is right, -1 if only `a1`, else 0. A question the
    judge left open in either arm (correct None) is left out; it is not a flip."""
    return [int(r[a2]["correct"]) - int(r[a1]["correct"]) for r in rows
            if r[a1]["correct"] is not None and r[a2]["correct"] is not None]


def flag_int(args: list[str], name: str, default: int) -> int:
    """Take `name N` out of args and return N."""
    if name not in args:
        return default
    k = args.index(name)
    v = int(args[k + 1])
    del args[k:k + 2]
    return v


CAPS = {"noise": 2.0, "brain": 5.0, "check": 5.0, "notes": 5.0, "open": 5.0, "near": 5.0, "place": 5.0, "wording": 5.0, "ends": 5.0, "dates": 5.0, "band": 5.0, "seats": 5.0, "names": 5.0, "count": 5.0, "countstyle": 5.0, "reader": 5.0, "disputes": 5.0, "datedir": 5.0}
# What each arm name in rows.jsonl and summary.json means. The keys are kept as they were (older runs use
# them); every summary now carries the meaning. Both arms answer from the brain service: "flat" is its
# retrieval layer on its own, so brain minus flat is what the upper layers add, not what memory adds.
ARM_LABELS = {
    "flat": "retrieval only: the brain service's own recall (stored turns, keyword and vector search, Jev rerank); "
            "ledger, counts, preferences and notes off. Not a no-memory baseline",
    "flat_a": "retrieval only, first pass (see flat)",
    "flat_b": "retrieval only, second pass on the same data (see flat)",
    "brain": "full brain: retrieval plus the ledger's facts, counts, preferences and the notes on excerpts",
    "nonotes": "full brain without the ledger notes on excerpts",
    "withcheck": "full brain with the memory check's line added",
    "plain": "full brain, but on open-route questions retrieval only (no ledger blocks, no notes)",
    "near1": "full brain, plus the turn before and after each excerpt in its session, screened, as context",
    "near2": "full brain, plus the two turns before and after each excerpt in its session, screened, as context",
    "perslot": "full brain over a second ingest of the same sessions in which a claim is placed by one Jev question "
               "per existing slot (ledger.PLACE_PER_SLOT) instead of fits + which",
    "near1soft": "near1 (the shipped pack) with the ledger notes worded softly: they point at the later statement "
                 "and do not say it replaced anything (core.NOTE_STYLE soft)",
    "near1off": "near1 (the shipped pack) without the ledger notes on excerpts (core.NOTE_STYLE off)",
    "near1band": "near1dates (the shipped pack since 2026-11-12) with Jev's rerank scores cut into steps of 0.1 "
                 "before the top ten are taken, the fused rank deciding inside a step (core.RERANK_BAND)",
    "near1evidence": "near1seats (the shipped pack) with the count route's list worded as further statements: no "
                     "number of items, and the sum said to be a sum (ledger.COUNT_STYLE evidence)",
    "near1nocount": "near1seats (the shipped pack) without the count route's gathered, code-counted list: on a count "
                    "question the excerpts answer alone (core.COUNT_BLOCK off)",
    "near1datedir": "near1seats (the shipped pack) read from a second ingest made with BRAIN_DATE_DIRECTION: a date "
                    "or month with no year written takes its year from whether the sentence is about the past or a plan",
    "near1disputes": "near1seats (the shipped pack) read from a second ingest made with BRAIN_RELATE_DISPUTES: a "
                     "\"contradicts\" marks two claims disputed only when the older one was judged to have stopped being true",
    "near1names": "near1seats (the shipped pack) read from a second ingest made with BRAIN_RESOLVE_NAMES: a proper "
                  "name is never linked tentatively to an entity known by a different proper name",
    "near1seats": "near1dates (the shipped pack since 2026-11-12) with the 30 candidates Jev reads chosen as "
                  "the dense top 25 and the keyword top 5, the rest by the fused rank (core.SHORTLIST_SEATS)",
    "near1dates": "near1 (the shipped pack) with each resolved date written beside its time phrase in the excerpts "
                  "and nearby turns (core.DATE_NOTES; dates stored at capture)",
    "near1ends": "near1 (the shipped pack) over a second ingest in which an update ends the older value only when "
                 "Jev also judges that the older value stopped being true (ledger.RELATE_ENDS)",
    "no_memory": "the answering model alone, nothing retrieved (not run by this script)",
}
CACHE = SERVICE / "runs" / "cache" / "client_claims.jsonl"
DECISIONS = SERVICE / "runs" / "cache" / "decisions.jsonl"     # every brain-mode run appends its decision logs
DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?: \(\w+\))?(?: (\d{2}):(\d{2}))?")
_lock = threading.Lock()


def session_ts(s: str) -> datetime:
    y, mo, d, hh, mm = DATE_RE.search(s).groups()
    return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), tzinfo=timezone.utc)


def pick(split: str, n: int, types: list[str], skip: int = 0) -> list[dict]:
    import route_eval
    rows = {r["question_id"]: r for r in json.loads(route_eval.S_FILE.read_text(encoding="utf-8"))}
    dev, test = route_eval.splits(route_eval.load_questions())
    pool = dev if split == "dev" else test
    out = []
    for t in types:
        qs = sorted((q for q in pool if (q["is_abs"] if t == "abstention" else q["qtype"] == t)),
                    key=lambda q: hashlib.sha256(q["qid"].encode()).hexdigest())
        out += [rows[q["qid"]] for q in qs[skip:skip + n]]
    return out


def load_cache() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["k"]] = r["claims"]
    return out


def main() -> int:
    args = sys.argv[1:]
    skip = flag_int(args, "--skip", 0)      # the next n per type by the same hash order (a fresh slice)
    reads = flag_int(args, "--reads", 1)    # each answer written this many times; the majority verdict counts
    only = None
    if "--only" in args:                    # question ids, comma-separated: the picked questions are cut to these
        k = args.index("--only")
        only = set(args[k + 1].split(","))
        del args[k:k + 2]
    cap = flag_int(args, "--cap", 0)        # whole dollars; 0 = the mode's own cap (CAPS). A larger run needs the owner's word
    memo = "--memo" in args                 # one Jev answer per distinct request for the whole run (MemoJev)
    if memo:
        args.remove("--memo")
        if reads != 1:
            sys.exit("--memo gives one answer per prompt, so it cannot be used with --reads")
    if len(args) < 3 or args[0] not in CAPS or args[1] not in ("dev", "test"):
        sys.exit(__doc__)
    mode, split, n = args[0], args[1], int(args[2])
    types = args[3:] or ["knowledge-update", "multi-session", "single-session-preference", "temporal-reasoning"]
    import env
    env.load_env()
    from prompts import lme_judge_prompt, parse_yes_no

    from brain import claims, core, dates, ledger, worker
    from brain.auth import Caller
    from brain.db import make_pool, migrate, tenant
    from brain.providers import ProviderError, make_providers
    from brain.schemas import ClaimIn, TurnIn
    from brain.settings import load
    from brain.spend import PgSpendStore, SpendMeter

    qs = pick(split, n, types, skip)
    if only:
        qs = [q for q in qs if q["question_id"] in only]
    print(f"{mode} {split}: {len(qs)} questions", flush=True)

    def fresh(name: str) -> str:
        with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
            c.execute(f"drop database if exists {name} with (force)")
            c.execute(f"create database {name}")
        u = db_url(name)
        migrate(u)
        return u

    meter_pool = make_pool(fresh(DB), max_size=2)
    meter = SpendMeter(PgSpendStore(meter_pool), float(cap) if cap else CAPS[mode])
    st = load()
    prov = make_providers(st, meter)
    if memo:
        prov.jev, prov.haiku = MemoJev(prov.jev), MemoHaiku(prov.haiku)
    try:
        prov.jev.ask({"text": "We moved last spring."}, {"q": {"type": "noul", "instructions": "Does the text mention a move?"}})
    except ProviderError as e:
        sys.exit(f"Jev preflight failed, nothing run: {e}")
    cache = load_cache() if mode != "noise" else {}
    new_cache: list[dict] = []
    system = claims.CLAIMS_CONTRACT + "\nReply with a JSON list of claims only; [] if the turn states none."

    def client_claims(text: str, prev: list[str], date: str) -> list:
        key = hashlib.sha256(json.dumps([system, text, prev, date]).encode()).hexdigest()[:24]
        with _lock:
            hit = cache.get(key)
        if hit is None:
            seen = {"speaker": "user", "role": "user", "channel": "chat", "session_date": date,
                    "previous_turns": prev, "known_slots": [], "turn": text[:claims.MAX_TURN_CHARS]}
            try:
                raw = prov.haiku.complete(system, json.dumps(seen, ensure_ascii=False), 2000).text
            except Exception:        # CapReached or ProviderError: this turn goes without claims (not cached)
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
        return out[:25]            # the capture API's limit per turn (schemas.MAX_CLAIMS_PER_TURN)

    def judge(q, pred):
        votes = []
        for _ in range(6):
            raw = prov.haiku.complete("", lme_judge_prompt(q["question_type"], q["question"], str(q["answer"]), pred,
                                                           q["question_id"].endswith("_abs")), 10).text
            v = parse_yes_no(raw)
            if v is not None:
                votes.append(v)
            if len(votes) == 3:
                break
        return (sum(votes) >= 2) if len(votes) == 3 else None, votes

    # (name, ledger_route, memory_check, ledger_notes); brain is the shipped default (memory check off since
    # exp-memory-check-v1); the second arm minus the first is the measured difference
    arms = {"noise": (("flat_a", False, False, False), ("flat_b", False, False, False)),
            "brain": (("flat", False, False, False), ("brain", True, False, True)),
            "check": (("brain", True, False, True), ("withcheck", True, True, True)),
            "notes": (("nonotes", True, False, False), ("brain", True, False, True)),
            # exp-open-route-v1: the second arm differs only on questions the router calls open
            "open": (("brain", True, False, True), ("plain", True, False, True, "plain")),
            # exp-more-excerpts-v1: the turns next to each excerpt as context; the summary compares the first
            # two arms, the third (near1) is in the rows
            "near": (("brain", True, False, True), ("near2", True, False, True, "full", 2),
                     ("near1", True, False, True, "full", 1)),
            # exp-ledger-stable-v1: the second arm reads a second ingest made with per-slot placement
            "place": (("brain", True, False, True), ("perslot", True, False, True)),
            # exp-ledger-wording-v1: the shipped pack (near1) against the same pack with soft notes; no notes in rows
            "wording": (("near1", True, False, True, "full", 1), ("near1soft", True, False, True, "full", 1, "soft"),
                        ("near1off", True, False, True, "full", 1, "off")),
            # exp-relation-ends-v1: the shipped pack over two ingests, the second with the ends check
            "ends": (("near1", True, False, True, "full", 1), ("near1ends", True, False, True, "full", 1)),
            # exp-when-locomo-v1, guard: one ingest with dates stored; the second arm writes them beside the phrase
            "dates": (("near1", True, False, True, "full", 1), ("near1dates", True, False, True, "full", 1, None, True)),
            # exp-stable-pack-v3, guard: the shipped pack against the same pack with a steadier top ten
            "band": (("near1dates", True, False, True, "full", 1, None, True),
                     ("near1band", True, False, True, "full", 1, None, True, 0.1)),
            # exp-shortlist-v1, guard: the shipped pack against the same pack with seats in the shortlist
            "seats": (("near1dates", True, False, True, "full", 1, None, True),
                      ("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5))),
            # exp-count-block-v1: the shipped pack against the same pack without the count route's gathered list
            "count": (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                      ("near1nocount", True, False, True, "full", 1, None, True, 0.0, (25, 5), False)),
            # exp-reader-swap-v1: retrieval only against the shipped pack on one capture; the packs are kept so
            # that scripts/reader_swap.py can give them to another reader
            "reader": (("flat", False, False, False),
                       ("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5))),
            # exp-count-style-v1: the shipped pack against the same pack with the count block worded as evidence
            "countstyle": (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                           ("near1evidence", True, False, True, "full", 1, None, True, 0.0, (25, 5), True, "evidence")),
            # exp-date-direction-v1: the shipped pack over two ingests, the second with BRAIN_DATE_DIRECTION
            "datedir": (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                        ("near1datedir", True, False, True, "full", 1, None, True, 0.0, (25, 5))),
            # exp-relate-disputes-v1: the shipped pack over two ingests, the second with BRAIN_RELATE_DISPUTES
            "disputes": (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                         ("near1disputes", True, False, True, "full", 1, None, True, 0.0, (25, 5))),
            # exp-resolve-names-v1: the shipped pack over two ingests, the second with BRAIN_RESOLVE_NAMES
            "names": (("near1seats", True, False, True, "full", 1, None, True, 0.0, (25, 5)),
                      ("near1names", True, False, True, "full", 1, None, True, 0.0, (25, 5)))}[mode]

    def one(item):
        if mode == "ends":
            row = one_db(item, arms[:1], None, "", False)
            second = one_db(item, arms[1:], None, "b", True)
            row.update(near1ends=second["near1ends"], write_path_ends=second["write_path"],
                       ledger_ends=second["ledger"])
            return row
        if mode == "datedir":
            row = one_db(item, arms[:1], None, "", None, None, None, False)
            second = one_db(item, arms[1:], None, "b", None, None, None, True)
            row.update(near1datedir=second["near1datedir"], write_path_datedir=second["write_path"],
                       ledger_datedir=second["ledger"])
            return row
        if mode == "disputes":
            row = one_db(item, arms[:1], None, "", None, None, False)
            second = one_db(item, arms[1:], None, "b", None, None, True)
            row.update(near1disputes=second["near1disputes"], write_path_disputes=second["write_path"],
                       ledger_disputes=second["ledger"])
            return row
        if mode == "names":
            row = one_db(item, arms[:1], None, "", None, False)
            second = one_db(item, arms[1:], None, "b", None, True)
            row.update(near1names=second["near1names"], write_path_names=second["write_path"],
                       ledger_names=second["ledger"], entities_names=second["entities"])
            return row
        if mode != "place":
            return one_db(item, arms, None, "")
        row = one_db(item, arms[:1], False, "")
        second = one_db(item, arms[1:], True, "b")
        row.update(perslot=second["perslot"], write_path_perslot=second["write_path"], ledger_perslot=second["ledger"])
        return row

    def one_db(item, arms, per_slot, suffix, ends=None, names=None, disputes=None, datedir=None):
        i, q = item
        name = f"{DB}_{i:03d}{suffix}"
        pool = make_pool(fresh(name), max_size=2)
        token = ledger.PLACE_PER_SLOT_OVERRIDE.set(per_slot)     # this thread's ingest only
        token_ends = ledger.RELATE_ENDS_OVERRIDE.set(ends)
        token_names = ledger.RESOLVE_NAMES_OVERRIDE.set(names)
        token_disputes = ledger.RELATE_DISPUTES_OVERRIDE.set(disputes)
        token_datedir = dates.DATE_DIRECTION_OVERRIDE.set(datedir)
        try:
            with pool.connection() as c:
                owner = c.execute("select id from brain.account where name = 'owner'").fetchone()[0]
            caller = Caller("eval", ("*",), ("*",), None, True, owner, "owner", False, True)
            ns = "u"
            for sid, date_s, sess in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
                base = session_ts(date_s)
                turns, prev = [], []
                for k, t in enumerate(sess or []):
                    text = (t.get("content") or "").strip() if isinstance(t, dict) else ""
                    if not text:
                        continue
                    role = t["role"] if t["role"] in ("user", "assistant") else "user"
                    cl = client_claims(text, prev[-2:], date_s[:10]) if mode != "noise" and role == "user" else []
                    turns.append(TurnIn(role=role, text=text[:100_000], ts=base + timedelta(seconds=k), claims=cl))
                    prev.append(f"{role}: {text[:400]}")
                for b in range(0, len(turns), 100):
                    with tenant(pool, owner) as conn:
                        core.capture(conn, prov, caller, ns, turns[b:b + 100], sid, None,
                                     write_path_enabled=(mode != "noise"), dates_enabled=(mode in ("dates", "band", "seats", "names", "count", "countstyle", "reader", "disputes", "datedir")))
            wp = {}
            if mode != "noise":                      # the night after: the worker's brain steps
                wp = worker.write_path(pool, prov, max_episodes=100_000)
                worker.embed_backfill(pool, prov.embedder, 100_000)
                worker.consolidate(pool, prov)
                worker.promote_preferences(pool, prov)
                worker.render_pages(pool)
            row = {"qid": q["question_id"], "qtype": q["question_type"], "question": q["question"],
                   "gold": str(q["answer"]), "write_path": {k: v for k, v in wp.items() if k != "ledger"},
                   "ledger": wp.get("ledger")}
            with pool.connection() as c:        # who the ledger holds, and how many claims are still tentative
                row["entities"] = {"count": c.execute("select count(*) from brain.entity where merged_into is null").fetchone()[0],
                                   "tentative_claims": c.execute("select count(*) from brain.claim where tentative").fetchone()[0]}
            for arm, routed, check, notes, *open_mode in arms:
                with tenant(pool, owner) as conn:      # the context once; the reader as core.answer calls it
                    ctx = core.context(conn, prov, caller, ns, q["question"], q["question_date"], "auto", None,
                                       when_route=False, ledger_route=routed, memory_check=check,
                                       ledger_notes=notes, open_route=open_mode[0] if open_mode else "full",
                                       neighbours=open_mode[1] if len(open_mode) > 1 else 0,
                                       note_style=open_mode[2] if len(open_mode) > 2 else None,
                                       date_notes=open_mode[3] if len(open_mode) > 3 else False,
                                       rerank_band=open_mode[4] if len(open_mode) > 4 else 0.0,
                                       seats=open_mode[5] if len(open_mode) > 5 else (0, 0),
                                       count_block=open_mode[6] if len(open_mode) > 6 else True,
                                       count_style=open_mode[7] if len(open_mode) > 7 else None)
                    sessions = sessions_of(conn, [it["chunk_id"] for it in ctx["excerpts"]])
                system = core.ANSWER_PREFERENCE if ctx["style"] == "preference" else core.ANSWER_DEFAULT
                with _lock:        # the pack itself, by its hash, for another reader to read (git-ignored)
                    PACKS_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
                    with PACKS_PARTIAL.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"sha": hashlib.sha256(ctx["prompt"].encode()).hexdigest()[:12],
                                            "prompt": ctx["prompt"]}) + "\n")
                gold_sessions = [str(s) for s in q.get("answer_session_ids") or []]
                row[arm] = {**majority_read(
                                lambda: prov.haiku.complete(system, ctx["prompt"], core.ANSWER_MAX_TOKENS).text,
                                lambda pred: judge(q, pred), reads),
                            "route": ctx["route"], "ledger": ctx["ledger"], "sessions": sessions, "nearby": len(ctx["nearby"]),
                            "style": ctx["style"],
                            "context_sha": hashlib.sha256(ctx["prompt"].encode()).hexdigest()[:12],
                            "excerpts": [it["chunk_id"] for it in ctx["excerpts"]],
                            "evidence_sessions_reached": sorted(set(sessions) & set(gold_sessions))}
                row["evidence_sessions"] = gold_sessions
            if unrouted([row], [a[0] for a in arms if a[1]]):
                raise RuntimeError("Jev did not answer (the router was skipped): the run is void, stopping")
            if mode != "noise":    # keep the brain's own decisions: the booster's training data (the booster plan 5.2)
                with pool.connection() as c:
                    decs = c.execute("""select stage, decider, model_version, input, output, probability, threshold,
                                               outcome from brain.decision_log
                                        where stage like 'write.%' or stage like 'nightly.%' or stage like 'recall.%'
                                        order by id""").fetchall()
                with _lock:
                    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
                    with DECISIONS.open("a", encoding="utf-8") as f:
                        f.write("".join(json.dumps({"qid": q["question_id"], "stage": s, "decider": dc, "model": m,
                                                    "input": i, "output": o, "p": p, "threshold": th, "outcome": oc},
                                                   default=str) + "\n" for s, dc, m, i, o, p, th, oc in decs))
            return row
        finally:
            dates.DATE_DIRECTION_OVERRIDE.reset(token_datedir)
            ledger.RELATE_DISPUTES_OVERRIDE.reset(token_disputes)
            ledger.RESOLVE_NAMES_OVERRIDE.reset(token_names)
            ledger.RELATE_ENDS_OVERRIDE.reset(token_ends)
            ledger.PLACE_PER_SLOT_OVERRIDE.reset(token)
            pool.close()
            if os.environ.get("BRAIN_EVAL_KEEP") != "1":      # keep them for scripts/page_audit.py, then drop there
                with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
                    c.execute(f"drop database if exists {name} with (force)")

    # a stopped run keeps its finished questions; a restart with the same arguments and code resumes
    from locomo_eval import partial_add, partial_load
    files = [SERVICE / "brain" / f for f in ("core.py", "ledger.py", "claims.py", "preferences.py", "router.py")]
    code = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files + [Path(__file__).resolve()]}
    # a stopped start is resumed only by the same code: every module of the brain, not only the five reported
    # above (2026-11-13: a run resumed 48 questions answered under an older brain/dates.py)
    code_all = hashlib.sha256(b"".join(f.read_bytes() for f in sorted((SERVICE / "brain").glob("*.py")))).hexdigest()[:16]
    run_key = hashlib.sha256(json.dumps([mode, split, n, skip, types, reads, memo, code, sorted(only or []), code_all]).encode()).hexdigest()[:12]
    partial = CACHE.parent / f"partial_brain_eval_{run_key}.jsonl"
    PACKS_PARTIAL = CACHE.parent / f"partial_packs_brain_eval_{run_key}.jsonl"    # survives a stopped start too
    start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    done, spent_before = partial_load(partial)
    if done:
        print(f"resuming: {len(done)} questions already saved by a stopped start (${spent_before} spent then)",
              flush=True)

    def one_saved(item):
        qid = item[1]["question_id"]
        if qid in done:
            return done[qid]
        row = one(item)
        with _lock:
            partial_add(partial, start, qid, row, meter.spent_today())
        return row

    rows = []
    with cf.ThreadPoolExecutor(4) as ex:
        for k, row in enumerate(ex.map(one_saved, enumerate(qs)), 1):
            rows.append(row)
            if k % 10 == 0:
                print(f"{k}/{len(qs)} done, spent ${meter.spent_today():.3f}", flush=True)
                if new_cache:
                    with _lock:
                        CACHE.parent.mkdir(parents=True, exist_ok=True)
                        with CACHE.open("a", encoding="utf-8") as f:
                            f.write("".join(json.dumps(r) + "\n" for r in new_cache))
                        new_cache.clear()
    if new_cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in new_cache))

    a1, a2 = arms[0][0], arms[1][0]

    def acc(rs, arm):
        v = [r[arm]["correct"] for r in rs if r[arm]["correct"] is not None]
        return round(sum(v) / len(v), 4) if v else None
    summary = {"mode": mode, "split": split, "n": len(rows), "arms": [a1, a2], "reads": reads,
               "read_splits": read_splits(rows, (a1, a2)),
               "jev_memo": {"hits": prov.jev.hits, "misses": prov.jev.misses} if memo else None,
               "arm_labels": {a: ARM_LABELS[a] for a in (a1, a2)}, "by_type": {}}
    for t in types + ["all"]:
        rs = [r for r in rows if t in ("all", "abstention") or r["qtype"] == t]
        if not rs:
            continue
        diffs = arm_diffs(rs, a1, a2)
        if not diffs:
            continue
        rnd = random.Random(7)
        boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
        summary["by_type"][t] = {"n": len(rs), "judged_in_both": len(diffs), a1: acc(rs, a1), a2: acc(rs, a2),
                                 "diff": round(sum(diffs) / len(diffs), 4), "ci95": [boots[50], boots[1949]],
                                 "flips": {f"{a2}_only": diffs.count(1), f"{a1}_only": diffs.count(-1)},
                                 "flip_rate": round(sum(d != 0 for d in diffs) / len(diffs), 4)}
    summary["spend_usd"] = round(meter.spent_today() + spent_before, 4)
    if done:
        summary["resumed"] = {"questions_from_stopped_starts": len(done), "their_spend_usd": spent_before}
    out = SERVICE / "runs" / f"brain_eval_{mode}_{split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"hashes": code,
                                                 "mode": mode, "split": split, "n_per_type": n, "skip": skip, "reads": reads,
                                                 "types": types,
                                                 "cap": float(cap) if cap else CAPS[mode]}, indent=2))
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                                    encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if PACKS_PARTIAL.exists():
        kept = {}
        for line in PACKS_PARTIAL.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            kept[d["sha"]] = d["prompt"]
        (CACHE.parent / f"packs_{out.name}.json").write_text(json.dumps(kept), encoding="utf-8")
        PACKS_PARTIAL.unlink()
    print(json.dumps(summary, indent=2))
    print("run dir:", out)
    partial.unlink(missing_ok=True)            # the run is whole now; its rows are in the run dir
    prov.close()
    meter_pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
