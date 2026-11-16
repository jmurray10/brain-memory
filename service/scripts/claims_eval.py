"""Build step 4 gate (the experiment log exp-claims-v1): claim extraction on real turns, audited by Claude Opus 5.5.

    python scripts/claims_eval.py dev      # 20 turns per source (one prompt-tuning round allowed)
    python scripts/claims_eval.py test     # 60 turns per source (the gate)

Sources: LongMemEval S (user turns), LoCoMo (both speakers; speaker_a is the user), the six complex-domain worlds
(user and attachment turns). Each sampled turn is captured with its two previous turns as one session, then the
real write path (brain/claims.py) runs on it in-process against the Supabase test branch. Dev and test are drawn
from one pool per source (dev first, seed 11; test from the rest, seed 23) and exclude the spike's turns.

Audit: Opus 5.5 (effort low, structured output, blind to verify scores) judges up to 200 kept claims and up to 50
verify-dropped claims. Writes runs/claims_<split>_<stamp>/ with rows, summary and review.md (every claim judged
not entailed plus 20 judged entailed, for the manual check). Spend caps: $2 for the pipeline, $3 for the audit.
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parent / "live-eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from testdb import admin_url, db_url  # noqa: E402

DB = "brain_claims"
PER_SOURCE = {"dev": 20, "test": 60, "test2": 60}
SEEDS = {"dev": 11, "test": 23, "test2": 31}
PIPELINE_CAP, AUDIT_CAP = 2.0, 3.0
AUDIT_KEPT, AUDIT_DROPPED, REVIEW_ENTAILED = 200, 50, 20
OPUS = "claude-opus-5-5"
OPUS_IN, OPUS_OUT = 4.0, 20.0          # USD per million tokens (claude-api models reference, Opus 5.5)
SPIKE = json.loads((SERVICE / "evals/claims_spike_turns.json").read_text(encoding="utf-8"))["turns"]


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _spike(text: str) -> bool:
    return any(text.startswith(p) or p.startswith(text[:80]) for p in SPIKE)


def pool_lme(rnd: random.Random, n: int) -> list[dict]:
    rows = json.loads((SERVICE.parent / "live-eval/data/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    out, seen = [], set()
    while len(out) < n:
        r = rnd.choice(rows)
        si = rnd.randrange(len(r["haystack_sessions"]))
        sess = r["haystack_sessions"][si]
        users = [i for i, t in enumerate(sess) if t.get("role") == "user" and (t.get("content") or "").strip()]
        if not users:
            continue
        ti = rnd.choice(users)
        text = sess[ti]["content"]
        if _key(text) in seen or _spike(text):
            continue
        seen.add(_key(text))
        prev = [{"role": t["role"], "speaker": "user" if t["role"] == "user" else "assistant", "text": t["content"]}
                for t in sess[max(0, ti - 2):ti] if (t.get("content") or "").strip()]
        out.append({"src": "lme", "date": r["haystack_dates"][si][:10].replace("/", "-"), "previous": prev,
                    "turn": {"role": "user", "speaker": "user", "channel": "chat", "text": text}})
    return out


def pool_locomo(rnd: random.Random, n: int) -> list[dict]:
    import re
    convs = json.loads((SERVICE.parents[1] / "locomo10.json").read_text(encoding="utf-8"))
    out, seen = [], set()
    while len(out) < n:
        c = rnd.choice(convs)["conversation"]
        s = rnd.choice([k for k in c if re.fullmatch(r"session_\d+", k) and c[k]])
        sess = c[s]
        ti = rnd.randrange(len(sess))
        text = sess[ti]["text"]
        if _key(text) in seen or _spike(text) or not text.strip():
            continue
        seen.add(_key(text))
        d = datetime.strptime(c[f"{s}_date_time"].split(" on ")[-1].strip(), "%d %B, %Y").date().isoformat()
        prev = [{"role": "user", "speaker": t["speaker"], "text": t["text"]} for t in sess[max(0, ti - 2):ti]]
        out.append({"src": "locomo", "date": d, "previous": prev,
                    "turn": {"role": "user", "speaker": sess[ti]["speaker"], "channel": "chat", "text": text}})
    return out


def pool_worlds(rnd: random.Random, n: int) -> list[dict]:
    worlds = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((SERVICE / "evals/worlds").glob("*.json"))]
    out, seen = [], set()
    while len(out) < n:
        w = rnd.choice(worlds)
        s = rnd.choice(w["sessions"])
        idx = [i for i, t in enumerate(s["turns"]) if t["role"] in ("user", "attachment")]
        if not idx:
            continue
        ti = rnd.choice(idx)
        t = s["turns"][ti]
        if _key(t["text"]) in seen or _spike(t["text"]):
            continue
        seen.add(_key(t["text"]))
        prev = [{"role": x["role"], "speaker": x.get("speaker") or x["role"], "text": x["text"],
                 "channel": x.get("channel", "chat")} for x in s["turns"][max(0, ti - 2):ti]]
        out.append({"src": f"world:{w['world']}", "date": s["date"], "previous": prev,
                    "turn": {"role": t["role"], "speaker": t.get("speaker") or t["role"],
                             "channel": t.get("channel", "chat"), "text": t["text"]}})
    return out


def sample(split: str) -> list[dict]:
    """dev and test come from one seed-7 pool per source. test2 (exp-claims-v2) is fresh: a seed-8 pool with every
    dev, test and spike turn removed, then 60 per source with seed 31."""
    out = []
    for name, fn in (("lme", pool_lme), ("locomo", pool_locomo), ("worlds", pool_worlds)):
        pool = fn(random.Random(7), PER_SOURCE["dev"] + PER_SOURCE["test"] + 40)
        dev = random.Random(SEEDS["dev"]).sample(pool, PER_SOURCE["dev"])
        rest = [p for p in pool if p not in dev]
        test = random.Random(SEEDS["test"]).sample(rest, PER_SOURCE["test"])
        if split == "test2":
            used = {_key(p["turn"]["text"]) for p in dev + test}
            fresh = [p for p in fn(random.Random(8), 400) if _key(p["turn"]["text"]) not in used]
            out += random.Random(SEEDS["test2"]).sample(fresh, PER_SOURCE["test2"])
        else:
            out += dev if split == "dev" else test
    return out


AUDIT_SYSTEM = """You audit claims that a memory system extracted from one turn of a conversation, note or document.
Judge whether the claim is entailed: the turn (read with its speaker and previous turns, which only resolve references) states this claim.
A claim is entailed only if all of these hold:
- subject: "mention" refers to the right person or thing. Mentions are copied as written in the turn ("you", "she", "the lot"); a later step resolves them to named entities, so a mention as written is right when the words refer to the claim's true subject. An empty mention means the speaker of the turn ("I", "me", "my");
- slot: the slot name describes what the value is (slot names are free proposals; judge only whether they fit the value);
- value: the turn says this value; nothing is added, changed or made more specific than the turn says; units, negations and conditions kept;
- polarity: asserted, negated, planned, hypothetical, corrected or reported matches how the turn says it;
- quote: the quoted sentence is in the turn and it is the sentence that states the claim.
Ignore style and wording choices that keep the meaning. Be strict on meaning."""

AUDIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["entailed", "failure", "reason"],
                "properties": {
                    "entailed": {"type": "boolean"},
                    "failure": {"type": "string", "enum": ["none", "wrong_subject", "wrong_slot", "value_not_stated",
                                                           "value_changed_or_added", "value_incomplete",
                                                           "wrong_polarity", "quote_does_not_state_it"]},
                    "reason": {"type": "string"}}}


class Auditor:
    def __init__(self, key: str):
        self.http = httpx.Client(timeout=httpx.Timeout(120, connect=10))
        self.key, self.usd = key, 0.0

    def judge(self, item: dict, max_tokens: int = 4000) -> dict:
        if self.usd > AUDIT_CAP:
            return {"entailed": None, "failure": None, "reason": "audit cap reached"}
        body = {"model": OPUS, "max_tokens": max_tokens, "system": AUDIT_SYSTEM,
                "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": AUDIT_SCHEMA}},
                "messages": [{"role": "user", "content": json.dumps(item, ensure_ascii=False)}]}
        import time
        r = None
        for attempt in range(4):
            r = self.http.post("https://api.anthropic.com/v1/messages", json=body,
                               headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"})
            if r.status_code not in (429, 500, 502, 503, 529):
                break
            time.sleep(2 ** attempt * 2)
        if r is None or r.status_code != 200:
            return {"entailed": None, "failure": None,
                    "reason": f"HTTP {getattr(r, 'status_code', None)}: {getattr(r, 'text', '')[:200]}"}
        j = r.json()
        u = j.get("usage") or {}
        self.usd += u.get("input_tokens", 0) / 1e6 * OPUS_IN + u.get("output_tokens", 0) / 1e6 * OPUS_OUT
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        try:
            return json.loads(text)
        except ValueError:
            if j.get("stop_reason") == "max_tokens" and body["max_tokens"] < 16000:   # thinking used the room
                return self.judge(item, max_tokens=16000)
            return {"entailed": None, "failure": None,
                    "reason": f"unparsed (stop_reason {j.get('stop_reason')}): {text[:200]}"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else ""
    client_name = "haiku"     # the yardstick: a low-end client, thinking off (the brain is model-agnostic)
    if split not in PER_SOURCE:
        sys.exit(__doc__)
    import env
    env.load_env()
    from brain import claims
    from brain.auth import Caller
    from brain.core import capture
    from brain.db import make_pool, migrate, tenant
    from brain.providers import ProviderError, make_providers
    from brain.schemas import ClaimIn, TurnIn
    from brain.settings import load
    from brain.spend import PgSpendStore, SpendMeter

    items = sample(split)
    print(f"{split}: {len(items)} turns", {s: sum(i['src'].split(':')[0] == s for i in items)
                                          for s in ('lme', 'locomo', 'world')}, flush=True)
    with psycopg.connect(admin_url(), autocommit=True, prepare_threshold=None) as c:
        c.execute(f"drop database if exists {DB} with (force)")
        c.execute(f"create database {DB}")
    url = db_url(DB)
    migrate(url)
    pool = make_pool(url, max_size=4)
    meter = SpendMeter(PgSpendStore(make_pool(url, max_size=2)), PIPELINE_CAP)
    st = load()
    prov = make_providers(st, meter)
    try:
        prov.jev.ask({"text": "We moved last spring."}, {"q": {"type": "noul", "instructions": "Does the text mention a move?"}})
    except ProviderError as e:
        sys.exit(f"Jev preflight failed, nothing run: {e}")
    with pool.connection() as c:
        owner = c.execute("select id from brain.account where name = 'owner'").fetchone()[0]
    caller = Caller("eval", ("*",), ("*",), None, True, owner, "owner", False, True)

    # The stand-in user model (Haiku 5.5, thinking off; outside the brain) reads each turn under the capture tool's claims
    # contract and sends what it extracts with the capture call, as a plugin client would. The brain never calls a model.
    system = claims.CLAIMS_CONTRACT + "\nReply with a JSON list of claims only; [] if the turn states none."

    def client_claims(it):
        t = it["turn"]
        seen = {"speaker": t["speaker"], "role": t["role"], "channel": t["channel"], "session_date": it["date"],
                "previous_turns": [f"{p['speaker']}: {p['text'][:400]}" for p in it["previous"]], "known_slots": [],
                "turn": t["text"][:claims.MAX_TURN_CHARS]}
        user = json.dumps(seen, ensure_ascii=False)
        try:
            raw = prov.haiku.complete(system, user, 2000).text
        except Exception as e:     # CapReached or ProviderError: this turn goes without claims
            print("client failed:", e, flush=True)
            return [], 1
        parsed, bad = claims.parse_claims(raw)
        ok, invalid = [], 0
        for c in parsed:
            try:
                ok.append(ClaimIn(**c))
            except ValueError:
                invalid += 1
        return ok, bad + invalid
    with cf.ThreadPoolExecutor(6) as ex:
        sent = list(ex.map(client_claims, items))
    client_stats = {"turns_with_claims": sum(bool(s[0]) for s in sent), "claims_sent": sum(len(s[0]) for s in sent),
                    "malformed_or_invalid": sum(s[1] for s in sent)}
    print("client:", client_stats, flush=True)

    # capture each turn (with its claims) after its previous turns, as one session; then the brain's write path
    targets: dict[int, dict] = {}
    for i, (it, (cl, _)) in enumerate(zip(items, sent)):
        base = datetime.fromisoformat(it["date"]).replace(hour=12, tzinfo=timezone.utc)
        turns = [TurnIn(role=p["role"], speaker=p["speaker"], channel=p.get("channel", "chat"), text=p["text"][:50_000],
                        ts=base + timedelta(minutes=k)) for k, p in enumerate(it["previous"])]
        t = it["turn"]
        turns.append(TurnIn(role=t["role"], speaker=t["speaker"], channel=t["channel"], text=t["text"][:50_000],
                            ts=base + timedelta(minutes=len(it["previous"])), claims=cl))
        ns = it["src"].split(":")[0]
        with tenant(pool, owner) as conn:
            out = capture(conn, prov, caller, ns, turns, f"s{i:04d}", write_path_enabled=True)
        targets[out["episodes"][-1]["id"]] = it
    ids = list(targets)
    totals = claims.Outcome()
    for b in range(0, len(ids), 20):
        with tenant(pool, owner) as conn:
            o = claims.process(conn, prov, ids[b:b + 20])
        for k, v in o.__dict__.items():
            setattr(totals, k, getattr(totals, k) + v)
        print(f"{min(b + 20, len(ids))}/{len(ids)} turns processed, pipeline spend ${meter.spent_today():.3f}", flush=True)

    with pool.connection() as c:
        cands = c.execute("""select id, episode_id, mention, slot, value, time_phrase, polarity, quote, status, drop_reason,
                                    p_hallucinated, p_off_target, p_incomplete from brain.claim_candidate order by id""").fetchall()
        kept_turns = {int(r[0]) for r in c.execute("select distinct episode_id from brain.claim_candidate")}
        spend_by = {r[0]: float(r[1]) for r in c.execute("select provider, usd from brain.spend")}
    cols = ["id", "episode_id", "mention", "slot", "value", "time_phrase", "polarity", "quote", "status", "drop_reason",
            "p_hallucinated", "p_off_target", "p_incomplete"]
    rows = [dict(zip(cols, r)) for r in cands]
    for r in rows:
        r["src"] = targets[r["episode_id"]]["src"]
    kept = [r for r in rows if r["status"] == "pending"]
    vdrop = [r for r in rows if (r["drop_reason"] or "").startswith("verify")]
    rnd = random.Random(5)
    audit = rnd.sample(kept, min(AUDIT_KEPT, len(kept))) + rnd.sample(vdrop, min(AUDIT_DROPPED, len(vdrop)))

    auditor = Auditor(st.anthropic_key)

    def one(r):
        it = targets[r["episode_id"]]
        item = {"speaker": it["turn"]["speaker"], "role": it["turn"]["role"], "session_date": it["date"],
                "previous_turns": [f"{p['speaker']}: {p['text'][:1500]}" for p in it["previous"]],
                "turn": it["turn"]["text"][:claims.MAX_TURN_CHARS],
                "claim": {k: r[k] for k in ("mention", "slot", "value", "time_phrase", "polarity", "quote")}}
        return {**r, "audit": auditor.judge(item)}
    with cf.ThreadPoolExecutor(6) as ex:
        judged = list(ex.map(one, audit))

    def rate(rs):
        v = [r["audit"]["entailed"] for r in rs if r["audit"]["entailed"] is not None]
        return {"n": len(v), "entailed": sum(v), "rate": round(sum(v) / len(v), 4) if v else None,
                "wilson95": wilson(sum(v), len(v))}
    jk = [r for r in judged if r["status"] == "pending"]
    jd = [r for r in judged if r["status"] != "pending"]
    by_src = {}
    for s in sorted({targets[i]["src"].split(":")[0] for i in targets}):
        src_ids = [i for i in targets if targets[i]["src"].split(":")[0] == s]
        by_src[s] = {"turns": len(src_ids), "kept_turns": sum(i in kept_turns for i in src_ids),
                     "kept_claims": sum(r["src"].split(":")[0] == s for r in kept),
                     "audit": rate([r for r in jk if r["src"].split(":")[0] == s])}
    failures = {}
    for r in jk:
        if r["audit"]["entailed"] is False:
            failures[r["audit"]["failure"]] = failures.get(r["audit"]["failure"], 0) + 1
    gate = rate(jk)
    summary = {"split": split, "client": client_stats, "brain": totals.as_dict(), "kept_claims": len(kept),
               "verify_dropped": len(vdrop),
               "claims_per_turn_with_claims": round(totals.received / max(1, client_stats["turns_with_claims"]), 2),
               "audit_kept": gate, "audit_failures": failures, "by_source": by_src,
               "verify_precision": rate(jd) | {"note": "rate here = share the auditor judged entailed; lower is better"},
               "gate": {"bar": 0.95, "pass": gate["rate"] is not None and gate["rate"] >= 0.95},
               "client_model": "claude-haiku-5-5",
               "spend_usd": {"client_haiku": round(spend_by.get("haiku", 0.0), 4),
                             "brain_jev_voyage": round(spend_by.get("jev", 0.0) + spend_by.get("voyage", 0.0), 4),
                             "audit": round(auditor.usd, 4)},
               "brain_usd_per_1000_turns": round((spend_by.get("jev", 0.0) + spend_by.get("voyage", 0.0))
                                                 / max(1, len(items)) * 1000, 3)}
    out = SERVICE / "runs" / f"claims_{split}_{client_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    files = [SERVICE / "brain/claims.py", Path(__file__).resolve()]
    (out / "config.json").write_text(json.dumps({"hashes": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files},
                                                 "auditor": OPUS, "split": split, "seeds": SEEDS}, indent=2))
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in judged),
                                    encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bad = [r for r in jk if r["audit"]["entailed"] is False]
    ok_rows = [r for r in jk if r["audit"]["entailed"]]
    good = random.Random(9).sample(ok_rows, min(REVIEW_ENTAILED, len(ok_rows)))
    lines = ["# Manual review: every kept claim judged not entailed, then 20 judged entailed", ""]
    for title, rs in (("Not entailed", bad), ("Entailed (sample)", good)):
        lines += [f"## {title} ({len(rs)})", ""]
        for r in rs:
            it = targets[r["episode_id"]]
            lines += [f"### claim {r['id']} [{r['src']}] speaker {it['turn']['speaker']}",
                      f"- turn: {it['turn']['text'][:600]!r}",
                      f"- claim: mention={r['mention']!r} slot={r['slot']} value={r['value']!r} polarity={r['polarity']} "
                      f"time={r['time_phrase']!r}", f"- quote: {r['quote']!r}",
                      f"- auditor: {r['audit']['failure']} - {r['audit']['reason']}", "- my verdict: ", ""]
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("run dir:", out)
    prov.close()
    pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
