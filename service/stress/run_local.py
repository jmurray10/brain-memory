"""System stress, local (phase C): hostile input, races, isolation fuzz, failure drills.

Runs a real server (python -m brain.serve) on a fresh database in the local Docker Postgres, offline stubs by
default (no spend), and checks the service's behavior from the outside. Prints PASS/FAIL per check and writes
runs/stress_local_<stamp>/report.json.

    python stress/run_local.py                     # all groups
    python stress/run_local.py input races         # some groups: input, races, isolation, drills
"""
from __future__ import annotations

import os as _os

_os.environ.setdefault("BRAIN_LOG_SOURCE", "benchmark")   # decision_log.source: never booster training data

import concurrent.futures as cf
import json
import os
import random
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE / "scripts"))
from testdb import admin_url, db_url  # noqa: E402
ADMIN_DB = admin_url()
DB_NAME = "brain_stress"
DB_URL = db_url(DB_NAME)
PORT = 8768
BASE = f"http://127.0.0.1:{PORT}"
ADMIN = "stress-admin-" + secrets.token_hex(16)
RESULTS: list[dict] = []


def check(group: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"group": group, "check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {group}/{name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


class Server:
    def __init__(self, **env):
        self.env = {**os.environ, "DATABASE_URL": DB_URL, "BRAIN_OFFLINE": "1", "BRAIN_ADMIN_KEY": ADMIN,
                    "PORT": str(PORT), "HOST": "127.0.0.1", "BRAIN_RATE_PER_MIN": "1000000",
                    "BRAIN_LOG_LEVEL": "WARNING", **env}
        self.proc = None

    def __enter__(self):
        self.log = open(RUN_DIR / f"server_{len(list(RUN_DIR.glob('server_*.log')))}.log", "w")
        self.proc = subprocess.Popen([sys.executable, "-m", "brain.serve"], cwd=SERVICE, env=self.env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        wait_healthy()
        return self

    def __exit__(self, *_):
        self.proc.terminate()
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def wait_healthy(seconds: int = 60) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            if httpx.get(BASE + "/healthz", timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def fresh_db() -> None:
    with psycopg.connect(ADMIN_DB, autocommit=True, prepare_threshold=None) as c:
        c.execute(f'drop database if exists "{DB_NAME}" with (force)')
        c.execute(f'create database "{DB_NAME}"')


def client(key: str = ADMIN) -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=60, headers={"Authorization": f"Bearer {key}"})


def new_account(name: str) -> str:
    with client() as h:
        r = h.post("/v1/admin/accounts", json={"name": name})
        r.raise_for_status()
        return r.json()["key"]


def cap(h, text, ns=None, **kw):
    turn = {"role": "user", "text": text, **{k: v for k, v in kw.items() if k != "session_id"}}
    body = {"turns": [turn]}
    if ns:
        body["namespace"] = ns
    if "session_id" in kw:
        body["session_id"] = kw["session_id"]
    return h.post("/v1/capture", json=body)


def db(sql, *args):
    with psycopg.connect(DB_URL) as c:
        return c.execute(sql, args).fetchall()


# ---------------------------------------------------------------- 1. hostile input

def group_input():
    g = "input"
    key = new_account("inp")
    with client(key) as h:
        for field, body in {
            "text": {"turns": [{"role": "user", "text": "before\u0000after"}]},
            "speaker": {"turns": [{"role": "user", "text": "x", "speaker": "a\u0000b"}]},
            "session_id": {"session_id": "s\u0000", "turns": [{"role": "user", "text": "x"}]},
            "idempotency_key": {"turns": [{"role": "user", "text": "x", "idempotency_key": "k\u0000"}]},
            "meta": {"turns": [{"role": "user", "text": "x", "meta": {"k": "v\u0000"}}]},
        }.items():
            r = h.post("/v1/capture", json=body)
            check(g, f"null byte in {field} is a 4xx, not a crash", 400 <= r.status_code < 500, f"HTTP {r.status_code}")
        r = h.post("/v1/recall", json={"query": "find\u0000me"})
        check(g, "null byte in recall query is not a crash", r.status_code < 500, f"HTTP {r.status_code}")

        raw = b'{"turns": [{"role": "user", "text": "lone \\ud800 surrogate"}]}'
        r = h.post("/v1/capture", content=raw, headers={"content-type": "application/json"})
        check(g, "lone surrogate is a 4xx, not a crash", 400 <= r.status_code < 500, f"HTTP {r.status_code}")
        r = h.post("/v1/capture", content=b'{"turns": [{"role": "user", "text": "\xff\xfe bad"}]}',
                   headers={"content-type": "application/json"})
        check(g, "invalid UTF-8 body is a 4xx", 400 <= r.status_code < 500, f"HTTP {r.status_code}")
        deep = '{"a":' * 1000 + "1" + "}" * 1000
        raw = ('{"turns": [{"role": "user", "text": "deep", "meta": ' + deep + "}]}").encode()
        r = h.post("/v1/capture", content=raw, headers={"content-type": "application/json"})
        check(g, "meta nested 1000 deep is not a crash", r.status_code < 500, f"HTTP {r.status_code}")

        r = cap(h, "x" * 100_000)
        check(g, "100,000-character turn accepted", r.status_code == 200, f"HTTP {r.status_code}")
        r = cap(h, "x" * 100_001)
        check(g, "100,001-character turn refused", r.status_code == 422, f"HTTP {r.status_code}")
        r = h.post("/v1/capture", json={"turns": [{"role": "user", "text": "y" * 600_000},
                                                  {"role": "user", "text": "z" * 600_000}]})
        check(g, "body over 1 MB refused with 413", r.status_code == 413, f"HTTP {r.status_code}")
        r = h.post("/v1/capture", json={"turns": [{"role": "user", "text": "t"}] * 101})
        check(g, "101 turns refused", r.status_code == 422, f"HTTP {r.status_code}")

        texts = ["张三的论文引用了定理 3.2",           # Chinese
                 "הפסק דין מ-2021 בוטל",  # Hebrew (RTL)
                 "Eigenvalue λ₁ = 2√3, ∀x∈ℝ: f(x) ≥ 0",            # math symbols
                 "Café vs Café; zero​width; tab\there",                               # combining, ZW
                 "'; drop table brain.episode; -- 100% _like_ \\ backslash"]                          # SQL-ish
        sid = "unicode"
        ok = all(cap(h, t, session_id=sid).status_code == 200 for t in texts)
        exported = [json.loads(ln)["text"] for ln in h.get("/v1/export?format=jsonl").text.splitlines()
                    if json.loads(ln).get("session_id") == sid]
        check(g, "mixed scripts, math symbols and SQL-like text round-trip exactly", ok and exported == texts,
              f"{len(exported)} of {len(texts)} back")
        r = h.post("/v1/recall", json={"query": "'; drop table brain.episode; --"})
        check(g, "SQL-like query is just a query", r.status_code == 200 and db("select count(*) from brain.episode")[0][0] > 0)
        for q in ("the of and", "???!!!", "x" * 10_000):
            r = h.post("/v1/recall", json={"query": q, "k": 50})
            check(g, f"odd query {q[:12]!r}... works", r.status_code == 200, f"HTTP {r.status_code}")
        for ts in ("0001-01-01T00:00:00Z", "9999-12-31T23:59:59Z", "2026-11-08T12:00:00+14:00", "2026-11-08T12:00:00"):
            r = cap(h, f"time edge {ts}", ts=ts)
            check(g, f"timestamp {ts} accepted", r.status_code == 200, f"HTTP {r.status_code}")
        r = h.post("/v1/recall", json={"query": "time edge", "time_from": "0001-01-01T00:00:00Z",
                                       "time_to": "9999-12-31T23:59:59Z"})
        check(g, "recall over the whole time range works", r.status_code == 200 and len(r.json()["items"]) >= 4)
        for ns, want in (("a" * 63, 200), ("a" * 64, 400), ("Upper", 400), ("-dash", 400)):
            r = cap(h, "ns edge", ns=ns)
            check(g, f"namespace {ns[:10]!r} -> {want}", r.status_code == want, f"HTTP {r.status_code}")


# ---------------------------------------------------------------- 2. races

def group_races():
    g = "races"
    key = new_account("race")
    with client(key) as h, cf.ThreadPoolExecutor(32) as pool:
        rs = list(pool.map(lambda _: cap(h, "Same turn sent fifty times.", idempotency_key="dup-1"), range(50)))
        codes = {r.status_code for r in rs}
        n = db("select count(*) from brain.episode where idempotency_key = 'dup-1'")[0][0]
        chunks = db("""select count(*) from brain.chunk k join brain.episode e on e.id = k.ref_id
                       where e.idempotency_key = 'dup-1'""")[0][0]
        check(g, "50 concurrent captures with one idempotency key -> one episode, one chunk",
              codes == {200} and n == 1 and chunks == 1, f"codes {codes}, episodes {n}, chunks {chunks}")

        rs = list(pool.map(lambda i: cap(h, f"writer line {i}", ns="race", session_id=f"w{i % 20}"), range(400)))
        n = db("select count(*) from brain.episode where namespace = 'race' and body like 'writer line%%'")[0][0]
        check(g, "20 sessions x 400 concurrent captures all land", all(r.status_code == 200 for r in rs) and n == 400,
              f"episodes {n}")

        def writer(i):
            return cap(h, f"racing line {i}", ns="race", session_id="victim")
        futs = [pool.submit(writer, i) for i in range(100)]
        forgets = [pool.submit(h.post, "/v1/forget", json={"namespace": "race", "session_id": "victim"})
                   for _ in range(5)]
        cf.wait(futs + forgets)
        h.post("/v1/forget", json={"namespace": "race", "session_id": "victim"})
        orphan = db("""select count(*) from brain.chunk k join brain.episode e on e.id = k.ref_id
                       where e.deleted_at is not null""")[0][0]
        live = db("select count(*) from brain.episode where session_id = 'victim' and deleted_at is null")[0][0]
        exported = sum("racing line" in ln for ln in h.get("/v1/export?namespace=race&format=jsonl").text.splitlines())
        check(g, "forget racing capture leaves no chunk of a forgotten episode",
              all(f.result().status_code == 200 for f in forgets) and orphan == 0 and live == 0 and exported == 0,
              f"orphan chunks {orphan}, live {live}, exported {exported}")

        eid = cap(h, "restore me twice", ns="race").json()["episodes"][0]["id"]
        h.post("/v1/forget", json={"namespace": "race", "episode_ids": [eid]})
        rs = list(pool.map(lambda _: h.post("/v1/admin/restore", json={"namespace": "race", "episode_ids": [eid]}),
                           range(10)))
        restored = sum(len(r.json().get("restored", [])) for r in rs if r.status_code == 200)
        chunks = db("select count(*) from brain.chunk where ref_id = %s", eid)[0][0]
        check(g, "10 concurrent restores of one episode restore it once", restored == 1 and chunks == 1,
              f"restored {restored}, chunks {chunks}, codes {sorted({r.status_code for r in rs})}")

        eid = cap(h, "flip flop", ns="race").json()["episodes"][0]["id"]
        ops = [lambda: h.post("/v1/forget", json={"namespace": "race", "episode_ids": [eid]}),
               lambda: h.post("/v1/admin/restore", json={"namespace": "race", "episode_ids": [eid]})] * 20
        random.Random(5).shuffle(ops)
        rs = [f.result() for f in [pool.submit(op) for op in ops]]
        deleted, chunks = db("""select e.deleted_at is not null, (select count(*) from brain.chunk k
                                 where k.ref_id = e.id) from brain.episode e where e.id = %s""", eid)[0]
        consistent = (deleted and chunks == 0) or (not deleted and chunks == 1)
        check(g, "40 interleaved forgets and restores end consistent",
              consistent and all(r.status_code < 500 for r in rs),
              f"deleted {deleted}, chunks {chunks}, 5xx {sum(r.status_code >= 500 for r in rs)}")


# ---------------------------------------------------------------- 3. isolation fuzz

def group_isolation(n_accounts: int = 30, ops: int = 1500):
    g = "isolation"
    accts = {f"iso{i:02d}": new_account(f"iso{i:02d}") for i in range(n_accounts)}
    marker = {a: f"MARK{a.upper()}" for a in accts}
    rnd = random.Random(42)
    leaks, errors, seen_ids = [], [], {a: [] for a in accts}

    def one(i):
        a = rnd.choice(list(accts))
        with client(accts[a]) as h:
            op = rnd.choice(["capture"] * 4 + ["recall"] * 4 + ["export", "directive", "session", "forget_other",
                                                                  "answer"])
            if op == "capture":
                r = cap(h, f"{marker[a]} note {i} shared words lab protocol theorem", ns="shared", session_id="s1")
                if r.status_code == 200:
                    seen_ids[a].extend(e["id"] for e in r.json()["episodes"])
            elif op == "recall":
                r = h.post("/v1/recall", json={"namespace": "shared", "query": "shared words lab protocol theorem",
                                               "k": 50})
            elif op == "answer":
                r = h.post("/v1/answer", json={"namespace": "shared", "question": "what shared words?"})
            elif op == "export":
                r = h.get("/v1/export?namespace=shared&format=jsonl")
            elif op == "directive":
                r = h.post("/v1/directives", json={"namespace": "shared", "text": f"{marker[a]} rule {i}"})
            elif op == "session":
                r = h.get("/v1/session_start?namespace=shared")
            else:                    # try to forget another account's episode ids
                other = rnd.choice([b for b in accts if b != a])
                ids = seen_ids[other][-5:]
                r = h.post("/v1/forget", json={"namespace": "shared", "episode_ids": ids or [1]})
                if r.status_code == 200 and r.json()["forgotten"]["episodes"]:
                    leaks.append((a, "forgot another account's episode"))
            if r.status_code >= 500:
                errors.append((op, r.status_code))
            body = r.text
            foreign = [m for b, m in marker.items() if b != a and m in body]
            if foreign:
                leaks.append((a, op, foreign[:3]))

    with cf.ThreadPoolExecutor(24) as pool:
        list(pool.map(one, range(ops)))
    check(g, f"{n_accounts} accounts x {ops} random concurrent ops: no foreign marker in any reply",
          not leaks, f"leaks {leaks[:3]}")
    check(g, "no 5xx during the fuzz", not errors, f"{errors[:5]}")
    per = db("""select a.name, count(*) from brain.episode e join brain.account a on a.id = e.account_id
                where a.name like 'iso%%' group by 1""")
    wrong = db("""select count(*) from brain.episode e join brain.account a on a.id = e.account_id
                  where a.name like 'iso%%' and position('MARK' || upper(a.name) in e.body) = 0""")[0][0]
    check(g, "every stored episode carries its own account's marker", wrong == 0, f"{len(per)} accounts, {wrong} wrong")


# ---------------------------------------------------------------- 4. failure drills

def group_drills():
    g = "drills"
    key = new_account("drill")
    # Database restart in the middle of traffic: errors are fine while it is down; it must recover by itself.
    stop = time.time() + 25
    codes: list[int] = []

    def traffic():
        with client(key) as h:
            while time.time() < stop:
                try:
                    codes.append(h.post("/v1/recall", json={"query": "anything"}, timeout=15).status_code)
                except httpx.HTTPError:
                    codes.append(0)
                time.sleep(0.05)
    with cf.ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(traffic) for _ in range(4)]
        time.sleep(4)
        subprocess.run(["docker", "restart", "service-db-1"], capture_output=True, timeout=120)
        cf.wait(futs)
    after = []
    with client(key) as h:
        for _ in range(10):
            after.append(h.post("/v1/recall", json={"query": "anything"}).status_code)
            time.sleep(0.3)
    check(g, "after a database restart the API recovers without a restart of its own",
          after[-5:] == [200] * 5, f"last 10 after: {after}; during: {sorted(set(codes))}")
    check(g, "first requests after the restart succeed (no stale pooled connections)", after[:3] == [200] * 3,
          f"{after[:3]}")
    hz = httpx.get(BASE + "/healthz", timeout=10)
    check(g, "healthz ok after the restart", hz.status_code == 200)


def group_degraded():
    """No provider keys, real (not stub) clients: every provider call fails, the API must degrade, not crash."""
    g = "degraded"
    with Server(BRAIN_OFFLINE="0", VOYAGE_API_KEY="", TYPESAFE_API_KEY="", ANTHROPIC_API_KEY="",
                VOYAGE_API_KEY_NEW="", JEV_API_KEY="", 
                ):
        key = new_account("degr")
        with client(key) as h:
            r = cap(h, "No vectors today: the spectrometer calibration drifted 0.3 nm.")
            check(g, "capture without Voyage degrades to no_embed", r.status_code == 200 and
                  r.json()["degraded"] == "no_embed", f"HTTP {r.status_code}")
            r = h.post("/v1/recall", json={"query": "spectrometer calibration drift"})
            j = r.json() if r.status_code == 200 else {}
            check(g, "recall without Voyage or Jev still finds the turn (keyword only, unscreened)",
                  r.status_code == 200 and j["dense"] is False and j["degraded"] == "no_rerank"
                  and any("spectrometer" in it["text"] for it in j["items"]), f"HTTP {r.status_code}")
            r = h.post("/v1/answer", json={"question": "What drifted?"})
            check(g, "answer without Haiku is a clean 503", r.status_code == 503, f"HTTP {r.status_code}")


def run_group(name: str) -> None:
    print(f"--- {name}", flush=True)
    try:
        globals()[f"group_{name}"]()
    except Exception as e:    # a crashed group is a failure, not the end of the run
        check(name, "group ran to the end", False, f"{type(e).__name__}: {str(e)[:200]}")


def main() -> int:
    groups = sys.argv[1:] or ["input", "races", "isolation", "drills", "degraded"]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fresh_db()
    with Server():
        for name in groups:
            if name != "degraded":
                run_group(name)
    if "degraded" in groups:
        run_group("degraded")
    (RUN_DIR / "report.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    failed = [r for r in RESULTS if not r["pass"]]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed; report {RUN_DIR / 'report.json'}")
    return 1 if failed else 0


RUN_DIR = SERVICE / "runs" / f"stress_local_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

if __name__ == "__main__":
    sys.exit(main())
