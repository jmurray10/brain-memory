"""Provider price table and the daily spend meter (REFERENCE.md).

Prices ported from live-eval/prices.py, checked 2026-11-06 (USD per million tokens):
- Claude Haiku 5.5: 0.10 in / 0.50 out for prompts up to 100K tokens; 0.50 / 2.50 above.
- voyage-4-lite: 0.02 per million input tokens (free-tier credits ignored, so this over-counts).
- TypeSafe Jev: 0.042 per million input tokens, 0.00 output.

The meter enforces BRAIN_DAILY_CAP_USD per UTC day across all providers. Before a call it refuses when
today's recorded spend, plus the projected cost of calls still in flight in this process, plus this call's
projected cost would exceed the cap (CapReached). After the call it records the real usage. Spend lives in
the `spend` table so the cap survives restarts; MemorySpendStore is for offline tests and tools.
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Generator, Protocol

PROVIDERS = ("voyage", "jev", "haiku")

HAIKU = {"in": 0.10, "out": 0.50, "in_long": 0.50, "out_long": 2.50, "long_threshold": 100_000}
VOYAGE_PER_M = 0.02
JEV_IN_PER_M = 0.042
JEV_OUT_PER_M = 0.0


def est_tokens(s: str) -> int:
    """Same rough estimate the measured harness used (live-eval/text.py)."""
    return max(1, len(s) // 4)


def haiku_cost(in_tok: int, out_tok: int) -> float:
    if in_tok > HAIKU["long_threshold"]:
        return in_tok / 1e6 * HAIKU["in_long"] + out_tok / 1e6 * HAIKU["out_long"]
    return in_tok / 1e6 * HAIKU["in"] + out_tok / 1e6 * HAIKU["out"]


def voyage_cost(tokens: int) -> float:
    return tokens / 1e6 * VOYAGE_PER_M


def jev_cost(in_tok: int, out_tok: int = 0) -> float:
    return in_tok / 1e6 * JEV_IN_PER_M + out_tok / 1e6 * JEV_OUT_PER_M


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


class CapReached(RuntimeError):
    """The daily provider cap would be exceeded. Capture and recall degrade; answer returns 503."""


# The account the current work is for: db.tenant() sets it for its block, so every paid call made inside an
# account's transaction is recorded against that account and checked against the per-account cap. None: the
# operator's own calls and work done for every account at once (recorded as account 0, global cap only).
ACCOUNT: contextvars.ContextVar = contextvars.ContextVar("spend_account", default=None)


# ---------------------------------------------------------------- stores

class SpendStore(Protocol):
    def spent(self, day: date, account: int | None = None) -> float: ...       # None: every account together
    def add(self, day: date, provider: str, usd: float, tokens_in: int, tokens_out: int, calls: int,
            account: int | None = None) -> None: ...
    def by_account(self, day: date) -> dict[int, float]: ...
    def rows_since(self, day: date) -> list[dict]: ...


class PgSpendStore:
    """The `spend` table: one row per (UTC day, provider, account), incremented with an upsert. Account 0 is
    spend that belongs to no account (migration 015)."""

    def __init__(self, pool):
        self.pool = pool

    def spent(self, day: date, account: int | None = None) -> float:
        with self.pool.connection() as c:
            return float(c.execute("""select coalesce(sum(usd), 0) from brain.spend
                                      where day = %s and (%s::bigint is null or account_id = %s)""",
                                   (day, account, account)).fetchone()[0])

    def add(self, day: date, provider: str, usd: float, tokens_in: int, tokens_out: int, calls: int,
            account: int | None = None) -> None:
        with self.pool.connection() as c:
            c.execute(
                """insert into brain.spend as s (day, provider, account_id, usd, calls, tokens_in, tokens_out)
                   values (%s, %s, %s, %s, %s, %s, %s)
                   on conflict (day, provider, account_id) do update set
                       usd = s.usd + excluded.usd, calls = s.calls + excluded.calls,
                       tokens_in = s.tokens_in + excluded.tokens_in, tokens_out = s.tokens_out + excluded.tokens_out""",
                (day, provider, account or 0, usd, calls, tokens_in, tokens_out))

    def by_account(self, day: date) -> dict[int, float]:
        with self.pool.connection() as c:
            return {int(a): round(float(u), 6) for a, u in c.execute(
                "select account_id, sum(usd) from brain.spend where day = %s group by 1 order by 1", (day,)).fetchall()}

    def rows_since(self, day: date) -> list[dict]:
        with self.pool.connection() as c:
            rows = c.execute(
                """select day, provider, sum(usd), sum(calls), sum(tokens_in), sum(tokens_out) from brain.spend
                   where day >= %s group by day, provider order by day desc, provider""", (day,)).fetchall()
        return [{"day": r[0].isoformat(), "provider": r[1], "usd": round(float(r[2]), 6), "calls": r[3],
                 "tokens_in": r[4], "tokens_out": r[5]} for r in rows]


class MemorySpendStore:
    """In-process store for offline unit tests and tools that run without a database."""

    def __init__(self):
        self._rows: dict[tuple[date, str, int], dict] = {}
        self._lock = threading.Lock()

    def spent(self, day: date, account: int | None = None) -> float:
        with self._lock:
            return sum(r["usd"] for (d, _, a), r in self._rows.items() if d == day and account in (None, a))

    def add(self, day: date, provider: str, usd: float, tokens_in: int, tokens_out: int, calls: int,
            account: int | None = None) -> None:
        with self._lock:
            r = self._rows.setdefault((day, provider, account or 0),
                                      {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0})
            r["usd"] += usd
            r["calls"] += calls
            r["tokens_in"] += tokens_in
            r["tokens_out"] += tokens_out

    def by_account(self, day: date) -> dict[int, float]:
        with self._lock:
            out: dict[int, float] = {}
            for (d, _, a), r in self._rows.items():
                if d == day:
                    out[a] = round(out.get(a, 0.0) + r["usd"], 6)
            return out

    def rows_since(self, day: date) -> list[dict]:
        with self._lock:
            merged: dict[tuple[date, str], dict] = {}
            for (d, p, _), r in self._rows.items():
                if d >= day:
                    m = merged.setdefault((d, p), {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0})
                    for k in m:
                        m[k] += r[k]
            items = sorted(((d, p, r) for (d, p), r in merged.items()), key=lambda t: (-t[0].toordinal(), t[1]))
        return [{"day": d.isoformat(), "provider": p, "usd": round(r["usd"], 6), **{k: r[k] for k in
                 ("calls", "tokens_in", "tokens_out")}} for d, p, r in items]


# ---------------------------------------------------------------- meter

@dataclass
class Charge:
    """Usage reported for one metered call (or one parallel batch of calls)."""
    provider: str
    projected: float
    usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0

    def record(self, usd: float, tokens_in: int = 0, tokens_out: int = 0, calls: int = 1) -> None:
        self.usd += usd
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.calls += calls


class SpendMeter:
    def __init__(self, store: SpendStore, cap_usd: float, today: Callable[[], date] = utc_today,
                 account_cap_usd: float | None = None):
        """`cap_usd` is the whole service's daily cap. `account_cap_usd` (None: no such cap) is each account's own
        daily cap inside it, so one account cannot spend the day's budget of all the others."""
        self.store, self.cap, self.today = store, float(cap_usd), today
        self.account_cap = None if account_cap_usd is None else float(account_cap_usd)
        self._reserved = 0.0          # projected cost of calls in flight in this process
        self._reserved_by: dict[int, float] = {}        # the same, per account
        self._lock = threading.Lock()

    def spent_today(self, account: int | None = None) -> float:
        """Today's spend: one account's, or with None the whole service's."""
        return self.store.spent(self.today(), account)

    def today_by_account(self) -> dict[int, float]:
        return self.store.by_account(self.today())

    def _check_locked(self, projected: float) -> None:
        spent = self.store.spent(self.today())
        if spent + self._reserved + projected > self.cap:
            raise CapReached(f"daily cap ${self.cap:.2f} reached: spent ${spent:.4f}, in flight "
                             f"${self._reserved:.4f}, projected ${projected:.4f}")
        account = ACCOUNT.get()
        if self.account_cap is not None and account:
            mine = self.store.spent(self.today(), account)
            held = self._reserved_by.get(account, 0.0)
            if mine + held + projected > self.account_cap:
                raise CapReached(f"this account's daily cap ${self.account_cap:.2f} reached: spent ${mine:.4f}, "
                                 f"in flight ${held:.4f}, projected ${projected:.4f}")

    def check(self, projected: float) -> None:
        """Refuse (CapReached) if a call of this projected cost would exceed today's cap. Records nothing."""
        with self._lock:
            self._check_locked(projected)

    @contextmanager
    def charge(self, provider: str, projected: float) -> Generator[Charge, None, None]:
        """Reserve `projected` against the cap for the duration of the call, then record what the call
        reported through Charge.record(). Usage recorded before an exception is still written."""
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        account = ACCOUNT.get() or 0
        with self._lock:
            self._check_locked(projected)
            self._reserved += projected
            self._reserved_by[account] = self._reserved_by.get(account, 0.0) + projected
        ch = Charge(provider, projected)
        try:
            yield ch
        finally:
            try:
                if ch.calls:
                    self.store.add(self.today(), provider, ch.usd, ch.tokens_in, ch.tokens_out, ch.calls, account)
            finally:
                with self._lock:
                    self._reserved = max(0.0, self._reserved - projected)
                    self._reserved_by[account] = max(0.0, self._reserved_by.get(account, 0.0) - projected)

    def by_day(self, days: int) -> list[dict]:
        return self.store.rows_since(self.today() - timedelta(days=max(1, days) - 1))
