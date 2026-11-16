"""Spend cap math (offline, in-memory store) and the Postgres store (needs TEST_DATABASE_URL)."""
import threading
from datetime import date

import pytest

from brain.spend import (CapReached, MemorySpendStore, PgSpendStore, SpendMeter, est_tokens, haiku_cost,
                         jev_cost, voyage_cost)

D1, D2 = date(2026, 11, 7), date(2026, 11, 8)


class Clock:
    def __init__(self, d):
        self.d = d

    def __call__(self):
        return self.d


def test_price_table_matches_live_eval():
    assert haiku_cost(50_000, 10_000) == pytest.approx(50_000 / 1e6 * 0.10 + 10_000 / 1e6 * 0.50)
    assert haiku_cost(100_000, 0) == pytest.approx(0.01)            # at the threshold: short price
    assert haiku_cost(100_001, 1_000_000) == pytest.approx(100_001 / 1e6 * 0.50 + 2.50)  # above: long price
    assert voyage_cost(1_000_000) == pytest.approx(0.02)
    assert jev_cost(1_000_000, 1_000_000) == pytest.approx(0.042)   # output is free
    assert est_tokens("") == 1 and est_tokens("x" * 400) == 100


def test_check_refuses_when_projection_crosses_cap():
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0, today=Clock(D1))
    m.check(1.0)                       # exactly at the cap is allowed
    with pytest.raises(CapReached):
        m.check(1.0001)
    m.store.add(D1, "haiku", 0.6, 1, 1, 1)
    m.check(0.4)
    with pytest.raises(CapReached):
        m.check(0.41)


def test_cap_is_shared_across_providers_and_resets_each_utc_day():
    clock = Clock(D1)
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0, today=clock)
    with m.charge("voyage", 0.1) as ch:
        ch.record(0.5, tokens_in=10)
    with m.charge("jev", 0.1) as ch:
        ch.record(0.4, tokens_in=10)
    assert m.spent_today() == pytest.approx(0.9)
    with pytest.raises(CapReached):
        m.check(0.2)
    clock.d = D2
    assert m.spent_today() == 0.0
    m.check(1.0)


def test_charge_records_real_usage_not_projection():
    m = SpendMeter(MemorySpendStore(), cap_usd=5.0, today=Clock(D1))
    with m.charge("haiku", 0.5) as ch:
        ch.record(0.01, tokens_in=100, tokens_out=20)
    assert m.spent_today() == pytest.approx(0.01)
    assert m.by_day(1) == [{"day": "2026-11-07", "provider": "haiku", "usd": 0.01, "calls": 1,
                            "tokens_in": 100, "tokens_out": 20}]


def test_in_flight_reservations_count_against_cap():
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0, today=Clock(D1))
    with m.charge("jev", 0.7):
        with pytest.raises(CapReached):
            m.check(0.4)                # 0.7 in flight + 0.4 > 1.0
        m.check(0.3)
    m.check(1.0)                        # reservation released, nothing recorded


def test_usage_recorded_before_an_error_is_kept_and_reservation_released():
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0, today=Clock(D1))
    with pytest.raises(RuntimeError):
        with m.charge("jev", 0.5) as ch:
            ch.record(0.2, tokens_in=5)
            raise RuntimeError("provider failed mid-batch")
    assert m.spent_today() == pytest.approx(0.2)
    m.check(0.8)                        # reservation of 0.5 was released


def test_unknown_provider_rejected():
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0)
    with pytest.raises(ValueError):
        with m.charge("openai", 0.1):
            pass


def test_concurrent_charges_never_exceed_cap():
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0, today=Clock(D1))
    gate = threading.Barrier(20)
    ok, refused = [], []

    def work():
        gate.wait()
        try:
            with m.charge("voyage", 0.1) as ch:
                ch.record(0.1, tokens_in=1)
            ok.append(1)
        except CapReached:
            refused.append(1)

    threads = [threading.Thread(target=work) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 10 and len(refused) == 10
    assert m.spent_today() == pytest.approx(1.0)


# ---------------------------------------------------------------- Postgres store

def test_pg_store_upserts_and_reports(pool):
    store = PgSpendStore(pool)
    store.add(D1, "jev", 0.001, 100, 0, 1)
    store.add(D1, "jev", 0.002, 200, 0, 2)
    store.add(D1, "haiku", 0.01, 10, 5, 1)
    store.add(D2, "voyage", 0.0005, 25, 0, 1)
    assert store.spent(D1) == pytest.approx(0.013)
    assert store.spent(D2) == pytest.approx(0.0005)
    rows = store.rows_since(D1)
    assert [(r["day"], r["provider"]) for r in rows] == [("2026-11-08", "voyage"), ("2026-11-07", "haiku"),
                                                         ("2026-11-07", "jev")]
    jev = rows[2]
    assert jev["calls"] == 3 and jev["tokens_in"] == 300 and jev["usd"] == pytest.approx(0.003)
    assert store.rows_since(D2) == rows[:1]


def test_pg_meter_cap_survives_a_new_meter(pool):
    """The cap is in the database, so a restarted process sees today's spend."""
    SpendMeter(PgSpendStore(pool), 0.05, today=Clock(D1)).store.add(D1, "haiku", 0.05, 1, 1, 1)
    fresh = SpendMeter(PgSpendStore(pool), 0.05, today=Clock(D1))
    with pytest.raises(CapReached):
        fresh.check(0.000001)


def test_an_account_cap_stops_that_account_and_not_the_others():
    """Release fix B2: inside the service's daily cap each account has its own, so one cannot spend the others'."""
    from brain import spend
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0, today=Clock(D1), account_cap_usd=0.10)

    def as_account(account, usd):
        token = spend.ACCOUNT.set(account)
        try:
            with m.charge("jev", usd) as ch:
                ch.record(usd, 10, 0)
        finally:
            spend.ACCOUNT.reset(token)

    as_account(1, 0.10)                            # exactly at its cap is allowed
    with pytest.raises(CapReached, match="this account's daily cap"):
        as_account(1, 0.01)
    as_account(2, 0.05)                            # another account is untouched
    with m.charge("voyage", 0.30) as ch:           # work for no account: the service's cap only
        ch.record(0.30, 10, 0)
    assert m.today_by_account() == {0: 0.30, 1: 0.10, 2: 0.05} and m.spent_today() == pytest.approx(0.45)
    token = spend.ACCOUNT.set(2)
    try:
        with pytest.raises(CapReached, match="^daily cap"):     # the service's cap still binds everyone
            SpendMeter(m.store, cap_usd=0.45, today=Clock(D1), account_cap_usd=0.10).check(0.01)
    finally:
        spend.ACCOUNT.reset(token)


def test_without_an_account_cap_nothing_changes():
    from brain import spend
    m = SpendMeter(MemorySpendStore(), cap_usd=1.0, today=Clock(D1))
    token = spend.ACCOUNT.set(7)
    try:
        with m.charge("jev", 0.9) as ch:
            ch.record(0.9, 1, 0)
    finally:
        spend.ACCOUNT.reset(token)
    assert m.today_by_account() == {7: 0.9}


def test_pg_store_keeps_spend_per_account(pool):
    store = PgSpendStore(pool)
    store.add(D1, "jev", 0.001, 100, 0, 1, account=1)
    store.add(D1, "jev", 0.002, 200, 0, 2, account=2)
    store.add(D1, "jev", 0.004, 400, 0, 1)
    assert store.spent(D1) == pytest.approx(0.007) and store.spent(D1, 2) == pytest.approx(0.002)
    assert store.by_account(D1) == {0: 0.004, 1: 0.001, 2: 0.002}
    rows = store.rows_since(D1)                      # the report by day and provider adds the accounts together
    assert len(rows) == 1 and rows[0]["calls"] == 4 and rows[0]["usd"] == pytest.approx(0.007)


def test_a_tenant_block_marks_paid_calls_as_that_accounts(pool):
    from brain import spend
    from brain.db import tenant
    owner = pool.connection().__enter__().execute("select id from brain.account where name = 'owner'").fetchone()[0]
    assert spend.ACCOUNT.get() is None
    with tenant(pool, owner):
        assert spend.ACCOUNT.get() == owner
    assert spend.ACCOUNT.get() is None
