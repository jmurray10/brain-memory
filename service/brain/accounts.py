"""Accounts (REFERENCE.md): one per customer. Operator only; these run as the owner, outside RLS.

Creating an account also creates its first key: an account admin named "account-admin" with every namespace of
that account and a default namespace (the account name unless given). The key is returned once.
"""
from __future__ import annotations

from typing import Literal

import psycopg

from .auth import OWNER_ACCOUNT, check_ns, create_client, valid_ns
from .errors import ApiError

FIRST_CLIENT = "account-admin"
_COLS = "id, name, kind, created_at, disabled_at"


def _row(r) -> dict:
    return {"id": r[0], "name": r[1], "kind": r[2], "created_at": r[3].isoformat(),
            "disabled_at": r[4].isoformat() if r[4] else None}


def account_id(conn: psycopg.Connection, name: str) -> int:
    r = conn.execute("select id from brain.account where name = %s", (name,)).fetchone()
    if r is None:
        raise ApiError(404, "not_found", "no account with that name")
    return r[0]


def create_account(conn: psycopg.Connection, name: str, kind: Literal["individual", "business"] = "individual",
                   default_ns: str | None = None) -> tuple[dict, dict, str]:
    """Returns (account, first client, key)."""
    if not valid_ns(name):
        raise ApiError(400, "bad_name", "account name must match ^[a-z0-9][a-z0-9_-]{0,62}$")
    if kind == "business":
        raise ApiError(501, "not_built", "business accounts (several members) come after individual accounts")
    ns = check_ns(default_ns or name)
    try:
        r = conn.execute(f"insert into brain.account (name, kind) values (%s, %s) returning {_COLS}",
                         (name, kind)).fetchone()
    except psycopg.errors.UniqueViolation:
        raise ApiError(409, "name_taken", "an account with that name already exists") from None
    client, key = create_client(conn, r[0], FIRST_CLIENT, ["*"], ["*"], ns, is_admin=True,
                                speaks_as_user=True)     # the account owner's own key
    return _row(r), client, key


def list_accounts(conn: psycopg.Connection) -> list[dict]:
    rows = conn.execute(f"""select {_COLS}, (select count(*) from brain.api_client k
                                            where k.account_id = a.id and k.revoked_at is null)
                            from brain.account a order by name""").fetchall()
    return [{**_row(r), "active_keys": r[5]} for r in rows]


def spend_by_name(pool, by_account: dict[int, float]) -> list[dict]:
    """The operator's report: today's spend per account by name. Account 0 is spend that belongs to no account
    (the operator's own calls and work done for every account at once)."""
    with pool.connection() as conn:
        names = dict(conn.execute("select id, name from brain.account").fetchall())
    return [{"account": names.get(a, "(none)" if a == 0 else f"#{a}"), "usd": usd} for a, usd in sorted(by_account.items())]


def set_disabled(conn: psycopg.Connection, name: str, disabled: bool) -> dict:
    if name == OWNER_ACCOUNT and disabled:
        raise ApiError(400, "bad_request", "the owner account cannot be disabled")
    r = conn.execute(f"""update brain.account set disabled_at = case when %s then coalesce(disabled_at, now())
                                                                      else null end
                         where name = %s returning {_COLS}""", (disabled, name)).fetchone()
    if r is None:
        raise ApiError(404, "not_found", "no account with that name")
    return _row(r)
