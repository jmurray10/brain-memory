"""Bearer keys, namespaces and scopes (REFERENCE.md).

- Client keys are "bk_" + 32 random URL-safe bytes, shown once at creation and stored as sha256.
- BRAIN_ADMIN_KEY (env) also works: name "admin", the operator, every scope, admin of the owner account (it acts
  there for data). Compared with hmac.compare_digest.
- Every key belongs to one account (migration 003). An account admin (is_admin) manages that account's keys and
  restores within it; only the operator creates accounts and sees provider spend. A disabled account's keys stop.
- A key has read_ns and write_ns lists; "*" means every namespace. A call without a namespace uses the key's
  default_ns. Missing scope is 403 with the same message whether or not the namespace has any data.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, replace

import psycopg

from .errors import ApiError

NS_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
KEY_PREFIX = "bk_"
ADMIN_NAME = "admin"
OWNER_ACCOUNT = "owner"
ALL = "*"


def valid_ns(ns: str) -> bool:
    return isinstance(ns, str) and bool(NS_RE.match(ns))


def check_ns(ns: str) -> str:
    if not valid_ns(ns):
        raise ApiError(400, "bad_namespace", "namespace must match ^[a-z0-9][a-z0-9_-]{0,62}$")
    return ns


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def parse_bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, token = header.strip().partition(" ")
    token = token.strip()
    return token if scheme.lower() == "bearer" and token else None


@dataclass(frozen=True)
class Caller:
    name: str
    read_ns: tuple[str, ...]
    write_ns: tuple[str, ...]
    default_ns: str | None
    is_admin: bool
    account_id: int | None = None   # None only without a database
    account: str = OWNER_ACCOUNT
    is_operator: bool = False
    speaks_as_user: bool = False    # may store role=user turns as the user's own words (migration 014)

    def can(self, ns: str, write: bool = False) -> bool:
        scopes = self.write_ns if write else self.read_ns
        return ALL in scopes or ns in scopes

    def namespace(self, ns: str | None, write: bool = False) -> str:
        """The namespace a call acts on: the one given, else the key's default. 403 without scope."""
        ns = ns or self.default_ns
        if not ns:
            raise ApiError(400, "namespace_required", "no namespace given and this key has no default namespace")
        check_ns(ns)
        if not self.can(ns, write):
            raise ApiError(403, "forbidden", f"this key has no {'write' if write else 'read'} scope for that namespace")
        return ns

    def require_admin(self) -> None:
        if not self.is_admin:
            raise ApiError(403, "forbidden", "admin key required")

    def require_operator(self) -> None:
        if not self.is_operator:
            raise ApiError(403, "forbidden", "operator key required")

    @property
    def label(self) -> str:
        """account/name, for the access log (client names are unique only within an account)."""
        return f"{self.account}/{self.name}"


# The operator key is the user's own key (it is never handed to a model), so it speaks as the user.
ADMIN_CALLER = Caller(ADMIN_NAME, (ALL,), (ALL,), None, True, None, OWNER_ACCOUNT, True, True)


def authenticate(pool, admin_key: str | None, header: str | None) -> Caller:
    token = parse_bearer(header)
    if not token:
        raise ApiError(401, "unauthorized", "missing bearer token",
                       {"WWW-Authenticate": "Bearer"})
    is_operator = bool(admin_key) and hmac.compare_digest(token.encode(), admin_key.encode())
    if is_operator and pool is None:
        return ADMIN_CALLER
    if pool is None:
        raise ApiError(503, "unavailable", "database not configured")
    # Runs as the owner, outside RLS: the account is not known until the key is.
    with pool.connection() as c:
        if is_operator:
            row = c.execute("select id from brain.account where name = %s", (OWNER_ACCOUNT,)).fetchone()
            return replace(ADMIN_CALLER, account_id=row[0] if row else None)
        row = c.execute("""select k.name, k.read_ns, k.write_ns, k.default_ns, k.is_admin, a.id, a.name, a.disabled_at,
                                  k.speaks_as_user
                           from brain.api_client k join brain.account a on a.id = k.account_id
                           where k.key_hash = %s and k.revoked_at is null""", (hash_key(token),)).fetchone()
    if row is None:
        raise ApiError(401, "unauthorized", "invalid or revoked key", {"WWW-Authenticate": "Bearer"})
    if row[7] is not None:
        raise ApiError(403, "account_disabled", "this account is disabled")
    return Caller(row[0], tuple(row[1]), tuple(row[2]), row[3], bool(row[4]), row[5], row[6], False, bool(row[8]))


# ---------------------------------------------------------------- admin: client keys

def _scopes(values: list[str], label: str) -> list[str]:
    out: list[str] = []
    for v in values:
        if v != ALL and not valid_ns(v):
            raise ApiError(400, "bad_namespace", f"{label}: {v!r} is not '*' or a valid namespace name")
        if v not in out:
            out.append(v)
    return out


def _client_row(r) -> dict:
    return {"name": r[0], "read_ns": list(r[1]), "write_ns": list(r[2]), "default_ns": r[3], "is_admin": r[4],
            "created_at": r[5].isoformat(), "revoked_at": r[6].isoformat() if r[6] else None,
            "speaks_as_user": bool(r[7])}


_COLS = "name, read_ns, write_ns, default_ns, is_admin, created_at, revoked_at, speaks_as_user"


def create_client(conn: psycopg.Connection, account_id: int, name: str, read_ns: list[str], write_ns: list[str],
                  default_ns: str | None = None, is_admin: bool = False,
                  speaks_as_user: bool = False) -> tuple[dict, str]:
    """Insert a client in an account and return (client, key). The key is returned only here; only its hash is
    stored. Names are unique within an account and never reused there. speaks_as_user: this client's role=user
    turns are the user's own words (off by default: they are stored as agent_reported)."""
    if not valid_ns(name):
        raise ApiError(400, "bad_name", "client name must match ^[a-z0-9][a-z0-9_-]{0,62}$")
    if name == ADMIN_NAME:
        raise ApiError(400, "bad_name", "'admin' is reserved for the bootstrap key")
    read, write = _scopes(read_ns, "read_ns"), _scopes(write_ns, "write_ns")
    if default_ns is not None:
        check_ns(default_ns)
        if not ({ALL, default_ns} & (set(read) | set(write))):
            raise ApiError(400, "bad_default_ns", "default_ns must be inside read_ns or write_ns")
    key = generate_key()
    try:
        r = conn.execute(f"""insert into brain.api_client (account_id, name, key_hash, read_ns, write_ns, default_ns,
                                                         is_admin, speaks_as_user)
                             values (%s, %s, %s, %s, %s, %s, %s, %s) returning {_COLS}""",
                         (account_id, name, hash_key(key), read, write, default_ns, is_admin,
                          bool(speaks_as_user))).fetchone()
    except psycopg.errors.UniqueViolation:
        raise ApiError(409, "name_taken", "a client with that name already exists in this account (names are never reused)") from None
    return _client_row(r), key


def list_clients(conn: psycopg.Connection, account_id: int) -> list[dict]:
    return [_client_row(r) for r in conn.execute(
        f"select {_COLS} from brain.api_client where account_id = %s order by name", (account_id,)).fetchall()]


def revoke_client(conn: psycopg.Connection, account_id: int, name: str) -> dict:
    r = conn.execute(f"""update brain.api_client set revoked_at = now()
                         where account_id = %s and name = %s and revoked_at is null returning {_COLS}""",
                     (account_id, name)).fetchone()
    if r is None:
        raise ApiError(404, "not_found", "no active client with that name")
    return _client_row(r)


def set_speaks_as_user(conn: psycopg.Connection, account_id: int, name: str, allow: bool) -> dict:
    """Grant or withdraw a client's right to store role=user turns as the user's own words. An admin call
    (REST or the CLI), never an MCP tool: the user decides this, not a model."""
    r = conn.execute(f"""update brain.api_client set speaks_as_user = %s
                         where account_id = %s and name = %s and revoked_at is null returning {_COLS}""",
                     (bool(allow), account_id, name)).fetchone()
    if r is None:
        raise ApiError(404, "not_found", "no active client with that name")
    return _client_row(r)
