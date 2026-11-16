"""Request models shared by REST (api.py) and, from M3, MCP and the CLI. Limits from REFERENCE.md
Models live at module level: FastAPI cannot resolve postponed annotations of local classes."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_TURNS = 100
MAX_TURN_CHARS = 100_000
MAX_K = 50
MAX_QUERY_CHARS = 10_000
MAX_NAMESPACES = 20

Role = Literal["user", "assistant", "system", "tool", "attachment"]
Channel = Literal["chat", "voice", "email", "meeting", "agent_log", "notes", "document", "image"]


MAX_DEPTH = 32   # nesting depth of a request (meta included)


def _check_values(data) -> None:
    """Walk the raw request without recursion: refuse NUL characters (Postgres text cannot hold U+0000) and
    nesting deeper than MAX_DEPTH, each as a clear 422 instead of a 500 (found by stress/run_local.py)."""
    stack = [(data, 0)]
    while stack:
        v, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise ValueError(f"the request is nested more than {MAX_DEPTH} levels deep")
        if isinstance(v, str):
            if chr(0) in v:
                raise ValueError("a value contains a NUL character (U+0000), which cannot be stored; remove it")
        elif isinstance(v, dict):
            stack.extend((k, depth + 1) for k in v)
            stack.extend((x, depth + 1) for x in v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend((x, depth + 1) for x in v)


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _safe_values(cls, data):
        _check_values(data)
        return data


MAX_CLAIMS_PER_TURN = 25
Polarity = Literal["asserted", "negated", "planned", "hypothetical", "corrected", "reported"]


class ClaimIn(_In):
    """One claim the user's model read in this turn (DESIGN.md write step 4, amended 2026-11-09; the W3 contract).
    The brain checks it: the quote must be an exact substring of the turn and the value's words must come from the
    quote, then Jev verifies it."""
    mention: str = Field(default="", max_length=300)      # as written in the turn; "" = the speaker
    slot: str = Field(min_length=1, max_length=100)        # a known slot, or "new:<snake_case_name>"
    value: str = Field(min_length=1, max_length=1000)
    time_phrase: str | None = Field(default=None, max_length=200)
    polarity: Polarity = "asserted"
    quote: str = Field(min_length=3, max_length=2000)


class TurnIn(_In):
    role: Role
    text: str = Field(min_length=1, max_length=MAX_TURN_CHARS)
    ts: datetime | None = None              # ISO 8601; default now; no offset means UTC
    speaker: str | None = Field(default=None, max_length=200)
    channel: Channel = "chat"
    meta: dict = Field(default_factory=dict)  # filename, page, url, ... (attachments: the client reads the file)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)
    claims: list[ClaimIn] = Field(default_factory=list, max_length=MAX_CLAIMS_PER_TURN)


class ResolveIn(_In):
    namespace: str | None = None
    keep_claim_id: int = Field(ge=1)


class CaptureIn(_In):
    namespace: str | None = None
    session_id: str | None = Field(default=None, max_length=200)
    turns: list[TurnIn] = Field(min_length=1, max_length=MAX_TURNS)


class RecallIn(_In):
    namespace: str | None = None
    namespaces: list[str] | None = Field(default=None, min_length=1, max_length=MAX_NAMESPACES)
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    k: int = Field(default=10, ge=1, le=MAX_K)
    session_id: str | None = Field(default=None, max_length=200)
    channels: list[Channel] | None = None
    time_from: datetime | None = None
    time_to: datetime | None = None
    rerank: bool = True
    screen: bool = True


class AnswerIn(_In):
    namespace: str | None = None
    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    question_date: str | None = Field(default=None, max_length=100)   # default today (UTC), as YYYY-MM-DD
    style: Literal["auto", "default", "preference"] = "auto"


class ForgetIn(_In):
    """Forget archives (reversible with restore). purge erases the turns' text for good and needs confirm_erase."""
    namespace: str | None = None
    episode_ids: list[int] | None = Field(default=None, min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    entity: str | None = Field(default=None, min_length=1, max_length=300)   # archive what is known about it
    purge: bool = False                     # erase the text now (cannot be undone)
    confirm_erase: bool = False             # required with purge: the user asked to delete permanently


class RestoreIn(_In):
    namespace: str | None = None
    episode_ids: list[int] | None = Field(default=None, min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    entity: str | None = Field(default=None, min_length=1, max_length=300)


class NamespaceIn(_In):                     # session_start and directives over MCP
    namespace: str | None = None


class EntityIn(_In):                        # page, current and history over MCP
    namespace: str | None = None
    entity: str = Field(min_length=1, max_length=300)
    slot: str | None = Field(default=None, min_length=1, max_length=100)


class DirectiveIn(_In):
    namespace: str | None = None
    text: str = Field(min_length=1, max_length=2000)
    scope: str = "all"                      # 'all' or a client name


class AccountIn(_In):
    name: str = Field(max_length=63)
    kind: Literal["individual", "business"] = "individual"   # business: later (501 until then)
    default_ns: str | None = None           # the first key's default namespace; default: the account name


class ClientIn(_In):
    name: str = Field(max_length=63)
    account: str | None = None              # operator only: create the key in another account
    read_ns: list[str] = Field(default_factory=list, max_length=100)
    write_ns: list[str] = Field(default_factory=list, max_length=100)
    default_ns: str | None = None
    is_admin: bool = False
    speaks_as_user: bool = False            # this client's role=user turns are the user's own words


class ConfirmDirectiveIn(_In):
    namespace: str | None = None
    all_clients: bool = False               # widen from the proposing client to every client
