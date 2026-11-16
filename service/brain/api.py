"""FastAPI app: REST over core.py (REFERENCE.md), bearer auth with namespaces and scopes, admin
client keys, spend report, request ids, access log, body limit and rate limit; MCP at /mcp (mcp_server.py)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from typing import Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from . import __version__, accounts, core
from .auth import Caller, authenticate, create_client, list_clients, revoke_client, set_speaks_as_user
from .db import make_pool, migrate, tenant
from .errors import ApiError, install as install_errors
from .mcp_server import LocalBackend, McpGate, build_server, http_app
from .middleware import RateLimiter, RequestContext
from .providers import make_providers
from .schemas import (AccountIn, AnswerIn, CaptureIn, ClientIn, ConfirmDirectiveIn, DirectiveIn, ForgetIn, RecallIn,
                      ResolveIn, RestoreIn)
from .settings import Settings, load
from .spend import PgSpendStore, SpendMeter

BUILT = ["healthz", "migrations", "auth", "admin_clients", "spend_meter", "providers", "capture", "recall",
         "answer", "forget", "restore", "directives", "session_start", "export", "decision_log", "mcp", "cli", "accounts", "rls",
         "claims", "ledger", "current", "history", "count", "preferences", "page", "resolve_conflict", "context"]


def admit(state, header: str | None) -> Caller:
    """Bearer auth plus the per-key rate limit; shared by REST (the `caller` dependency) and /mcp (McpGate).
    The limit is kept per account and key name: names are unique only within an account, and every account's
    first key is called account-admin."""
    c = authenticate(state.pool, state.settings.admin_key, header)
    ok, retry_after = state.limiter.allow(c.label)
    if not ok:
        raise ApiError(429, "rate_limited", f"over {state.settings.rate_per_min} requests per minute for this key",
                       {"Retry-After": str(retry_after)})
    return c


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = app.state.spend_pool = app.state.meter = app.state.providers = None
        if settings.database_url:
            if settings.migrate_on_start:
                migrate(settings.database_url)
            app.state.pool = make_pool(settings.database_url, max_size=settings.db_pool_max)
            # The meter gets its own small pool: a request holds a pooled connection for its whole transaction
            # and meters its provider calls inside it, so sharing one pool deadlocks once every connection is
            # held by a request waiting for a second one (found by stress/run_local.py, 2026-11-08).
            app.state.spend_pool = make_pool(settings.database_url, max_size=2)
            app.state.meter = SpendMeter(PgSpendStore(app.state.spend_pool), settings.daily_cap_usd,
                                         account_cap_usd=settings.account_cap_usd)
            app.state.providers = make_providers(settings, app.state.meter)
        # A fresh MCP server per start: the SDK's session manager can only run once.
        app.state.mcp_manager = http_app(build_server(LocalBackend(app)))
        async with app.state.mcp_manager.run():
            yield
        app.state.mcp_manager = None
        if app.state.providers is not None:
            app.state.providers.close()
        for p in (app.state.pool, app.state.spend_pool):
            if p is not None:
                p.close()

    app = FastAPI(title="Brain-Memory", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_per_min)
    app.add_middleware(RequestContext, max_body_bytes=settings.max_body_bytes)
    install_errors(app)

    # ------------------------------------------------------------ dependencies

    def caller(request: Request) -> Caller:
        c = admit(request.app.state, request.headers.get("authorization"))
        request.state.client = c.label
        return c

    def operator(c: Caller = Depends(caller)) -> Caller:
        c.require_operator()
        return c

    def admin(c: Caller = Depends(caller)) -> Caller:
        c.require_admin()
        return c

    def pool(request: Request):
        p = request.app.state.pool
        if p is None:
            raise ApiError(503, "unavailable", "database not configured")
        return p

    def rid(request: Request) -> str:
        return request.state.request_id

    def providers(request: Request):
        prov = request.app.state.providers
        if prov is None:
            raise ApiError(503, "unavailable", "database not configured")
        return prov

    def logged(request: Request, out: dict) -> dict:
        """Put the namespace into the access-log line."""
        ns = out.get("namespace") or ",".join(out.get("namespaces") or [])
        if ns:
            request.state.namespace = ns
        return out

    # ------------------------------------------------------------ routes

    @app.get("/healthz")
    def healthz(request: Request):
        p = request.app.state.pool
        if p is None:
            return JSONResponse({"ok": False, "error": "DATABASE_URL not set"}, status_code=503)
        try:
            with p.connection() as c:
                vec = c.execute("select extversion from pg_extension where extname = 'vector'").fetchone()
                mig = [r[0] for r in c.execute("select name from brain.schema_migrations order by name").fetchall()]
            return {"ok": True, "version": __version__, "pgvector": vec[0] if vec else None, "migrations": mig}
        except Exception as e:  # report the class only; messages can include connection details
            return JSONResponse({"ok": False, "error": type(e).__name__}, status_code=503)

    @app.get("/v1/status")
    def status(request: Request, c: Caller = Depends(caller)):
        meter = request.app.state.meter
        # A key sees its own account's spend; the whole service's bill is the operator's to see (auth.py).
        mine = None if c.is_operator else c.account_id
        return {"version": __version__, "keys": settings.key_status(), "offline": settings.offline,
                "daily_cap_usd": settings.daily_cap_usd,
                "spent_today_usd": round(meter.spent_today(mine), 6) if meter else None,
                "client": c.name, "account": c.account, "operator": c.is_operator, "default_ns": c.default_ns,
                "built": BUILT,
                "not_built": {k: f"DESIGN.md build step {v}" for k, v in core.NOT_BUILT.items()},
                "write_path_enabled": settings.write_path_enabled,
                "request_id": rid(request)}

    # ------------------------------------------------------------ core (section 7)

    @app.post("/v1/capture")
    def capture(body: CaptureIn, request: Request, c: Caller = Depends(caller), p=Depends(pool),
                prov=Depends(providers)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.capture(conn, prov, c, body.namespace, body.turns, body.session_id,
                                                rid(request), settings.write_path_enabled, settings.dates_enabled))

    @app.post("/v1/recall")
    def recall(body: RecallIn, request: Request, c: Caller = Depends(caller), p=Depends(pool),
               prov=Depends(providers)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.recall(
                conn, prov, c, body.namespace, body.query, body.k, body.namespaces, body.session_id,
                body.channels, body.time_from, body.time_to, body.rerank, body.screen, rid(request)))

    @app.post("/v1/answer")
    def answer(body: AnswerIn, request: Request, c: Caller = Depends(caller), p=Depends(pool),
               prov=Depends(providers)):
        with tenant(p, c.account_id) as conn:
            out = core.answer(conn, prov, c, body.namespace, body.question, body.question_date, body.style,
                              rid(request), when_route=settings.when_route_enabled,
                              ledger_route=settings.ledger_route_enabled,
                              memory_check=settings.memory_check_enabled)
        return logged(request, out)

    @app.post("/v1/forget")
    def forget(body: ForgetIn, request: Request, c: Caller = Depends(caller), p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.forget(conn, c, body.namespace, body.episode_ids, body.session_id,
                                               body.entity, body.purge, settings.forget_grace_days,
                                               rid(request), confirm_erase=body.confirm_erase))

    @app.post("/v1/restore")
    def restore(body: RestoreIn, request: Request, c: Caller = Depends(caller), p=Depends(pool),
                prov=Depends(providers)):
        """Undo a forget in the caller's own namespaces (forget archives; only erase cannot be undone)."""
        with tenant(p, c.account_id) as conn:
            return logged(request, core.restore(conn, prov, c, body.namespace, body.episode_ids, body.session_id,
                                                settings.write_path_enabled, rid(request), settings.dates_enabled,
                                                entity=body.entity))

    @app.post("/v1/context")
    def context(body: AnswerIn, request: Request, c: Caller = Depends(caller), p=Depends(pool),
                prov=Depends(providers)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.context(conn, prov, c, body.namespace, body.question, body.question_date,
                                                body.style, rid(request), when_route=settings.when_route_enabled,
                                                ledger_route=settings.ledger_route_enabled,
                              memory_check=settings.memory_check_enabled))

    @app.get("/v1/directives")
    def directives(request: Request, namespace: str | None = None, c: Caller = Depends(caller), p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.list_directives(conn, c, namespace, rid(request)))

    @app.post("/v1/directives", status_code=201)
    def add_directive(body: DirectiveIn, request: Request, c: Caller = Depends(caller), p=Depends(pool),
                      prov=Depends(providers)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.add_directive(conn, c, body.namespace, body.text, body.scope, rid(request),
                                                      prov))

    @app.post("/v1/admin/directives/{directive_id}/confirm")
    def admin_confirm_directive(directive_id: int, body: ConfirmDirectiveIn, request: Request,
                                c: Caller = Depends(admin), p=Depends(pool)):
        """The user's confirmation of a pending directive. Admin keys only; not an MCP tool."""
        with tenant(p, c.account_id) as conn:
            return logged(request, core.confirm_directive(conn, c, body.namespace, directive_id, body.all_clients,
                                                          rid(request)))

    @app.delete("/v1/directives/{directive_id}")
    def retire_directive(directive_id: int, request: Request, namespace: str | None = None,
                         c: Caller = Depends(caller), p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.retire_directive(conn, c, namespace, directive_id, rid(request)))

    @app.get("/v1/session_start")
    def session_start(request: Request, namespace: str | None = None, c: Caller = Depends(caller),
                      p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.session_start(conn, c, namespace, rid(request)))

    @app.get("/v1/export")
    def export(request: Request, namespace: str | None = None,
               format: Literal["markdown", "jsonl"] = "markdown",  # noqa: A002  (the spec's query name)
               c: Caller = Depends(caller), p=Depends(pool)):
        ns, media, lines = core.export(p, c, namespace, format)
        request.state.namespace = ns
        ext = "md" if format == "markdown" else "jsonl"
        return StreamingResponse(lines, media_type=media,
                                 headers={"Content-Disposition": f'attachment; filename="brain-{ns}.{ext}"'})

    @app.get("/v1/admin/export")
    def export_account(request: Request, c: Caller = Depends(caller), p=Depends(pool)):
        """The whole account as NDJSON (every namespace: turns, claims sent, the ledger's facts, directives)."""
        lines = core.export_account(p, c)
        return StreamingResponse(lines, media_type="application/x-ndjson", headers={
            "Content-Disposition": f'attachment; filename="brain-account-{c.account}.jsonl"'})

    # ------------------------------------------------------------ the ledger (build steps 5 and 8; was section 7.7)

    @app.get("/v1/current")
    def current(request: Request, entity: str = Query(min_length=1, max_length=300), slot: str | None = None,
                namespace: str | None = None, c: Caller = Depends(caller), p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.get_current(conn, c, namespace, entity, slot, rid(request)))

    @app.get("/v1/history")
    def history(request: Request, entity: str = Query(min_length=1, max_length=300),
                slot: str = Query(min_length=1, max_length=100), namespace: str | None = None,
                c: Caller = Depends(caller), p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.get_history(conn, c, namespace, entity, slot, rid(request)))

    @app.get("/v1/page")
    def page(request: Request, entity: str = Query(min_length=1, max_length=300), namespace: str | None = None,
             c: Caller = Depends(caller), p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.get_page(conn, c, namespace, entity, rid(request)))

    @app.post("/v1/resolve_conflict")
    def resolve_conflict(body: ResolveIn, request: Request, c: Caller = Depends(caller), p=Depends(pool)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.resolve_conflict(conn, c, body.namespace, body.keep_claim_id, rid(request)))

    # ------------------------------------------------------------ admin

    def target_account(conn, c: Caller, name: str | None) -> int:
        """The caller's own account, or (operator only) the account named."""
        if name is None or name == c.account:
            return c.account_id
        c.require_operator()
        return accounts.account_id(conn, name)

    @app.post("/v1/admin/clients", status_code=201)
    def admin_create_client(body: ClientIn, request: Request, c: Caller = Depends(admin), p=Depends(pool)):
        with p.connection() as conn:
            acct = target_account(conn, c, body.account)
        with tenant(p, acct) as conn:
            client, key = create_client(conn, acct, body.name, body.read_ns, body.write_ns, body.default_ns,
                                        body.is_admin, body.speaks_as_user)
        return {"client": client, "key": key, "note": "Store this key now; it is not shown again.",
                "request_id": rid(request)}

    @app.post("/v1/admin/restore")
    def admin_restore(body: RestoreIn, request: Request, c: Caller = Depends(admin), p=Depends(pool),
                      prov=Depends(providers)):
        with tenant(p, c.account_id) as conn:
            return logged(request, core.restore(conn, prov, c, body.namespace, body.episode_ids, body.session_id,
                                                settings.write_path_enabled, rid(request), settings.dates_enabled,
                                                entity=body.entity))

    @app.get("/v1/admin/clients")
    def admin_list_clients(request: Request, account: str | None = None, c: Caller = Depends(admin),
                           p=Depends(pool)):
        with p.connection() as conn:
            acct = target_account(conn, c, account)
        with tenant(p, acct) as conn:
            return {"clients": list_clients(conn, acct), "request_id": rid(request)}

    @app.post("/v1/admin/clients/{name}/speaks-as-user")
    def admin_speaks_as_user(name: str, request: Request, allow: bool = True, account: str | None = None,
                             c: Caller = Depends(admin), p=Depends(pool)):
        with p.connection() as conn:
            acct = target_account(conn, c, account)
        with tenant(p, acct) as conn:
            return {"client": set_speaks_as_user(conn, acct, name, allow), "request_id": rid(request)}

    @app.post("/v1/admin/clients/{name}/revoke")
    def admin_revoke_client(name: str, request: Request, account: str | None = None, c: Caller = Depends(admin),
                            p=Depends(pool)):
        with p.connection() as conn:
            acct = target_account(conn, c, account)
        with tenant(p, acct) as conn:
            return {"client": revoke_client(conn, acct, name), "request_id": rid(request)}

    # ------------------------------------------------------------ operator: accounts and provider spend

    @app.post("/v1/admin/accounts", status_code=201)
    def admin_create_account(body: AccountIn, request: Request, _: Caller = Depends(operator), p=Depends(pool)):
        with p.connection() as conn:
            account, client, key = accounts.create_account(conn, body.name, body.kind, body.default_ns)
        return {"account": account, "client": client, "key": key,
                "note": "This is the account admin key. Store it now; it is not shown again.",
                "request_id": rid(request)}

    @app.get("/v1/admin/accounts")
    def admin_list_accounts(request: Request, _: Caller = Depends(operator), p=Depends(pool)):
        with p.connection() as conn:
            return {"accounts": accounts.list_accounts(conn), "request_id": rid(request)}

    @app.post("/v1/admin/accounts/{name}/disable")
    def admin_disable_account(name: str, request: Request, _: Caller = Depends(operator), p=Depends(pool)):
        with p.connection() as conn:
            return {"account": accounts.set_disabled(conn, name, True), "request_id": rid(request)}

    @app.post("/v1/admin/accounts/{name}/enable")
    def admin_enable_account(name: str, request: Request, _: Caller = Depends(operator), p=Depends(pool)):
        with p.connection() as conn:
            return {"account": accounts.set_disabled(conn, name, False), "request_id": rid(request)}

    @app.get("/v1/admin/spend")
    def admin_spend(request: Request, days: int = Query(7, ge=1, le=366), _: Caller = Depends(operator),
                    p=Depends(pool)):
        meter = request.app.state.meter
        rows = meter.by_day(days)
        return {"days": rows, "total_usd": round(sum(r["usd"] for r in rows), 6),
                "today_usd": round(meter.spent_today(), 6), "daily_cap_usd": settings.daily_cap_usd,
                "account_cap_usd": settings.account_cap_usd,
                "today_by_account": accounts.spend_by_name(p, meter.today_by_account()), "request_id": rid(request)}

    # ------------------------------------------------------------ MCP (section 9)

    app.router.routes.append(Route("/mcp", endpoint=McpGate(app, admit)))

    return app


app = create_app()
