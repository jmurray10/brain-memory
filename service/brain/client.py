"""A small REST client for the brain, used by the CLI (brain/cli.py) and the stdio MCP proxy (brain/mcp_stdio.py).

Reads BRAIN_URL (default http://localhost:8000) and BRAIN_KEY. The key goes only into the Authorization header; it
is never printed or put into an error message.
"""
from __future__ import annotations

import os
from urllib.parse import quote
from typing import Any, Iterator

import httpx

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)   # answer and recall can wait on Jev and Haiku


class BrainError(Exception):
    """A non-2xx reply, carrying the API's {"error", "message", "request_id"} body."""

    def __init__(self, status: int, code: str, message: str, request_id: str | None = None, body: dict | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.request_id = status, code, message, request_id
        self.body = body or {}


class BrainClient:
    """Thin wrapper over httpx. Pass `http` to use an existing client (tests pass FastAPI's TestClient)."""

    def __init__(self, url: str | None = None, key: str | None = None, http: httpx.Client | None = None):
        self.url = (url or os.environ.get("BRAIN_URL") or DEFAULT_URL).rstrip("/")
        key = key if key is not None else os.environ.get("BRAIN_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._own = http is None
        self.http = http or httpx.Client(base_url=self.url, timeout=TIMEOUT)
        self.headers = headers

    def close(self) -> None:
        if self._own:
            self.http.close()

    def __enter__(self) -> BrainClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def _raise(r: httpx.Response) -> None:
        try:
            body = r.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        raise BrainError(r.status_code, str(body.get("error") or "http_error"),
                         str(body.get("message") or r.reason_phrase or "request failed"), body.get("request_id"), body)

    def request(self, method: str, path: str, *, json: Any = None, params: dict | None = None) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            r = self.http.request(method, path, json=json, params=params or None, headers=self.headers)
        except httpx.HTTPError as e:   # connection refused, timeout: name the class and the host, not the key
            raise BrainError(0, "unreachable", f"{type(e).__name__} talking to {self.url}") from None
        if r.status_code >= 400:
            self._raise(r)
        return r.json()

    def stream(self, path: str, params: dict | None = None) -> Iterator[str]:
        """Yield text pieces of a streamed reply (export)."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            with self.http.stream("GET", path, params=params or None, headers=self.headers) as r:
                if r.status_code >= 400:
                    r.read()
                    self._raise(r)
                yield from r.iter_text()
        except httpx.HTTPError as e:
            raise BrainError(0, "unreachable", f"{type(e).__name__} talking to {self.url}") from None

    # ------------------------------------------------------------ one method per route (REFERENCE.md)

    def status(self) -> dict:
        return self.request("GET", "/v1/status")

    def capture(self, body: dict) -> dict:
        return self.request("POST", "/v1/capture", json=body)

    def recall(self, body: dict) -> dict:
        return self.request("POST", "/v1/recall", json=body)

    def answer(self, body: dict) -> dict:
        return self.request("POST", "/v1/answer", json=body)

    def forget(self, body: dict) -> dict:
        return self.request("POST", "/v1/forget", json=body)

    def directives(self, namespace: str | None = None) -> dict:
        return self.request("GET", "/v1/directives", params={"namespace": namespace})

    def add_directive(self, body: dict) -> dict:
        return self.request("POST", "/v1/directives", json=body)

    def confirm_directive(self, directive_id: int, namespace: str | None = None, all_clients: bool = False) -> dict:
        return self.request("POST", f"/v1/admin/directives/{int(directive_id)}/confirm",
                            json={"namespace": namespace, "all_clients": all_clients})

    def retire_directive(self, directive_id: int, namespace: str | None = None) -> dict:
        return self.request("DELETE", f"/v1/directives/{int(directive_id)}", params={"namespace": namespace})

    def session_start(self, namespace: str | None = None) -> dict:
        return self.request("GET", "/v1/session_start", params={"namespace": namespace})

    def restore(self, body: dict) -> dict:
        return self.request("POST", "/v1/restore", json=body)

    def context(self, body: dict) -> dict:
        return self.request("POST", "/v1/context", json=body)

    def page(self, entity: str, namespace: str | None = None) -> dict:
        return self.request("GET", "/v1/page", params={"entity": entity, "namespace": namespace})

    def current(self, entity: str, slot: str | None = None, namespace: str | None = None) -> dict:
        return self.request("GET", "/v1/current", params={"entity": entity, "slot": slot, "namespace": namespace})

    def history(self, entity: str, slot: str, namespace: str | None = None) -> dict:
        return self.request("GET", "/v1/history", params={"entity": entity, "slot": slot, "namespace": namespace})

    def export(self, namespace: str | None = None, fmt: str = "markdown") -> Iterator[str]:
        return self.stream("/v1/export", params={"namespace": namespace, "format": fmt})

    def export_account(self) -> Iterator[str]:
        return self.stream("/v1/admin/export")

    def create_client(self, body: dict) -> dict:
        return self.request("POST", "/v1/admin/clients", json=body)

    def list_clients(self, account: str | None = None) -> dict:
        return self.request("GET", "/v1/admin/clients", params={"account": account})

    def set_speaks_as_user(self, name: str, allow: bool = True, account: str | None = None) -> dict:
        return self.request("POST", f"/v1/admin/clients/{quote(name, safe='')}/speaks-as-user",
                            params={"allow": "true" if allow else "false", "account": account})

    def revoke_client(self, name: str, account: str | None = None) -> dict:
        return self.request("POST", f"/v1/admin/clients/{quote(name, safe='')}/revoke", params={"account": account})

    def create_account(self, body: dict) -> dict:
        return self.request("POST", "/v1/admin/accounts", json=body)

    def list_accounts(self) -> dict:
        return self.request("GET", "/v1/admin/accounts")

    def set_account_disabled(self, name: str, disabled: bool) -> dict:
        action = "disable" if disabled else "enable"
        return self.request("POST", f"/v1/admin/accounts/{quote(name, safe='')}/{action}")

    def spend(self, days: int = 7) -> dict:
        return self.request("GET", "/v1/admin/spend", params={"days": days})

    def admin_restore(self, body: dict) -> dict:
        """Operator: restore in another account (admin endpoint)."""
        return self.request("POST", "/v1/admin/restore", json=body)
