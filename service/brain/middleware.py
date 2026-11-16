"""Request id, access log, body-size limit and the per-key rate limiter (REFERENCE.md).

The access log has method, path, status, latency, client name, namespace and request id. Never bodies, query
strings, headers or keys.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import deque
from typing import Callable

log = logging.getLogger("brain.access")


def new_request_id() -> str:
    return "req_" + secrets.token_hex(8)


class RateLimiter:
    """Sliding 60-second window per key (the caller passes account/name), in process (one API replica in v0)."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute, self.clock = per_minute, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """(allowed, seconds until the next request would be allowed)."""
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= 60.0:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, max(1, int(60.0 - (now - q[0])) + 1)
            q.append(now)
            return True, 0


class RequestContext:
    """Pure ASGI middleware: assigns a request id (x-request-id response header), enforces the body limit
    before the app reads anything, and writes one access-log line per request."""

    def __init__(self, app, max_body_bytes: int):
        self.app, self.max_body = app, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = new_request_id()
        state = scope.setdefault("state", {})
        state["request_id"] = rid
        status = [500]
        started = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status[0] = message["status"]
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", rid.encode())]}
            await send(message)

        try:
            headers = dict(scope.get("headers") or [])
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    too_big = int(length) > self.max_body
                except ValueError:
                    return await self._reject(send_wrapper, rid, 400, "bad_request", "invalid content-length")
                if too_big:
                    return await self._reject(send_wrapper, rid, 413, "body_too_large",
                                              f"request body over {self.max_body} bytes")
                return await self.app(scope, receive, send_wrapper)
            if headers.get(b"transfer-encoding", b"").lower() == b"chunked":
                body, more = b"", True
                while more:
                    msg = await receive()
                    if msg["type"] != "http.request":
                        return  # client went away
                    body += msg.get("body", b"")
                    more = msg.get("more_body", False)
                    if len(body) > self.max_body:
                        return await self._reject(send_wrapper, rid, 413, "body_too_large",
                                                  f"request body over {self.max_body} bytes")
                replayed = [False]

                async def replay():
                    if not replayed[0]:
                        replayed[0] = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await receive()
                return await self.app(scope, replay, send_wrapper)
            return await self.app(scope, receive, send_wrapper)
        finally:
            log.info("%s %s %s %.0fms client=%s ns=%s rid=%s", scope.get("method"), scope.get("path"), status[0],
                     (time.perf_counter() - started) * 1000, state.get("client", "-"), state.get("namespace", "-"), rid)

    @staticmethod
    async def _reject(send, rid: str, status: int, code: str, message: str):
        body = json.dumps({"error": code, "message": message, "request_id": rid}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
