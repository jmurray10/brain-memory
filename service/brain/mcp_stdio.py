"""Stdio MCP proxy (REFERENCE.md): the same tools as /mcp, forwarded to the REST API.

For Claude Desktop, other stdio-only clients and a Claude plugin. Configure with BRAIN_URL and BRAIN_KEY:

    BRAIN_URL=https://<domain> BRAIN_KEY=bk_... python -m brain.mcp_stdio

Stdout carries the MCP protocol, so this module never prints; problems go to stderr through logging.
"""
from __future__ import annotations

import functools
import logging

import anyio
import anyio.to_thread
import pydantic
from mcp.server.mcpserver import Context

from .client import BrainClient, BrainError
from .mcp_server import build_server, tool_error


class RestBackend:
    def __init__(self, client: BrainClient):
        self.client = client

    def _send(self, op: str, body: pydantic.BaseModel) -> dict:
        c = self.client
        data = body.model_dump(mode="json", exclude_none=True)
        if op == "session_start":
            return c.session_start(data.get("namespace"))
        if op == "directives":
            return c.directives(data.get("namespace"))
        if op == "add_directive":
            return c.add_directive(data)
        if op == "page":
            return c.page(data["entity"], data.get("namespace"))
        if op == "current":
            return c.current(data["entity"], data.get("slot"), data.get("namespace"))
        if op == "history":
            return c.history(data["entity"], data.get("slot") or "", data.get("namespace"))
        if op == "restore":
            return c.restore(data)
        if op in ("capture", "recall", "answer", "forget", "context"):
            return getattr(c, op)(data)
        raise ValueError(f"unknown op {op}")

    async def call(self, op: str, body: pydantic.BaseModel, ctx: Context) -> dict:
        try:
            return await anyio.to_thread.run_sync(functools.partial(self._send, op, body))
        except BrainError as e:
            extra = {k: v for k, v in e.body.items() if k not in ("error", "message", "request_id")}
            raise tool_error(e.code, e.message, e.request_id, extra) from None


def main() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one INFO line per request is noise on stderr
    client = BrainClient()
    try:
        build_server(RestBackend(client)).run("stdio")
    finally:
        client.close()


if __name__ == "__main__":
    main()
