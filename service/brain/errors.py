"""JSON errors: {"error": code, "message": text, "request_id": id} for every failure (REFERENCE.md)."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("brain.errors")

_HTTP_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
               405: "method_not_allowed", 413: "body_too_large", 429: "rate_limited", 501: "not_built",
               503: "unavailable"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None,
                 extra: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}
        self.extra = extra or {}     # more fields for the error body, e.g. {"build_step": 5}


def request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_body(code: str, message: str, rid: str | None) -> dict:
    return {"error": code, "message": message, "request_id": rid}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(request: Request, e: ApiError):
        return JSONResponse({**error_body(e.code, e.message, request_id(request)), **e.extra}, e.status,
                            headers=e.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, e: RequestValidationError):
        # Field locations and messages only; never echo the submitted values.
        problems = "; ".join(f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}"
                             for err in e.errors())
        return JSONResponse(error_body("invalid_request", problems[:1000], request_id(request)), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, e: StarletteHTTPException):
        code = _HTTP_CODES.get(e.status_code, "http_error")
        return JSONResponse(error_body(code, str(e.detail), request_id(request)), e.status_code,
                            headers=getattr(e, "headers", None))

    @app.exception_handler(Exception)
    async def _crash(request: Request, e: Exception):
        log.error("unhandled %s rid=%s", type(e).__name__, request_id(request), exc_info=e)
        return JSONResponse(error_body("internal", "internal error", request_id(request)), 500)
