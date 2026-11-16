"""Run the API: python -m brain.serve (the Docker image's command).

Configures logging first (brain/logs.py), then uvicorn with its own logging config and access log off: the
brain.access line (method, path, status, latency, account/client, namespace, request id) replaces it.
"""
from __future__ import annotations

import os

import uvicorn

from . import logs
from .settings import load, load_env_file


def main() -> None:
    load_env_file()
    logs.configure(load().log_level)
    uvicorn.run("brain.api:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")),
                proxy_headers=True, forwarded_allow_ips="*", log_config=None, access_log=False)


if __name__ == "__main__":
    main()
