"""One command from a fresh clone to a connected client.

    python -m brain.setup                      # first run: writes .env (an operator key; trial mode if no provider keys)
    python -m brain.setup                      # with the service up: creates a client key and prints how to connect
    python -m brain.setup --desktop            # the same, and adds the server to Claude Desktop's config
    python -m brain.setup --client cursor --namespace work

Run it from the `service/` folder. It is safe to run again: it never replaces a value already in `.env`, never
reuses a client name, and backs up Claude Desktop's config before changing it. The operator key is written to
`.env` and never printed. A client key is printed once, because that is the only time it exists in the clear.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from collections.abc import Mapping
from pathlib import Path

from .client import DEFAULT_URL, BrainClient, BrainError
from .settings import read_env_file

PROVIDER_KEYS = ("TYPESAFE_API_KEY", "VOYAGE_API_KEY")


def write_private(path: Path, text: str) -> None:
    """Write a file that holds a key so that only its owner can read it (0600), whether it is new or was there
    already. On Windows the mode has no effect and the user profile's own permissions apply."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    try:
        os.chmod(path, 0o600)       # an existing file keeps its old mode through os.open
    except OSError:
        pass


# The settings the published results were measured with (REFERENCE.md, "measured on"). The code defaults are off,
# so that a change of behaviour is always a choice; a .env this command creates makes that choice for a new
# install, so it behaves like the brain the results describe. An existing .env is never given these.
MEASURED_ON = (
    "WRITE_PATH_ENABLED=1", "BRAIN_LEDGER_ROUTE=1", "BRAIN_NEIGHBOURS=1", "BRAIN_DATES=1", "BRAIN_DATE_NOTES=1",
    "BRAIN_DATE_DIRECTION=1", "BRAIN_DATE_COUNTED=1", "BRAIN_CONTEXT_PARALLEL=1", "BRAIN_SHORTLIST_SEATS=25,5",
    "BRAIN_PLACE_PER_SLOT=1", "BRAIN_PLACE_CHECK=1", "BRAIN_RELATE_ENDS=1", "BRAIN_RESOLVE_NAMES=1",
)


def read_env(path: Path) -> dict[str, str]:
    """The names set in a .env file, with their values (never printed by this module)."""
    return read_env_file(path)


def ensure_env(path: Path) -> list[str]:
    """Add what a first run needs to `.env` and return the names added. Existing lines are never changed.
    An operator key is generated when none is set. With no provider key set, trial mode (stub providers) is
    turned on so the service starts and can be tried; setting the keys later and removing that line goes live."""
    new = not path.exists()
    have = read_env(path)
    add: list[str] = []
    if not have.get("BRAIN_ADMIN_KEY"):
        add.append(f"BRAIN_ADMIN_KEY={secrets.token_urlsafe(32)}")
    if "BRAIN_OFFLINE" not in have and not any(have.get(k) for k in PROVIDER_KEYS):
        add.append("BRAIN_OFFLINE=1")
    if new:
        add.extend(MEASURED_ON)
    if add:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        write_private(path, text + "\n".join(add) + "\n")
        give_to_folder_owner(path)
    return [a.split("=", 1)[0] for a in add]


def give_to_folder_owner(path: Path) -> None:
    """Run as root inside a container, setup would leave a root-owned .env that the user cannot read. Hand the
    file to whoever owns the folder it is in. Does nothing when not root, and on Windows."""
    if getattr(os, "geteuid", lambda: 1)() != 0:
        return
    try:
        st = os.stat(path.parent)
        os.chown(path, st.st_uid, st.st_gid)
    except OSError:
        pass


def desktop_config_path(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    """Where Claude Desktop keeps its config on this machine."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return Path(env.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Claude" / "claude_desktop_config.json"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "Claude" / "claude_desktop_config.json"


def desktop_entry(service_dir: Path, url: str, key: str, python: str | None = None) -> dict:
    """The server entry Claude Desktop runs: the stdio proxy, from this checkout, with this client's key."""
    return {"command": python or sys.executable, "args": ["-m", "brain.mcp_stdio"],
            "env": {"BRAIN_URL": url, "BRAIN_KEY": key, "PYTHONPATH": str(service_dir)}}


DOCKER_IMAGE = "brain-memory"


def docker_entry(url: str, key: str, image: str = DOCKER_IMAGE) -> dict:
    """The server entry for Claude Desktop when the brain runs in Docker: the stdio proxy in a throwaway container
    of the same image, so nothing has to be installed on the machine. From inside a container the host's
    `localhost` is `host.docker.internal`; the --add-host line makes that name work on Linux too."""
    inside = url.replace("://localhost", "://host.docker.internal").replace("://127.0.0.1", "://host.docker.internal")
    return {"command": "docker",
            "args": ["run", "-i", "--rm", "--add-host", "host.docker.internal:host-gateway",
                     "-e", "BRAIN_URL", "-e", "BRAIN_KEY", image, "python", "-m", "brain.mcp_stdio"],
            "env": {"BRAIN_URL": inside, "BRAIN_KEY": key}}


def add_to_desktop(cfg_path: Path, entry: dict, name: str = "brain") -> Path | None:
    """Add the server to Claude Desktop's config; the file as it was is kept beside it. Returns the backup's path
    (None when there was no config yet). Refuses to replace a server of the same name."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    servers = cfg.setdefault("mcpServers", {})
    if name in servers:
        raise ValueError(f"Claude Desktop already has a server named {name}")
    backup = None
    if cfg_path.exists():
        backup = cfg_path.with_name(f"{cfg_path.stem}.before-{name}{cfg_path.suffix}")
        write_private(backup, cfg_path.read_text(encoding="utf-8"))     # it may hold other servers' keys
    servers[name] = entry
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    write_private(cfg_path, json.dumps(cfg, indent=2))
    return backup


def connect(client: BrainClient, name: str, namespace: str) -> str:
    """Create a client key that reads and writes one namespace and speaks as the user. Returns the key."""
    made = client.create_client({"name": name, "read_ns": [namespace], "write_ns": [namespace],
                                 "default_ns": namespace, "is_admin": False, "account": None, "speaks_as_user": True})
    return made["key"]


def main(argv: list[str] | None = None, cwd: Path | None = None, http=None, out=print) -> int:
    p = argparse.ArgumentParser(prog="python -m brain.setup", description="Set the brain up and connect a client.")
    p.add_argument("--client", default="claude", help="name for the client key (default: claude)")
    p.add_argument("--namespace", default="me", help="the namespace it reads and writes (default: me)")
    p.add_argument("--url", help=f"where the service is (default: BRAIN_URL in .env, else {DEFAULT_URL})")
    p.add_argument("--public-url", help="the address clients use, when it differs from --url (set by Docker Compose)")
    p.add_argument("--docker", action="store_true", default=os.environ.get("BRAIN_SETUP_DOCKER") == "1",
                   help="the brain runs in Docker: connect Claude Desktop through Docker too (set by Docker Compose)")
    p.add_argument("--desktop", action="store_true", help="also add the server to Claude Desktop's config")
    args = p.parse_args(argv)
    cwd = Path.cwd() if cwd is None else cwd
    env_path = cwd / ".env"

    added = ensure_env(env_path)
    if added:
        shown = [a for a in added if a in ("BRAIN_ADMIN_KEY", "BRAIN_OFFLINE")]
        more = len(added) - len(shown)
        out(f"Wrote {', '.join(shown)}" + (f" and the {more} measured settings" if more else "") + f" to {env_path.name}.")
        if "BRAIN_OFFLINE" in added:
            out("No provider keys are set, so this is trial mode (stub providers). For real answers set "
                "TYPESAFE_API_KEY and VOYAGE_API_KEY in .env, change that line to BRAIN_OFFLINE=0 and restart.")
    env = read_env(env_path)
    url = (args.url or os.environ.get("BRAIN_SETUP_URL") or env.get("BRAIN_URL") or DEFAULT_URL).rstrip("/")
    public = (args.public_url or os.environ.get("BRAIN_SETUP_PUBLIC_URL") or url).rstrip("/")

    c = BrainClient(url, env["BRAIN_ADMIN_KEY"], http=http)
    try:
        try:
            c.request("GET", "/healthz")
        except BrainError as e:
            if e.code != "unreachable":
                out(f"The service at {url} answered {e.status} ({e.code}). Check it, then run this again.")
                return 1
            out("The service is not running yet. Start it, then run this again:")
            out("    docker compose up -d --build" if args.docker else
                "    docker compose up -d --build        # or: python -m brain.serve")
            return 0
        try:
            key = connect(c, args.client, args.namespace)
        except BrainError as e:
            if e.code == "name_taken":
                out(f"A client named {args.client} already exists (names are never reused). "
                    f"Pick another with --client.")
            elif e.status in (401, 403):
                out(f"The service at {url} did not accept the operator key in {env_path.name}. "
                    "It was started with a different BRAIN_ADMIN_KEY: restart it so it reads this file.")
            else:
                out(f"Could not create the client key: {e.status} {e.code}.")
            return 1
    finally:
        c.close()

    out(f"Created client {args.client} for namespace {args.namespace}. Its key, shown this once:")
    out(f"    {key}")
    out("Claude Code:")
    out(f'    claude mcp add --transport http brain {public}/mcp --header "Authorization: Bearer {key}"')
    entry = docker_entry(public, key) if args.docker else desktop_entry(cwd, public, key)
    if args.desktop and args.docker:
        out("Setup is running in a container, so it cannot reach Claude Desktop's config on your machine. "
            "Add the entry below by hand.")
    elif args.desktop:
        cfg_path = desktop_config_path()
        try:
            backup = add_to_desktop(cfg_path, entry)
        except ValueError as e:
            out(f"{e}; its config was not changed. The entry to add by hand is below.")
        else:
            out(f"Added the server to Claude Desktop ({cfg_path})" + (f"; the file as it was: {backup.name}." if backup else "."))
            out("Restart Claude Desktop to see the memory tools.")
            return 0
    out('Claude Desktop (Settings, Developer, Edit Config), under "mcpServers":')
    out("    " + json.dumps({"brain": entry}, indent=2).replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
