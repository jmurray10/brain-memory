"""All configuration comes from environment variables. Values are never logged; key_status() reports booleans only."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    admin_key: str | None
    daily_cap_usd: float
    offline: bool                 # stub providers (tests, local dev without keys)
    migrate_on_start: bool
    voyage_key: str | None
    jev_key: str | None
    anthropic_key: str | None
    voyage_model: str = "voyage-4-lite"
    voyage_dim: int = 1024
    jev_model: str = "jev-1.13.0"
    answer_model: str = "claude-haiku-5-5"
    provider_retries: int = 3     # retries on 429, 529, 5xx and network errors
    rate_per_min: int = 120       # per key, in process (one replica in v0)
    max_body_bytes: int = 1_000_000
    write_path_enabled: bool = False  # queue write_path jobs on capture (DESIGN.md build steps 2-9)
    forget_grace_days: int = 0        # 0: forgotten (archived) text is kept until the user erases it; N > 0: purged after N days
    decision_log_days: int = 0        # 0 keeps every decision_log row (the booster's training data); > 0 deletes older ones
    backfill_max: int = 20_000        # chunks the worker embeds per run at most (the daily cap also applies)
    backup_dir: str | None = None     # the worker writes a nightly backup here when set (brain/backup.py)
    backup_keep: int = 14             # backups kept; older ones are deleted, so forgotten text ages out of them
    backup_bucket: str | None = None  # Supabase Storage bucket for nightly backups (private), when set
    supabase_url: str | None = None   # https://<ref>.supabase.co, for Storage only
    supabase_service_key: str | None = None   # worker only; never logged
    log_level: str = "INFO"
    db_pool_max: int = 10             # request connections per API process (the meter has its own 2)
    dates_enabled: bool = False       # build step 3: time phrases -> time_mention at capture (BRAIN_DATES=1)
    when_route_enabled: bool = False  # build step 3: answer runs the router and adds computed dates (BRAIN_WHEN_ROUTE=1)
    ledger_route_enabled: bool = False  # build step 5: current/history questions get ledger facts (BRAIN_LEDGER_ROUTE=1)
    memory_check_enabled: bool = False  # the evidence check's line; off, exp-memory-check-v1 (BRAIN_MEMORY_CHECK=1)
    cold_fraction: float = 0.0          # build step 9: share of episode lines kept out of the hot vector index
    cold_policy: str = "structural"     # build step 9: structural or recency (the baseline)
    cold_min_age_days: int = 30         # build step 9: lines younger than this never go cold

    def key_status(self) -> dict:
        return {"database": bool(self.database_url), "admin": bool(self.admin_key),
                "voyage": bool(self.voyage_key), "jev": bool(self.jev_key), "anthropic": bool(self.anthropic_key),
                "backup_storage": bool(self.backup_bucket and self.supabase_url and self.supabase_service_key)}
    # each account's own daily cap inside the service's (BRAIN_ACCOUNT_CAP_USD); None: no such cap
    account_cap_usd: float | None = None

def read_env_file(path) -> dict[str, str]:
    """The names a .env file sets, with their values. Comments and blank lines are skipped; quotes are removed."""
    from pathlib import Path
    out: dict[str, str] = {}
    path = Path(path)
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_env_file(path=".env") -> list[str]:
    """Put a .env file's settings into the environment for this process and return the names it set. A variable
    that is already set wins, so a container or a host that sets its own is never overridden. Called by the two
    entry points (serve, worker) so that `python -m brain.serve` in a folder with a .env works as documented; an
    image built from the Dockerfile holds no .env, so nothing changes there."""
    took = []
    for k, v in read_env_file(path).items():
        if k not in os.environ:
            os.environ[k] = v
            took.append(k)
    return took


def load() -> Settings:
    return Settings(
        database_url=_first("DATABASE_URL"),
        admin_key=_first("BRAIN_ADMIN_KEY"),
        daily_cap_usd=float(os.environ.get("BRAIN_DAILY_CAP_USD", "5")),
        account_cap_usd=float(os.environ["BRAIN_ACCOUNT_CAP_USD"]) if os.environ.get("BRAIN_ACCOUNT_CAP_USD") else None,
        offline=os.environ.get("BRAIN_OFFLINE", "0") == "1",
        migrate_on_start=os.environ.get("BRAIN_MIGRATE_ON_START", "1") == "1",
        voyage_key=_first("VOYAGE_API_KEY", "VOYAGE_API_KEY_NEW"),
        jev_key=_first("TYPESAFE_API_KEY", "JEV_API_KEY"),
        anthropic_key=_first("ANTHROPIC_API_KEY"),
        jev_model=os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"),
        rate_per_min=int(os.environ.get("BRAIN_RATE_PER_MIN", "120")),
        write_path_enabled=os.environ.get("WRITE_PATH_ENABLED", "0") == "1",
        forget_grace_days=int(os.environ.get("BRAIN_FORGET_GRACE_DAYS", "0")),
        decision_log_days=int(os.environ.get("DECISION_LOG_DAYS", "0")),
        backfill_max=int(os.environ.get("BRAIN_BACKFILL_MAX", "20000")),
        backup_dir=_first("BRAIN_BACKUP_DIR"),
        backup_keep=int(os.environ.get("BRAIN_BACKUP_KEEP", "14")),
        backup_bucket=_first("BRAIN_BACKUP_BUCKET"),
        supabase_url=_first("SUPABASE_URL"),
        supabase_service_key=_first("SUPABASE_SERVICE_ROLE_KEY"),
        log_level=os.environ.get("BRAIN_LOG_LEVEL", "INFO").upper(),
        db_pool_max=int(os.environ.get("BRAIN_DB_POOL_MAX", "10")),
        dates_enabled=os.environ.get("BRAIN_DATES", "0") == "1",
        when_route_enabled=os.environ.get("BRAIN_WHEN_ROUTE", "0") == "1",
        ledger_route_enabled=os.environ.get("BRAIN_LEDGER_ROUTE", "0") == "1",
        memory_check_enabled=os.environ.get("BRAIN_MEMORY_CHECK", "0") == "1",
        cold_fraction=float(os.environ.get("BRAIN_COLD_FRACTION", "0")),
        cold_policy=os.environ.get("BRAIN_COLD_POLICY", "structural"),
        cold_min_age_days=int(os.environ.get("BRAIN_COLD_MIN_AGE_DAYS", "30")),
    )
