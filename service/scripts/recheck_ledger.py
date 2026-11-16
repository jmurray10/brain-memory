"""Re-check what the old ledger rules decided, or take one merge night back (room review v2, items 1b and 1e).

It works on the database DATABASE_URL names (the service's own setting), under the ledger lock, per account.
Production runs need the owner's go: --dry-run and --apply call Jev (paid; the daily cap in settings applies).

    python scripts/recheck_ledger.py                      # survey: counts only, no Jev call, nothing written
    python scripts/recheck_ledger.py --dry-run            # ask Jev, report what would change, write nothing
    python scripts/recheck_ledger.py --apply              # reverse the links the shared rule would not write
    python scripts/recheck_ledger.py --undo-merge RUN_ID  # take one merge_slots run back (nightly.undo_merge_run)

    --account ID   one account only (default: every account that is not disabled)
    --limit N      at most N links per account (default 2000, oldest first)

Prints one JSON line per account. The rules are in brain/recheck.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain import nightly, recheck, worker  # noqa: E402
from brain.db import make_pool, tenant  # noqa: E402
from brain.providers import make_providers  # noqa: E402
from brain.settings import load  # noqa: E402
from brain.spend import PgSpendStore, SpendMeter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="ask Jev and report; write nothing")
    mode.add_argument("--apply", action="store_true", help="ask Jev and write the changes")
    mode.add_argument("--undo-merge", metavar="RUN_ID", help="take one merge_slots run back")
    p.add_argument("--account", type=int, help="one account id (default: all)")
    p.add_argument("--all-accounts", action="store_true", help="with --undo-merge: undo the run in every account")
    p.add_argument("--limit", type=int, default=2000)
    args = p.parse_args(argv)
    if args.undo_merge and not (args.account or args.all_accounts):
        p.error("--undo-merge needs --account ID, or --all-accounts to undo the run everywhere")
    settings = load()
    if not settings.database_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2
    pool = make_pool(settings.database_url, max_size=2)
    providers = make_providers(settings, SpendMeter(PgSpendStore(pool), settings.daily_cap_usd))
    writes = bool(args.apply or args.undo_merge)
    try:
        with pool.connection() as c:
            accounts = [args.account] if args.account else [r[0] for r in c.execute(
                "select id from brain.account where disabled_at is null order by id")]
        with worker.ledger_lock(pool, worker.CONSOLIDATE_LOCK_WAIT_S if writes else 0) as held:
            if writes and not held:
                print("the ledger lock is held by another run; try again", file=sys.stderr)
                return 1
            for acct in accounts:
                with tenant(pool, acct) as conn:
                    if args.undo_merge:
                        out = nightly.undo_merge_run(conn, args.undo_merge)
                    elif args.apply or args.dry_run:
                        out = recheck.recheck(conn, providers, apply=args.apply, limit=args.limit)
                    else:
                        out = recheck.survey(conn)
                print(json.dumps({"account": acct, **out}, default=str))
    finally:
        providers.close()
        pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
