"""Command line client (REFERENCE.md): python -m brain.cli <command> [options].

Uses BRAIN_URL (default http://localhost:8000) and BRAIN_KEY, or --url and --key. Human-readable output by default,
--json for scripts. Exit codes: 0 ok, 1 the API returned an error, 2 bad usage.

    python -m brain.cli capture "My sister lives in Austin." --namespace jeff
    python -m brain.cli capture --file notes.pdf.txt --namespace jeff          # role attachment, channel document
    echo "text" | python -m brain.cli capture --namespace jeff
    python -m brain.cli recall "Where does my sister live?"
    python -m brain.cli answer "Where does my sister live?"
    python -m brain.cli forget --episode 12 --episode 13
    python -m brain.cli directives list | add "Use metric units." | retire 4
    python -m brain.cli export --format jsonl --out brain.jsonl
    python -m brain.cli admin create-client claude-code --read jeff --write jeff --default jeff   # one per client
    python -m brain.cli admin speaks-as-user voice-bot | confirm-directive 4 --all-clients
    python -m brain.cli admin create-account alice                         # operator: a new customer account
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .client import BrainClient, BrainError
from .schemas import MAX_TURN_CHARS

CHANNELS = ["chat", "voice", "email", "meeting", "agent_log", "notes", "document", "image"]
ROLES = ["user", "assistant", "system", "tool", "attachment"]


# ---------------------------------------------------------------- helpers

def pieces(text: str, limit: int = MAX_TURN_CHARS) -> list[str]:
    """Split text longer than one turn allows into turns of at most `limit` characters, on paragraph boundaries
    where possible. The server splits each turn again into recall windows."""
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        if len(para) > limit:                        # a paragraph longer than a turn: fill, then hard cut
            para, cur = (f"{cur}\n\n{para}" if cur else para), ""
            while len(para) > limit:
                out.append(para[:limit])
                para = para[limit:]
        joined = f"{cur}\n\n{para}" if cur else para
        if len(joined) > limit:
            out.append(cur)
            cur = para
        else:
            cur = joined
    if cur:
        out.append(cur)
    return [p for p in out if p.strip()]


def _short(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _print_json(out: Any) -> None:
    print(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- human-readable output

def show_capture(out: dict) -> None:
    eps = out.get("episodes", [])
    ids = ", ".join(str(e["id"]) for e in eps)
    print(f"saved {len(eps)} turn(s) in {out.get('namespace')}: episode {ids}" if eps else "nothing saved")
    if out.get("deduped"):
        print(f"{out['deduped']} already saved (same idempotency key)")
    if out.get("degraded"):
        print(f"degraded: {out['degraded']} (stored without vectors; found by keyword search now, embedded later)")


def show_recall(out: dict) -> None:
    items = out.get("items", [])
    if not items:
        print("no matches")
    for it in items:
        s = it.get("scores", {})
        score = f"rerank {s['rerank']:.2f}" if s.get("rerank") is not None else f"rrf {s.get('rrf', 0):.4f}"
        print(f"[chunk {it['chunk_id']} / episode {it['episode_id']}] {score}")
        print(f"  {_short(it['text'], 400)}")
    flags = []
    if out.get("degraded"):
        flags.append(f"degraded: {out['degraded']}")
    if out.get("screened") is False:
        flags.append("not screened for prompt injection")
    if out.get("dense") is False:
        flags.append("keyword search only")
    if flags:
        print("(" + "; ".join(flags) + ")")


def show_answer(out: dict) -> None:
    print(out.get("answer", ""))
    if out.get("citations"):
        print("citations: " + ", ".join(f"chunk {c}" for c in out["citations"]))


def show_forget(out: dict) -> None:
    f = out.get("forgotten", {})
    print(f"forgot {f.get('episodes', 0)} episode(s), {f.get('chunks', 0)} chunk(s) in {out.get('namespace')}")
    if f.get("purged"):
        print(f"text erased now for {f['purged']} episode(s)")
    elif out.get("restorable_until"):
        print(f"restorable until {out['restorable_until']} (then the text is erased)")
    elif out.get("restorable"):
        print("archived: restore it any time (brain restore ...)")
    if out.get("backups"):
        print(out["backups"])


def show_directives(out: dict) -> None:
    ds = out.get("directives", [])
    if not ds:
        print("no directives")
    for d in ds:
        state = "" if d.get("status", "confirmed") == "confirmed" else f" {d['status'].upper()}"
        print(f"[{d['id']}]{state} ({d.get('scope', 'all')}) {d['text']}")


def show_one_directive(out: dict) -> None:
    d = out.get("directive", {})
    state = ("retired" if out.get("retired") else "rejected" if out.get("rejected")
             else "confirmed" if out.get("confirmed") else "pending" if out.get("pending") else "pinned")
    print(f"{state} directive {d.get('id')}: {d.get('text', '')}")
    if out.get("note"):
        print(out["note"])
    s = d.get("screen")
    if state == "pending":
        print("screen: " + (f"injection {s['injection']}, works against the user {s['subverts_user']}" if s
                            else "not screened (Jev was not available)"))


def show_session_start(out: dict) -> None:
    print(out.get("prompt") or "(no directives yet)")


def show_status(out: dict) -> None:
    print(f"brain {out.get('version')} as {out.get('client')} (default namespace: {out.get('default_ns') or '-'})")
    print(f"offline: {out.get('offline')}; spent today ${out.get('spent_today_usd') or 0:.4f} "
          f"of ${out.get('daily_cap_usd')}")
    keys = out.get("keys", {})
    print("keys: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in keys.items()))
    print("built: " + ", ".join(out.get("built", [])))


def show_client(out: dict) -> None:
    c = out.get("client", {})
    print(f"client {c.get('name')}: read {c.get('read_ns')}, write {c.get('write_ns')}, "
          f"default {c.get('default_ns')}, admin {c.get('is_admin')}"
          + (f", revoked {c['revoked_at']}" if c.get("revoked_at") else ""))
    if out.get("key"):
        print(f"key (shown once, store it now): {out['key']}")


def show_clients(out: dict) -> None:
    for c in out.get("clients", []):
        show_client({"client": c})


def show_account(out: dict) -> None:
    a = out.get("account", {})
    print(f"account {a.get('name')} ({a.get('kind')})" + (f", disabled {a['disabled_at']}" if a.get("disabled_at")
                                                          else ""))
    if out.get("client"):
        show_client(out)


def show_accounts(out: dict) -> None:
    for a in out.get("accounts", []):
        print(f"{a['name']:<20} {a['kind']:<10} keys {a.get('active_keys', 0)}"
              + (f"  disabled {a['disabled_at']}" if a.get("disabled_at") else ""))


def show_spend(out: dict) -> None:
    for d in out.get("days", []):
        print(f"{d.get('day')} {d.get('provider'):<10} ${d.get('usd', 0):.4f}")
    print(f"total ${out.get('total_usd', 0):.4f}; today ${out.get('today_usd', 0):.4f} of ${out.get('daily_cap_usd')}")


def show_restore(out: dict) -> None:
    r = out.get("restored", [])
    print(f"restored {len(r)} episode(s) in {out.get('namespace')}")
    if out.get("not_restorable_erased"):
        print(f"already erased, cannot restore: {out['not_restorable_erased']}")


# ---------------------------------------------------------------- commands

def _read_text(args) -> tuple[str, dict]:
    # NUL characters (common in text extracted from PDFs) cannot be stored; the API refuses them, so drop them here.
    if args.file:
        path = Path(args.file)
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", ""), {"filename": path.name}
    if args.text:
        return " ".join(args.text), {}
    if sys.stdin is None or sys.stdin.isatty():
        raise SystemExit("capture: give text, --file or stdin")
    return sys.stdin.read().replace("\x00", ""), {}


def cmd_capture(c: BrainClient, args) -> dict:
    text, meta = _read_text(args)
    if not text.strip():
        raise SystemExit("capture: the text is empty")
    role = args.role or ("attachment" if args.file else "user")
    channel = args.channel or ("document" if args.file else "chat")
    parts = pieces(text)
    episodes, deduped, degraded, ns, rid = [], 0, None, None, None
    for i, part in enumerate(parts):      # one request per piece keeps each body well under the 1 MB limit
        turn: dict[str, Any] = {"role": role, "text": part, "channel": channel}
        if args.speaker:
            turn["speaker"] = args.speaker
        if args.ts:
            turn["ts"] = args.ts
        m = dict(meta)
        if len(parts) > 1:
            m["part"] = f"{i + 1}/{len(parts)}"
        if m:
            turn["meta"] = m
        if args.idempotency_key:
            turn["idempotency_key"] = args.idempotency_key if len(parts) == 1 else f"{args.idempotency_key}:{i + 1}"
        out = c.capture({"namespace": args.namespace, "session_id": args.session, "turns": [turn]})
        episodes += out.get("episodes", [])
        deduped += out.get("deduped", 0)
        degraded = degraded or out.get("degraded")
        ns, rid = out.get("namespace"), out.get("request_id")
    return {"namespace": ns, "episodes": episodes, "deduped": deduped, "degraded": degraded, "request_id": rid}


def cmd_recall(c: BrainClient, args) -> dict:
    return c.recall({"namespace": args.namespace, "query": " ".join(args.query), "k": args.k,
                     "session_id": args.session, "time_from": args.time_from, "time_to": args.time_to,
                     "rerank": not args.no_rerank})


def cmd_answer(c: BrainClient, args) -> dict:
    return c.answer({"namespace": args.namespace, "question": " ".join(args.question),
                     "question_date": args.date, "style": args.style})


def cmd_forget(c: BrainClient, args) -> dict:
    if not args.episode and not args.session:
        raise SystemExit("forget: give --episode (repeatable) or --session")
    if args.purge and not args.confirm_erase:
        raise SystemExit("forget --purge erases the text for good and cannot be undone; add --confirm-erase only if "
                         "the user asked to delete permanently (plain forget can be restored)")
    return c.forget({"namespace": args.namespace, "episode_ids": args.episode or None, "session_id": args.session,
                     "purge": args.purge, "confirm_erase": args.confirm_erase})


def cmd_directives(c: BrainClient, args) -> dict:
    if args.action == "list":
        return c.directives(args.namespace)
    if args.action == "add":
        if not args.value:
            raise SystemExit("directives add: give the directive text")
        return c.add_directive({"namespace": args.namespace, "text": " ".join(args.value), "scope": args.scope})
    if len(args.value) != 1 or not args.value[0].isdigit():
        raise SystemExit("directives retire: give one directive id")
    return c.retire_directive(int(args.value[0]), args.namespace)


def cmd_session_start(c: BrainClient, args) -> dict:
    return c.session_start(args.namespace)


def cmd_export(c: BrainClient, args) -> None:
    out = open(args.out, "w", encoding="utf-8", newline="") if args.out else sys.stdout
    try:
        for piece in (c.export_account() if args.account else c.export(args.namespace, args.format)):
            out.write(piece)
    finally:
        if args.out:
            out.close()
    return None


def cmd_status(c: BrainClient, _args) -> dict:
    return c.status()


def cmd_admin(c: BrainClient, args) -> dict:
    if args.admin_cmd == "create-client":
        return c.create_client({"name": args.name, "read_ns": args.read or [], "write_ns": args.write or [],
                                "default_ns": args.default, "is_admin": args.is_admin, "account": args.account,
                                "speaks_as_user": args.speaks_as_user})
    if args.admin_cmd == "speaks-as-user":
        return c.set_speaks_as_user(args.name, not args.off, args.account)
    if args.admin_cmd == "confirm-directive":
        return c.confirm_directive(args.id, args.namespace, args.all_clients)
    if args.admin_cmd == "list":
        return c.list_clients(args.account)
    if args.admin_cmd == "revoke":
        return c.revoke_client(args.name, args.account)
    if args.admin_cmd == "create-account":
        return c.create_account({"name": args.name, "default_ns": args.default})
    if args.admin_cmd == "accounts":
        return c.list_accounts()
    if args.admin_cmd in ("disable-account", "enable-account"):
        return c.set_account_disabled(args.name, args.admin_cmd == "disable-account")
    if args.admin_cmd == "spend":
        return c.spend(args.days)
    if not args.episode and not args.session:
        raise SystemExit("admin restore: give --episode (repeatable) or --session")
    return c.admin_restore({"namespace": args.namespace, "episode_ids": args.episode or None,
                            "session_id": args.session})


SHOW: dict[str, Callable[[dict], None]] = {
    "capture": show_capture, "recall": show_recall, "answer": show_answer, "forget": show_forget,
    "session-start": show_session_start, "status": show_status,
    "create-client": show_client, "list": show_clients, "revoke": show_client, "spend": show_spend,
    "speaks-as-user": show_client, "confirm-directive": show_one_directive,
    "restore": show_restore, "create-account": show_account, "accounts": show_accounts,
    "disable-account": show_account, "enable-account": show_account,
}


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m brain.cli", description="Brain-Memory command line client.")
    p.add_argument("--url", help="API base URL (default: BRAIN_URL, else http://localhost:8000)")
    p.add_argument("--key", help="API key (default: BRAIN_KEY); prefer the env var so it stays out of shell history")
    p.add_argument("--json", action="store_true", help="print the raw JSON reply")
    sub = p.add_subparsers(dest="cmd", required=True)

    def ns(sp):
        sp.add_argument("--namespace", "-n", help="namespace (default: the key's default namespace)")

    sp = sub.add_parser("capture", help="save text: an argument, --file or stdin")
    sp.add_argument("text", nargs="*")
    sp.add_argument("--file", "-f", help="read a text file (role attachment, channel document)")
    sp.add_argument("--role", choices=ROLES)
    sp.add_argument("--channel", choices=CHANNELS)
    sp.add_argument("--speaker")
    sp.add_argument("--session", help="session id")
    sp.add_argument("--ts", help="when it was said, ISO 8601 (default now)")
    sp.add_argument("--idempotency-key")
    ns(sp)
    sp.set_defaults(fn=cmd_capture)

    sp = sub.add_parser("recall", help="search memory")
    sp.add_argument("query", nargs="+")
    sp.add_argument("-k", type=int, default=10)
    sp.add_argument("--session")
    sp.add_argument("--from", dest="time_from", help="ISO 8601")
    sp.add_argument("--to", dest="time_to", help="ISO 8601")
    sp.add_argument("--no-rerank", action="store_true")
    ns(sp)
    sp.set_defaults(fn=cmd_recall)

    sp = sub.add_parser("answer", help="answer a question from memory")
    sp.add_argument("question", nargs="+")
    sp.add_argument("--date", help="the question's date, YYYY-MM-DD (default today)")
    sp.add_argument("--style", choices=["auto", "default", "preference"], default="auto")
    ns(sp)
    sp.set_defaults(fn=cmd_answer)

    sp = sub.add_parser("forget", help="forget episodes or a session")
    sp.add_argument("--episode", type=int, action="append")
    sp.add_argument("--session")
    sp.add_argument("--purge", action="store_true", help="erase the text permanently now (cannot be undone); without it, forget archives and restore undoes it")
    sp.add_argument("--confirm-erase", action="store_true", help="required with --purge: the user asked to delete permanently")
    ns(sp)
    sp.set_defaults(fn=cmd_forget)

    sp = sub.add_parser("directives", help="list, add or retire standing directives")
    sp.add_argument("action", choices=["list", "add", "retire"])
    sp.add_argument("value", nargs="*", help="add: the text; retire: the id")
    sp.add_argument("--scope", default="all", help="add: 'all' or a client name")
    ns(sp)
    sp.set_defaults(fn=cmd_directives)

    sp = sub.add_parser("session-start", help="print the session-start prompt")
    ns(sp)
    sp.set_defaults(fn=cmd_session_start)

    sp = sub.add_parser("export", help="export a namespace (streams to stdout or --out)")
    sp.add_argument("--format", choices=["markdown", "jsonl"], default="markdown")
    sp.add_argument("--account", action="store_true",
                    help="the whole account as JSON lines: every namespace's turns, claims, facts and directives (admin key)")
    sp.add_argument("--out", "-o")
    ns(sp)
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("status", help="version, keys present, spend, built features")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("admin", help="admin: client keys, restore; operator: accounts, spend")
    asub = sp.add_subparsers(dest="admin_cmd", required=True)
    a = asub.add_parser("create-client")
    a.add_argument("name")
    a.add_argument("--read", action="append", help="namespace or '*' (repeatable)")
    a.add_argument("--write", action="append", help="namespace or '*' (repeatable)")
    a.add_argument("--default")
    a.add_argument("--admin", dest="is_admin", action="store_true")
    a.add_argument("--speaks-as-user", action="store_true",
                   help="this client's role=user turns are the user's own words (default: stored as agent_reported)")
    a.add_argument("--account", help="operator: create the key in this account")
    a = asub.add_parser("list")
    a.add_argument("--account", help="operator: list this account's keys")
    a = asub.add_parser("revoke")
    a.add_argument("name")
    a.add_argument("--account", help="operator: revoke in this account")
    a = asub.add_parser("speaks-as-user", help="let a client store role=user turns as the user's own words")
    a.add_argument("name")
    a.add_argument("--off", action="store_true", help="withdraw it")
    a.add_argument("--account")
    a = asub.add_parser("confirm-directive", help="confirm a pending directive (the user's decision, not a model's)")
    a.add_argument("id", type=int)
    a.add_argument("--all-clients", action="store_true", help="serve it to every client, not only the one that proposed it")
    ns(a)
    a = asub.add_parser("create-account", help="operator: new individual account and its admin key")
    a.add_argument("name")
    a.add_argument("--default", help="the admin key's default namespace (default: the account name)")
    asub.add_parser("accounts", help="operator: list accounts")
    for cmd in ("disable-account", "enable-account"):
        a = asub.add_parser(cmd)
        a.add_argument("name")
    a = asub.add_parser("spend")
    a.add_argument("--days", type=int, default=7)
    a = asub.add_parser("restore", help="undo a forget within the grace period")
    a.add_argument("--episode", type=int, action="append")
    a.add_argument("--session")
    ns(a)
    sp.set_defaults(fn=cmd_admin)
    return p


def main(argv: list[str] | None = None, client: BrainClient | None = None) -> int:
    args = build_parser().parse_args(argv)
    c = client or BrainClient(url=args.url, key=args.key)
    try:
        out = args.fn(c, args)
    except BrainError as e:
        msg = {"error": e.code, "message": e.message, "status": e.status, "request_id": e.request_id}
        if args.json:
            print(json.dumps(msg), file=sys.stderr)
        else:
            print(f"error {e.status} {e.code}: {e.message}" + (f" ({e.request_id})" if e.request_id else ""),
                  file=sys.stderr)
        return 1
    finally:
        if client is None:
            c.close()
    if out is None:
        return 0
    if args.json:
        _print_json(out)
    else:
        name = args.admin_cmd if args.cmd == "admin" else args.cmd
        if name == "directives":
            (show_directives if args.action == "list" else show_one_directive)(out)
        else:
            SHOW.get(name, _print_json)(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
