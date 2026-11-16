"""CLI (REFERENCE.md): commands against the in-process API through BrainClient with the
TestClient as its HTTP client. `pieces` runs offline; the rest needs TEST_DATABASE_URL."""
import io
import json

import pytest

from brain.cli import main, pieces
from brain.client import BrainClient
from conftest import ADMIN_KEY


def run(client, capsys, *argv, key=ADMIN_KEY):
    code = main(list(argv), client=BrainClient(key=key, http=client))
    out, err = capsys.readouterr()
    return code, out, err


def test_pieces_split_on_paragraphs_and_keep_all_text():
    assert pieces("short") == ["short"]
    paras = ["a" * 60, "b" * 60, "c" * 60]
    out = pieces("\n\n".join(paras), limit=130)
    assert out == ["a" * 60 + "\n\n" + "b" * 60, "c" * 60]
    long = pieces("x" * 250, limit=100)
    assert [len(p) for p in long] == [100, 100, 50]
    assert all(len(p) <= 100 for p in pieces(("word " * 30 + "\n\n") * 10, limit=100))


def test_capture_recall_answer(client, capsys):
    code, out, _ = run(client, capsys, "capture", "My sister lives in Austin.", "-n", "jeff", "--session", "s1")
    assert code == 0 and out.startswith("saved 1 turn(s) in jeff: episode 1")
    code, out, _ = run(client, capsys, "recall", "Where", "does", "my", "sister", "live?", "-n", "jeff")
    assert code == 0 and "[chunk 1 / episode 1]" in out and "Austin" in out
    code, out, _ = run(client, capsys, "--json", "recall", "sister", "-n", "jeff", "-k", "1")
    assert code == 0 and json.loads(out)["items"][0]["episode_id"] == 1
    code, out, _ = run(client, capsys, "answer", "Where does my sister live?", "-n", "jeff")
    assert code == 0 and "citations: chunk 1" in out


def test_capture_from_stdin_and_file(client, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO("Piped note about the garden."))
    code, out, _ = run(client, capsys, "--json", "capture", "-n", "jeff")
    assert code == 0 and len(json.loads(out)["episodes"]) == 1

    f = tmp_path / "lease.txt"
    f.write_text("Lease ends June 30.\n\n" + "Clause. " * 20000, encoding="utf-8")   # over one turn: 2 pieces
    code, out, _ = run(client, capsys, "--json", "capture", "--file", str(f), "-n", "jeff")
    eps = json.loads(out)["episodes"]
    assert code == 0 and len(eps) == 2
    code, out, _ = run(client, capsys, "--json", "export", "--format", "jsonl", "-n", "jeff")
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    lease = [r for r in rows if r.get("meta", {}).get("filename") == "lease.txt"]
    assert [r["meta"]["part"] for r in lease] == ["1/2", "2/2"]
    assert all(r["role"] == "attachment" and r["channel"] == "document" for r in lease)


def test_admin_create_client_then_use_the_key(client, capsys):
    code, out, _ = run(client, capsys, "admin", "create-client", "claude-code", "--read", "jeff", "--write", "jeff",
                       "--default", "jeff")
    assert code == 0 and "client claude-code" in out
    key = out.split("store it now): ")[1].strip()
    assert key.startswith("bk_")
    code, out, _ = run(client, capsys, "capture", "The boat is called Juniper.", key=key)
    assert code == 0 and "in jeff" in out
    code, out, _ = run(client, capsys, "status", key=key)
    assert code == 0 and "as claude-code" in out and key not in out
    code, _, err = run(client, capsys, "recall", "boat", "-n", "work", key=key)
    assert code == 1 and "403 forbidden" in err
    code, _, err = run(client, capsys, "admin", "list", key=key)
    assert code == 1 and "admin key required" in err
    code, out, _ = run(client, capsys, "admin", "list")
    assert code == 0 and "client claude-code" in out and key not in out
    code, out, _ = run(client, capsys, "admin", "revoke", "claude-code")
    assert code == 0 and "revoked" in out
    code, _, err = run(client, capsys, "status", key=key)
    assert code == 1 and "401 unauthorized" in err


def test_forget_restore_directives_session_start(client, capsys):
    run(client, capsys, "capture", "Temporary note.", "-n", "jeff")
    code, out, _ = run(client, capsys, "forget", "--episode", "1", "-n", "jeff")
    assert code == 0 and "forgot 1 episode(s)" in out and "archived: restore it any time" in out
    code, out, _ = run(client, capsys, "recall", "temporary note", "-n", "jeff")
    assert "no matches" in out
    code, out, _ = run(client, capsys, "admin", "restore", "--episode", "1", "-n", "jeff")
    assert code == 0 and "restored 1 episode(s)" in out
    code, out, _ = run(client, capsys, "recall", "temporary note", "-n", "jeff")
    assert "Temporary note." in out

    code, out, _ = run(client, capsys, "directives", "add", "Use", "metric", "units.", "-n", "jeff")
    assert code == 0 and out.startswith("pending directive 1") and "confirm-directive" in out
    assert "screen: injection" in out
    code, out, _ = run(client, capsys, "directives", "list", "-n", "jeff")
    assert "[1] PENDING (admin) Use metric units." in out
    code, out, _ = run(client, capsys, "session-start", "-n", "jeff")
    assert "Use metric units." not in out                       # pending: served to nobody
    code, out, _ = run(client, capsys, "admin", "confirm-directive", "1", "--all-clients", "-n", "jeff")
    assert code == 0 and out.startswith("confirmed directive 1")
    code, out, _ = run(client, capsys, "directives", "list", "-n", "jeff")
    assert "[1] (all) Use metric units." in out
    code, out, _ = run(client, capsys, "session-start", "-n", "jeff")
    assert "Use metric units." in out
    code, out, _ = run(client, capsys, "directives", "retire", "1", "-n", "jeff")
    assert code == 0 and out.startswith("retired directive 1")


def test_export_to_file_and_usage_errors(client, capsys, tmp_path):
    run(client, capsys, "capture", "Exported line.", "-n", "jeff", "--session", "s9")
    dest = tmp_path / "brain.md"
    code, out, _ = run(client, capsys, "export", "-n", "jeff", "--out", str(dest))
    assert code == 0 and out == "" and "Exported line." in dest.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        run(client, capsys, "forget", "-n", "jeff")          # needs --episode or --session
    with pytest.raises(SystemExit):
        run(client, capsys, "forget", "--episode", "1", "--purge", "-n", "jeff")   # erase needs --confirm-erase
    code, _, err = run(client, capsys, "--json", "recall", "x", "-n", "Bad NS")
    assert code == 1 and json.loads(err)["error"] == "bad_namespace"


def test_unreachable_api_is_a_clean_error(capsys):
    code = main(["status"], client=BrainClient(url="http://127.0.0.1:9", key="bk_secret_should_not_print"))
    _, err = capsys.readouterr()
    assert code == 1 and "unreachable" in err and "bk_secret_should_not_print" not in err


def test_operator_account_commands(client, capsys):
    code, out, _ = run(client, capsys, "admin", "create-account", "alice")
    assert code == 0 and "account alice (individual)" in out and "client account-admin" in out
    key = out.split("store it now): ")[1].strip()
    code, out, _ = run(client, capsys, "status", key=key)
    assert code == 0 and "as account-admin" in out
    code, out, _ = run(client, capsys, "admin", "accounts")
    assert code == 0 and "alice" in out and "owner" in out
    code, _, err = run(client, capsys, "admin", "accounts", key=key)
    assert code == 1 and "operator key required" in err
    code, out, _ = run(client, capsys, "admin", "create-client", "phone", "--read", "alice", "--account", "alice")
    assert code == 0 and "client phone" in out
    code, out, _ = run(client, capsys, "admin", "list", key=key)
    assert "client phone" in out and "client account-admin" in out
    code, out, _ = run(client, capsys, "admin", "disable-account", "alice")
    assert code == 0 and "disabled" in out
    code, _, err = run(client, capsys, "status", key=key)
    assert code == 1 and "account_disabled" in err


def test_capture_file_drops_nul_characters(client, capsys, tmp_path):
    f = tmp_path / "extracted.txt"
    f.write_text("Section 2.1\u0000 Lemma: every bounded sequence has a convergent subsequence.", encoding="utf-8")
    code, out, _ = run(client, capsys, "--json", "capture", "--file", str(f), "-n", "jeff")
    assert code == 0 and len(json.loads(out)["episodes"]) == 1
