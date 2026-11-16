"""The pre-commit secret scan (scripts/secret_scan.py). Fake secrets are built by joining pieces, so this file
passes the scan itself. No database, no keys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import secret_scan  # noqa: E402

FILL = "a1B2c3D4e5F6g7H8i9J0" * 3
FAKES = {
    "Anthropic API key": "sk-" + "ant-" + FILL,
    "Voyage API key": "VOYAGE = 'pa" + "-" + FILL + "'",
    "GitHub token": "gh" + "p_" + FILL,
    "AWS access key": "AK" + "IA" + "ABCDEFGHIJKLMNOP",
    "Supabase or JWT token": "ey" + "J" + FILL[:20] + "." + FILL[:20] + "." + FILL[:20],
    "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "database URL with a password": "postgres" + "ql://postgres.ref:" + FILL[:16] + "@db.example.com:5432/postgres",
    "assigned secret": "JEV_API" + "_KEY=" + FILL,
}


def test_each_kind_of_secret_is_found():
    for kind, line in FAKES.items():
        assert kind in secret_scan.scan_line(line), kind


def test_ordinary_lines_placeholders_and_allowed_lines_pass():
    clean = [
        'ADMIN_KEY = "test-admin-key-" + "k" * 40',
        "key = os.environ.get('ANTHROPIC_API_KEY')",
        "DATABASE_URL=postgresql://user:password@localhost:5432/brain",
        "BRAIN_ADMIN_KEY=<your-admin-key-goes-here-and-is-long>",
        "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}",
        "the token budget is 30000 and the password rule is unchanged",
        "export TEST_DATABASE_URL=\"$(grep '^TEST_DATABASE_URL=' ../../.env.deploy | cut -d= -f2-)\"",
    ]
    for line in clean:
        assert secret_scan.scan_line(line) == [], line
    assert secret_scan.scan_line(FAKES["Anthropic API key"] + "  # " + secret_scan.ALLOW) == []


def test_only_added_lines_of_a_diff_are_scanned_and_reported_without_the_secret():
    secret = FAKES["Anthropic API key"]
    diff = "\n".join([
        "diff --git a/service/brain/x.py b/service/brain/x.py",
        "--- a/service/brain/x.py",
        "+++ b/service/brain/x.py",
        "@@ -10,1 +10,3 @@",
        "-old = '" + secret + "'",
        "+import os",
        "+KEY = '" + secret + "'",
        "+n = 1",
        "diff --git a/gone.py b/gone.py",
        "--- a/gone.py",
        "+++ /dev/null",
        "@@ -1,1 +0,0 @@",
        "-" + secret,
    ])
    found = secret_scan.scan_diff(diff)
    assert found == [("service/brain/x.py", 11, "Anthropic API key")]
    assert all(secret not in str(f) for f in found)


def test_a_staged_env_file_is_refused_but_the_example_is_not():
    def staged(path):
        return secret_scan.scan_diff(f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,1 @@\n+BRAIN_OFFLINE=1\n")
    assert staged(".env") == [(".env", 0, "an env file is staged")]
    assert staged("service/.env.deploy")[0][2] == "an env file is staged"
    assert staged("service/.env.example") == []
    assert secret_scan.scan_text("notes.md", "fine\n" + FAKES["private key"] + "\n") == [("notes.md", 2, "private key")]


def test_the_scan_passes_on_its_own_sources():
    here = Path(__file__).resolve()
    for p in (here, here.parents[1] / "scripts" / "secret_scan.py"):
        assert secret_scan.scan_text(p.name, p.read_text(encoding="utf-8")) == [], p.name
