"""Secret scan for the pre-commit hook: refuse a commit whose added lines look like a key or a database URL.

    python service/scripts/secret_scan.py            # scan what is staged (git diff --cached)
    python service/scripts/secret_scan.py FILE ...   # scan whole files

Prints the file, the line number and the kind of secret; never the text that matched. Exit 1 when something is
found. A line that is a known false positive can end with the marker `secret-scan: allow`.
Also refuses a staged `.env` file (any `.env*` except `.env.example`).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import PurePosixPath

ALLOW = "secret-scan: allow"

PATTERNS = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("Voyage API key", re.compile(r"\bpa-[A-Za-z0-9_\-]{30,}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Supabase or JWT token", re.compile(r"\beyJ[A-Za-z0-9_\-]{15,}\.[A-Za-z0-9_\-]{15,}\.[A-Za-z0-9_\-]{10,}")),
    ("Supabase secret key", re.compile(r"\bsb_secret_[A-Za-z0-9_\-]{20,}")),
    ("Railway or bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{30,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("database URL with a password", re.compile(r"(?i)\bpostgres(?:ql)?://[^\s:/@'\"]+:[^\s@'\"]{6,}@")),
    ("assigned secret", re.compile(
        r"(?i)\b[A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD|ADMIN_KEY)[A-Z0-9_]*\s*[=:]\s*['\"]?[A-Za-z0-9_\-\.]{24,}")),
]
# Values that are plainly not secrets: placeholders, and names that only refer to an environment variable.
PLACEHOLDER = re.compile(r"(?i)(?:example|placeholder|changeme|your[_-]|xxxx|<[^>]+>|\$\{?[A-Z_]+\}?|os\.environ|"
                         r"user:pass(?:word)?@|:password@|:\*+@)")


def scan_line(line: str) -> list[str]:
    """The kinds of secret one line looks like (empty when clean)."""
    if ALLOW in line:
        return []
    found = []
    for kind, pattern in PATTERNS:
        m = pattern.search(line)
        if m and not PLACEHOLDER.search(m.group(0)):
            found.append(kind)
    return found


def is_env_file(path: str) -> bool:
    name = PurePosixPath(path.replace("\\", "/")).name
    return name.startswith(".env") and name != ".env.example"


def scan_diff(diff: str) -> list[tuple[str, int, str]]:
    """Findings in the added lines of a unified diff (git diff -U0): (file, line number, kind)."""
    out, path, lineno = [], None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            if path and is_env_file(path):
                out.append((path, 0, "an env file is staged"))
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
        elif raw.startswith("+") and path:
            out += [(path, lineno, kind) for kind in scan_line(raw[1:])]
            lineno += 1
    return out


def scan_text(path: str, text: str) -> list[tuple[str, int, str]]:
    return [(path, i, kind) for i, line in enumerate(text.splitlines(), 1) for kind in scan_line(line)]


def main(argv: list[str]) -> int:
    if argv:
        found = []
        for p in argv:
            with open(p, encoding="utf-8", errors="replace") as f:
                found += scan_text(p, f.read())
    else:
        diff = subprocess.run(["git", "diff", "--cached", "-U0", "--no-color", "--no-ext-diff"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
        found = scan_diff(diff)
    for path, lineno, kind in found:
        print(f"secret scan: {path}:{lineno}: looks like {kind}")
    if found:
        print(f"secret scan: {len(found)} finding(s). Remove them, or end a false positive's line with "
              f"'{ALLOW}'.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
