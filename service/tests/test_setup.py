"""`python -m brain.setup`: from a fresh clone to a connected client. File handling runs offline; the run against
the API needs TEST_DATABASE_URL (the in-process API through the TestClient)."""
import json

import pytest

from brain import setup
from conftest import ADMIN_KEY, bearer


def test_first_run_writes_an_operator_key_and_trial_mode_and_a_second_run_changes_nothing(tmp_path):
    env = tmp_path / ".env"
    measured = [m.split("=")[0] for m in setup.MEASURED_ON]
    assert setup.ensure_env(env) == ["BRAIN_ADMIN_KEY", "BRAIN_OFFLINE", *measured]
    first = env.read_text(encoding="utf-8")
    have = setup.read_env(env)
    assert len(have["BRAIN_ADMIN_KEY"]) >= 32 and have["BRAIN_OFFLINE"] == "1"
    # a new install gets the settings the published results were measured with, and each is a real setting
    assert have["WRITE_PATH_ENABLED"] == "1" and have["BRAIN_LEDGER_ROUTE"] == "1" and have["BRAIN_SHORTLIST_SEATS"] == "25,5"
    import pathlib
    code = "".join(p.read_text(encoding="utf-8") for p in (pathlib.Path(setup.__file__).parent).glob("*.py") if p.name != "setup.py")
    assert all(f'"{name}"' in code for name in measured), [n for n in measured if f'"{n}"' not in code]
    assert setup.ensure_env(env) == [] and env.read_text(encoding="utf-8") == first


def test_setup_never_replaces_what_env_already_holds_and_stays_live_when_a_provider_key_is_set(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# mine\nBRAIN_ADMIN_KEY=already-chosen\nVOYAGE_API_KEY=pa-x", encoding="utf-8")     # no newline at the end
    assert setup.ensure_env(env) == []                      # a provider key is set: no trial mode
    assert env.read_text(encoding="utf-8") == "# mine\nBRAIN_ADMIN_KEY=already-chosen\nVOYAGE_API_KEY=pa-x"
    env.write_text("VOYAGE_API_KEY=pa-x", encoding="utf-8")
    assert setup.ensure_env(env) == ["BRAIN_ADMIN_KEY"]
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "VOYAGE_API_KEY=pa-x" and lines[1].startswith("BRAIN_ADMIN_KEY=") and len(lines) == 2
    env.write_text("BRAIN_OFFLINE=0\n", encoding="utf-8")   # the user turned trial mode off: left as it is
    assert setup.ensure_env(env) == ["BRAIN_ADMIN_KEY"] and setup.read_env(env)["BRAIN_OFFLINE"] == "0"


def test_the_desktop_config_is_backed_up_and_a_server_of_the_same_name_is_never_replaced(tmp_path):
    cfg = tmp_path / "Claude" / "claude_desktop_config.json"
    entry = setup.desktop_entry(tmp_path, "http://localhost:8000", "bk_test", python="py")
    assert entry == {"command": "py", "args": ["-m", "brain.mcp_stdio"],
                     "env": {"BRAIN_URL": "http://localhost:8000", "BRAIN_KEY": "bk_test", "PYTHONPATH": str(tmp_path)}}
    assert setup.add_to_desktop(cfg, entry) is None         # no config yet: created, nothing to back up
    assert json.loads(cfg.read_text(encoding="utf-8")) == {"mcpServers": {"brain": entry}}
    with pytest.raises(ValueError, match="already has a server named brain"):
        setup.add_to_desktop(cfg, entry)
    cfg.write_text(json.dumps({"mcpServers": {"files": {"command": "npx"}}, "preferences": {"x": 1}}), encoding="utf-8")
    before = cfg.read_text(encoding="utf-8")
    backup = setup.add_to_desktop(cfg, entry)
    assert backup.name == "claude_desktop_config.before-brain.json" and backup.read_text(encoding="utf-8") == before
    now = json.loads(cfg.read_text(encoding="utf-8"))
    assert now["preferences"] == {"x": 1} and set(now["mcpServers"]) == {"files", "brain"}


def test_the_desktop_config_is_found_on_each_platform(tmp_path):
    win = setup.desktop_config_path({"APPDATA": str(tmp_path)}, "win32")
    assert win == tmp_path / "Claude" / "claude_desktop_config.json"
    assert setup.desktop_config_path({}, "darwin").parts[-4:] == ("Library", "Application Support", "Claude", "claude_desktop_config.json")
    assert setup.desktop_config_path({"XDG_CONFIG_HOME": str(tmp_path)}, "linux") == tmp_path / "Claude" / "claude_desktop_config.json"


def test_with_the_service_down_setup_says_how_to_start_it(tmp_path):
    said = []
    assert setup.main(["--url", "http://127.0.0.1:9"], cwd=tmp_path, out=said.append) == 0
    text = "\n".join(said)
    assert "Wrote BRAIN_ADMIN_KEY, BRAIN_OFFLINE and the 13 measured settings to .env" in text
    assert "not running" in text and "docker compose up" in text
    assert setup.read_env(tmp_path / ".env")["BRAIN_ADMIN_KEY"] not in text       # the operator key is never printed


def test_with_the_service_up_setup_creates_a_working_client_key_and_connects_claude_desktop(client, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"BRAIN_ADMIN_KEY={ADMIN_KEY}\nBRAIN_OFFLINE=1\n", encoding="utf-8")
    cfg = tmp_path / "desktop" / "claude_desktop_config.json"
    monkeypatch.setattr(setup, "desktop_config_path", lambda: cfg)
    said = []
    assert setup.main(["--client", "claude", "--namespace", "me", "--desktop"], cwd=tmp_path, http=client, out=said.append) == 0
    text = "\n".join(said)
    assert "Created client claude for namespace me" in text and "claude mcp add --transport http brain" in text
    assert ADMIN_KEY not in text and "Restart Claude Desktop" in text
    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["brain"]
    key = entry["env"]["BRAIN_KEY"]
    assert key.startswith("bk_") and key in text and entry["env"]["PYTHONPATH"] == str(tmp_path)
    r = client.get("/v1/session_start", headers=bearer(key))                    # the key it made works, in its namespace
    assert r.status_code == 200 and r.json()["namespace"] == "me"
    said.clear()                                                                # a name is never reused
    assert setup.main(["--client", "claude"], cwd=tmp_path, http=client, out=said.append) == 1
    assert "already exists" in "\n".join(said)
    said.clear()                                                                # Desktop already has the server
    assert setup.main(["--client", "claude-2", "--desktop"], cwd=tmp_path, http=client, out=said.append) == 0
    assert "its config was not changed" in "\n".join(said) and '"brain"' in "\n".join(said)


def test_a_service_started_with_another_operator_key_is_explained(client, tmp_path):
    (tmp_path / ".env").write_text("BRAIN_ADMIN_KEY=not-the-key-the-service-has\n", encoding="utf-8")
    said = []
    assert setup.main([], cwd=tmp_path, http=client, out=said.append) == 1
    assert "did not accept the operator key" in "\n".join(said)


def test_files_that_hold_a_key_are_readable_by_their_owner_only(tmp_path):
    """.env holds the operator key, and Claude Desktop's config and its backup hold client keys: 0600, also when
    the file was there before with a wider mode. (On Windows the mode means nothing, so only the content is checked.)"""
    import os
    import stat
    import sys

    def mode(path):
        return stat.S_IMODE(os.stat(path).st_mode)

    env = tmp_path / ".env"
    env.write_text("VOYAGE_API_KEY=pa-x" + chr(10), encoding="utf-8")
    os.chmod(env, 0o644)
    assert setup.ensure_env(env) == ["BRAIN_ADMIN_KEY"]
    cfg = tmp_path / "claude_desktop_config.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"env": {"TOKEN": "t"}}}}), encoding="utf-8")
    os.chmod(cfg, 0o644)
    backup = setup.add_to_desktop(cfg, setup.desktop_entry(tmp_path, "http://localhost:8000", "bk_test"))
    assert json.loads(backup.read_text(encoding="utf-8"))["mcpServers"]["other"]["env"]["TOKEN"] == "t"
    assert set(json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]) == {"other", "brain"}
    if sys.platform != "win32":
        assert mode(env) == 0o600 and mode(cfg) == 0o600 and mode(backup) == 0o600


def test_a_env_file_reaches_the_process_and_never_overrides_what_is_already_set(tmp_path, monkeypatch):
    """`python -m brain.serve` and `python -m brain.worker` read a .env in their folder. A variable the host or the
    container already set wins."""
    from brain import settings
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nBRAIN_NEW_SETTING=from-file\nBRAIN_SET_BY_HOST=from-file\nQUOTED="a b"\nnot a setting\n',
                   encoding="utf-8")
    monkeypatch.setenv("BRAIN_SET_BY_HOST", "from-host")
    monkeypatch.delenv("BRAIN_NEW_SETTING", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    import os
    try:
        assert settings.load_env_file(env) == ["BRAIN_NEW_SETTING", "QUOTED"]
        assert os.environ["BRAIN_NEW_SETTING"] == "from-file" and os.environ["QUOTED"] == "a b"
        assert os.environ["BRAIN_SET_BY_HOST"] == "from-host"
    finally:
        os.environ.pop("BRAIN_NEW_SETTING", None)
        os.environ.pop("QUOTED", None)
    assert settings.load_env_file(tmp_path / "missing.env") == []


def test_an_existing_env_is_not_given_the_measured_settings(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=postgresql://example\n", encoding="utf-8")
    assert setup.ensure_env(env) == ["BRAIN_ADMIN_KEY", "BRAIN_OFFLINE"]
    assert "WRITE_PATH_ENABLED" not in setup.read_env(env)


def test_with_docker_claude_desktop_runs_the_proxy_in_a_container_of_the_same_image():
    """No Python on the machine: the entry starts a throwaway container, and the host's localhost is reached by
    the name a container has for it."""
    e = setup.docker_entry("http://localhost:8000", "bk_test")
    assert e["command"] == "docker" and e["args"][-4:] == ["brain-memory", "python", "-m", "brain.mcp_stdio"]
    assert e["args"][:3] == ["run", "-i", "--rm"] and "host.docker.internal:host-gateway" in e["args"]
    assert e["env"] == {"BRAIN_URL": "http://host.docker.internal:8000", "BRAIN_KEY": "bk_test"}
    assert "bk_test" not in " ".join(e["args"])                     # the key is passed by name, never on the command line
    assert setup.docker_entry("https://brain.example.com", "k")["env"]["BRAIN_URL"] == "https://brain.example.com"


def test_setup_run_by_docker_compose_talks_to_the_api_inside_and_prints_the_address_clients_use(client, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"BRAIN_ADMIN_KEY={ADMIN_KEY}\nBRAIN_OFFLINE=1\n", encoding="utf-8")
    monkeypatch.setenv("BRAIN_SETUP_DOCKER", "1")                   # what the Compose service sets
    monkeypatch.setenv("BRAIN_SETUP_URL", "http://api:8000")
    monkeypatch.setenv("BRAIN_SETUP_PUBLIC_URL", "http://localhost:8000")
    said = []
    assert setup.main(["--client", "claude", "--desktop"], cwd=tmp_path, http=client, out=said.append) == 0
    text = "\n".join(said)
    assert "brain http://localhost:8000/mcp" in text and "api:8000" not in text
    assert '"command": "docker"' in text and "http://host.docker.internal:8000" in text
    assert "cannot reach Claude Desktop's config" in text           # --desktop cannot work from inside a container


def test_with_the_service_down_docker_setup_names_only_the_docker_command(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_SETUP_DOCKER", "1")
    said = []
    assert setup.main(["--url", "http://127.0.0.1:9"], cwd=tmp_path, out=said.append) == 0
    text = "\n".join(said)
    assert "docker compose up -d --build" in text and "brain.serve" not in text and "BRAIN_OFFLINE=0" in text
    setup.give_to_folder_owner(tmp_path / ".env")                   # not root here: does nothing, raises nothing

