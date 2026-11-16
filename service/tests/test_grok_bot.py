"""scripts/grok_bot.py: a second model on the same memory. No keys and no network: the HTTP call is replaced."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import grok_bot  # noqa: E402

CFG = {"XAI_API_KEY": "xai-test", "BRAIN_URL": "https://brain.example.com/", "BRAIN_KEY": "bk_test"}


def test_settings_come_from_the_file_and_the_environment_wins(tmp_path):
    f = tmp_path / "grok-bot.env"
    f.write_text("# mine\nBRAIN_URL=https://from-file\nBRAIN_KEY='bk_file'\n", encoding="utf-8")
    cfg = grok_bot.read_config(f, {"BRAIN_KEY": "bk_env", "GROK_MODEL": "grok-x"})
    assert cfg == {"BRAIN_URL": "https://from-file", "BRAIN_KEY": "bk_env", "GROK_MODEL": "grok-x"}
    assert grok_bot.missing(cfg) == ["XAI_API_KEY"]
    assert grok_bot.read_config(tmp_path / "none.env", {}) == {}


def test_the_brain_is_handed_to_grok_as_a_remote_mcp_server_with_its_own_key():
    body = grok_bot.request_body(CFG, [{"role": "user", "content": "hi"}])
    assert body["model"] == grok_bot.MODEL and body["input"] == [{"role": "user", "content": "hi"}]
    assert body["tools"] == [{"type": "mcp", "server_url": "https://brain.example.com/mcp", "server_label": "brain",
                              "server_description": "The user's long-term memory, shared with their other assistants.",
                              "headers": {"Authorization": "Bearer bk_test"}}]
    assert "memory_session_start" in body["instructions"] and "not instructions" in body["instructions"]
    assert "xai-test" not in str(body)                      # the xAI key travels only in the request's own header


def test_a_turn_sends_the_whole_conversation_and_shows_the_reply_and_the_memory_tools_used():
    sent, said = [], []
    replies = iter([
        {"output": [{"type": "mcp_call", "name": "memory_context"},
                    {"type": "message", "content": [{"type": "output_text", "text": "She lives in Austin."}]}]},
        {"output_text": "You're welcome."},
    ])

    def post(url, headers, body):
        sent.append((url, headers, body))
        return next(replies)

    history = []
    assert grok_bot.turn(CFG, history, "Where does my sister live?", post, said.append) == "She lives in Austin."
    assert said == ["  [memory: memory_context]", "She lives in Austin."]
    assert grok_bot.turn(CFG, history, "Thanks", post, said.append) == "You're welcome."
    url, headers, body = sent[1]
    assert url == "https://api.x.ai/v1/responses" and headers["Authorization"] == "Bearer xai-test"
    assert body["input"] == [{"role": "user", "content": "Where does my sister live?"},
                             {"role": "assistant", "content": "She lives in Austin."},
                             {"role": "user", "content": "Thanks"}]


def test_with_a_setting_missing_the_bot_names_it_and_calls_nothing(tmp_path, monkeypatch):
    for k in grok_bot.NEEDED:
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "grok-bot.env"
    f.write_text("BRAIN_URL=https://brain.example.com\nBRAIN_KEY=bk_test\n", encoding="utf-8")
    said = []

    def post(*a):
        raise AssertionError("nothing may be sent")

    assert grok_bot.main(["--config", str(f), "--once", "hi"], post=post, out=said.append) == 2
    assert "Not set: XAI_API_KEY" in said[0] and "bk_test" not in said[0]


def test_once_sends_one_message_and_exits(tmp_path, monkeypatch):
    for k in grok_bot.NEEDED:
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "grok-bot.env"
    f.write_text("XAI_API_KEY=xai-test\nBRAIN_URL=https://brain.example.com\nBRAIN_KEY=bk_test\n", encoding="utf-8")
    said = []
    assert grok_bot.main(["--config", str(f), "--once", "hi"], post=lambda *a: {"output_text": "hello"}, out=said.append) == 0
    assert said == ["hello"]
