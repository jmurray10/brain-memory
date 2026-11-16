"""Booster client (brain/booster.py): offline, mock servers for both vLLM and text-generation-inference."""
import json
import math

import httpx

from brain.booster import END, BoosterClient, chat_prompt

CONFIG = {"served_tasks": ["route"], "defer_below_confidence": {"route": 0.79},
          "system_prompts": {"route": "Task: choose the lookup route."}}
ROUTE = {"route": "when", "entity": "user", "slot": None, "time_from": None, "time_to": None, "as_of": None,
         "unknown_target": False}


def vllm(lps, text=None, finish="stop", status=200):
    seen = {}

    def handler(req):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "/repository"}]})
        seen["body"] = json.loads(req.content)
        if status != 200:
            return httpx.Response(status, json={"error": "x"})
        return httpx.Response(200, json={"choices": [{
            "text": text if text is not None else json.dumps(ROUTE) + END, "finish_reason": finish,
            "logprobs": {"token_logprobs": lps}}]})
    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def tgi(lps):
    def handler(req):
        if req.url.path == "/v1/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"generated_text": json.dumps(ROUTE), "details": {
            "finish_reason": "eos_token", "tokens": [{"logprob": lp} for lp in lps]}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_prompt_is_the_training_string():
    p = chat_prompt("SYS", "USER")
    assert p == ("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_vllm_confident_answer_is_kept_and_request_matches_training():
    http, seen = vllm([-0.01, -0.05, -0.02])
    a = BoosterClient("https://ep", "tok", CONFIG, http=http).route("When did I move?", "2026-01-02")
    assert a.answer == ROUTE and not a.deferred and a.error is None
    assert abs(a.confidence - math.exp(-0.05)) < 1e-9
    b = seen["body"]
    assert b["temperature"] == 0 and b["logprobs"] == 1 and b["include_stop_str_in_output"] is True
    user = b["prompt"].split("<|im_start|>user\n")[1].split(END)[0]
    assert json.loads(user) == {"question": "When did I move?", "question_date": "2026-01-02",
                                "roster": "user (me)", "slots": ""}


def test_low_confidence_cutoff_invalid_json_and_errors_all_defer():
    cases = [vllm([-0.01, -0.9])[0],                         # exp(-0.9) = 0.41 < 0.79
             vllm([-0.01], finish="length")[0],              # cut off: confidence 0
             vllm([-0.01], text="not json")[0],
             vllm([-0.01], status=503)[0]]
    for http in cases:
        a = BoosterClient("https://ep", "tok", CONFIG, http=http).route("q", "2026-01-02")
        assert a.deferred
    def boom(req):
        raise httpx.ConnectTimeout("cold start")
    a = BoosterClient("https://ep", "tok", CONFIG, http=httpx.Client(transport=httpx.MockTransport(boom))).route("q", "d")
    assert a.deferred and a.error == "ConnectTimeout" and a.answer is None


def test_tgi_backend_and_unserved_task():
    a = BoosterClient("https://ep", "tok", CONFIG, http=tgi([-0.001, -0.01])).route("q", "2026-01-02")
    assert a.answer == ROUTE and not a.deferred
    assert BoosterClient("https://ep", "tok", CONFIG, http=tgi([0.0])).ask("extract", {}).error == "task not served"


def test_served_model_keeps_only_a_successful_answer(monkeypatch):
    """A failed /v1/models (cold endpoint) is asked again after MODEL_RETRY_S, not remembered as "no model"."""
    from brain import booster
    state = {"models": 503, "calls": {"/v1/models": 0, "/v1/completions": 0, "/generate": 0}}

    def handler(req):
        state["calls"][req.url.path] += 1
        if req.url.path == "/v1/models":
            if state["models"] == "boom":
                raise httpx.ConnectError("down")
            return (httpx.Response(200, json={"data": [{"id": "/repository"}]}) if state["models"] == 200
                    else httpx.Response(state["models"]))
        if req.url.path == "/generate":
            return httpx.Response(404)
        return httpx.Response(200, json={"choices": [{"text": json.dumps(ROUTE) + END, "finish_reason": "stop",
                                                      "logprobs": {"token_logprobs": [-0.01]}}]})
    clock = {"t": 1000.0}
    monkeypatch.setattr(booster.time, "monotonic", lambda: clock["t"])
    c = BoosterClient("https://ep", "tok", CONFIG, http=httpx.Client(transport=httpx.MockTransport(handler)))

    a = c.route("When did I move?", "2026-01-02")            # cold: /v1/models 503, falls back, defers
    assert a.deferred and a.error == "HTTP 404" and state["calls"]["/v1/models"] == 1
    clock["t"] += booster.MODEL_RETRY_S - 1
    assert c.route("q", "2026-01-02").deferred and state["calls"]["/v1/models"] == 1      # inside the wait: not re-asked
    clock["t"] += 2
    state["models"] = "boom"                                  # a network error is a failure too
    assert c.route("q", "2026-01-02").deferred and state["calls"]["/v1/models"] == 2
    clock["t"] += booster.MODEL_RETRY_S + 1
    state["models"] = 200                                     # the endpoint is up: found, and kept
    b = c.route("q", "2026-01-02")
    assert not b.deferred and b.answer == ROUTE and state["calls"]["/v1/models"] == 3
    state["models"] = 503
    clock["t"] += 10 * booster.MODEL_RETRY_S
    assert not c.route("q", "2026-01-02").deferred and state["calls"]["/v1/models"] == 3  # a success is not re-asked
    assert state["calls"]["/v1/completions"] == 2
