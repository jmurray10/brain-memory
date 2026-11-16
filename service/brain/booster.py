"""The booster (the booster plan): a small trained model in front of Jev at the brain's decision points.

The model (`your-account/brain-navigator-booster`: Qwen3-1.7B, fine-tuned on the navigator-v4 contract) runs on a
Hugging Face Inference Endpoint (text-generation-inference). For one decision it gets the task's system prompt and
the inputs as JSON, exactly as in training (non-thinking chat format, greedy), and answers with JSON. Its
confidence is the probability of the least certain generated token, and each task has a threshold below which
the decision goes to Jev instead (`booster_config.json` on the Hub, calibrated to 99% accuracy on familiar data).

Never on the critical path by surprise: every call has a short timeout, and any failure (cold start, error,
invalid JSON) is a deferral to Jev, not an error. Step 2 runs it in shadow mode only (scripts/booster_shadow.py).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import httpx

REPO = "your-account/brain-navigator-booster"
MODEL_RETRY_S = 60.0     # after /v1/models fails, ask again this much later (the endpoint may still be starting)
END = "<|im_end|>"


def chat_prompt(system: str, user: str) -> str:
    """Qwen3's chat template with enable_thinking=False and the generation prompt (the training string)."""
    return (f"<|im_start|>system\n{system}{END}\n<|im_start|>user\n{user}{END}\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


@dataclass(frozen=True)
class BoosterAnswer:
    task: str
    answer: dict | None        # parsed JSON, None if the call failed or the JSON was invalid
    confidence: float          # probability of the least certain generated token (0 on any failure)
    deferred: bool             # True: Jev decides (low confidence, failure or invalid answer)
    latency_ms: float
    error: str | None = None


class BoosterClient:
    def __init__(self, url: str, token: str, config: dict, http: httpx.Client | None = None,
                 timeout_s: float = 8.0):
        self.url = url.rstrip("/")
        self.config = config
        self.http = http or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))
        self._headers = {"Authorization": f"Bearer {token}"}

    @classmethod
    def from_hub(cls, url: str, token: str, **kw) -> "BoosterClient":
        r = httpx.get(f"https://huggingface.co/{REPO}/resolve/main/booster_config.json",
                      headers={"Authorization": f"Bearer {token}"}, follow_redirects=True, timeout=60)
        r.raise_for_status()
        return cls(url, token, r.json(), **kw)

    def threshold(self, task: str) -> float:
        return float(self.config["defer_below_confidence"][task])

    def _served_model(self) -> str | None:
        """vLLM's OpenAI-compatible server names the model; text-generation-inference has no /v1/models.

        Only a successful answer is kept for good. A failure (a cold endpoint, a timeout, a 5xx, a 404 from
        text-generation-inference) is not remembered as "no model": it is asked again after MODEL_RETRY_S, so one
        bad first call cannot send every later request to the wrong API for the life of the process."""
        model = getattr(self, "_model", None)
        if model:
            return model
        now = time.monotonic()
        if now < getattr(self, "_model_retry_at", 0.0):
            return None
        try:
            r = self.http.get(self.url + "/v1/models", headers=self._headers)
            model = r.json()["data"][0]["id"] if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            model = None
        if model:
            self._model = model
        else:
            self._model_retry_at = now + MODEL_RETRY_S
        return model

    def _generate(self, prompt: str, max_new_tokens: int) -> tuple[str, list[float], bool] | str:
        """(text, per-token logprobs including the end token, cut off) or an error string."""
        model = self._served_model()
        if model:      # vLLM: /v1/completions with logprobs; keep the stop token so it counts, as in training
            r = self.http.post(self.url + "/v1/completions", headers=self._headers, json={
                "model": model, "prompt": prompt, "max_tokens": max_new_tokens, "temperature": 0,
                "logprobs": 1, "stop": [END], "include_stop_str_in_output": True})
            if r.status_code != 200:
                return f"HTTP {r.status_code}"
            c = r.json()["choices"][0]
            lps = [lp for lp in (c.get("logprobs") or {}).get("token_logprobs") or [] if lp is not None]
            return c.get("text", ""), lps, c.get("finish_reason") == "length"
        r = self.http.post(self.url + "/generate", headers=self._headers, json={
            "inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "do_sample": False,
                                             "details": True, "stop": [END]}})
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        j = r.json()
        d = j.get("details") or {}
        return j.get("generated_text", ""), [t["logprob"] for t in d.get("tokens") or []], d.get("finish_reason") == "length"

    def ask(self, task: str, inputs: dict, max_new_tokens: int = 160) -> BoosterAnswer:
        if task not in self.config.get("served_tasks", []):
            return BoosterAnswer(task, None, 0.0, True, 0.0, "task not served")
        prompt = chat_prompt(self.config["system_prompts"][task], json.dumps(inputs, ensure_ascii=False))
        t0 = time.perf_counter()
        try:
            out = self._generate(prompt, max_new_tokens)
            ms = (time.perf_counter() - t0) * 1000
            if isinstance(out, str):
                return BoosterAnswer(task, None, 0.0, True, ms, out)
            raw, lps, cut = out
            text = raw.replace(END, "").strip()
            conf = math.exp(min(lps)) if lps else 0.0
            if cut:
                conf = 0.0                                      # cut off: as in training, no confidence
            try:
                ans = json.loads(text)
            except ValueError:
                return BoosterAnswer(task, None, 0.0, True, ms, "invalid JSON")
            return BoosterAnswer(task, ans, conf, conf < self.threshold(task), ms)
        except httpx.HTTPError as e:
            return BoosterAnswer(task, None, 0.0, True, (time.perf_counter() - t0) * 1000, type(e).__name__)

    def route(self, question: str, question_date: str, roster: str = "user (me)", slots: str = "") -> BoosterAnswer:
        """R2. Until claims exist (build steps 4-5) the brain has no roster or slot list to send."""
        return self.ask("route", {"question": question, "question_date": question_date, "roster": roster,
                                  "slots": slots})

    def close(self) -> None:
        self.http.close()
