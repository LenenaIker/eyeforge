"""Minimal client for Nebius Token Factory (OpenAI-compatible chat API), stdlib only.

It runs in every runtime (including Isaac Sim's Python) without extra packages. Every call is appended
to a usage log (model, tokens, latency, purpose) so a job can report where Token Factory was used.

Configuration (environment, or the repo's .env file for the key):
    NEBIUS_API_KEY           Token Factory API key
    SIM2CHIP_LLM_BASE_URL    default https://api.tokenfactory.nebius.com/v1
    SIM2CHIP_TEXT_MODEL      model for text tasks (interview); default: a Nemotron text model
    SIM2CHIP_VISION_MODEL    model for image tasks (asset QA, labeling); default: a Nemotron omni/VL model
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .workspace import REPO

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
# Preferred model families per role, matched against the live /models list (ids change over time).
PREFERENCES = {
    "text": [r"nemotron-3-super", r"nemotron-3.*nano-30b", r"nemotron"],
    "vision": [r"nemotron.*omni", r"nemotron.*vl", r"nemotron.*nano.*v"],
}


class LLMError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("NEBIUS_API_KEY", "")
    env_file = REPO / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("NEBIUS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise LLMError("NEBIUS_API_KEY is not set (environment or .env)")
    return key


def image_part(path: Path | str, detail: str = "auto") -> dict:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}", "detail": detail}}


def parse_json(text: str):
    """JSON from a model answer: drops reasoning blocks and code fences, takes the outermost object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON in answer: {text[:200]!r}")
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start:end + 1])


@dataclass
class Client:
    base_url: str = field(default_factory=lambda: os.environ.get("SIM2CHIP_LLM_BASE_URL", DEFAULT_BASE_URL))
    key: str | None = None
    usage_log: Path | None = None
    timeout: float = 120.0
    retries: int = 4
    _models: list[str] | None = None

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = self.base_url.rstrip("/") + path
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.key or api_key()}", "Content-Type": "application/json"}
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise LLMError(f"{method} {path}: HTTP {e.code} {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise LLMError(f"{method} {path}: {e}") from None
        raise AssertionError("unreachable")

    def models(self) -> list[str]:
        if self._models is None:
            self._models = sorted(m["id"] for m in self._request("GET", "/models").get("data", []))
        return self._models

    def model_for(self, role: str) -> str:
        env = os.environ.get(f"SIM2CHIP_{role.upper()}_MODEL")
        if env:
            return env
        ids = self.models()
        for pattern in PREFERENCES[role]:
            hits = [m for m in ids if re.search(pattern, m, flags=re.I)]
            if hits:
                return hits[0]
        raise LLMError(f"no {role} model found among {len(ids)} models; set SIM2CHIP_{role.upper()}_MODEL")

    def chat(self, messages: list[dict], model: str, purpose: str, temperature: float = 0.2,
             max_tokens: int = 2048, json_mode: bool = False) -> str:
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        resp = self._request("POST", "/chat/completions", body)
        latency = time.time() - t0
        try:
            text = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise LLMError(f"unexpected answer: {str(resp)[:300]}") from None
        self._log(model, purpose, resp.get("usage", {}), latency)
        return text

    def _log(self, model: str, purpose: str, usage: dict, latency: float) -> None:
        if self.usage_log is None:
            return
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        rec = {"t": round(time.time(), 3), "model": model, "purpose": purpose, "latency_s": round(latency, 3),
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        with open(self.usage_log, "a") as f:
            f.write(json.dumps(rec) + "\n")


def usage_summary(log: Path) -> dict:
    """Totals per purpose from a usage log."""
    out: dict[str, dict] = {}
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        r = json.loads(line)
        s = out.setdefault(r["purpose"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
                                          "models": []})
        s["calls"] += 1
        s["prompt_tokens"] += r.get("prompt_tokens") or 0
        s["completion_tokens"] += r.get("completion_tokens") or 0
        s["latency_s"] = round(s["latency_s"] + r["latency_s"], 2)
        if r["model"] not in s["models"]:
            s["models"].append(r["model"])
    return out
