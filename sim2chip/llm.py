"""Minimal client for Nebius Token Factory (OpenAI-compatible chat API), stdlib only.

It runs in every runtime (including Isaac Sim's Python) without extra packages. Every call is appended
to a usage log (model, tokens, latency, purpose) so a job can report where Token Factory was used.

Configuration (environment, or the repo's .env file for the key):
    NEBIUS_API_KEY           Token Factory API key
    SIM2CHIP_LLM_BASE_URL    default https://api.tokenfactory.nebius.com/v1
    SIM2CHIP_TEXT_MODEL      model for text tasks (interview); default: a Nemotron text model
    SIM2CHIP_VISION_MODEL    model for image tasks (asset QA, labeling); default: see PREFERENCES
    SIM2CHIP_DESCRIBE_MODEL  describer for the describe_then_decide QA strategy
    SIM2CHIP_SCENE_MODEL     model that looks at the user's photos during the interview (scene_profile)
    SIM2CHIP_COMPARE_MODEL   vision model that compares renders with the user's photos (match.py)
    SIM2CHIP_REASON_MODEL    reasoning model that turns those differences into spec changes (match.py)
    SIM2CHIP_LLM_DAILY_USD   spend cap per day on this machine (unset: none); a call over it raises LLMError
    SIM2CHIP_LOCAL_VLM_URL   where models named "local/<id>" answer (sim2chip.local_vlm, started on demand); default
                             http://127.0.0.1:8765/v1
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


DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
LOCAL = "local/"  # model prefix: answered by this machine's GPU (sim2chip.local_vlm), free
# The vision model of the checks that run where the GPU is (the scene route's): NVIDIA's Cosmos Reason 2 on
# this machine, free. User 2026-10-15: local wherever it can be (Token Factory's Qwen cost 47 $ in a fortnight).
LOCAL_VLM = "local/nvidia/Cosmos-Reason2-8B"
# Preferred model families per role, matched against the live /models list (ids change over time).
PREFERENCES = {
    "text": [r"nemotron-3-super", r"nemotron-3.*nano-30b", r"nemotron"],
    # The interviewer, conversation and spec alike (user, 2026-10-16): Nemotron 3.5 Super VL, one NVIDIA model
    # that also sees photos. interview_bench arm NV: 11/12 right specs, 13 s, 0.004 $ (experiments/NOTES.md).
    "interview": [r"nemotron.*super.*vl", r"nemotron-3-super"],
    # Kimi K2.6 conversing while Nemotron wrote the spec (interview.SPLIT_NOTE), the interview until 2026-10-16.
    "chat": [r"Kimi-K2\.6", r"nemotron-3-super"],
    # No Nemotron on Token Factory accepts images (checked 2026-09-24); prefer one if it appears. GLM-5.3-Flash
    # agreed best with the human asset review (88.9%), see experiments/NOTES.md.
    "vision": [r"nemotron.*omni", r"nemotron.*vl", r"GLM-5\.3-Flash", r"MiniCPM-V"],
    # Fast describer for describe_then_decide, where Nemotron makes the decision.
    "describe": [r"MiniCPM-V", r"GLM-5\.3-Flash"],
    # The user's photos while they wait in the interview (scene_profile, and ranking photos for 3D models:
    # Kimi 11 s for 6 photos, GLM timed out on all). 2026-09-30, one facade photo: Kimi-K2.6
    # 10 s and right (brick facade, two balconies, several pots), MiniCPM-V 1.5 s but saw one pot on a railing,
    # gemma-3-27b 86 s, GLM-5.3-Flash no answer in 120 s.
    # 2026-10-16, 9 street photos: Nemotron 3.5 Super VL 34 s against Kimi's 86 s, same reading of place and light,
    # sees small far cars Kimi missed, but once saw people where a dog was (experiments/NOTES.md). User's choice.
    "scene": [r"nemotron.*super.*vl", r"nemotron.*omni", r"Kimi-K2\.6", r"GLM-5\.3-Flash", r"MiniCPM-V"],
    # Renders beside photos: a strong vision model. Its differences go to the biggest Nemotron to reason on.
    # Kimi-K2.6: as concrete as K3 and 30 s instead of 4 to 24 min per comparison (2026-09-29).
    "compare": [r"nemotron.*omni", r"Kimi-K2\.6", r"Kimi-K3", r"GLM-5\.3-Flash"],
    "reason": [r"nemotron-3-ultra", r"nemotron-3-super"],
}


# A day's spend on this machine, every process's calls (for the daily cap): one line per call in a file per day.
SPEND_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "llm_spend"
PRICES_FILE = Path(__file__).resolve().parents[1] / "data" / "cache" / "token_prices.json"


def daily_cap() -> float | None:
    from .config import env
    v = env("SIM2CHIP_LLM_DAILY_USD", "")
    return float(v) if v else None


def spent_today() -> float:
    f = SPEND_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"
    if not f.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in f.read_text().splitlines() if line.strip())


def _price(model: str) -> dict:
    try:
        return json.loads(PRICES_FILE.read_text())["prices"].get(model, {})
    except (OSError, ValueError, KeyError):
        return {}


# Model calls in flight at once per stage (photos to label, assets to judge): Token Factory rate limits.
WORKERS = int(os.environ.get("SIM2CHIP_LLM_WORKERS", "8"))


# One photo looked at while the user waits in the interview: a slow answer is dropped, not retried for minutes.
# Seen 2026-09-30: GLM-5.3-Flash took 75 to 571 s per photo and the default 120 s x 5 attempts, twice, kept
# the user 20 minutes on "Looking at your photos".
PHOTO_CALL = {"timeout": 45.0, "retries": 1}
# Answer budget of a reasoning model that looks at a photo: at 4096 tokens Kimi K2.6 often spent them all on its
# reasoning and never answered (25 of 32 scene descriptions, so every chat fell back to the default scene;
# 2026-10-09). At 16384 the same photo got its answer in 18 s.
REASONING_TOKENS = 16384


def map_within(fn, items: list, seconds: float, workers: int | None = None, progress=None) -> list:
    """fn over items in threads, for at most `seconds`: results in the order of items, None for an item that
    failed or had not finished in time (its thread is left to end on its own, nothing waits for it).
    progress(done, total) is called as items finish, in any order."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as Late

    out: list = [None] * len(items)
    if not items:
        return out
    pool = ThreadPoolExecutor(max_workers=workers or WORKERS)
    futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
    done = 0
    try:
        for f in as_completed(futures, timeout=seconds):
            done += 1
            try:
                out[futures[f]] = f.result()
            except Exception:  # one photo failing must not lose the others
                pass
            if progress:
                progress(done, len(items))
    except Late:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out


class LLMError(RuntimeError):
    pass


def api_key() -> str:
    from .config import env
    key = env("NEBIUS_API_KEY", "")
    if not key:
        raise LLMError("NEBIUS_API_KEY is not set (environment or .env)")
    return key


def image_part(path: Path | str, detail: str = "auto") -> dict:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}", "detail": detail}}


def ask_json(client, msg: list, model: str, purpose: str, check=lambda v: True, tries: int = 2, **kw) -> dict | None:
    """A JSON object from the model that passes `check`, asking again on a bad answer; None if none came.
    Model errors (LLMError) are not bad answers: they propagate."""
    for _ in range(tries):
        try:
            v = parse_json(client.chat(msg, model, purpose=purpose, **kw))
            if isinstance(v, dict) and check(v):
                return v
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    return None


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
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # Models sometimes add a second object or a remark after the JSON: keep the first complete value.
        return json.JSONDecoder().raw_decode(text[start:])[0]


def local_vlm_url() -> str:
    from .config import env
    return env("SIM2CHIP_LOCAL_VLM_URL", "http://127.0.0.1:8765/v1").rstrip("/")


def _answers(base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/models", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def unload_local_vlm() -> None:
    """Frees the GPU memory of the local vision model, if its server is up (best effort, never starts it)."""
    base = local_vlm_url()
    if not _answers(base):
        return
    try:
        urllib.request.urlopen(urllib.request.Request(base + "/unload", data=b"{}", method="POST"), timeout=120)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


_START = __import__("threading").Lock()  # threads of one process start the local server once


def ensure_local_vlm(base: str, wait: float = 60.0) -> None:
    """Starts sim2chip.local_vlm in the teachers' Python (SIM2CHIP_SAM3_PYTHON) when nothing answers at a local
    address, so a job on a fresh GPU machine needs no extra step. A remote address is left alone."""
    with _START:
        _ensure_local_vlm(base, wait)


def _ensure_local_vlm(base: str, wait: float) -> None:
    import subprocess
    import urllib.parse
    if _answers(base):
        return
    u = urllib.parse.urlparse(base)
    from .config import env
    py = env("SIM2CHIP_SAM3_PYTHON", "")
    if u.hostname not in ("127.0.0.1", "localhost") or not py or not Path(py).exists():
        return  # the request then fails with the usual connection error
    root = Path(__file__).resolve().parents[1]
    log = root / "data" / "cache" / "local_vlm" / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:  # two processes starting it at once: the second one's bind fails, and it waits for the first
        subprocess.Popen([py, "-m", "sim2chip.local_vlm", "--port", str(u.port or 80)], cwd=root,
                         stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    except OSError:
        return
    t0 = time.time()
    while time.time() - t0 < wait and not _answers(base):
        time.sleep(1)


@dataclass
class Client:
    base_url: str = field(default_factory=lambda: os.environ.get("SIM2CHIP_LLM_BASE_URL", DEFAULT_BASE_URL))
    key: str | None = None
    usage_log: Path | None = None
    timeout: float = 120.0
    retries: int = 4
    _models: list[str] | None = None

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None,
                 retries: int | None = None) -> dict:
        local = bool(body) and str(body.get("model", "")).startswith(LOCAL)
        if local:
            base = local_vlm_url()
            ensure_local_vlm(base)
            url = base + path
            headers = {"Content-Type": "application/json"}
        else:
            url = self.base_url.rstrip("/") + path
            headers = {"Authorization": f"Bearer {self.key or api_key()}", "Content-Type": "application/json"}
        data = json.dumps(body).encode() if body is not None else None
        retries = self.retries if retries is None else retries
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise LLMError(f"{method} {path}: HTTP {e.code} {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise LLMError(f"{method} {path}: {e}") from None
        raise AssertionError("unreachable")

    def models(self) -> list[str]:
        if self._models is None:
            self._models = sorted(m["id"] for m in self._request("GET", "/models").get("data", []))
        return self._models

    def prices(self) -> dict[str, dict[str, float]]:
        """Dollars per token of each model, input and output, as Token Factory lists them."""
        out = {}
        for m in self._request("GET", "/models?verbose=true").get("data", []):
            p = m.get("pricing") or {}
            try:
                out[m["id"]] = {"input": float(p["prompt"]), "output": float(p["completion"])}
            except (KeyError, TypeError, ValueError):
                continue
        return out

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
             max_tokens: int = 2048, json_mode: bool = False, timeout: float | None = None,
             retries: int | None = None, thinking: bool | None = None) -> str:
        """timeout and retries: tighter than the client's for calls a user waits on (see PHOTO_CALL).
        thinking=False turns reasoning off (Nemotron's chat template); None keeps the model's default."""
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        cap = None if model.startswith(LOCAL) else daily_cap()
        if cap is not None and (done := spent_today()) >= cap:
            raise LLMError(f"daily Token Factory cap reached: {done:.2f} of {cap:.2f} USD (SIM2CHIP_LLM_DAILY_USD)")
        t0 = time.time()
        resp = self._request("POST", "/chat/completions", body, timeout=timeout, retries=retries)
        latency = time.time() - t0
        try:
            text = resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise LLMError(f"unexpected answer: {str(resp)[:300]}") from None
        self._log(model, purpose, resp.get("usage", {}), latency)
        return text

    def _log(self, model: str, purpose: str, usage: dict, latency: float) -> None:
        if daily_cap() is not None and not model.startswith(LOCAL):
            p = _price(model)
            usd = (usage.get("prompt_tokens") or 0) * p.get("input", 0) + (usage.get("completion_tokens") or 0) * p.get("output", 0)
            SPEND_DIR.mkdir(parents=True, exist_ok=True)
            with open(SPEND_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl", "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "model": model, "purpose": purpose, "usd": round(usd, 6)}) + "\n")
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
