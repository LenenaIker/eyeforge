"""What a job used of Nebius and NVIDIA, for the job page: Token Factory models (calls, tokens, dollars) and GPU
time (Nebius AI Cloud or this machine's GPU).

Model calls and Tavily web searches come from the job's logs/llm_usage.jsonl (its stages) and
logs/interview_usage.jsonl (the interview that wrote its spec); GPU time from logs/cloud_usage.jsonl (Nebius jobs) and logs/timeline.jsonl (local stages).
Prices per token come from Token Factory's model list, kept in cache/token_prices.json (refresh_prices).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .research import USD_PER_SEARCH
from .workspace import Job

PRICES = "token_prices.json"
PRICES_MAX_AGE_S = 24 * 3600
# What each logged purpose is part of, as the user knows the steps.
STEP = {"interview": "interview", "interview_spec": "interview", "english_queries": "interview",
        "pick_photos": "interview", "scene_profile": "interview", "research": "interview", "label": "labeling",
        "label_lesson": "labeling", "label_rules": "labeling",
        "qa_assets": "3D model check", "qa_describe": "3D model check", "qa_decide": "3D model check"}


def refresh_prices(cache: Path, client=None) -> dict:
    """Token Factory's prices, asked again when the saved ones are a day old. Never fails: the page shows
    tokens without dollars when there are no prices."""
    f = cache / PRICES
    try:
        saved = json.loads(f.read_text())
        if time.time() - saved["t"] < PRICES_MAX_AGE_S:
            return saved["prices"]
    except (OSError, ValueError, KeyError):
        saved = None
    try:
        from .llm import Client
        prices = (client or Client()).prices()
    except Exception:  # no key or no network: the old prices, if any
        return saved["prices"] if saved else {}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"t": time.time(), "prices": prices}, indent=1))
    return prices


def saved_prices(cache: Path) -> dict:
    try:
        return json.loads((cache / PRICES).read_text())["prices"]
    except (OSError, ValueError, KeyError):
        return {}


def _records(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def spent(prices: dict, calls: list[Path] = (), cloud: Path | None = None) -> float:
    """Dollars Token Factory, Tavily and Nebius charged us for these logs: model calls (usage files of the
    Client) and cloud GPU time (cloud_usage.jsonl). A model without a known price counts nothing."""
    usd = 0.0
    for f in calls:
        for r in _records(f):
            if r.get("model") == "tavily/search":
                usd += USD_PER_SEARCH
            elif p := prices.get(r.get("model")):
                usd += (r.get("prompt_tokens") or 0) * p["input"] + (r.get("completion_tokens") or 0) * p["output"]
    if cloud is not None:
        usd += sum(r.get("usd_estimate") or 0 for r in _records(cloud))
    return round(usd, 6)


def summary(job: Job, prices: dict, gpu_name: str = "L40S") -> dict:
    from .stages import BY_NAME

    models: dict[str, dict] = {}
    for r in _records(job.logs / "interview_usage.jsonl") + _records(job.logs / "llm_usage.jsonl"):
        m = models.setdefault(r["model"], {"model": r["model"], "name": r["model"].split("/")[-1], "steps": [],
                                           "calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        step = STEP.get(r["purpose"], r["purpose"])
        if step not in m["steps"]:
            m["steps"].append(step)
        m["calls"] += 1
        m["input_tokens"] += r.get("prompt_tokens") or 0
        m["output_tokens"] += r.get("completion_tokens") or 0
    for m in models.values():
        p = prices.get(m["model"])
        m["usd"] = round(m["input_tokens"] * p["input"] + m["output_tokens"] * p["output"], 4) if p else None
        m["nvidia"] = m["model"].lower().startswith("nvidia/")
        m["service"] = "Nebius Token Factory"
        if m["model"] == "tavily/search":  # web research: priced per search, not per token
            m.update(name="Tavily search", service="Tavily", usd=round(m["calls"] * USD_PER_SEARCH, 4))

    gpu = []
    cloud = _records(job.logs / "cloud_usage.jsonl")
    if cloud:
        gpu.append({"where": f"Nebius AI Cloud, {gpu_name}", "minutes": round(sum(r["hours"] for r in cloud) * 60, 1),
                    "usd": round(sum(r["usd_estimate"] for r in cloud), 4)})
    local = [r for r in _records(job.logs / "timeline.jsonl") if r.get("where") == "local"
             and r["step"] in BY_NAME and BY_NAME[r["step"]].gpu and not r.get("failed")]
    if local:
        gpu.append({"where": "your GPU", "minutes": round(sum(r["seconds"] for r in local) / 60, 1), "usd": 0.0})

    known = [m["usd"] for m in models.values() if m["usd"] is not None] + [g["usd"] for g in gpu]
    return {"models": sorted(models.values(), key=lambda m: -(m["input_tokens"] + m["output_tokens"])),
            "gpu": gpu, "usd": round(sum(known), 4),
            "priced": all(m["usd"] is not None for m in models.values())}
