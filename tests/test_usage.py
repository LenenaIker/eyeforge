"""The usage panel: Token Factory calls priced per model, Nebius and local GPU time, the interview credited."""
import json

from sim2chip import usage
from sim2chip.workspace import Job

PRICES = {"nvidia/nemotron-3-super-120b-a12b": {"input": 3e-7, "output": 9e-7},
          "zai-org/GLM-5.3-Flash": {"input": 1.5e-7, "output": 5e-7}}


def write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_summary(tmp_path):
    job = Job(tmp_path / "j")
    write(job.logs / "interview_usage.jsonl", [
        {"model": "nvidia/nemotron-3-super-120b-a12b", "purpose": "interview", "prompt_tokens": 1_000_000,
         "completion_tokens": 100_000},
        {"model": "zai-org/GLM-5.3-Flash", "purpose": "scene_profile", "prompt_tokens": 1000, "completion_tokens": 0}])
    write(job.logs / "llm_usage.jsonl", [
        {"model": "zai-org/GLM-5.3-Flash", "purpose": "qa_describe", "prompt_tokens": 9000, "completion_tokens": 2000},
        {"model": "some/unpriced", "purpose": "x", "prompt_tokens": 5, "completion_tokens": None}])
    write(job.logs / "cloud_usage.jsonl", [{"hours": 0.5, "usd_estimate": 0.78}])
    write(job.logs / "timeline.jsonl", [{"t": 1, "step": "render", "seconds": 120, "where": "local"},
                                        {"t": 2, "step": "ingest", "seconds": 600, "where": "local"},  # CPU
                                        {"t": 3, "step": "train", "seconds": 60, "where": "local", "failed": True}])
    s = usage.summary(job, PRICES)
    nemo = s["models"][0]
    assert nemo["name"] == "nemotron-3-super-120b-a12b" and nemo["nvidia"] and nemo["steps"] == ["interview"]
    assert nemo["usd"] == 0.39
    glm = next(m for m in s["models"] if m["model"].startswith("zai"))
    assert glm["steps"] == ["interview", "3D model check"] and glm["calls"] == 2 and not glm["nvidia"]
    assert s["gpu"] == [{"where": "Nebius AI Cloud, L40S", "minutes": 30.0, "usd": 0.78},
                        {"where": "your GPU", "minutes": 2.0, "usd": 0.0}]
    assert not s["priced"] and s["usd"] == round(0.39 + glm["usd"] + 0.78, 4)


def test_prices_are_kept_a_day(tmp_path):
    class Client:
        asked = 0

        def prices(self):
            Client.asked += 1
            return PRICES
    assert usage.refresh_prices(tmp_path, Client()) == PRICES
    assert usage.refresh_prices(tmp_path, Client()) == PRICES and Client.asked == 1
    assert usage.saved_prices(tmp_path) == PRICES

    class Down:
        def prices(self):
            raise OSError("no network")
    assert usage.refresh_prices(tmp_path / "empty", Down()) == {}


def test_tavily_searches_are_priced_per_search(tmp_path):
    job = Job(tmp_path / "j")
    write(job.logs / "interview_usage.jsonl", [{"model": "tavily/search", "purpose": "research", "prompt_tokens": 0,
                                                "completion_tokens": 0}] * 3)
    [m] = usage.summary(job, {})["models"]
    assert (m["name"], m["service"], m["steps"], m["calls"], m["usd"]) == ("Tavily search", "Tavily", ["interview"], 3,
                                                                           0.024)
