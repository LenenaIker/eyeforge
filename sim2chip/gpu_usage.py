"""How much of the GPU we pay for does work: the cloud job script samples nvidia-smi every SAMPLE_S seconds
into SAMPLES (worker_script, warm_script), and the run sums it up per step of its timeline (logs/gpu_usage.json),
the warm GPU for its whole life (warm/<id>/gpu.json). Started 2026-10-13, when nothing said whether the L40S
works or waits (the timeline put about half of a run's time in start-up and copies)."""
from __future__ import annotations

import json
from pathlib import Path

SAMPLES = Path("/tmp/s2c/gpu.log")  # lines "epoch util% memory_MB", from the job script
SAMPLE_S = 2
BUSY = 50  # a sample at this utilization or more counts as busy

# The job script's sampler, started first thing: the time before Python (kits, code) is paid for too.
SAMPLER = (f"( while true; do echo \"$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used "
           f"--format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ,)\"; sleep {SAMPLE_S}; done "
           f"> {SAMPLES} 2>/dev/null & )")


def read(path: Path = SAMPLES) -> list[tuple[float, float, float]]:
    """(epoch, util %, memory MB) per sample; lines without a GPU reading are skipped."""
    out = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            try:
                out.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return out


def window(samples: list[tuple[float, float, float]], start: float, end: float) -> dict | None:
    """Mean utilization, share of busy samples and peak memory between start and end, or None with no sample."""
    s = [x for x in samples if start <= x[0] <= end]
    if not s:
        return None
    return {"util": round(sum(x[1] for x in s) / len(s)), "busy": round(sum(x[1] >= BUSY for x in s) / len(s), 2),
            "mem_mb": round(max(x[2] for x in s)), "samples": len(s)}


def summary(samples: list[tuple[float, float, float]], timeline: list[dict], start: float, end: float) -> dict:
    """The whole window and each of the timeline's steps in it (steps end at "t" and last "seconds")."""
    steps = {}
    for r in timeline:
        if r["t"] >= start and r.get("where") == "nebius":
            w = window(samples, r["t"] - r["seconds"], r["t"])
            if w:
                steps[r["step"]] = w
    return {"all": window(samples, start, end), "steps": steps}


def write(job_logs: Path, start: float, end: float, samples_path: Path = SAMPLES) -> dict | None:
    """logs/gpu_usage.json of a run, from the samples and its timeline. None when there were no samples."""
    samples = read(samples_path)
    if not samples:
        return None
    f = job_logs / "timeline.jsonl"
    recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []
    out = {**summary(samples, recs, start, end), "start": start, "end": end}
    (job_logs / "gpu_usage.json").write_text(json.dumps(out, indent=1))
    return out
