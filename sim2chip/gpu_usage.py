"""How much of the machine we pay for does work: the cloud job script samples the GPU (nvidia-smi), the CPU and
the RAM every SAMPLE_S seconds into SAMPLES (worker_script, warm_script), and the run sums it up per step of its
timeline (logs/gpu_usage.json), the warm GPU for its whole life (warm/<id>/gpu.json). Started 2026-10-13, when
nothing said whether the L40S works or waits (the timeline put about half of a run's time in start-up and copies);
the CPU and RAM the same day, the user suspecting runs on Nebius are slow for want of CPU, not of GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Lines "epoch gpu_util% gpu_memory_MB cpu% iowait% ram_used_MB" from the job script ("-" for the GPU when
# nvidia-smi gives nothing); the first runs wrote only the first three.
SAMPLES = Path("/tmp/s2c/gpu.log")
SAMPLE_S = 2
BUSY = 50  # a sample at this GPU utilization or more counts as busy
CPU_FULL = 90  # a sample at this CPU use or more (all cores together) counts as the CPU being the limit

# The job script's sampler, started first thing: the time before Python (kits, code) is paid for too. CPU and
# iowait are the shares of all cores' time since the last sample (/proc/stat), RAM is MemTotal - MemAvailable.
SAMPLER = ("( pt=0; pi=0; pw=0; while true; do read -r _ u n s i w q sq st _ < /proc/stat; "
           "t=$((u+n+s+i+w+q+sq+st)); c=-; io=-; "
           "if [ $pt -gt 0 ] && [ $t -gt $pt ]; then c=$((100*((t-pt)-(i+w-pi))/(t-pt))); "
           "io=$((100*(w-pw)/(t-pt))); fi; pt=$t; pi=$((i+w)); pw=$w; "
           "g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null "
           "| head -1 | tr -d ,); [ -n \"$g\" ] || g=\"- -\"; "
           "m=$(awk '/^MemTotal/{a=$2} /^MemAvailable/{b=$2} END{print int((a-b)/1024)}' /proc/meminfo); "
           f"echo \"$(date +%s) $g $c $io $m\"; sleep {SAMPLE_S}; done > {SAMPLES} 2>/dev/null & )")

Sample = tuple  # (epoch, gpu util %, gpu memory MB, cpu % or None, iowait % or None, ram MB or None)


def read(path: Path = SAMPLES) -> list[Sample]:
    """One tuple per sample with a GPU reading; the CPU fields are None on the first runs' lines."""
    out = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) not in (3, 6):
            continue
        try:
            head = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            continue
        rest = tuple(None if x == "-" else float(x) for x in parts[3:]) if len(parts) == 6 else (None,) * 3
        out.append(head + rest)
    return out


def window(samples: list[Sample], start: float, end: float) -> dict | None:
    """Between start and end: the GPU's mean use, share of busy samples and peak memory; with CPU readings, the
    CPU's mean use, share of samples with it full, mean iowait and peak RAM. None with no sample."""
    s = [x for x in samples if start <= x[0] <= end]
    if not s:
        return None
    out = {"util": round(sum(x[1] for x in s) / len(s)), "busy": round(sum(x[1] >= BUSY for x in s) / len(s), 2),
           "mem_mb": round(max(x[2] for x in s)), "samples": len(s)}
    c = [x for x in s if x[3] is not None]
    if c:
        out.update({"cpu": round(sum(x[3] for x in c) / len(c)),
                    "cpu_full": round(sum(x[3] >= CPU_FULL for x in c) / len(c), 2),
                    "iowait": round(sum(x[4] for x in c) / len(c)),
                    "ram_mb": round(max(x[5] for x in c if x[5] is not None))})
    return out


def summary(samples: list[Sample], timeline: list[dict], start: float, end: float) -> dict:
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
    out = {**summary(samples, recs, start, end), "start": start, "end": end, "cpus": os.cpu_count()}
    (job_logs / "gpu_usage.json").write_text(json.dumps(out, indent=1))
    return out
