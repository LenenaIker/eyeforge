"""Measure how Nebius Serverless AI starts several jobs at once (queueing, image pull, Isaac cold start).

Launches N jobs with the Isaac Sim image in parallel. Each one prints when its container started, its GPU
and how long a headless Isaac Sim app takes to start cold, then exits. The script polls the jobs and
records when each state change was first seen.

    python scripts/nebius_parallel_probe.py --jobs 4
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim2chip.backends.nebius import RUNTIMES, TERMINAL, NebiusCLI  # noqa: E402

SCRIPT = """set -e
echo "PROBE container_start $(date +%s.%N)"
nvidia-smi -L
cat > /tmp/probe.py <<'PY'
import time
t = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print(f"PROBE isaac_ready_s {time.time() - t:.1f}", flush=True)
app.close()
PY
/isaac-sim/python.sh /tmp/probe.py 2>&1 | grep PROBE || true
echo "PROBE container_end $(date +%s.%N)"
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cli = NebiusCLI()
    rt = RUNTIMES["isaac"]
    script = Path(tempfile.mkdtemp()) / "probe.sh"
    script.write_text(SCRIPT)
    t0 = time.time()

    def launch(i: int) -> str:
        a = ["ai", "job", "create", "--name", f"s2c-probe-{int(t0)}-{i}", "--image", rt.image,
             "--platform", "gpu-l40s-a", "--preset", "1gpu-8vcpu-32gb", "--on-demand", "--timeout", "1h",
             *cli._parent(), "--inject-file", f"{script}:/opt/probe.sh", "--container-command", "bash",
             "--args", "/opt/probe.sh", "--async"]
        for k, v in rt.env.items():
            a += ["--env", f"{k}={v}"]
        out = cli(*a)
        return next(l.split("Job ID:", 1)[1].strip() for l in out.splitlines() if "Job ID:" in l)

    with ThreadPoolExecutor(args.jobs) as pool:
        ids = list(pool.map(launch, range(args.jobs)))
    print(f"launched {len(ids)} jobs in {time.time() - t0:.0f} s", flush=True)
    seen = {j: {} for j in ids}
    while True:
        with ThreadPoolExecutor(len(ids)) as pool:
            states = dict(zip(ids, pool.map(cli.job_state, ids)))
        for j, s in states.items():
            if s not in seen[j]:
                seen[j][s] = round(time.time() - t0)
                print(f"{time.time() - t0:6.0f} s  job {ids.index(j)}: {s}", flush=True)
        if all(s in TERMINAL for s in states.values()) or time.time() - t0 > 1800:
            break
        time.sleep(5)
    report = []
    for i, j in enumerate(ids):
        logs = cli("ai", "job", "logs", j, check=False)
        probe = {l.split()[1]: l.split()[2] for l in logs.splitlines() if l.startswith("PROBE ")}
        gpu = next((l for l in logs.splitlines() if l.startswith("GPU 0")), "")
        start = float(probe["container_start"]) - t0 if "container_start" in probe else None
        report.append({"job": i, "states_seen_at_s": seen[j], "container_start_s": round(start) if start else None,
                       "isaac_ready_s": probe.get("isaac_ready_s"), "gpu": gpu[:60]})
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
