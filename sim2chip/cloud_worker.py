"""The whole pipeline inside one cloud job (Nebius backend, worker mode).

Getting an L40S in eu-north1 can take many minutes, or fail, so a run asks for one GPU once and runs every
stage on it with the local backend, instead of one job per stage. Inside the container:

    python -m sim2chip.cloud_worker --home /tmp/s2c/home --job <name> [--from STAGE] [--only STAGE] [--force]

It pulls the job (spec, input, earlier stage outputs) and the shared knowledge (asset reviews, teacher
answers) from Object Storage, runs `sim2chip run` locally, pushes progress every PROGRESS_S seconds (the
log and done markers, so the machine that launched it can follow) and everything at the end.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .cloud_sync import S3Store
from .workspace import Job

PROGRESS_S = 20
LOG = "logs/cloud_pipeline.log"
SHARED = ("db/", "labels/")  # small, append-only knowledge shared by all jobs


def sync_progress(store: S3Store, bucket: str, job: Path) -> None:
    """Log and done markers only: cheap enough to send often."""
    for f in [job / LOG, job / "logs" / "timeline.jsonl"] + sorted(job.glob("*/_done.json")):
        if f.exists():
            store.s3.upload_file(str(f), bucket, f.relative_to(job).as_posix())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--from", dest="start")
    p.add_argument("--only")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    job_bucket, cache_bucket = os.environ["S2C_JOB_BUCKET"], os.environ["S2C_CACHE_BUCKET"]
    store = S3Store()
    job, cache = args.home / "jobs" / args.job, args.home / "cache"
    t0 = time.time()
    n = store.pull(job_bucket, [k for k in store.listing(job_bucket) if not k.startswith("_code/")], job)
    m = store.pull(cache_bucket, [k for k in store.listing(cache_bucket) if k.startswith(SHARED)], cache)
    print(f"[sim2chip] worker: pulled {n} job files and {m} shared cache files", flush=True)
    timeline = Job(job).timeline
    setup = os.environ.get("S2C_SETUP_S")
    if setup:  # code download and Python environment, measured by the job script
        timeline("cloud: set up environment", float(setup), "nebius")
    timeline("cloud: pull job data", time.time() - t0, "nebius", files=n + m)

    cmd = [sys.executable, "-m", "sim2chip.cli", "--home", str(args.home), "run", str(job / "spec.json"),
           "--job", args.job, "--backend", "local"]
    for flag, value in (("--from", args.start), ("--only", args.only)):
        if value:
            cmd += [flag, value]
    if args.force:
        cmd.append("--force")
    (job / "logs").mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def progress():
        while not stop.wait(PROGRESS_S):
            try:
                sync_progress(store, job_bucket, job)
            except Exception as e:  # a failed progress upload must not kill the run
                print(f"[sim2chip] progress upload failed: {e}", flush=True)
    t = threading.Thread(target=progress, daemon=True)
    t.start()
    with open(job / LOG, "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}
                              ).returncode
    stop.set()
    t.join()
    print((job / LOG).read_text(errors="replace")[-3000:], flush=True)
    t0 = time.time()
    m = store.upload_dir(cache_bucket, cache, "")
    timeline("cloud: push cache", time.time() - t0, "nebius", files=m)
    n = store.upload_dir(job_bucket, job, "")
    sync_progress(store, job_bucket, job)
    print(f"[sim2chip] worker: pushed {n} job files and {m} cache files, exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
