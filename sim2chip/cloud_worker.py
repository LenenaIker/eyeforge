"""The whole pipeline inside one cloud job (Nebius backend, worker mode).

Getting an L40S in eu-north1 can take many minutes, or fail, so a run asks for one GPU once and runs every
stage on it with the local backend, instead of one job per stage. Inside the container:

    python -m sim2chip.cloud_worker --home /tmp/s2c/home --job <name> [--from STAGE] [--only STAGE] [--force]

It plans the run first and pulls only what the stages about to run read: the input, their upstream
outputs and what a stage reads back from its own last run (label's verdicts with the user's corrections),
plus the shared knowledge (asset reviews, teacher answers). Before each stage that uses the asset cache it
pulls that job's cached files (USD conversions, skies, textures) so nothing is converted or downloaded
twice. It runs the pipeline with the local backend, pushes progress every PROGRESS_S seconds (the
log and done markers, so the machine that launched it can follow) and everything at the end.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from . import spec as spec_mod
from .backends.local import LocalBackend
from .cloud_sync import S3Store, asset_cache_keys
from .pipeline import Pipeline, StageFailed
from .workspace import DONE, Cache, Job

PROGRESS_S = 20
LOG = "logs/cloud_pipeline.log"
SHARED = ("db/", "labels/")  # small, append-only knowledge shared by all jobs
# Isaac Sim compiles its renderer's shaders on the first start in a fresh container: about 180 s on an L40S
# (2026-09-27), then 3 s for every later start. The compiled caches go to the cache bucket and come back on
# the next run, so only the first run pays. Bucket prefix -> path ("~" is $HOME, /isaac-sim in the image).
GPU_CACHES = {"gpu/ov-shaders": "~/.cache/ov/shaders", "gpu/ov-ddc": "~/.cache/ov/DerivedDataCache",
              "gpu/kit": "/isaac-sim/kit/cache", "gpu/compute": "~/.nv/ComputeCache",
              "gpu/gl": "~/.cache/nvidia/GLCache"}
# Where the caches really land is checked after each run: sizes of these folders go to the worker log.
CACHE_PROBE = ("~/.cache", "~/.cache/ov", "~/.nv", "/isaac-sim/kit/cache", "/isaac-sim/kit/data")


def gpu_cache(local: str) -> Path:
    return Path(local).expanduser()


def probe_caches() -> None:
    for top in CACHE_PROBE:
        d = gpu_cache(top)
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                try:
                    mb = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file()) / 1e6
                except OSError:  # unreadable folders must not stop the run
                    continue
                print(f"[sim2chip] cache probe {sub}: {mb:.0f} MB", flush=True)


def sync_progress(store: S3Store, bucket: str, job: Path) -> None:
    """Log and done markers only: cheap enough to send often."""
    for f in [job / LOG, job / "logs" / "timeline.jsonl"] + sorted(job.glob("*/_done.json")):
        if f.exists():
            store.s3.upload_file(str(f), bucket, f.relative_to(job).as_posix())


def wanted(key: str, todo: list) -> bool:
    """Job files the stages about to run read: the input, the outputs of their upstream stages and what they
    read back from their own last run. Everything else stays in the bucket."""
    top, _, rest = key.partition("/")
    names = {st.name for st in todo}
    if top == "input":
        return any(st.reads_input for st in todo)
    if top in names:
        return any(st.name == top and rest in st.rereads for st in todo)
    if top == "fetch_assets" and any(st.uses_cache for st in todo):
        return True  # its manifest says which cached files the asset stages need
    return any(top in st.after for st in todo)


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
    job, cache = Job(args.home / "jobs" / args.job), Cache(args.home / "cache")
    t0 = time.time()
    listing = store.listing(job_bucket)
    store.pull(job_bucket, [k for k in listing if k == "spec.json" or k.endswith("/" + DONE)], job.root)
    pipe = Pipeline(spec_mod.load(job.spec_path), job, cache)
    todo = [pl.stage for pl in pipe.plan(args.start, args.only, args.force) if pl.action == "run"]
    n = store.pull(job_bucket, [k for k in listing if wanted(k, todo)], job.root)
    m = store.pull(cache_bucket, [k for k in store.listing(cache_bucket) if k.startswith(SHARED)], cache.root)
    listing = store.listing(cache_bucket, "gpu/")
    g = sum(store.pull(cache_bucket, [k for k in listing if k.startswith(prefix + "/")], gpu_cache(local),
                       strip=prefix) for prefix, local in GPU_CACHES.items())
    print(f"[sim2chip] worker: pulled {n} job files, {m} shared cache files and {g} GPU cache files, "
          f"{len(todo)} stages to run", flush=True)
    setup = os.environ.get("S2C_SETUP_S")
    if setup:  # code download and Python environment, measured by the job script
        job.timeline("cloud: set up environment", float(setup), "nebius")
    job.timeline("cloud: pull job data", time.time() - t0, "nebius", files=n + m + g)

    def pull_asset_cache(stage, job: Job, cache: Cache) -> None:
        if not stage.uses_cache:
            return
        t = time.time()
        k = store.pull(cache_bucket, asset_cache_keys(store.listing(cache_bucket), job.root), cache.root)
        if k:
            job.timeline(f"cloud: pull cache for {stage.name}", time.time() - t, "nebius", files=k)

    job.logs.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def progress():
        while not stop.wait(PROGRESS_S):
            try:
                sync_progress(store, job_bucket, job.root)
            except Exception as e:  # a failed progress upload must not kill the run
                print(f"[sim2chip] progress upload failed: {e}", flush=True)
    t = threading.Thread(target=progress, daemon=True)
    t.start()
    code = 1
    with open(job.root / LOG, "w", buffering=1) as log, contextlib.redirect_stdout(log), \
            contextlib.redirect_stderr(log):
        try:
            pipe.run(LocalBackend(before_stage=pull_asset_cache), start=args.start, only=args.only, force=args.force)
            code = 0
        except StageFailed as e:
            print(f"FAILED: {e}")
        except Exception:
            traceback.print_exc()
    stop.set()
    t.join()
    print((job.root / LOG).read_text(errors="replace")[-3000:], flush=True)
    probe_caches()
    t0 = time.time()
    m = store.upload_dir(cache_bucket, cache.root, "")
    m += sum(store.upload_dir(cache_bucket, gpu_cache(local), prefix) for prefix, local in GPU_CACHES.items())
    job.timeline("cloud: push cache", time.time() - t0, "nebius", files=m)
    n = store.upload_dir(job_bucket, job.root, "")
    sync_progress(store, job_bucket, job.root)
    print(f"[sim2chip] worker: pushed {n} job files and {m} cache files, exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
