"""The whole pipeline inside one cloud job (Nebius backend, worker mode).

Getting an L40S in eu-north1 can take many minutes, or fail, so a run asks for one GPU once and runs every
stage on it with the local backend, instead of one job per stage. Inside the container:

    python -m sim2chip.cloud_worker --home /tmp/s2c/home --job <name> [--from STAGE] [--only STAGE] [--force]
    python -m sim2chip.cloud_worker --home ... --job <name> --prefetch    (the teacher's answers only, warm GPU)

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
from .backends.nebius import KIT_TIMES
from .cloud_sync import SHARED, S3Store, asset_cache_keys, wanted
from .pipeline import Pipeline, StageFailed
from .workspace import DONE, Cache, Job
from .workspace import say

PROGRESS_S = 20
LOG = "logs/cloud_pipeline.log"
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
                say(f"cache probe {sub}: {mb:.0f} MB")


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
    p.add_argument("--prefetch", action="store_true", help="only the teacher's answers for every photo (warm GPU)")
    args = p.parse_args()
    job_bucket, cache_bucket = os.environ["S2C_JOB_BUCKET"], os.environ["S2C_CACHE_BUCKET"]
    store = S3Store()
    job, cache = Job(args.home / "jobs" / args.job), Cache(args.home / "cache")
    if args.prefetch:
        return prefetch(store, job_bucket, cache_bucket, job, cache)
    t0 = time.time()
    listing = store.listing(job_bucket)
    store.pull(job_bucket, [k for k in listing if k == "spec.json" or k.endswith("/" + DONE)], job.root)
    pipe = Pipeline(spec_mod.load(job.spec_path), job, cache)
    todo = [pl.stage for pl in pipe.plan(args.start, args.only, args.force) if pl.action == "run"]
    n = store.pull(job_bucket, [k for k in listing if wanted(k, todo)], job.root)
    m = store.pull(cache_bucket, [k for k in store.listing(cache_bucket) if k.startswith(SHARED)], cache.root)
    g = 0
    if os.environ.get("SIM2CHIP_ISAAC_PYTHON"):  # Isaac's caches, only in its image (a run that renders)
        listing = store.listing(cache_bucket, "gpu/")
        g = sum(store.pull(cache_bucket, [k for k in listing if k.startswith(prefix + "/")], gpu_cache(local),
                           strip=prefix) for prefix, local in GPU_CACHES.items())
    say(f"worker: pulled {n} job files, {m} shared cache files and {g} GPU cache files, {len(todo)} stages to run")
    setup = os.environ.get("S2C_SETUP_S")
    if setup:  # code download and Python environment, measured by the job script
        job.timeline("cloud: set up environment", float(setup), "nebius")
    kits = Path(KIT_TIMES)
    warm = bool(os.environ.get("S2C_WARM"))  # on the warm GPU the kits were unpacked once, for an earlier run
    for line in kits.read_text().splitlines() if kits.exists() and not warm else []:  # the Python kits, all at once
        i, start, end, size = line.split()
        seconds, mb = float(end) - float(start), int(size) / 1e6
        job.timeline(f"cloud: kit {i} download and unpack", seconds, "nebius", mb=round(mb),
                     mb_s=round(mb / max(seconds, 0.01)))
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
                say(f"progress upload failed: {e}")
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
    if os.environ.get("SIM2CHIP_ISAAC_PYTHON"):
        m += sum(store.upload_dir(cache_bucket, gpu_cache(local), prefix) for prefix, local in GPU_CACHES.items())
    job.timeline("cloud: push cache", time.time() - t0, "nebius", files=m)
    n = store.upload_dir(job_bucket, job.root, "")
    sync_progress(store, job_bucket, job.root)
    say(f"worker: pushed {n} job files and {m} cache files, exit {code}")
    return code


def prefetch(store: S3Store, job_bucket: str, cache_bucket: str, job: Job, cache: Cache) -> int:
    """The teacher's answers for every photo of a job being calibrated (stages.label.Label.prefetch), pushed to
    the cache bucket for the next rounds and the first run. The job's own folders are not touched."""
    from .stages import Context
    from .stages.label import Label

    listing = store.listing(job_bucket)
    store.pull(job_bucket, [k for k in listing if k == "spec.json" or k.startswith("input/")
                            or k == "label/verdicts.json"], job.root)
    store.pull(cache_bucket, [k for k in store.listing(cache_bucket) if k.startswith(SHARED)], cache.root)
    code = 1
    try:
        n = Label().prefetch(Context(spec_mod.load(job.spec_path), job, cache))
        say(f"prefetch: {n} photos answered")
        code = 0
    except Exception:
        traceback.print_exc()
    m = sum(store.upload_dir(cache_bucket, cache.root / p.rstrip("/"), p.rstrip("/")) for p in SHARED)
    say(f"prefetch: pushed {m} cache files, exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
