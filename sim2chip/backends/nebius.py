"""Nebius backend: the pipeline runs in Serverless AI jobs on an L40S, with its data in Object Storage.

Worker mode (default): one job runs every stage (cloud_worker.py). An L40S can take many minutes to get, or
not come at all (measured 2026-09-26: 3 of 4 on-demand jobs failed for lack of capacity), so a run asks for
one GPU once. Stage mode (SIM2CHIP_NEBIUS_MODE=stage): one job per stage, each in its own image.

Data stays in Object Storage between stages: the user uploads the spec and input once, every job pulls
what its stage reads to local disk, runs it and pushes what it wrote (see cloud_sync.py). Only small
results (done markers, reports, the exported model) come back to the local job folder, which keeps
working as the pipeline's view of the job.

    job bucket    s3://<prefix>-<job>   spec.json, input/, <stage>/, logs/, _code/
    cache bucket  s3://<prefix>-cache   objaverse, usd, hdri, shared by jobs

Configuration (environment or the repo's .env):
    SIM2CHIP_NEBIUS_CLI        nebius CLI (default: on PATH or ~/.nebius/bin/nebius)
    SIM2CHIP_NEBIUS_PROJECT    project id (default: the CLI profile's parent-id)
    SIM2CHIP_BUCKET_PREFIX     bucket name prefix (default sim2chip)
    SIM2CHIP_S3_SECRET         MysteryBox secret with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for the jobs
    SIM2CHIP_NEBIUS_SECRET     MysteryBox secret holding NEBIUS_API_KEY, for stages that call Token Factory
    SIM2CHIP_MAX_USD           spend cap for one run, estimated from GPU time (default 5)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   Object Storage key, to upload input and fetch results
"""
from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_hex

from .. import config
from ..cloud_sync import REGION, S3Store
from ..stages import Stage
from ..workspace import DONE, LOG_PREFIX as PREFIX, REPO, Cache, Job

# The GPU and its price go together: set all three to try another platform (for instance when eu-north1 has
# no L40S left). Isaac Sim needs a GPU with RT cores (L40S yes, H100/H200 no).
PLATFORM = config.env("SIM2CHIP_NEBIUS_PLATFORM", "gpu-l40s-a")
PRESET = config.env("SIM2CHIP_NEBIUS_PRESET", "1gpu-8vcpu-32gb")
USD_PER_HOUR = float(config.env("SIM2CHIP_USD_PER_HOUR", "1.55"))  # on-demand L40S (Intel), nebius.com/prices, 2026-09
WORKER_HOURS = int(config.env("SIM2CHIP_WORKER_HOURS", "3"))  # timeout of a whole run in one job
UV_URL = "https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz"
WORKER_DEPS = ["tensorflow[and-cuda]==2.18.1", "numpy<2.1", "pillow", "objaverse", "boto3", "usd-core==26.8",
               "trimesh==5.1.0"]  # the last two convert models (glb2usd)
# The teacher (SAM 3, label stage) gets its own environment: torch in the same one as TensorFlow, each with its
# own CUDA libraries, is not worth the risk. Only installed when the run labels photos.
TEACHER_DEPS = ["torch==2.11.0", "torchvision==0.26.0", "transformers==5.17.0", "accelerate", "pillow"]
# Brought back for the web (status, report, review) after a worker run; the rest stays in the bucket.
WORKER_RESULTS = ("*.json", "export/*", "label/*", "ingest/test/*", "qa_assets/sheets/*", "preview_assets/*_0.png",
                  "generate_assets/*", "logs/*")
RENDER_SAMPLE = 48
JOB_DIR, CACHE_DIR = "/tmp/s2c/job", "/tmp/s2c/cache"
CLOUD_JOB = "cloud_job.json"  # in logs/: the Nebius job of a run in progress


@dataclass(frozen=True)
class Runtime:
    image: str
    python: str
    setup: str = ""
    env: dict = field(default_factory=dict)


# One L40S preset for every stage: the eu-north1 quota has no CPU-only instances, and Isaac needs RT cores.
RUNTIMES = {
    "isaac": Runtime("nvcr.io/nvidia/isaac-sim:5.1.0", "/isaac-sim/python.sh",
                     env={"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"}),
    "tf": Runtime("tensorflow/tensorflow:2.18.0-gpu", "python3", "pip install -q pillow"),
    "cpu": Runtime("tensorflow/tensorflow:2.18.0-gpu", "python3",
                   "pip install -q objaverse pillow numpy usd-core==26.8 trimesh==5.1.0"),
}
TERMINAL = {"COMPLETED", "SUCCEEDED", "FAILED", "ERROR", "CANCELLED", "CANCELED", "STOPPED", "DELETED"}


_env = config.env


def code_archive() -> bytes:
    """The sim2chip package as a tar.gz, unpacked by every stage job (no custom image needed)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted((REPO / "sim2chip").rglob("*.py")):
            tar.add(f, arcname=str(f.relative_to(REPO)))
    return buf.getvalue()


def bootstrap(stage: Stage, fp: str) -> str:
    """Shell script that runs one stage inside its container: fetch the code, pull, run, push.

    The code comes from a presigned URL in S2C_CODE_URL; boto3 (for the sync only) goes to its own folder
    so it cannot clash with the runtime's packages.
    """
    rt = RUNTIMES[stage.runtime]
    dirs = f"--job {JOB_DIR} --cache {CACHE_DIR}"
    sync = f"PYTHONPATH=/tmp/s2c/code:/tmp/s2c/deps {rt.python} -m sim2chip.cloud_sync"
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        f"mkdir -p /tmp/s2c/code /tmp/s2c/deps {JOB_DIR}/logs {CACHE_DIR}",
        f"{rt.python} -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "
        '"$S2C_CODE_URL" /tmp/s2c/code.tgz',
        "tar xzf /tmp/s2c/code.tgz -C /tmp/s2c/code",
        f"{rt.python} -m pip install -q --target /tmp/s2c/deps boto3",
        rt.setup,
        "export PYTHONUNBUFFERED=1",
        f"{sync} pull {stage.name} {dirs}",
        "set +e",
        f"PYTHONPATH=/tmp/s2c/code {rt.python} -m sim2chip.run_stage {stage.name} {dirs} --fingerprint {fp} 2>&1 "
        f"| tee {JOB_DIR}/logs/{stage.name}.log",
        "set -e",
        f"{sync} push {stage.name} {dirs}",
    ]
    return "\n".join(l for l in lines if l) + "\n"


def print_timeline(job: Job, log=print, since: float | None = None) -> None:
    """Where the time of the last run went, longest first."""
    f = job.logs / "timeline.jsonl"
    if not f.exists():
        return
    recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if since is not None:
        recs = [r for r in recs if r["t"] >= since]
    if not recs:
        return
    # Steps nest (the job's running state holds every stage): shares are of the wall time, not of the sum.
    total = max(r["t"] for r in recs) - min(r["t"] - r["seconds"] for r in recs) or 1
    log(f"  where the time went ({total / 60:.1f} min in all):")
    for r in sorted(recs, key=lambda r: -r["seconds"]):
        log(f"    {r['seconds'] / 60:5.1f} min  {100 * r['seconds'] / total:3.0f}%  {r['step']} ({r['where']})")


def cloud_spec(spec, log=print):
    """The spec without what the cloud cannot run yet, decided before asking for a GPU: 3D models from the
    user's photos need the TRELLIS runtime, which only exists locally (2026-09-27: a run found out inside the
    job, with the GPU already paid). Those classes use the 3D model search instead, and the log says so."""
    from .. import spec as spec_mod

    data = json.loads(spec.to_json())
    for c in data.get("classes") or []:
        a = c.get("assets")
        if not a or not a.get("photos"):
            continue
        a["photos"] = None
        if a.get("source") == "photos":
            a["source"] = "search"
            a["queries"] = a.get("queries") or [c["id"]]
        log(f"WARNING: 3D models from photos are not available on Nebius yet: class {c['id']} uses the "
            f"3D model search ({', '.join(a.get('queries') or a.get('categories') or [])})")
    return spec_mod.from_dict(data)


def portable_spec(spec, job: Job):
    """The spec with every local file or folder it names (user photos, QA lists) linked into the job's input/
    and named relative to it: input/ is uploaded, and absolute paths of this machine mean nothing on Nebius.
    Paths are rewritten before the pipeline fingerprints its stages, so both sides plan the same run."""
    import hashlib

    from .. import spec as spec_mod
    from ..workspace import link_input

    def portable(path: str | None) -> str | None:
        if not path or not Path(path).is_absolute() or not Path(path).exists():
            return path
        name = f"{Path(path).name}_{hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:8]}"
        link_input(job, name, Path(path))
        return name

    data = json.loads(spec.to_json())
    ud = data.get("user_data") or {}
    for k in ("raw", "labeled", "test"):
        if k in ud:
            ud[k] = portable(ud[k])
    for c in data.get("classes") or []:
        if c.get("assets"):
            c["assets"]["photos"] = portable(c["assets"].get("photos"))
    qa = data.get("qa") or {}
    for k in ("exclude", "class_override", "no_color"):
        if k in qa:
            qa[k] = portable(qa[k])
    return spec_mod.from_dict(data)


def worker_script(job_name: str, plan_args: list[str], teacher: bool = False) -> str:
    """Shell script for the worker job, in the stock Isaac Sim image: the code from a presigned URL, a Python
    with TensorFlow next to Isaac's own (uv, a couple of minutes), then the whole pipeline. teacher: also a
    Python for SAM 3, when the run labels the user's photos."""
    isaac = RUNTIMES["isaac"].python
    # A download can meet a passing 503 (2026-10-03, after the GPU was paid): retry before giving up.
    fetch = (f"{isaac} -c 'import sys, time, urllib.request\n"
             "for i in range(5):\n"
             "    try: urllib.request.urlretrieve(sys.argv[1], sys.argv[2]); break\n"
             "    except Exception as e:\n"
             "        if i == 4: raise\n"
             "        print(\"download failed, retrying:\", e, flush=True); time.sleep(5 * 2 ** i)'")
    py = "/tmp/s2c/py/bin/python"
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        "S2C_T0=$(date +%s)",
        "mkdir -p /tmp/s2c/code /tmp/s2c/bin /tmp/s2c/home",
        f'{fetch} "$S2C_CODE_URL" /tmp/s2c/code.tgz',
        "tar xzf /tmp/s2c/code.tgz -C /tmp/s2c/code",
        f"{fetch} {UV_URL} /tmp/s2c/uv.tgz",
        "tar xzf /tmp/s2c/uv.tgz -C /tmp/s2c/bin --strip-components 1",
        "export UV_CACHE_DIR=/tmp/s2c/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/s2c/pythons",
        "/tmp/s2c/bin/uv venv -q --python 3.11 /tmp/s2c/py",
        f"/tmp/s2c/bin/uv pip install -q --python {py} " + " ".join(shlex.quote(d) for d in WORKER_DEPS),
    ]
    if teacher:
        lines += ["/tmp/s2c/bin/uv venv -q --python 3.11 /tmp/s2c/sam3",
                  "/tmp/s2c/bin/uv pip install -q --python /tmp/s2c/sam3/bin/python "
                  + " ".join(shlex.quote(d) for d in TEACHER_DEPS),
                  "export SIM2CHIP_SAM3_PYTHON=/tmp/s2c/sam3/bin/python HF_HUB_CACHE=/tmp/s2c/hf"]
    lines += [
        f"export PYTHONUNBUFFERED=1 SIM2CHIP_ISAAC_PYTHON={isaac} SIM2CHIP_PYTHON={py} PYTHONPATH=/tmp/s2c/code "
        "SIM2CHIP_RUNS_ON=nebius S2C_SETUP_S=$(( $(date +%s) - S2C_T0 ))",
        f"{py} -m sim2chip.cloud_worker --home /tmp/s2c/home --job {shlex.quote(job_name)} "
        + " ".join(shlex.quote(a) for a in plan_args),
    ]
    return "\n".join(lines) + "\n"


class NebiusCLI:
    def __init__(self, exe: str | None = None, project: str | None = None, dry_run: bool = False):
        self.exe = exe or _env("SIM2CHIP_NEBIUS_CLI") or shutil.which("nebius") or str(
            Path.home() / ".nebius/bin/nebius")
        self.project = project or _env("SIM2CHIP_NEBIUS_PROJECT")
        self.dry_run = dry_run

    def __call__(self, *args: str, check: bool = True) -> str:
        cmd = [self.exe, *args]
        if self.dry_run:
            print("  $ " + shlex.join(cmd), flush=True)
            return ""
        r = subprocess.run(cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{shlex.join(cmd[:4])}...: {r.stderr.strip()[-500:]}")
        return r.stdout

    def _parent(self) -> list[str]:
        return ["--parent-id", self.project] if self.project else []

    def ensure_bucket(self, name: str) -> None:
        if self.dry_run:
            self("storage", "bucket", "create", "--name", name, *self._parent())
            return
        if not self("storage", "bucket", "get-by-name", "--name", name, *self._parent(), check=False).strip():
            try:
                self("storage", "bucket", "create", "--name", name, *self._parent())
            except RuntimeError as e:  # the lookup can fail for a moment (seen 2026-09-27) on a bucket we own
                if "BucketAlreadyOwnedByYou" not in str(e):
                    raise

    def create_job(self, name: str, runtime: Runtime, env: dict[str, str], script: Path, timeout_h: int,
                   env_secrets: list[str]) -> str:
        args = ["ai", "job", "create", "--name", name, "--image", runtime.image, "--platform", PLATFORM,
                "--preset", PRESET, "--on-demand", "--timeout", f"{timeout_h}h", *self._parent(),
                "--inject-file", f"{script}:/opt/s2c_stage.sh", "--container-command", "bash",
                "--args", "/opt/s2c_stage.sh", "--async"]
        for k, v in {**runtime.env, **env}.items():
            args += ["--env", f"{k}={v}"]
        for s in env_secrets:
            args += ["--env-secret", s]
        out = self(*args)
        if self.dry_run:
            return "<job id>"
        for line in out.splitlines():
            if "Job ID:" in line:
                return line.split("Job ID:", 1)[1].strip()
        raise RuntimeError(f"no job id in: {out[-300:]}")

    def job_state(self, job_id: str) -> str:
        for line in self("ai", "job", "get", job_id).splitlines():
            if line.strip().startswith("State:"):
                return line.split(":", 1)[1].strip()
        return "UNKNOWN"

    def cancel(self, job_id: str) -> None:
        self("ai", "job", "cancel", job_id, check=False)

    def job_logs(self, job_id: str) -> str:
        return self("ai", "job", "logs", job_id, check=False)


class NebiusBackend:
    name = "nebius"

    def __init__(self, verbose: bool = False, dry_run: bool = False, max_usd: float | None = None,
                 cli: NebiusCLI | None = None, store=None, poll_s: float = 20.0):
        self.verbose, self.dry_run, self.poll_s = verbose, dry_run, poll_s
        self.max_usd = max_usd if max_usd is not None else float(_env("SIM2CHIP_MAX_USD", "5"))
        self.cli = cli or NebiusCLI(dry_run=dry_run)
        self._store = store
        self.prefix = _env("SIM2CHIP_BUCKET_PREFIX", "sim2chip")
        self.spent_usd = 0.0
        self._ready: set[str] = set()

    @property
    def store(self):
        if self._store is None:
            self._store = S3Store(_env("AWS_ACCESS_KEY_ID"), _env("AWS_SECRET_ACCESS_KEY"))
        return self._store

    def buckets(self, job: Job) -> tuple[str, str]:
        """The job's own bucket and the shared cache. A job name can come back (another server, a wiped
        home), so a new job adds a random id kept in its folder; jobs that already ran in the cloud keep
        their old name, and their files in it."""
        id_file = job.root / "cloud_id"
        if not id_file.exists() and not (job.logs / "cloud_usage.jsonl").exists():
            id_file.parent.mkdir(parents=True, exist_ok=True)
            id_file.write_text(token_hex(4))
        cloud_id = id_file.read_text().strip() if id_file.exists() else ""
        suffix = f"-{cloud_id}" if cloud_id else ""
        base = f"{self.prefix}-{job.name}".lower().replace("_", "-")[:63 - len(suffix)]
        return base + suffix, f"{self.prefix}-cache"

    def _upload_job(self, job: Job, cache: Cache, bucket: str, cache_bucket: str) -> str:
        """Buckets, code, spec and input (once per run). Returns the code's presigned URL."""
        if self.dry_run:
            print(f"  upload code, spec.json and input/ to s3://{bucket}", flush=True)
            return "<presigned url>"
        if bucket not in self._ready:
            for b in (bucket, cache_bucket):
                self.cli.ensure_bucket(b)
            self.store.put(bucket, "_code/sim2chip.tar.gz", code_archive())
            if job.input.exists():
                n = self.store.upload_dir(bucket, job.input, "input")
                print(f"  uploaded {n} input files to s3://{bucket}", flush=True)
            # Input the job no longer has must not reach the worker (it would label and train on it).
            # Walked as upload_dir walks it: input folders are symlinks, which rglob does not enter (it found
            # nothing there and deleted every photo just uploaded, 2026-09-27).
            local = {(Path(d) / n).relative_to(job.root).as_posix()
                     for d, _, names in os.walk(job.input, followlinks=True) for n in names}
            for key in self.store.listing(bucket, "input/"):
                if key not in local:
                    self.store.delete(bucket, key)
            self.store.upload_dir(cache_bucket, cache.db, "db")  # share what this machine learned about assets
            self._ready.add(bucket)
        self.store.put(bucket, "spec.json", job.spec_path.read_bytes())
        return self.store.url(bucket, "_code/sim2chip.tar.gz")

    def prepare(self, pipe):
        """Local folders named in the spec travel as job input, and what the cloud cannot do yet is swapped
        for what it can (3D from photos). The stages are built anew from the new spec."""
        from ..pipeline import Pipeline
        return Pipeline(portable_spec(cloud_spec(pipe.spec), pipe.job), pipe.job, pipe.cache)

    def run(self, pipe, start: str | None = None, only: str | None = None, force: bool = False) -> None:
        if self.worker_mode:  # the whole run in one cloud job
            self.run_pipeline(pipe, start=start, only=only, force=force)
        elif self.dry_run:  # stage mode: print each stage's job
            pipe.job.spec_path.write_text(pipe.spec.to_json())
            for p in pipe.plan(start, only, force):
                if p.action == "run":
                    print(f"> {p.stage.name} ({p.stage.powered_by(pipe.spec, self.name)})")
                    self.run_stage(p.stage, pipe.job, pipe.cache, p.fingerprint)
        else:
            pipe.run(self, start=start, only=only, force=force)

    @property
    def worker_mode(self) -> bool:
        return _env("SIM2CHIP_NEBIUS_MODE", "worker") != "stage"

    def run_pipeline(self, pipe, start: str | None = None, only: str | None = None, force: bool = False,
                     log=print) -> None:
        """Worker mode: the whole run in one job. Progress lines and done markers come back while it runs."""
        from ..pipeline import StageFailed

        plans = pipe.plan(start, only, force)
        todo = [p.stage for p in plans if p.action == "run"]
        for p in plans:
            if p.action == "skip" and only is None:
                log(f"= {p.stage.name}: up to date")
        if not todo:
            return
        job, cache = pipe.job, pipe.cache
        teacher = any(st.name == "label" for st in todo)
        if teacher and pipe.spec.task == "count" and not self.dry_run:
            # SAM 3's weights are gated: they will come inside its own Nebius image, with no token (user,
            # 2026-10-03). Until that image exists, stop here instead of failing after the GPU is paid.
            raise RuntimeError("counting labels the photos with SAM 3, which has no Nebius image yet: label them "
                               "on a machine with a GPU (SIM2CHIP_SAM3_PYTHON) or bring labeled photos")
        if self.spent_usd + WORKER_HOURS * USD_PER_HOUR > self.max_usd:
            raise RuntimeError(f"a cloud run can cost up to ${WORKER_HOURS * USD_PER_HOUR:.2f} (timeout "
                               f"{WORKER_HOURS} h): over the cap of ${self.max_usd:.2f} (SIM2CHIP_MAX_USD)")
        job.spec_path.write_text(pipe.spec.to_json())
        job_bucket, cache_bucket = self.buckets(job)
        t_up = time.time()
        code_url = self._upload_job(job, cache, job_bucket, cache_bucket)
        if not self.dry_run:
            self.store.upload_dir(cache_bucket, cache.root / "labels", "labels")  # teacher answers known here
            for st in todo:  # stale markers must not pass as success
                self.store.delete(job_bucket, f"{st.name}/{DONE}")
                job.clear_done(st.name)
            for old in ("logs/cloud_pipeline.log", "logs/timeline.jsonl"):  # nor an old run's log as progress
                self.store.delete(job_bucket, old)
            (job.logs / "cloud_pipeline.log").unlink(missing_ok=True)
        plan_args = [a for flag, v in (("--from", start), ("--only", only)) if v for a in (flag, v)]
        plan_args += ["--force"] if force else []
        script = job.logs / "worker.nebius.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(worker_script(job.name, plan_args, teacher))
        secrets = self._secrets(any(st.calls_llm(pipe.spec) for st in todo))
        env = {"S2C_JOB_BUCKET": job_bucket, "S2C_CACHE_BUCKET": cache_bucket, "S2C_CODE_URL": code_url,
               "SIM2CHIP_REGION": REGION}
        if os.environ.get("SIM2CHIP_LABEL_SAMPLE"):  # calibration: the teacher marks only a few photos
            env["SIM2CHIP_LABEL_SAMPLE"] = os.environ["SIM2CHIP_LABEL_SAMPLE"]
        name = f"s2c-{job.name}-run".lower().replace("_", "-")[:63]
        job.timeline("upload job", time.time() - t_up, "local")
        t0 = time.time()
        job_id = self.cli.create_job(name, RUNTIMES["isaac"], env, script, WORKER_HOURS, secrets)
        if self.dry_run:
            return
        log(f"  nebius job {job_id} ({PLATFORM}, {PRESET}), whole pipeline in one job")
        # While this file exists a cloud job may be running and billing: the web reads it (and can cancel the
        # job) even when the process that launched it is gone.
        marker = job.logs / CLOUD_JOB
        marker.write_text(json.dumps({"id": job_id, "started": time.time(), "hours": WORKER_HOURS + 1}))
        state = self._follow(job_id, job, job_bucket, log, [st.name for st in todo])  # if this fails the marker stays, to find the job
        marker.unlink(missing_ok=True)
        hours = (time.time() - t0) / 3600
        self.spent_usd += hours * USD_PER_HOUR
        self._record(job, "run", job_id, state, hours)
        (job.logs / "worker.job.log").write_text(self.cli.job_logs(job_id))
        t1 = time.time()
        self._fetch_results(job, cache, job_bucket, cache_bucket)
        job.timeline("download results", time.time() - t1, "local")
        print_timeline(job, log, since=t_up)
        log(f"  {state} after {hours * 60:.1f} min, about ${hours * USD_PER_HOUR:.2f}")
        missing = [st.name for st in todo if not job.done(st.name)]
        if missing:
            raise StageFailed(f"cloud run {state}: stages not done {missing}, see {job.logs / 'worker.job.log'} "
                              f"and {job.logs / 'cloud_pipeline.log'}")

    def _follow(self, job_id: str, job: Job, bucket: str, log, stages: list[str] | None = None) -> str:
        """Waits for the job, echoing new lines of the cloud pipeline log (the web reads them from the local
        log) and bringing done markers back as stages finish."""
        seen, last, since = 0, None, time.time()
        # The job times out after WORKER_HOURS of running; the extra hour covers the queue and the image pull.
        deadline = since + (WORKER_HOURS + 1) * 3600
        misses, store_error = 0, None
        while True:
            try:
                state = self.cli.job_state(job_id)
            except RuntimeError as e:  # a transient CLI failure must not orphan a job that is still billing
                log(f"  could not read the job state: {e}")
                state = "UNKNOWN"
            misses = misses + 1 if state == "UNKNOWN" else 0
            if misses >= 10 or time.time() > deadline:
                raise RuntimeError(f"lost track of Nebius job {job_id} (state {state}): check it with "
                                   f"'nebius ai job get {job_id}' and cancel it if it still runs")
            if state != last and state != "UNKNOWN":
                now = time.time()
                if last is not None:  # how long the job stayed in each state: queue, image pull, run
                    job.timeline(f"nebius job: {last.lower()}", now - since, "nebius")
                log(f"  state {state}")
                last, since = state, now
            try:
                self.store.download(bucket, "logs", job.logs, ("cloud_pipeline.log",))
                # Only the markers of the stages that run: listing the bucket means thousands of render frames.
                keys = [f"{st}/{DONE}" for st in stages] if stages is not None else \
                    [k for k in self.store.listing(bucket) if k.endswith("/" + DONE)]
                for key in keys:
                    try:
                        (job.root / key).parent.mkdir(parents=True, exist_ok=True)  # the stage's folder is new here
                        self.store.s3.download_file(bucket, key, str(job.root / key))
                    except Exception as e:  # not written yet
                        if "404" not in str(e) and "NoSuchKey" not in str(e) and "Not Found" not in str(e):
                            raise
            except Exception as e:  # nothing uploaded yet, or a transient error; say it once
                if store_error is None and "NoSuchKey" not in str(e) and "404" not in str(e):
                    log(f"  could not read the job bucket yet: {e}")
                store_error = e
            cloud_log = job.logs / "cloud_pipeline.log"
            lines = cloud_log.read_text(errors="replace").splitlines() if cloud_log.exists() else []
            if len(lines) < seen:  # a new log replaced an older one
                seen = 0
            for line in lines[seen:]:
                if line.startswith(("> ", "= ", "  ", "FAILED", "model:", "real test")):
                    log(line)
            seen = max(seen, len(lines))
            if state in TERMINAL:
                return state
            time.sleep(self.poll_s)

    def _fetch_results(self, job: Job, cache: Cache, bucket: str, cache_bucket: str) -> None:
        import fnmatch
        import random

        keys = list(self.store.listing(bucket))
        want = [k for k in keys if any(fnmatch.fnmatch(k, p) for p in WORKER_RESULTS) and k != "logs/timeline.jsonl"
                and (not k.startswith("render/") or k in ("render/labels.json", f"render/{DONE}"))]  # not 2 json per frame
        frames = sorted(k for k in keys if k.startswith("render/rgb_"))
        want += random.Random(0).sample(frames, min(RENDER_SAMPLE, len(frames)))  # for review
        for key in want:
            out = job.root / key
            out.parent.mkdir(parents=True, exist_ok=True)
            self.store.s3.download_file(bucket, key, str(out))
        if "logs/timeline.jsonl" in keys:  # the cloud's steps go after this machine's
            cloud = self.store.s3.get_object(Bucket=bucket, Key="logs/timeline.jsonl")["Body"].read().decode()
            with open(job.logs / "timeline.jsonl", "a") as f:
                f.write(cloud)
        for prefix in ("db", "labels"):  # what the run learned, for later runs here
            self.store.download(cache_bucket, prefix, cache.root / prefix, ("*",), overwrite=False)

    def _secrets(self, llm: bool) -> list[str]:
        s3_secret = _env("SIM2CHIP_S3_SECRET", "sim2chip-s3")
        secrets = [f"AWS_ACCESS_KEY_ID={s3_secret}", f"AWS_SECRET_ACCESS_KEY={s3_secret}"]
        if llm:
            secret = _env("SIM2CHIP_NEBIUS_SECRET")
            if not secret and not self.dry_run:
                raise RuntimeError("the run calls Token Factory: set SIM2CHIP_NEBIUS_SECRET")
            secrets.append(f"NEBIUS_API_KEY={secret or '<secret>'}")
        return secrets

    def run_stage(self, stage: Stage, job: Job, cache: Cache, fp: str) -> None:
        if stage.runtime not in RUNTIMES:
            raise NotImplementedError(f"stage {stage.name} (runtime {stage.runtime}) has no cloud image yet: "
                                      "run it with the local backend")
        timeout_h = stage.max_hours
        if self.spent_usd + timeout_h * USD_PER_HOUR > self.max_usd:
            raise RuntimeError(f"stage {stage.name} could cost up to ${timeout_h * USD_PER_HOUR:.2f} "
                               f"(timeout {timeout_h} h) and ${self.spent_usd:.2f} is already spent: over the "
                               f"cap of ${self.max_usd:.2f} (SIM2CHIP_MAX_USD)")
        job_bucket, cache_bucket = self.buckets(job)
        code_url = self._upload_job(job, cache, job_bucket, cache_bucket)
        if not self.dry_run:
            self.store.delete(job_bucket, f"{stage.name}/{DONE}")  # a stale marker must not pass as success

        script = job.logs / f"{stage.name}.nebius.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(bootstrap(stage, fp))
        secrets = self._secrets(stage.needs_llm)
        name = f"s2c-{job.name}-{stage.name}".lower().replace("_", "-")[:63]
        t0 = time.time()
        env = {"S2C_JOB_BUCKET": job_bucket, "S2C_CACHE_BUCKET": cache_bucket, "S2C_CODE_URL": code_url,
               "SIM2CHIP_REGION": REGION}
        job_id = self.cli.create_job(name, RUNTIMES[stage.runtime], env, script, timeout_h, secrets)
        if self.dry_run:
            return
        print(f"  nebius job {job_id} ({PLATFORM}, {PRESET})", flush=True)
        state = self._wait(job_id)
        hours = (time.time() - t0) / 3600
        self.spent_usd += hours * USD_PER_HOUR
        self._record(job, stage.name, job_id, state, hours)

        # The job log (nebius ai job logs) is only its last lines: the full stage log comes from the bucket.
        tail = self.cli.job_logs(job_id)
        (job.logs / f"{stage.name}.job.log").write_text(tail)
        stage_log = job.logs / f"{stage.name}.log"
        stage_log.unlink(missing_ok=True)
        self.store.download(job_bucket, "logs", job.logs, (stage_log.name, "*.jsonl"))  # + Token Factory usage
        lines = stage_log.read_text(errors="replace").splitlines() if stage_log.exists() else []
        for line in lines + [l for l in tail.splitlines() if l not in lines]:
            if self.verbose or line.startswith(PREFIX):
                print("  " + line.removeprefix(PREFIX).strip(), flush=True)
        self.store.download(job_bucket, stage.name, job.stage_dir(stage.name), (DONE, *stage.results))
        if stage.uses_db:  # new asset reviews made in the cloud
            self.store.download(cache_bucket, "db", cache.db, ("*.jsonl",), overwrite=False)
        print(f"  {state} after {hours * 60:.1f} min, about ${hours * USD_PER_HOUR:.2f} "
              f"(run total ${self.spent_usd:.2f})", flush=True)

    def _wait(self, job_id: str) -> str:
        last = None
        while True:
            state = self.cli.job_state(job_id)
            if state != last:
                print(f"  state {state}", flush=True)
                last = state
            if state in TERMINAL:
                return state
            time.sleep(self.poll_s)

    @staticmethod
    def _record(job: Job, stage: str, job_id: str, state: str, hours: float) -> None:
        rec = {"t": round(time.time()), "stage": stage, "job_id": job_id, "state": state, "platform": PLATFORM,
               "preset": PRESET, "hours": round(hours, 4), "usd_estimate": round(hours * USD_PER_HOUR, 4)}
        with open(job.logs / "cloud_usage.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
