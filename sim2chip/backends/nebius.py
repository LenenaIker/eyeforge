"""Nebius backend: each stage runs as a Serverless AI job on an L40S, with the job and cache buckets mounted.

Data stays in Object Storage between stages: the user uploads the spec and input once, every stage
reads and writes the mounted buckets, and only small results (done markers, reports, the exported
model) come back to the local job folder, which keeps working as the pipeline's view of the job.

    job bucket    s3://<prefix>-<job>   mounted at /s2c/job     (spec.json, input/, <stage>/, logs/)
    cache bucket  s3://<prefix>-cache   mounted at /s2c/cache   (objaverse, usd, hdri), shared by jobs

Configuration (environment or the repo's .env):
    SIM2CHIP_NEBIUS_CLI        nebius CLI (default: on PATH or ~/.nebius/bin/nebius)
    SIM2CHIP_NEBIUS_PROJECT    project id (default: the CLI profile's parent-id)
    SIM2CHIP_BUCKET_PREFIX     bucket name prefix (default sim2chip)
    SIM2CHIP_NEBIUS_SECRET     MysteryBox secret holding NEBIUS_API_KEY, for stages that call Token Factory
    SIM2CHIP_MAX_USD           spend cap for one run, estimated from GPU time (default 5)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   Object Storage key, to upload input and fetch results
"""
from __future__ import annotations

import fnmatch
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

from ..stages import Stage
from ..workspace import DONE, REPO, Cache, Job

REGION = "eu-north1"
S3_ENDPOINT = f"https://storage.{REGION}.nebius.cloud"
PLATFORM, PRESET = "gpu-l40s-a", "1gpu-8vcpu-32gb"
USD_PER_HOUR = 1.55  # on-demand L40S (Intel), nebius.com/prices, 2026-09
JOB_MOUNT, CACHE_MOUNT = "/s2c/job", "/s2c/cache"
PREFIX = "[sim2chip]"


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
    "cpu": Runtime("tensorflow/tensorflow:2.18.0-gpu", "python3", "pip install -q objaverse pillow numpy"),
}
TERMINAL = {"COMPLETED", "SUCCEEDED", "FAILED", "ERROR", "CANCELLED", "CANCELED", "STOPPED", "DELETED"}


def _env(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


def code_archive() -> bytes:
    """The sim2chip package as a tar.gz, unpacked by every stage job (no custom image needed)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted((REPO / "sim2chip").rglob("*.py")):
            tar.add(f, arcname=str(f.relative_to(REPO)))
    return buf.getvalue()


def bootstrap(stage: Stage, fp: str) -> str:
    """Shell script that runs one stage inside its container."""
    rt = RUNTIMES[stage.runtime]
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        "mkdir -p /tmp/s2c && tar xzf %s/_code/sim2chip.tar.gz -C /tmp/s2c" % JOB_MOUNT,
        "export PYTHONPATH=/tmp/s2c PYTHONUNBUFFERED=1",
        rt.setup,
        "mkdir -p %s/logs" % JOB_MOUNT,
        "%s -m sim2chip.run_stage %s --job %s --cache %s --fingerprint %s 2>&1 | tee /tmp/stage.log" % (
            rt.python, stage.name, JOB_MOUNT, CACHE_MOUNT, fp),
        "cp /tmp/stage.log %s/logs/%s.log" % (JOB_MOUNT, stage.name),
    ]
    return "\n".join(l for l in lines if l) + "\n"


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

    def bucket_id(self, name: str, create: bool = True) -> str:
        if self.dry_run:
            self("storage", "bucket", "get-by-name", "--name", name, *self._parent())
            return f"<id of {name}>"
        out = self("storage", "bucket", "get-by-name", "--name", name, *self._parent(), "--format", "json",
                   check=False)
        if not out.strip() and create:
            self("storage", "bucket", "create", "--name", name, *self._parent())
            out = self("storage", "bucket", "get-by-name", "--name", name, *self._parent(), "--format", "json")
        if not out.strip():
            raise RuntimeError(f"bucket {name} not found")
        return json.loads(out)["metadata"]["id"]

    def create_job(self, name: str, runtime: Runtime, volumes: list[str], script: Path, timeout_h: int,
                   env_secrets: list[str]) -> str:
        args = ["ai", "job", "create", "--name", name, "--image", runtime.image, "--platform", PLATFORM,
                "--preset", PRESET, "--on-demand", "--timeout", f"{timeout_h}h", *self._parent(),
                "--inject-file", f"{script}:/opt/s2c_stage.sh", "--container-command", "bash",
                "--args", "/opt/s2c_stage.sh", "--async"]
        for v in volumes:
            args += ["--volume", v]
        for k, v in runtime.env.items():
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

    def job_logs(self, job_id: str) -> str:
        return self("ai", "job", "logs", job_id, check=False)


class S3Store:
    """Object Storage through boto3 (orchestrator side only; stages see the mounted buckets)."""

    def __init__(self):
        try:
            import boto3
        except ImportError:
            raise RuntimeError("the Nebius backend needs boto3: pip install -e .[cloud]") from None
        key, secret = _env("AWS_ACCESS_KEY_ID"), _env("AWS_SECRET_ACCESS_KEY")
        if not key or not secret:
            raise RuntimeError("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (Object Storage key) are not set")
        self.s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name=REGION, aws_access_key_id=key,
                               aws_secret_access_key=secret)

    def listing(self, bucket: str, prefix: str = "") -> dict[str, int]:
        out = {}
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                out[o["Key"]] = o["Size"]
        return out

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=bucket, Key=key, Body=data)

    def upload_dir(self, bucket: str, src: Path, prefix: str) -> int:
        """Uploads files missing or with a different size (symlinks followed). Returns the count."""
        have = self.listing(bucket, prefix)
        n = 0
        for f in sorted(src.rglob("*")):
            if f.is_file():
                key = f"{prefix}/{f.relative_to(src).as_posix()}"
                if have.get(key) != f.stat().st_size:
                    self.s3.upload_file(str(f), bucket, key)
                    n += 1
        return n

    def download(self, bucket: str, prefix: str, dst: Path, patterns: tuple[str, ...]) -> int:
        n = 0
        for key in self.listing(bucket, prefix):
            rel = key[len(prefix):].lstrip("/")
            if any(fnmatch.fnmatch(rel, p) for p in patterns):
                out = dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(bucket, key, str(out))
                n += 1
        return n

    def delete(self, bucket: str, key: str) -> None:
        self.s3.delete_object(Bucket=bucket, Key=key)


class NebiusBackend:
    def __init__(self, verbose: bool = False, dry_run: bool = False, max_usd: float | None = None,
                 cli: NebiusCLI | None = None, store=None, poll_s: float = 20.0):
        self.verbose, self.dry_run, self.poll_s = verbose, dry_run, poll_s
        self.max_usd = max_usd if max_usd is not None else float(_env("SIM2CHIP_MAX_USD", "5"))
        self.cli = cli or NebiusCLI(dry_run=dry_run)
        self._store = store
        self.prefix = _env("SIM2CHIP_BUCKET_PREFIX", "sim2chip")
        self.spent_usd = 0.0
        self._uploaded: set[str] = set()

    @property
    def store(self):
        if self._store is None:
            self._store = S3Store()
        return self._store

    def buckets(self, job: Job) -> tuple[str, str]:
        name = f"{self.prefix}-{job.name}".lower().replace("_", "-")[:63]
        return name, f"{self.prefix}-cache"

    def _upload_job(self, job: Job, bucket: str) -> None:
        if self.dry_run:
            print(f"  upload code, spec.json and input/ to s3://{bucket}", flush=True)
            return
        if bucket in self._uploaded:
            self.store.put(bucket, "spec.json", job.spec_path.read_bytes())
            return
        self.store.put(bucket, "_code/sim2chip.tar.gz", code_archive())
        self.store.put(bucket, "spec.json", job.spec_path.read_bytes())
        if job.input.exists():
            n = self.store.upload_dir(bucket, job.input, "input")
            print(f"  uploaded {n} input files to s3://{bucket}", flush=True)
        self._uploaded.add(bucket)

    def run_stage(self, stage: Stage, job: Job, cache: Cache, fp: str) -> None:
        timeout_h = stage.max_hours
        if self.spent_usd + timeout_h * USD_PER_HOUR > self.max_usd:
            raise RuntimeError(f"stage {stage.name} could cost up to ${timeout_h * USD_PER_HOUR:.2f} "
                               f"(timeout {timeout_h} h) and ${self.spent_usd:.2f} is already spent: over the "
                               f"cap of ${self.max_usd:.2f} (SIM2CHIP_MAX_USD)")
        job_bucket, cache_bucket = self.buckets(job)
        job_id_b = self.cli.bucket_id(job_bucket)
        cache_id_b = self.cli.bucket_id(cache_bucket)
        self._upload_job(job, job_bucket)
        if not self.dry_run:
            self.store.delete(job_bucket, f"{stage.name}/{DONE}")  # a stale marker must not pass as success

        script = job.logs / f"{stage.name}.nebius.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(bootstrap(stage, fp))
        secrets = []
        if stage.needs_llm:
            secret = _env("SIM2CHIP_NEBIUS_SECRET")
            if not secret and not self.dry_run:
                raise RuntimeError(f"stage {stage.name} calls Token Factory: set SIM2CHIP_NEBIUS_SECRET")
            secrets.append(f"NEBIUS_API_KEY={secret or '<secret>'}")
        name = f"s2c-{job.name}-{stage.name}".lower().replace("_", "-")[:63]
        t0 = time.time()
        job_id = self.cli.create_job(name, RUNTIMES[stage.runtime],
                                     [f"{job_id_b}:{JOB_MOUNT}", f"{cache_id_b}:{CACHE_MOUNT}"], script,
                                     timeout_h, secrets)
        if self.dry_run:
            return
        print(f"  nebius job {job_id} ({PLATFORM}, {PRESET})", flush=True)
        state = self._wait(job_id)
        hours = (time.time() - t0) / 3600
        self.spent_usd += hours * USD_PER_HOUR
        self._record(job, stage.name, job_id, state, hours)

        logs = self.cli.job_logs(job_id)
        (job.logs / f"{stage.name}.log").write_text(logs)
        for line in logs.splitlines():
            if self.verbose or line.startswith(PREFIX):
                print("  " + line.removeprefix(PREFIX).strip(), flush=True)
        self.store.download(job_bucket, stage.name, job.stage_dir(stage.name), (DONE, *stage.results))
        self.store.download(job_bucket, "logs", job.logs, ("*.jsonl",))  # Token Factory usage
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
