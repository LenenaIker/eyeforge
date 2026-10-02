"""Job folders and the shared asset cache.

Stages only see plain folders. Locally they live under the data root; on Nebius the worker copies the job
bucket to a local folder with the same layout (cloud_worker.py), so stage code does not change.

    <home>/jobs/<job>/spec.json, input/, logs/, <stage>/..., <stage>/_done.json
    <home>/jobs/<job>/versions/v<n>/  earlier models of the job (snapshot())
    <home>/cache/objaverse, cache/usd/{raw,norm}, cache/hdri, cache/textures   (disposable files)
    <home>/cache/db                                                          (asset knowledge base, keep)
    <home>/cache/labels/<prompt>/<model>/<photo>.json                        (teacher answers per photo)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DONE = "_done.json"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # photos the user can give
LOG_PREFIX = "[sim2chip]"  # marks our lines in the noisy output of Isaac Sim and TensorFlow; backends keep only these


def content_key(path: Path) -> str:
    """Identity of a photo by its bytes: the same photo in another folder or job has the same key."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def default_home() -> Path:
    return Path(os.environ.get("SIM2CHIP_HOME", REPO / "data"))


@dataclass(frozen=True)
class Cache:
    root: Path

    @property
    def objaverse(self) -> Path:
        return self.root / "objaverse"

    @property
    def usd_raw(self) -> Path:
        return self.root / "usd" / "raw"

    @property
    def usd_glb2usd(self) -> Path:
        return self.root / "usd" / "glb2usd"

    @property
    def usd_norm(self) -> Path:
        return self.root / "usd" / "norm"

    @property
    def hdri(self) -> Path:
        return self.root / "hdri"

    @property
    def db(self) -> Path:
        """Asset knowledge base (reviews of assets, not files). Everything else in the cache is disposable."""
        return self.root / "db"


@dataclass(frozen=True)
class Job:
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def spec_path(self) -> Path:
        return self.root / "spec.json"

    @property
    def input(self) -> Path:
        return self.root / "input"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def resolve_input(self, rel: str) -> Path:
        """Spec paths are absolute or relative to the job's input folder (and stay inside it)."""
        p = Path(rel)
        if p.is_absolute():
            return p
        if ".." in p.parts:
            raise ValueError(f"input path '{rel}' leaves the job's input folder")
        return self.input / p

    def done(self, stage: str) -> dict | None:
        f = self.stage_dir(stage) / DONE
        return json.loads(f.read_text()) if f.exists() else None

    def mark_done(self, stage: str, fingerprint: str, info: dict | None = None) -> None:
        d = self.stage_dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        (d / DONE).write_text(json.dumps({"fingerprint": fingerprint, **(info or {})}, indent=2))

    def clear_done(self, stage: str) -> None:
        (self.stage_dir(stage) / DONE).unlink(missing_ok=True)

    def timeline(self, step: str, seconds: float, where: str, **extra) -> None:
        """Appends how long a step took to logs/timeline.jsonl: stages, and the cloud's own phases (waiting
        for a GPU, pulling the image, setting up, moving data), to see where a run's time goes."""
        import time
        self.logs.mkdir(parents=True, exist_ok=True)
        rec = {"t": round(time.time(), 1), "step": step, "seconds": round(seconds, 1), "where": where, **extra}
        with open(self.logs / "timeline.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


VERSION_FILES = ("model_int8.tflite", "model.tflite", "labels.txt", "model_data.h", "report.json")
CURRENT_VERSION = "version.json"  # input/version.json: number and reason of the model export/ holds or will hold
PREVIOUS = "previous"             # input/previous/: the model a rerun replaces, scored again on the new test set


def current_version(job: Job) -> dict:
    f = job.input / CURRENT_VERSION
    return json.loads(f.read_text()) if f.exists() else {"n": 1, "reason": "first model"}


def versions(job: Job) -> list[dict]:
    """Earlier models of the job, oldest first (versions/v<n>/meta.json)."""
    out = [json.loads(f.read_text()) for f in (job.root / "versions").glob("v*/meta.json")]
    return sorted(out, key=lambda m: m["n"])


def snapshot(job: Job, reason: str) -> dict | None:
    """Called before a rerun that will replace the job's model (new photos, corrections, a changed spec).

    Keeps the current model in versions/v<n>/, copies it to input/previous/ for evaluate to score it on the
    same test photos as its successor, and records why the next version is made. Returns the kept version,
    or None when there is no finished model yet. A rerun that failed left the same model in export/: it is
    not kept twice, only the reason of the next version changes."""
    import shutil
    import time

    export = job.root / "export"
    report = export / "report.json"
    if not report.exists():
        return None
    cur = current_version(job)
    rep = json.loads(report.read_text())
    kept = next((m for m in versions(job) if m.get("report_key") == content_key(report)), None)
    if kept is None:
        real = rep.get("real_test") or {}
        shipped = (real.get("per_seed") or {}).get(str(rep.get("seed")))
        # The headline is the shipped model's score (the one the user got), not the mean of the seeds.
        kept = {"n": cur["n"], "reason": cur["reason"], "created": round(report.stat().st_mtime),
                "metric": rep.get("metric"),
                "headline": shipped if shipped is not None else real.get("mean", rep.get("val_score")),
                "shipped": shipped, "seeds_mean": real.get("mean"),
                "test_size": real.get("test_size"), "report_key": content_key(report),
                "model": rep.get("model_path", "model_int8.tflite")}
        d = job.root / "versions" / f"v{cur['n']}"
        d.mkdir(parents=True, exist_ok=True)
        for f in VERSION_FILES:
            if (export / f).exists():
                shutil.copy2(export / f, d / f)
        if job.spec_path.exists():
            shutil.copy2(job.spec_path, d / "spec.json")
        (d / "meta.json").write_text(json.dumps(kept, indent=1))
    prev = job.input / PREVIOUS
    shutil.rmtree(prev, ignore_errors=True)
    prev.mkdir(parents=True)
    for f in (rep.get("model_path", "model_int8.tflite"), "report.json"):
        shutil.copy2(export / f, prev / f)
    (prev / "meta.json").write_text(json.dumps({"n": kept["n"]}))
    (job.input / CURRENT_VERSION).write_text(json.dumps({"n": kept["n"] + 1, "reason": reason,
                                                         "started": round(time.time())}))
    return kept


def open_job(home: Path, name: str) -> Job:
    job = Job(home / "jobs" / name)
    for d in (job.root, job.input, job.logs):
        d.mkdir(parents=True, exist_ok=True)
    return job


def link_input(job: Job, name: str, src: Path) -> None:
    """Expose a user folder inside the job's input/ (symlink locally, upload on Nebius)."""
    dst = job.input / name
    src = src.resolve()
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src:
            return
        raise FileExistsError(f"{dst} already exists and points elsewhere")
    dst.symlink_to(src, target_is_directory=src.is_dir())


def copy_name(key: str) -> str:
    """Name of a photo's copy in label/<class>/: subfolder names are kept as a prefix."""
    p = key.split("/")
    return p[-1] if len(p) == 1 else f"{p[-2]}_{p[-1]}"


def objaverse_at(folder: Path):
    """The objaverse package, downloading into `folder` (it has no setting for this, only module globals)."""
    import objaverse
    objaverse.BASE_PATH = str(folder)
    objaverse._VERSIONED_PATH = str(Path(folder) / "hf-objaverse-v1")
    return objaverse


def say(msg: str) -> None:
    """A progress line for the logs (see LOG_PREFIX)."""
    print(f"{LOG_PREFIX} {msg}", flush=True)
