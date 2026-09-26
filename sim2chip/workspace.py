"""Job folders and the shared asset cache.

Stages only see plain folders. Locally they live under the data root; on Nebius the job bucket is
mounted at the same layout, so stage code does not change.

    <home>/jobs/<job>/spec.json, input/, logs/, <stage>/..., <stage>/_done.json
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
        """Spec paths are absolute or relative to the job's input folder."""
        p = Path(rel)
        return p if p.is_absolute() else self.input / p

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
