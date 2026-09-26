"""Common interface for pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..hardware import Hardware
from ..hardware import get as get_hardware
from ..spec import Spec
from ..workspace import Cache, Job


@dataclass
class Context:
    spec: Spec
    job: Job
    cache: Cache

    @property
    def hardware(self) -> Hardware:
        return get_hardware(self.spec.hardware)

    def out(self, stage: str):
        d = self.job.stage_dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def log(msg: str) -> None:
        # Prefixed so the local backend can show our lines and hide Isaac Sim's noise.
        print(f"[sim2chip] {msg}", flush=True)


RUNTIME_TECH = {"isaac": "NVIDIA Isaac Sim", "tf": "TensorFlow", "trellis": "TRELLIS"}


class Stage:
    """A pipeline step. It reads and writes only inside the job folder (and the shared cache).

    `uses` lists the spec sections the stage depends on and `after` the upstream stages whose
    output it reads: both go into the stage fingerprint, so a stage reruns only when something it
    depends on changed. Bump `version` when the stage code changes its output.
    """
    name: ClassVar[str]
    runtime: ClassVar[str] = "cpu"          # "cpu" | "tf" | "isaac" | "trellis"
    gpu: ClassVar[str | None] = None        # None | "any" | "rtx"
    uses: ClassVar[tuple[str, ...]] = ()
    after: ClassVar[tuple[str, ...]] = ()
    version: ClassVar[str] = "1"
    # Cloud execution: job timeout, output files (glob, relative to the stage folder) brought back to the
    # local job folder, and whether the stage calls Token Factory (then it gets the API key secret).
    max_hours: ClassVar[int] = 1
    results: ClassVar[tuple[str, ...]] = ("*.json",)
    needs_llm: ClassVar[bool] = False
    reads_input: ClassVar[bool] = False    # needs the job's input/ folder
    uses_cache: ClassVar[bool] = False     # reads or writes the shared asset cache
    uses_db: ClassVar[bool] = False        # reads or writes the asset knowledge base (cache/db)

    def enabled(self, spec: Spec) -> bool:
        return True

    def calls_llm(self, spec: Spec) -> bool:
        """Whether this run of the stage calls Token Factory (some stages only do in some modes)."""
        return self.needs_llm

    def powered_by(self, spec: Spec, backend: str) -> str:
        """What the stage runs on, as shown to the user: the technology and where it runs. Only what really
        happens: a local run says local, never Nebius AI Cloud."""
        tech = [t for t in (RUNTIME_TECH.get(self.runtime), "Nebius Token Factory" if self.calls_llm(spec) else None)
                if t]
        where = "Nebius AI Cloud, L40S" if backend == "nebius" else "local GPU" if self.gpu else "local CPU"
        return f"{' + '.join(tech)} | {where}" if tech else where

    def run(self, ctx: Context) -> dict | None:
        """Do the work. The optional returned dict is stored in the stage's _done.json."""
        raise NotImplementedError
