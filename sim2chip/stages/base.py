"""Common interface for pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from ..hardware import Hardware
from ..hardware import get as get_hardware
from ..spec import Spec
from ..workspace import Cache, Job
from ..workspace import say


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
        say(msg)


RuntimeName = Literal["cpu", "tf", "isaac", "trellis"]
RUNTIME_TECH = {"isaac": "NVIDIA Isaac Sim", "tf": "TensorFlow", "trellis": "TRELLIS"}


class Stage:
    """A pipeline step. It reads and writes only inside the job folder (and the shared cache).
    Template method: run() is the step, the other methods are defaults a stage may override
    (patterns: docs/architecture.md).

    `uses` lists the spec sections the stage depends on and `after` the upstream stages whose
    output it reads: both go into the stage fingerprint, so a stage reruns only when something it
    depends on changed. Bump `version` when the stage code changes its output.
    """
    name: ClassVar[str]
    runtime: ClassVar[RuntimeName] = "cpu"   # the Python it runs in (backends map each to an interpreter or image)
    gpu: ClassVar[Literal["any", "rtx"] | None] = None
    uses: ClassVar[tuple[str, ...]] = ()
    after: ClassVar[tuple[str, ...]] = ()
    extra: ClassVar[tuple[str, ...]] = ()  # stages of `after` it goes on without when they were left out
    version: ClassVar[str] = "1"
    # Cloud execution: job timeout, output files (glob, relative to the stage folder) brought back to the
    # local job folder, and whether the stage calls Token Factory (then it gets the API key secret).
    max_hours: ClassVar[int] = 1
    results: ClassVar[tuple[str, ...]] = ("*.json",)
    needs_llm: ClassVar[bool] = False
    reads_input: ClassVar[bool] = False    # needs the job's input/ folder
    uses_cache: ClassVar[bool] = False     # reads or writes the shared asset cache
    uses_db: ClassVar[bool] = False        # reads or writes the asset knowledge base (cache/db)
    rereads: ClassVar[tuple[str, ...]] = ()  # own output files (relative) a rerun reads back, e.g. user corrections
    # Whether "rerun from an upstream stage" (--from) drags this stage and its own dependents along. False for a
    # stage whose output is cached by content and must stay put across versions (models made from photos): it
    # still reruns when its own fingerprint changes or when it is named.
    follows_upstream: ClassVar[bool] = True
    # What the run can do without when the stage fails (user, 2026-10-09: the pipeline must not depend on every
    # step going right): "with photos", the synthetic branch, when the user's photos train the model anyway;
    # "always", a step the model does not need (the test on real photos). The run goes on, says why, and the
    # report lists what was left out.
    skippable: ClassVar[Literal["with photos", "always"] | None] = None

    def enabled(self, spec: Spec) -> bool:
        return True

    def can_skip(self, spec: Spec) -> bool:
        ud = spec.user_data
        return self.skippable == "always" or (self.skippable == "with photos" and bool(ud.labeled or ud.raw))

    def calls_llm(self, spec: Spec) -> bool:
        """Whether this run of the stage calls Token Factory (some stages only do in some modes)."""
        return self.needs_llm

    def powered_by(self, spec: Spec, backend: str, model: str | None = None) -> str:
        """What the stage runs on, as shown to the user. Only what really happens: a local run says local,
        never Nebius AI Cloud. The backend decides where the stage's own process runs; models called through
        Token Factory always run on Nebius, whatever the backend (they do not fit a local GPU)."""
        where = self.where(backend)
        if self.calls_llm(spec):
            name = model.split("/")[-1] if model else "vision model"
            return f"{name} on Nebius Token Factory, called from {where}"
        tech = RUNTIME_TECH.get(self.runtime)
        return f"{tech} | {where}" if tech else where

    def where(self, backend: str) -> str:
        """The machine the stage's own process runs on, as shown to the user."""
        from ..config import gpu_name
        return f"Nebius AI Cloud, {gpu_name()}" if backend == "nebius" else "local GPU" if self.gpu else "local CPU"

    def run(self, ctx: Context) -> dict | None:
        """Do the work. The optional returned dict is stored in the stage's _done.json."""
        raise NotImplementedError
