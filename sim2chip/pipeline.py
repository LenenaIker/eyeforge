"""Builds the stage list for a spec and runs it through a backend, skipping stages already done."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .spec import Spec, fingerprint
from .stages import STAGES, Stage
from .workspace import Cache, Job


class Backend(Protocol):
    """Strategy: where and how the stages run. With Stage it forms a bridge: any stage runs on any
    backend (patterns: docs/architecture.md)."""
    name: str  # where the stages' own processes run, as shown to the user: "local" or "nebius"

    def prepare(self, pipe: "Pipeline") -> "Pipeline":
        """The pipeline as this backend can run it (the cloud rewrites local paths and local-only options)."""

    def run(self, pipe: "Pipeline", start: str | None = None, only: str | None = None, force: bool = False) -> None:
        """Run the planned stages; raises StageFailed. Usually pipe.run(self, ...)."""

    def run_stage(self, stage: Stage, job: Job, cache: Cache, fp: str) -> None:
        """Run one stage to completion. It must leave job/<stage>/_done.json with `fp`, or raise."""


class StageFailed(RuntimeError):
    pass


@dataclass
class Plan:
    stage: Stage
    fingerprint: str
    action: str  # "run" | "skip"


class Pipeline:
    def __init__(self, spec: Spec, job: Job, cache: Cache, stages: list[Stage] | None = None):
        self.spec, self.job, self.cache = spec, job, cache
        pool = stages if stages is not None else STAGES
        known = {s.name for s in pool}
        for s in pool:
            unknown = [a for a in s.after if a not in known]
            if unknown:
                raise ValueError(f"stage {s.name} depends on unknown stages {unknown}")
        # Upstream stages disabled by the spec (e.g. no user photos -> no ingest) are simply absent.
        self.stages = [s for s in pool if s.enabled(spec)]

    def fingerprints(self) -> dict[str, str]:
        """Stage fingerprint: its code version, the spec sections it uses and its upstream fingerprints.

        Files referenced by the spec (user photos, review lists) are identified by path, not content:
        after editing them in place, rerun with --from. Empty sections are left out, so a stage can start
        using a new optional section (e.g. props) without invalidating existing jobs.
        """
        fps: dict[str, str] = {}
        for s in self.stages:
            used = {k: v for k in s.uses if (v := self.spec.section(k)) not in (None, [], {})}
            fps[s.name] = fingerprint(s.name, s.version, used,
                                      {a: fps[a] for a in s.after if a in fps})
        return fps

    def plan(self, start: str | None = None, only: str | None = None, force: bool = False) -> list[Plan]:
        names = [s.name for s in self.stages]
        for n in (start, only):
            if n is not None and n not in names:
                raise ValueError(f"unknown or disabled stage '{n}', stages for this spec: {names}")
        fps = self.fingerprints()
        # --from reruns the stage and what depends on it, not every stage listed after it: new user photos
        # must not re-render the synthetic data.
        forced = set(names) if force else {start} | _dependents(self.stages, start) if start else set()
        plans = []
        for s in self.stages:
            done = self.job.done(s.name)
            fresh = done is not None and done.get("fingerprint") == fps[s.name]
            if only is not None:
                action = "run" if s.name == only else "skip"
                if s.name != only and not fresh and names.index(s.name) < names.index(only) and s.name in _deps(
                        self.stages, only):
                    raise ValueError(f"--only {only}: upstream stage {s.name} is not done or is stale")
            else:
                action = "run" if s.name in forced or not fresh else "skip"
            plans.append(Plan(s, fps[s.name], action))
        return plans

    def run(self, backend: Backend, start: str | None = None, only: str | None = None, force: bool = False,
            log=print) -> None:
        self.job.spec_path.write_text(self.spec.to_json())
        for p in self.plan(start, only, force):
            if p.action == "skip":
                if only is None:
                    log(f"= {p.stage.name}: up to date")
                continue
            where = backend.name
            log(f"> {p.stage.name} ({p.stage.powered_by(self.spec, where)})")
            self.job.clear_done(p.stage.name)
            t0 = time.time()
            try:
                backend.run_stage(p.stage, self.job, self.cache, p.fingerprint)
            except StageFailed:
                raise
            except (RuntimeError, NotImplementedError, OSError) as e:  # could not run it at all (cloud, interpreter)
                self.job.timeline(p.stage.name, time.time() - t0, where, failed=True)
                raise StageFailed(f"stage {p.stage.name} could not run: {e}") from e
            seconds = time.time() - t0
            done = self.job.done(p.stage.name)
            if not done or done.get("fingerprint") != p.fingerprint:
                self.job.timeline(p.stage.name, seconds, where, failed=True)
                raise StageFailed(f"stage {p.stage.name} did not finish, see {self.job.logs / (p.stage.name + '.log')}")
            self.job.timeline(p.stage.name, seconds, where)
            log(f"  {p.stage.name}: done in {seconds:.0f} s")


def _dependents(stages: list[Stage], name: str) -> set[str]:
    """Stages that read `name`'s output, directly or through others, except those that do not follow an
    upstream rerun (and what hangs only from them)."""
    out, todo = set(), [name]
    while todo:
        top = todo.pop()
        for s in stages:
            if top in s.after and s.name not in out and s.follows_upstream:
                out.add(s.name)
                todo.append(s.name)
    return out


def _deps(stages: list[Stage], name: str) -> set[str]:
    by = {s.name: s for s in stages}
    out, todo = set(), [name]
    while todo:
        for a in by[todo.pop()].after:
            if a in by and a not in out:
                out.add(a)
                todo.append(a)
    return out
