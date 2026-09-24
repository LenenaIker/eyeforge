"""Builds the stage list for a spec and runs it through a backend, skipping stages already done."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .spec import Spec, fingerprint
from .stages import STAGES, Stage
from .workspace import Cache, Job


class Backend(Protocol):
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
        after editing them in place, rerun with --from.
        """
        fps: dict[str, str] = {}
        for s in self.stages:
            fps[s.name] = fingerprint(s.name, s.version, {k: self.spec.section(k) for k in s.uses},
                                      {a: fps[a] for a in s.after if a in fps})
        return fps

    def plan(self, start: str | None = None, only: str | None = None, force: bool = False) -> list[Plan]:
        names = [s.name for s in self.stages]
        for n in (start, only):
            if n is not None and n not in names:
                raise ValueError(f"unknown or disabled stage '{n}', stages for this spec: {names}")
        fps = self.fingerprints()
        plans, forced = [], force
        for s in self.stages:
            if start == s.name:
                forced = True
            done = self.job.done(s.name)
            fresh = done is not None and done.get("fingerprint") == fps[s.name]
            if only is not None:
                action = "run" if s.name == only else "skip"
                if s.name != only and not fresh and names.index(s.name) < names.index(only) and s.name in _deps(
                        self.stages, only):
                    raise ValueError(f"--only {only}: upstream stage {s.name} is not done or is stale")
            else:
                action = "run" if forced or not fresh else "skip"
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
            log(f"> {p.stage.name} ({p.stage.runtime}{', gpu ' + p.stage.gpu if p.stage.gpu else ''})")
            self.job.clear_done(p.stage.name)
            backend.run_stage(p.stage, self.job, self.cache, p.fingerprint)
            done = self.job.done(p.stage.name)
            if not done or done.get("fingerprint") != p.fingerprint:
                raise StageFailed(f"stage {p.stage.name} did not finish, see {self.job.logs / (p.stage.name + '.log')}")
            log(f"  {p.stage.name}: done")


def _deps(stages: list[Stage], name: str) -> set[str]:
    by = {s.name: s for s in stages}
    out, todo = set(), [name]
    while todo:
        for a in by[todo.pop()].after:
            if a in by and a not in out:
                out.add(a)
                todo.append(a)
    return out
