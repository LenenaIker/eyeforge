"""The work behind the web app, without HTTP: checking the teacher before the first run (calibration), asking the
warm GPU for teacher answers ahead of time, launching and cancelling runs, and the cloud budget.

server.py turns requests into these calls and their refusals into HTTP errors; anything else (the CLI, a test)
can drive the same work. One JobService per server process: it holds the runs and calibrations in flight.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import spec as spec_mod
from . import warm as warm_mod
from .workspace import REPO, Cache, Job, note_run

CALIBRATION = "calibration.json"  # a new job whose teacher the user checks before the first run
CALIBRATION_PHOTOS = 6  # photos the teacher marks per calibration round
LABELING = "labeling.json"  # a job that only labels a library dataset: {"dataset", "state"} (Datasets page)


def launch_run(home: Path, job: Job, request: Path, backend: str, max_usd: float | None,
               start: str | None = None) -> subprocess.Popen:
    """A pipeline run in its own process, logging to the job's pipeline.log."""
    cmd = [sys.executable, "-m", "sim2chip.cli", "--home", str(home), "run", str(request), "--job", job.name,
           "--backend", backend]
    if start:
        cmd += ["--from", start]
    if max_usd is not None:
        cmd += ["--max-usd", str(max_usd)]
    job.logs.mkdir(parents=True, exist_ok=True)
    with open(job.logs / "pipeline.log", "a") as log:  # the child keeps its own copy of the file
        from .version import engine
        log.write(f"EyeForge {engine()}\n")
        log.flush()
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)


def cloud_spent(home: Path, hours: float = 24) -> float:
    """Cloud GPU dollars of every job's runs that ended in the last `hours`, plus the runs still going
    (counted at their worst case, the whole timeout). The warm GPU counts once, for its whole life (warm.py):
    its tasks' records in the jobs are left out."""
    from .backends.nebius import USD_PER_HOUR
    since = time.time() - hours * 3600
    total = warm_mod.spent(home, since)
    for logs in (home / "jobs").glob("*/logs"):
        try:
            for line in (logs / "cloud_usage.jsonl").read_text().splitlines():
                rec = json.loads(line)
                if rec.get("t", 0) >= since and not rec.get("warm"):
                    total += rec.get("usd_estimate", 0.0)
        except (OSError, ValueError):
            pass
        cj = cloud_job(Job(logs.parent))
        if cj and not cj.get("warm"):
            total += cj["hours"] * USD_PER_HOUR
    return total


def cloud_job(job: Job) -> dict | None:
    """The Nebius job a run left running, while it can still be running (its timeout plus the queue)."""
    f = job.logs / "cloud_job.json"
    try:
        cj = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return cj if time.time() - cj["started"] < cj["hours"] * 3600 else None


class JobService:
    """launcher: how a run starts (launch_run, or a fake in tests). warm (warm.WarmGPU): the shared cloud GPU,
    or None. max_runs counts only on a public server."""

    def __init__(self, home: Path, backend: str = "local", max_usd: float | None = None, launcher=None,
                 warm=None, public: bool = False, max_runs: int = 2, daily_usd: float | None = None):
        self.home, self.backend, self.max_usd = home, backend, max_usd
        self.launcher = launcher or launch_run
        self.warm, self.public, self.max_runs, self.daily_usd = warm, public, max_runs, daily_usd
        self.runs: dict[str, subprocess.Popen] = {}
        self.calibrations: dict[str, threading.Thread] = {}
        self.prefetching: set[str] = set()
        self._lock = threading.Lock()  # calibration threads and requests may touch the same job at once

    @property
    def cache(self) -> Cache:
        return Cache(self.home / "cache")

    # Runs

    def start(self, job: Job, request: Path, start: str | None = None) -> None:
        (job.logs / "launched_at").write_text(f"{time.time():.3f}")
        note_run(job, start)
        self.runs[job.name] = self.launcher(self.home, job, request, self.backend, self.max_usd, start=start)

    def running(self, job: Job) -> bool:
        """This server's run process, or a cloud job left by an earlier server process (still in time)."""
        proc, cal = self.runs.get(job.name), self.calibrations.get(job.name)
        return (proc is not None and proc.poll() is None) or cloud_job(job) is not None or \
            (cal is not None and cal.is_alive())

    def busy(self) -> bool:
        """A public server already training max_runs models."""
        return self.public and sum(p.poll() is None for p in self.runs.values()) >= self.max_runs

    def cancel(self, job: Job) -> dict | None:
        """Stops the run: the local process and, on Nebius, the cloud job (which bills until it ends). Returns the
        cloud job it stopped, if any."""
        proc = self.runs.get(job.name)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        cj = cloud_job(job)
        if cj and cj.get("warm"):  # a task of the shared warm GPU: stop the task, not the GPU
            from .backends.nebius import NebiusBackend
            be = NebiusBackend()
            warm_mod.cancel(be.store, f"{be.prefix}-cache", cj["warm"], cj["task"])
            (job.logs / "cloud_job.json").unlink(missing_ok=True)
        elif cj:
            from .backends.nebius import NebiusCLI
            NebiusCLI().cancel(cj["id"])
            (job.logs / "cloud_job.json").unlink(missing_ok=True)
        with open(job.logs / "pipeline.log", "a") as f:
            f.write("FAILED: cancelled by the user\n")
        return cj

    # Budget and the warm GPU

    def over_budget(self) -> str | None:
        """Why no new run may start today (the cloud budget), or None."""
        if self.daily_usd is None or self.backend != "nebius":
            return None
        from .backends.nebius import USD_PER_HOUR, WORKER_HOURS
        spent, worst = cloud_spent(self.home), WORKER_HOURS * USD_PER_HOUR
        if spent + worst > self.daily_usd:
            return f"today's cloud budget is used up (${spent:.2f} of ${self.daily_usd:.2f}); try again tomorrow"
        return None

    def warm_up(self, reason: str, who: str | None = None) -> None:
        """Ask for the warm GPU, unless today's budget could not pay for it."""
        if self.warm is not None and self.over_budget() is None:
            self.warm.start(reason, who)

    # Calibration

    def calibrate(self, job: Job) -> None:
        """Before any training, the teacher marks a few of the user's photos. The user corrects them; the next
        round tunes the teacher to the corrections, marks the same photos again and a few new ones, so the user
        sees if it understood. The first run starts from these answers. The teacher (SAM 3) needs a GPU: this
        machine's when it has one, else a short Nebius job (the photos wait for the GPU to start)."""
        from . import teacher
        state = job.root / CALIBRATION

        def work():
            try:
                if teacher.available() or self.backend != "nebius" or self._answers_here(job):
                    self._calibrate_here(job)
                else:
                    self._calibrate_in_cloud(job)
                state.write_text(json.dumps({"state": "ready"}))
                if self.warm is not None:  # the prefetch follows at once; the user's check itself needs no GPU
                    warm_mod.note(self.home, "check", job.name)
                    self.prefetch(job)
            except Exception as e:  # shown on the job page, with the option to train anyway
                state.write_text(json.dumps({"state": "failed", "error": str(e)[:300]}))

        state.write_text(json.dumps({"state": "labeling"}))
        self.calibrations[job.name] = threading.Thread(target=work, daemon=True)
        self.calibrations[job.name].start()

    def _label_sample(self, job: Job):
        from .stages import Context
        from .stages.label import Label
        stage = Label()
        stage.sample = CALIBRATION_PHOTOS
        return stage, Context(spec_mod.load(job.root / "request.json"), job, self.cache)

    def _calibrate_here(self, job: Job) -> None:
        stage, ctx = self._label_sample(job)
        stage.run(ctx)

    def label_all(self, job: Job) -> None:
        """A labeling job: after the check, the teacher marks every photo (the label stage alone, on this
        machine's GPU or a cloud job). The library takes the marks in from the job (library.sync)."""
        from . import teacher
        state = job.root / LABELING

        def put(**change) -> None:  # whole or not at all: the job page reads it meanwhile
            tmp = state.with_suffix(".tmp")
            tmp.write_text(json.dumps({**json.loads(state.read_text()), **change}))
            tmp.replace(state)

        def work():
            try:
                if teacher.available() or self.backend != "nebius":
                    from .stages import Context
                    from .stages.label import Label
                    Label().run(Context(spec_mod.load(job.root / "request.json"), job, self.cache))
                else:
                    self._calibrate_in_cloud(job, sample=None)
                put(state="done")
            except Exception as e:
                put(state="failed", error=str(e)[:300])

        put(state="labeling")
        self.calibrations[job.name] = threading.Thread(target=work, daemon=True)
        self.calibrations[job.name].start()

    def _calibrate_in_cloud(self, job: Job, sample: int | None = CALIBRATION_PHOTOS) -> None:
        cmd = [sys.executable, "-m", "sim2chip.cli", "--home", str(self.home), "run", str(job.root / "request.json"),
               "--job", job.name, "--backend", self.backend, "--only", "label"]
        if self.max_usd is not None:
            cmd += ["--max-usd", str(self.max_usd)]
        job.logs.mkdir(parents=True, exist_ok=True)
        with open(job.logs / "pipeline.log", "a") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                    env={**os.environ, "SIM2CHIP_LABEL_SAMPLE": str(sample or 0)})
            self.runs[job.name] = proc  # cancel stops it like a run
            if proc.wait() != 0:
                lines = (job.logs / "pipeline.log").read_text(errors="replace").splitlines()
                raise RuntimeError(next((ln for ln in reversed(lines) if "error" in ln.lower() or "FAILED" in ln),
                                        f"the teacher's cloud job failed (exit {proc.returncode})"))
        if sample:
            job.clear_done("label")  # a sample, not the whole folder: the first run labels every photo

    def _answers_here(self, job: Job) -> bool:
        """Every answer the round needs is cached on this machine (the warm GPU's prefetch brought them): only
        the threshold or the centres change with the corrections, which takes seconds without a GPU."""
        try:
            stage, ctx = self._label_sample(job)
            return not stage.gpu_needed(ctx)
        except Exception as e:  # then the round asks the teacher for them
            print(f"calibration {job.name}: could not check the cached answers ({type(e).__name__}: {e})",
                  flush=True)
            return False

    def prefetch(self, job: Job) -> None:
        """Count calibration: while the user checks the first photos, the warm GPU answers for the rest, so the
        next rounds and the first run need no GPU work for the teacher (warm.py, Label.prefetch)."""
        from .stages import Context
        from .stages.label import Label
        from .tasks import task_of
        spec = spec_mod.load(job.root / "request.json")
        if not task_of(spec).points:
            return
        try:
            if not Label().gpu_needed(Context(spec, job, self.cache)):
                return  # every photo answered already
        except Exception as e:  # prefetch anyway: the warm GPU skips the photos it finds answered
            print(f"prefetch {job.name}: could not check the cached answers ({type(e).__name__}: {e})", flush=True)
        with self._lock:
            if job.name in self.prefetching:
                return
            self.prefetching.add(job.name)

        def go():
            try:
                from .backends.nebius import NebiusBackend
                NebiusBackend(max_usd=self.max_usd).prefetch_warm(job, self.cache, spec, CALIBRATION_PHOTOS)
            except Exception as e:  # the rounds and the run then ask the teacher themselves
                print(f"prefetch {job.name} failed: {e}", flush=True)
            finally:
                self.prefetching.discard(job.name)
        threading.Thread(target=go, daemon=True).start()
