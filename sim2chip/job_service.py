"""The work behind the web app, without HTTP: checking the teacher before the first run (calibration), asking the
warm GPU for teacher answers ahead of time, launching and cancelling runs, and the cloud budget.

server.py turns requests into these calls and their refusals into HTTP errors; anything else (the CLI, a test)
can drive the same work. One JobService per server process: it holds the runs and calibrations in flight.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import spec as spec_mod
from . import warm as warm_mod
from .workspace import REPO, Cache, Job, content_key, note_run

CALIBRATION = "calibration.json"  # a new job whose teacher the user checks before the first run
# Photos the teacher marks per calibration round (user, 2026-10-28: 6 were too few to correct it): at least
# CALIBRATION_PHOTOS, a share of a large dataset (10 % from 150 photos on), at most CALIBRATION_MAX a round.
CALIBRATION_PHOTOS = 15
CALIBRATION_SHARE = 0.1
CALIBRATION_MAX = 40


def _stopped(state: Path) -> bool:
    """The user stopped the teacher (JobService.cancel): its process ending in an error is no failure."""
    try:
        return json.loads(state.read_text()).get("state") == "stopped"
    except (OSError, ValueError):
        return False


def calibration_photos(total: int) -> int:
    """The photos one calibration round shows, for a dataset of `total` photos (all of a smaller one)."""
    n = max(CALIBRATION_PHOTOS, min(CALIBRATION_MAX, math.ceil(total * CALIBRATION_SHARE)))
    return min(n, total) if total else n


def job_photos(job: Job) -> int:
    """How many photos the job's labeling has (input/labeling.json), 0 without one."""
    from .library import KEEP
    try:
        return len(json.loads((job.input / KEEP).read_text())["photos"])
    except (OSError, ValueError, KeyError):
        return 0
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
    or None; scenes: the scene GPU (a WarmGPU of kind "isaac"), asked for when a run renders, or None. max_runs
    counts only on a public server."""

    def __init__(self, home: Path, backend: str = "local", max_usd: float | None = None, launcher=None,
                 warm=None, public: bool = False, max_runs: int = 2, daily_usd: float | None = None,
                 on_labels=None, scenes=None):
        self.home, self.backend, self.max_usd = home, backend, max_usd
        self.on_labels = on_labels  # on_labels(job): the teacher marked photos of the job (library.take_teacher)
        self.launcher = launcher or launch_run
        self.warm, self.public, self.max_runs, self.daily_usd = warm, public, max_runs, daily_usd
        self.scenes = scenes
        self.rendering: set[str] = set()  # the jobs whose run renders (the scene GPU's work)
        self.runs: dict[str, subprocess.Popen] = {}
        self.calibrations: dict[str, threading.Thread] = {}
        self.prefetching: set[str] = set()
        self.drafts: dict[str, tuple] = {}  # draft folder -> (photos, spec) waiting for prefetch_draft
        self._lock = threading.Lock()  # calibration threads and requests may touch the same job at once

    @property
    def cache(self) -> Cache:
        return Cache(self.home / "cache")

    # Runs

    def start(self, job: Job, request: Path, start: str | None = None) -> None:
        (job.logs / "launched_at").write_text(f"{time.time():.3f}")
        note_run(job, start)
        self.rendering.discard(job.name)
        if self.scenes is not None and self.over_budget() is None and self._renders(job, request, start):
            # Asked before the run's upload, so it is starting by the time the run looks for it (nebius.warm_for).
            # The runs that render now, this one too, pick its machine (nebius.scene_platforms).
            runs = 1 + sum(1 for name in self.rendering if (p := self.runs.get(name)) is not None and p.poll() is None)
            self.rendering.add(job.name)
            self.scenes.start("a run that renders", job.name, runs=runs)
        self.runs[job.name] = self.launcher(self.home, job, request, self.backend, self.max_usd, start=start)

    def _renders(self, job: Job, request: Path, start: str | None) -> bool:
        """The run has a stage to run that renders in Isaac Sim (the scene GPU's work)."""
        from .pipeline import Pipeline
        try:
            plans = Pipeline(spec_mod.load(request), job, self.cache).plan(start, retry=True)
        except Exception as e:  # the run says why itself; no scene GPU for it
            print(f"{job.name}: run not planned here ({type(e).__name__}: {e})", flush=True)
            return False
        return any(p.action == "run" and (p.stage.runtime == "isaac" or p.stage.gpu == "rtx") for p in plans)

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
        for name in (CALIBRATION, LABELING):  # the teacher's check or labeling: stopped, not failed
            f = job.root / name
            try:
                state = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if state.get("state") == "labeling":
                f.write_text(json.dumps({**state, "state": "stopped"}))
        job.logs.mkdir(parents=True, exist_ok=True)
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
                self._labels_done(job)
                state.write_text(json.dumps({"state": "ready"}))
                if self.warm is not None:  # the prefetch follows at once; the user's check itself needs no GPU
                    warm_mod.note(self.home, "check", job.name)
                    self.prefetch(job)
            except Exception as e:  # shown on the job page, with the option to train anyway
                if not _stopped(state):
                    state.write_text(json.dumps({"state": "failed", "error": str(e)[:300]}))

        state.write_text(json.dumps({"state": "labeling"}))
        self.calibrations[job.name] = threading.Thread(target=work, daemon=True)
        self.calibrations[job.name].start()

    def _labels_done(self, job: Job) -> None:
        if self.on_labels is not None:
            try:
                self.on_labels(job)
            except Exception as e:  # the library takes them in later (library.sync)
                print(f"{job.name}: the teacher's marks did not reach the library yet ({type(e).__name__}: {e})",
                      flush=True)

    def _label_sample(self, job: Job):
        from .stages import Context
        from .stages.label import Label
        stage = Label()
        stage.sample = calibration_photos(job_photos(job))
        return stage, Context(spec_mod.load(job.root / "request.json"), job, self.cache)

    def _calibrate_here(self, job: Job) -> None:
        stage, ctx = self._label_sample(job)
        stage.run(ctx)

    def label_all(self, job: Job) -> None:
        """A labeling job: after the check, the teacher marks every photo (the label stage alone, on this
        machine's GPU or a cloud job). The marks go to the job's labeling in the library (on_labels)."""
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
                self._labels_done(job)
                put(state="done")
            except Exception as e:
                if not _stopped(state):
                    put(state="failed", error=str(e)[:300])

        put(state="labeling")
        self.calibrations[job.name] = threading.Thread(target=work, daemon=True)
        self.calibrations[job.name].start()

    def _calibrate_in_cloud(self, job: Job, sample: int | None = -1) -> None:
        """sample: the photos of a calibration round (-1: the job's, calibration_photos), None for every photo."""
        if sample == -1:
            sample = calibration_photos(job_photos(job))
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
        """Count and presence calibration (SAM 3): while the user checks the first photos, the warm GPU answers for the rest, so the
        next rounds and the first run need no GPU work for the teacher (warm.py, Label.prefetch)."""
        from .stages import Context
        from .stages.label import Label
        from .tasks import finds_objects
        spec = spec_mod.load(job.root / "request.json")
        if not finds_objects(spec):
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
                NebiusBackend(max_usd=self.max_usd).prefetch_warm(job, self.cache, spec,
                                                                calibration_photos(job_photos(job)))
            except Exception as e:  # the rounds and the run then ask the teacher themselves
                print(f"prefetch {job.name} failed: {e}", flush=True)
            finally:
                self.prefetching.discard(job.name)
        threading.Thread(target=go, daemon=True).start()

    def prefetch_draft(self, owner: str, photos: list[Path], spec: spec_mod.Spec | None = None,
                       reason: str = "photos uploaded") -> None:
        """The teacher's work that needs no labeling yet, on the warm GPU while the user is still in the chat
        (user, 2026-10-28: the check came minutes after "Teach it" while the GPU had idled since the upload). With
        the photos only (an upload, a dataset the chat opened): SigLIP 2's embeddings, which every check uses to
        pick varied photos. With the proposed spec: also SAM 3's answers for its classes. Both are kept by photo
        content, so the job made from the chat finds them and its check runs here in seconds (_answers_here).
        One draft folder per owner, in drafts/ (no job of theirs); a call while one runs waits for it."""
        if self.warm is None or not photos or self.over_budget() is not None:
            return
        import hashlib

        from .tasks import finds_objects
        name = "draft_" + hashlib.sha256(owner.encode()).hexdigest()[:10]
        if spec is None or not finds_objects(spec):  # the embeddings only (Label.prefetch)
            spec = spec_mod.from_dict({"name": name, "background_class": None, "classes": [{"id": "a"}, {"id": "b"}],
                                       "user_data": {"raw": "photos"}, "generation": {"frames": 0}})
        else:
            data = json.loads(spec.to_json())
            data["user_data"] = {"raw": "photos"}
            data["generation"] = {"frames": 0}
            spec = spec_mod.from_dict(data)
        with self._lock:
            self.drafts[name] = (photos, spec)  # the newest wins: an earlier call not started yet is replaced
            if name in self.prefetching:
                return
            self.prefetching.add(name)
        self.warm_up(reason, f"draft:{name}")

        def go():
            from .backends.nebius import NebiusBackend, worker_runtime
            from .workspace import open_job
            try:
                while True:
                    with self._lock:
                        got = self.drafts.pop(name, None)
                    if got is None:
                        return
                    files, draft = got
                    job = open_job(self.home / "drafts", name)
                    shutil.rmtree(job.input / "photos", ignore_errors=True)
                    (job.input / "photos").mkdir(parents=True)
                    for f in files:  # by content: the same photo twice is one
                        dst = job.input / "photos" / f"{content_key(f)}{f.suffix.lower()}"
                        if not dst.exists():
                            try:
                                os.link(f, dst)
                            except OSError:
                                shutil.copyfile(f, dst)
                    be = NebiusBackend(max_usd=self.max_usd)
                    for _ in range(20):  # the warm GPU asked for just now: its record comes in a moment
                        if be.warm_for(self.cache, worker_runtime(False)) is not None:
                            break
                        time.sleep(3)
                    t0 = time.time()
                    ok = be.prefetch_warm(job, self.cache, draft, calibration_photos(len(files)))
                    print(f"prefetch {name}: {len(files)} photos, {'done' if ok else 'not done'} in "
                          f"{time.time() - t0:.0f} s", flush=True)
            except Exception as e:  # the check then asks the teacher itself
                print(f"prefetch {name} failed: {e}", flush=True)
            finally:
                with self._lock:
                    self.prefetching.discard(name)
        threading.Thread(target=go, daemon=True).start()
