"""Local backend: each stage runs as a subprocess with the interpreter of its runtime.

Interpreters come from environment variables, with defaults for the development machine:
    SIM2CHIP_ISAAC_PYTHON  Isaac Sim's python.sh
    SIM2CHIP_PYTHON        Python with TensorFlow and the asset tools (default: the current one)
    SIM2CHIP_TRELLIS_PYTHON  Python of the image-to-3D environment (TRELLIS, torch, transformers)
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .. import config
from ..locks import GPU, hold
from ..stages import Stage
from ..subproc import python_env, run_module
from ..workspace import LOG_PREFIX as PREFIX, REPO, Cache, Job

# Where this machine keeps the other runtimes: environment or .env (see README). Isaac Sim's default is the
# standard install folder; the image-to-3D runtime has no default. Its model weights go wherever HF_HUB_CACHE
# and TORCH_HOME point (set them in .env to keep the weights off the system disk).
ISAAC_PYTHON = str(Path.home() / "isaacsim" / "python.sh")
TRELLIS_ENV = ("HF_HUB_CACHE", "TORCH_HOME")


class LocalBackend:
    def __init__(self, verbose: bool = False, before_stage=None):
        self.verbose = verbose
        # Called with (stage, job, cache) before each stage: the Nebius worker brings cached files in.
        self.before_stage = before_stage
        # Inside a Nebius worker job this machine is Nebius AI Cloud: stage labels must say so.
        self.name = os.environ.get("SIM2CHIP_RUNS_ON", "local")

    def interpreter(self, runtime: str) -> list[str]:
        if runtime == "isaac":
            exe = config.env("SIM2CHIP_ISAAC_PYTHON", ISAAC_PYTHON)
            if not Path(exe).exists():
                raise FileNotFoundError(f"Isaac Sim python not found at {exe}, set SIM2CHIP_ISAAC_PYTHON")
            return [exe]
        if runtime == "trellis":
            exe = config.env("SIM2CHIP_TRELLIS_PYTHON")
            if not exe or not Path(exe).exists():
                raise FileNotFoundError(f"image-to-3D python not found at {exe}, set SIM2CHIP_TRELLIS_PYTHON")
            return [exe]
        return [config.env("SIM2CHIP_PYTHON", sys.executable)]

    def prepare(self, pipe):
        return pipe

    def run(self, pipe, start: str | None = None, only: str | None = None, force: bool = False) -> None:
        if self.name != "local":  # a Nebius worker: its job script samples the machine (gpu_usage.SAMPLER)
            return pipe.run(self, start=start, only=only, force=force)
        # How much of this machine the run uses, per stage, as on Nebius (logs/gpu_usage.json): a run that keeps
        # the CPU full here will want a good CPU there too (user, 2026-10-13).
        from .. import gpu_usage
        samples, t0 = pipe.job.logs / "usage_samples.log", time.time()
        try:
            with gpu_usage.sampling(samples):
                pipe.run(self, start=start, only=only, force=force)
        finally:
            try:
                use = gpu_usage.write(pipe.job.logs, t0, time.time(), samples, where="local")
                a = (use or {}).get("all")
                if a and "cpu" in a:
                    print(f"machine use: GPU {a['util']}% (busy {100 * a['busy']:.0f}%), CPU {a['cpu']}% of "
                          f"{use['cpus']} cores (full {100 * a['cpu_full']:.0f}%), iowait {a['iowait']}%, "
                          f"RAM {a['ram_mb'] / 1024:.1f} GB at most", flush=True)
            except Exception as e:  # a measure, never worth the run
                print(f"machine use not summed up: {e}", flush=True)

    def run_stage(self, stage: Stage, job: Job, cache: Cache, fp: str) -> None:
        if self.before_stage:
            self.before_stage(stage, job, cache)
        extra = {k: v for k in TRELLIS_ENV if (v := config.env(k))} if stage.runtime == "trellis" else {}
        log_path = job.logs / f"{stage.name}.log"

        def show(line: str) -> None:
            # Isaac Sim and TensorFlow are very chatty: show only our own lines unless verbose.
            if self.verbose or line.startswith(PREFIX):
                print("  " + line.removeprefix(PREFIX).strip(), flush=True)

        # A GPU stage waits while another process (a scene route step, another run) has this machine's GPU.
        gpu = nullcontext()
        if stage.gpu and self.name == "local":
            gpu = hold(GPU, f"{stage.name} of {job.root.name}", log=lambda m: print("  " + m, flush=True))
        with gpu:
            code, tail = run_module(self.interpreter(stage.runtime), "sim2chip.run_stage",
                                    [stage.name, "--job", str(job.root), "--cache", str(cache.root), "--fingerprint",
                                     fp], on_line=show, log_file=log_path, env=python_env(extra), tail=40, cwd=REPO)
        if job.done(stage.name) is None or job.done(stage.name).get("fingerprint") != fp:
            print(f"  {stage.name} failed (exit {code}), last lines of {log_path}:", flush=True)
            for line in tail:
                print("  | " + line, flush=True)
