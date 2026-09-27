"""Local backend: each stage runs as a subprocess with the interpreter of its runtime.

Interpreters come from environment variables, with defaults for the development machine:
    SIM2CHIP_ISAAC_PYTHON  Isaac Sim's python.sh
    SIM2CHIP_PYTHON        Python with TensorFlow and the asset tools (default: the current one)
    SIM2CHIP_TRELLIS_PYTHON  Python of the image-to-3D environment (TRELLIS, torch, transformers)
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from .. import config
from ..stages import Stage
from ..workspace import REPO, Cache, Job

# Where this machine keeps the other runtimes: environment or .env (see README). Isaac Sim's default is the
# standard install folder; the image-to-3D runtime has no default. Its model weights go wherever HF_HUB_CACHE
# and TORCH_HOME point (set them in .env to keep the weights off the system disk).
ISAAC_PYTHON = str(Path.home() / "isaacsim" / "python.sh")
TRELLIS_ENV = ("HF_HUB_CACHE", "TORCH_HOME")
PREFIX = "[sim2chip]"


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
        return [os.environ.get("SIM2CHIP_PYTHON", sys.executable)]

    def prepare(self, pipe):
        return pipe

    def run(self, pipe, start: str | None = None, only: str | None = None, force: bool = False) -> None:
        pipe.run(self, start=start, only=only, force=force)

    def run_stage(self, stage: Stage, job: Job, cache: Cache, fp: str) -> None:
        if self.before_stage:
            self.before_stage(stage, job, cache)
        cmd = self.interpreter(stage.runtime) + ["-m", "sim2chip.run_stage", stage.name, "--job", str(job.root),
                                                 "--cache", str(cache.root), "--fingerprint", fp]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
               "PYTHONUNBUFFERED": "1"}
        if stage.runtime == "trellis":
            env = {**{k: v for k in TRELLIS_ENV if (v := config.env(k))}, **env}
        log_path = job.logs / f"{stage.name}.log"
        tail: deque[str] = deque(maxlen=40)
        with open(log_path, "w") as log, subprocess.Popen(cmd, env=env, cwd=REPO, stdout=subprocess.PIPE,
                                                          stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                log.write(line)
                tail.append(line)
                # Isaac Sim and TensorFlow are very chatty: show only our own lines unless verbose.
                if self.verbose or line.startswith(PREFIX):
                    print("  " + line.removeprefix(PREFIX).strip(), flush=True)
        if job.done(stage.name) is None or job.done(stage.name).get("fingerprint") != fp:
            print(f"  {stage.name} failed (exit {proc.returncode}), last lines of {log_path}:", flush=True)
            for line in tail:
                print("  | " + line.rstrip(), flush=True)
