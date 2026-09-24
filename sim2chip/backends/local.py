"""Local backend: each stage runs as a subprocess with the interpreter of its runtime.

Interpreters come from environment variables, with defaults for the development machine:
    SIM2CHIP_ISAAC_PYTHON  Isaac Sim's python.sh
    SIM2CHIP_PYTHON        Python with TensorFlow and the asset tools (default: the current one)
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from ..stages import Stage
from ..workspace import REPO, Cache, Job

ISAAC_PYTHON = "/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh"
PREFIX = "[sim2chip]"


class LocalBackend:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def interpreter(self, runtime: str) -> list[str]:
        if runtime == "isaac":
            exe = os.environ.get("SIM2CHIP_ISAAC_PYTHON", ISAAC_PYTHON)
            if not Path(exe).exists():
                raise FileNotFoundError(f"Isaac Sim python not found at {exe}, set SIM2CHIP_ISAAC_PYTHON")
            return [exe]
        return [os.environ.get("SIM2CHIP_PYTHON", sys.executable)]

    def run_stage(self, stage: Stage, job: Job, cache: Cache, fp: str) -> None:
        cmd = self.interpreter(stage.runtime) + ["-m", "sim2chip.run_stage", stage.name, "--job", str(job.root),
                                                 "--cache", str(cache.root), "--fingerprint", fp]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
               "PYTHONUNBUFFERED": "1"}
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
