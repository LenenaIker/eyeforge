"""Run one of our modules in another Python (Isaac Sim's, the teachers', PaddlePaddle's) and follow its output.

Every runtime gets the repo on PYTHONPATH and unbuffered output, so progress lines arrive as they are printed.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from pathlib import Path

from .workspace import REPO


def python_env(extra: dict[str, str] | None = None, inherit_path: bool = True) -> dict[str, str]:
    """This process's environment for a child Python. inherit_path=False puts only the repo on PYTHONPATH (a
    runtime that must not see this environment's packages)."""
    path = [str(REPO)] + ([os.environ["PYTHONPATH"]] if inherit_path and os.environ.get("PYTHONPATH") else [])
    return {**os.environ, **(extra or {}), "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.pathsep.join(path)}


def run_module(exe: list[str], module: str, args: list[str] = (), request: dict | None = None, on_line=None,
               log_file: Path | None = None, env: dict[str, str] | None = None, tail: int = 20,
               cwd: Path | None = None) -> tuple[int, list[str]]:
    """`exe -m module args` (in cwd, default this process's), `request` as JSON on its stdin. Each output line goes to log_file (all
    of them) and to on_line(line), which shows the ones worth showing. Returns the exit code and the last `tail`
    lines, for the error message."""
    last: deque[str] = deque(maxlen=tail)
    stdin = subprocess.PIPE if request is not None else None
    with subprocess.Popen(list(exe) + ["-m", module, *args], stdin=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1, env=env or python_env(), cwd=cwd) as proc:
        if request is not None:
            proc.stdin.write(json.dumps(request))
            proc.stdin.close()
        log = open(log_file, "w") if log_file else None
        try:
            for line in proc.stdout:
                if log:
                    log.write(line)
                last.append(line.rstrip())
                if on_line:
                    on_line(line)
        finally:
            if log:
                log.close()
    return proc.returncode, list(last)
