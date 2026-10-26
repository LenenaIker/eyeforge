"""Machine-wide locks between processes: the local GPU (a training stage, a scene route step) and the WorldStereo VM
(one Nebius VM, one job at a time). The server's own queues only order the work of one process; the CLI and a
second server are other processes.

    with hold("gpu", "scene_render of job x"):
        ...

A lock is an flock on <SIM2CHIP_LOCKS or /tmp>/sim2chip_<name>.lock, released when the holder exits, even when it
is killed. The file says who holds it, for the waiting message."""
from __future__ import annotations

import fcntl
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .config import env

GPU = "gpu"
WS_VM = "ws_vm"


def lock_file(name: str) -> Path:
    return Path(env("SIM2CHIP_LOCKS") or tempfile.gettempdir()) / f"sim2chip_{name}.lock"


@contextmanager
def hold(name: str, who: str, log=print):
    """The lock for the block's duration, waiting for it when another process holds it."""
    f = lock_file(name)
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            log(f"waiting for the {name} lock, held by {fh.read().strip() or 'another process'}")
            t0 = time.time()
            fcntl.flock(fh, fcntl.LOCK_EX)
            log(f"got the {name} lock after {time.time() - t0:.0f} s")
        fh.seek(0)
        fh.truncate()
        fh.write(f"{who} (pid {os.getpid()})\n")
        fh.flush()
        try:
            yield
        finally:
            fh.seek(0)
            fh.truncate()
            fcntl.flock(fh, fcntl.LOCK_UN)
