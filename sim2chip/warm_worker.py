"""The warm GPU's loop, inside its Nebius job (see warm.py for the protocol):

    python -m sim2chip.warm_worker --id <warm id>

First the teachers' weights (SAM 3, SigLIP 2) come from the cache bucket, so no task waits for them. Then it
looks for tasks every POLL_S seconds and runs up to PARALLEL at once, each as `cloud_worker` in its own folder
with the code its run uploaded (a server deployed meanwhile sends the new code) and the shared cache, which keeps
the teachers' answers, the weights and the asset files between tasks. A task that trains PicoDet waits for that
kit, unpacked in the background after start-up. The task's output goes to its job bucket as logs/<task log>.
It stops after warm.IDLE_S without tasks once the keep-alive has passed, or shortly before its job's timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from . import warm
from .cloud_sync import S3Store
from .workspace import say

ROOT = Path("/tmp/s2c")
CACHE = ROOT / "home" / "cache"  # the cache every task shares (cloud_worker's --home is per task)


class Task:
    def __init__(self, spec: dict):
        self.spec, self.id = spec, spec["id"]
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.code: int | None = None
        self.started = time.time()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        d = ROOT / "tasks" / self.id
        try:
            shutil.rmtree(d, ignore_errors=True)
            (d / "code").mkdir(parents=True)
            (d / "home").mkdir()
            (d / "home" / "cache").symlink_to(CACHE)
            with urllib.request.urlopen(self.spec["code_url"], timeout=60) as r:
                (d / "code.tgz").write_bytes(r.read())
            with tarfile.open(d / "code.tgz") as tar:
                tar.extractall(d / "code")
            if self.spec.get("picodet"):
                while not Path(warm.LATE_READY).exists() and not self.cancelled:
                    time.sleep(1)
            env = {**os.environ, **(self.spec.get("env") or {}), "S2C_JOB_BUCKET": self.spec["bucket"],
                   "PYTHONPATH": str(d / "code"), "S2C_WARM": "1"}
            env.pop("S2C_SETUP_S", None)  # the start-up was paid once, not by this run
            cmd = [os.environ.get("SIM2CHIP_PYTHON", sys.executable), "-m", "sim2chip.cloud_worker",
                   "--home", str(d / "home"), "--job", self.spec["job"], *self.spec.get("args", [])]
            with open(d / "out.log", "w") as out:
                if not self.cancelled:
                    self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=d)  # no other code on the path
                    self.code = self.proc.wait()
                else:
                    self.code = 1
        except Exception as e:
            with open(d / "out.log", "a") as out:
                out.write(f"warm task failed: {type(e).__name__}: {e}\n")
            self.code = 1

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    args = p.parse_args()
    wid, bucket = args.id, os.environ["S2C_CACHE_BUCKET"]
    store = S3Store()
    t0 = time.time()
    deadline = t0 + warm.HOURS * 3600 - 600  # leave room to push the last task before the job's timeout
    CACHE.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Task] = {}
    finished: set[str] = set()

    def beat(state: str) -> None:
        try:
            warm.put(store, bucket, warm.key(wid, "state.json"),
                     {"state": state, "t": time.time(), "running": sorted(tasks), "since": t0,
                      "picodet": Path(warm.LATE_READY).exists()})
        except Exception as e:  # a missed heartbeat is not worth the run
            say(f"heartbeat failed: {e}")

    beat("setup")
    from . import teacher
    for repo in (teacher.MODEL, teacher.SIGLIP_MODEL):
        try:
            teacher.weights(repo, log=say)
        except Exception as e:  # the task that needs them tries again and says why
            say(f"weights for {repo}: {e}")
    say(f"warm GPU ready after {time.time() - t0:.0f} s")
    last_busy = time.time()
    stopping = False
    while True:
        listing = store.listing(bucket, warm.key(wid) + "/")
        for tid, t in list(tasks.items()):
            if warm.key(wid, "cancel", tid) in listing:
                t.cancel()
            if not t.thread.is_alive():
                log = ROOT / "tasks" / tid / "out.log"
                try:
                    if log.exists():
                        store.s3.upload_file(str(log), t.spec["bucket"], f"logs/{t.spec.get('log', 'worker.job.log')}")
                except Exception as e:
                    say(f"task {tid}: log upload failed: {e}")
                warm.put(store, bucket, warm.key(wid, "done", f"{tid}.json"),
                         {"code": t.code, "started": t.started, "ended": time.time()})
                warm.withdraw(store, bucket, wid, tid)
                shutil.rmtree(ROOT / "tasks" / tid, ignore_errors=True)
                say(f"task {tid} ({t.spec['job']}): exit {t.code} after {time.time() - t.started:.0f} s")
                del tasks[tid]
                finished.add(tid)
                last_busy = time.time()
        pending = sorted(k for k in listing if k.startswith(warm.key(wid, "tasks") + "/"))
        for k in pending:
            tid = Path(k).stem
            if tid in tasks or tid in finished or len(tasks) >= warm.PARALLEL or time.time() > deadline:
                continue
            spec = warm.read(store, bucket, k)
            if spec is None:
                continue
            tasks[tid] = Task(spec)
            warm.put(store, bucket, warm.key(wid, "running", f"{tid}.json"), {"t": time.time()})
            tasks[tid].thread.start()
            say(f"task {tid} ({spec['job']} {' '.join(spec.get('args', []))}) started")
        waiting = [k for k in pending if Path(k).stem not in tasks and Path(k).stem not in finished]
        if tasks:
            last_busy = time.time()
        keep = (warm.read(store, bucket, warm.key(wid, "keepalive.json")) or {}).get("until", 0)
        idle = not tasks and not waiting and time.time() > max(last_busy + warm.IDLE_S, keep)
        if idle or (time.time() > deadline and not tasks):
            if stopping:  # a second look after saying so: a task sent meanwhile is still taken
                break
            stopping = True
            beat("stopping")
        else:
            stopping = False
            beat("ready")
        time.sleep(warm.POLL_S)
    beat("stopped")
    say(f"warm GPU stopping after {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
