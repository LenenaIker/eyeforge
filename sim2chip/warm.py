"""The warm GPU: one Nebius job that stays up while users are around, so their runs skip the start-up.

A cloud run pays about three minutes before any work (the node, the image, the Python kits, the teacher's
weights), and a calibration round is half a minute of work. The server asks for this GPU as soon as a user
uploads photos in the interview (the start-up hides behind the questions), keeps it while they check the
teacher, and every run of a job (calibration rounds, the prefetch of the teacher's answers, training) goes to it
as a task instead of a job of its own. Tasks of several users run side by side on it. It stops by itself after
IDLE_S after its last task (user, 2026-10-10: no keep-alive after work, measured unused), or at HOURS. The
keep-alive covers only the wait for a first task: one asked for before the last task started is spent.

The server and the GPU talk through the cache bucket, under warm/<id>/ (the job's own data stays in its bucket):

    state.json        GPU   {"state": "setup" | "ready" | "stopping" | "stopped", "t", "running": [task ids]}
    keepalive.json    server {"until": epoch}: stay up at least until then
    tasks/<id>.json   server {"id", "job", "bucket", "code_url", "args", "env", "picodet", "log", "kind"}
    running/<id>.json GPU   the task started
    done/<id>.json    GPU   {"code", "started", "ended"}
    cancel/<id>       server stop that task

What users do with it, to tune KEEP_S and IDLE_S from data: logs/warm_events.jsonl in the server's home, "keep"
when a user renews the keep-alive (photos uploaded, a calibration round) and "task" when a run sends it work,
each with who (an interview session or a job; "job" links the two); gaps() reads it.
Its cost is the job's whole life, idle time included: logs/warm_usage.jsonl in the server's home (record_end).
A task's own records in its job's cloud_usage.jsonl say "warm": the usage panel shows them, the daily budget
counts the GPU once.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import config

PREFIX = "warm"
# Stop right after the work (user, 2026-10-10): on 7 GPUs of eyeforge.app no task ever came during the 2 to 5 min
# kept after the last one (a quarter of the GPU time); the check's rounds need no GPU once SAM 3's answers are
# prefetched. IDLE_S only bridges tasks sent back to back (a round, then its prefetch: 6 to 12 s apart).
IDLE_S = float(config.env("SIM2CHIP_WARM_IDLE_S", "20"))    # without tasks, past the keep-alive: stop
KEEP_S = float(config.env("SIM2CHIP_WARM_KEEP_S", "300"))   # an upload or a new job waits this long for a first task
HOURS = int(config.env("SIM2CHIP_WARM_HOURS", "2"))         # the job's timeout, a cap on any forgotten GPU
PARALLEL = 2      # tasks at once (two users); more queue
POLL_S = 3        # how often the GPU and a waiting run look at the bucket
STALE_S = 120     # a heartbeat older than this: the GPU is gone or stuck, runs go elsewhere
PICK_S = 180      # a task the GPU has not started by then, while it says ready, goes elsewhere
LATE_READY = "/tmp/s2c/late.ready"  # written when the kits fetched after start-up (PicoDet's) are unpacked


def enabled() -> bool:
    """Our own images and kits are needed (the stock image installs for minutes), and it can be switched off."""
    from .backends.nebius import IMAGE_TAG, REGISTRY
    return bool(REGISTRY and IMAGE_TAG) and config.env("SIM2CHIP_WARM", "1") != "0"


def key(wid: str, *parts: str) -> str:
    return "/".join([PREFIX, wid, *parts])


def worker_file(home: Path) -> Path:
    """This server's warm GPU: {"wid", "job_id", "started"}, while it may be up."""
    return home / "warm" / "worker.json"


def current(home: Path) -> dict | None:
    try:
        return json.loads(worker_file(home).read_text())
    except (OSError, ValueError):
        return None


def read(store, bucket: str, k: str) -> dict | None:
    try:
        return json.loads(store.s3.get_object(Bucket=bucket, Key=k)["Body"].read())
    except Exception:  # not written yet
        return None


def put(store, bucket: str, k: str, data: dict) -> None:
    store.put(bucket, k, json.dumps(data).encode())


def heartbeat(store, bucket: str, wid: str) -> dict | None:
    return read(store, bucket, key(wid, "state.json"))


def usable(store, bucket: str, wid: str) -> bool:
    """The GPU can take a task: starting (no heartbeat yet) or ready, not stopping, not silent."""
    hb = heartbeat(store, bucket, wid)
    if hb is None:
        return True
    return hb.get("state") in ("setup", "ready") and time.time() - hb.get("t", 0) < STALE_S


def keep_alive(store, bucket: str, wid: str, seconds: float) -> None:
    """Stay up at least `seconds` from now (never shortens an earlier promise)."""
    now = time.time()
    old = read(store, bucket, key(wid, "keepalive.json")) or {}
    put(store, bucket, key(wid, "keepalive.json"), {"until": max(now + seconds, old.get("until", 0)), "t": now})


def keep_until(doc: dict | None, last_started: float | None) -> float:
    """Until when a keep-alive holds the GPU: a promise made before the last task started was for that task
    (or one before it), and is spent."""
    doc = doc or {}
    if last_started is not None and doc.get("t", 0) <= last_started:
        return 0.0
    return doc.get("until", 0.0)


def submit(store, bucket: str, wid: str, task: dict) -> None:
    """Command: the task is a JSON file in the bucket that the warm GPU picks up and runs
    (patterns: docs/architecture.md)."""
    put(store, bucket, key(wid, "tasks", f"{task['id']}.json"), task)


def withdraw(store, bucket: str, wid: str, tid: str) -> None:
    store.delete(bucket, key(wid, "tasks", f"{tid}.json"))


def cancel(store, bucket: str, wid: str, tid: str) -> None:
    store.put(bucket, key(wid, "cancel", tid), b"")


def task_state(store, bucket: str, wid: str, tid: str, worker_ended: bool, submitted: float) -> str:
    """QUEUED, RUNNING, COMPLETED, FAILED, or LOST: the GPU ended or went silent before finishing it, or never
    started it while it could have (then the run goes elsewhere if it had not started)."""
    done = read(store, bucket, key(wid, "done", f"{tid}.json"))
    if done is not None:
        return "COMPLETED" if done.get("code") == 0 else "FAILED"
    started = read(store, bucket, key(wid, "running", f"{tid}.json")) is not None
    hb = heartbeat(store, bucket, wid)
    silent = hb is not None and time.time() - hb.get("t", 0) > STALE_S
    # "stopping" is not lost: the GPU looks once more for tasks before it stops, and takes this one.
    if worker_ended or silent or (hb and hb.get("state") == "stopped" and not started):
        return "LOST"
    if not started and hb and hb.get("state") == "ready" and time.time() - submitted > PICK_S:
        return "LOST"
    return "RUNNING" if started else "QUEUED"


def events_file(home: Path) -> Path:
    return home / "logs" / "warm_events.jsonl"


def note(home: Path, event: str, who: str | None = None, **fields) -> None:
    """One line of logs/warm_events.jsonl; never raises (telemetry)."""
    try:
        events_file(home).parent.mkdir(parents=True, exist_ok=True)
        with open(events_file(home), "a") as f:
            f.write(json.dumps({"t": round(time.time(), 1), "event": event, "who": who, **fields}) + "\n")
    except OSError:
        pass


def gaps(home: Path) -> dict:
    """From logs/warm_events.jsonl: per user (a session and the job it became count as one), the seconds from each
    keep-alive renewal to their next task ("gaps"; the keep-alive should cover most of them), and the renewals no
    task followed ("abandoned": each cost the GPU a keep-alive for nothing)."""
    try:
        lines = events_file(home).read_text().splitlines()
    except OSError:
        return {"gaps": [], "abandoned": 0}
    events = [json.loads(x) for x in lines if x.strip()]
    owner = {e["session"]: e["who"] for e in events if e["event"] == "job" and e.get("session")}
    pending, out = {}, []
    for e in events:
        who = owner.get(e.get("who"), e.get("who"))
        if e["event"] == "keep":
            pending[who] = e["t"]  # the keep-alive counts from the latest renewal
        elif e["event"] == "task" and who in pending:
            out.append(round(e["t"] - pending.pop(who), 1))
    return {"gaps": out, "abandoned": len(pending)}


def usage_file(home: Path) -> Path:
    return home / "logs" / "warm_usage.jsonl"


def spent(home: Path, since: float) -> float:
    """Dollars of warm GPUs that ended since `since`, plus the one up now at its elapsed time and a full idle
    period more (what it costs at least before it can stop)."""
    from .backends.nebius import USD_PER_HOUR
    total = 0.0
    try:
        for line in usage_file(home).read_text().splitlines():
            rec = json.loads(line)
            if rec.get("t", 0) >= since:
                total += rec.get("usd_estimate", 0.0)
    except (OSError, ValueError):
        pass
    w = current(home)
    if w:
        total += ((time.time() - w["started"]) + IDLE_S) / 3600 * USD_PER_HOUR
    return total


class WarmGPU:
    """The server's side: start the GPU when a user shows up, keep it while they are around, note its cost when it
    ends. One at a time; every method is safe to call from request threads and never raises (a warm GPU is a
    speed-up: without it runs start their own job as before)."""

    def __init__(self, home: Path, backend=None, log=print):
        self.home, self.log = home, log
        self._backend = backend
        self._lock = threading.Lock()
        self._watching = False

    @property
    def backend(self):
        if self._backend is None:
            from .backends.nebius import NebiusBackend
            self._backend = NebiusBackend()
        return self._backend

    def _alive(self) -> dict | None:
        """The current GPU if its job has not ended (else its cost is noted and it is forgotten)."""
        w = current(self.home)
        if w is None:
            return None
        from .backends.nebius import TERMINAL
        try:
            state = self.backend.cli.job_state(w["job_id"])
        except Exception:  # cannot tell: assume it is up; a run checks again before using it
            return w
        if state in TERMINAL:
            self.record_end(w, state)
            return None
        return w

    def start(self, reason: str = "", who: str | None = None) -> None:
        """Ask for the GPU unless one is up (in the background: the caller never waits for Nebius)."""
        note(self.home, "keep", who, why=reason)
        threading.Thread(target=self._start, args=(reason,), daemon=True).start()

    def _start(self, reason: str) -> None:
        with self._lock:
            try:
                w = self._alive()
                if w is not None and usable(self.backend.store, self._bucket(), w["wid"]):
                    keep_alive(self.backend.store, self._bucket(), w["wid"], KEEP_S)
                    return
                if w is not None:  # stopping or silent: let it end, a new one takes over
                    self.record_end(w, "REPLACED")
                w = self.backend.start_warm(self.home)
                keep_alive(self.backend.store, self._bucket(), w["wid"], KEEP_S)
                self.log(f"warm GPU {w['job_id']} asked for ({reason})")
            except Exception as e:  # never in the way of the user
                self.log(f"warm GPU: could not start it: {e}")
        self.watch()

    def touch(self, seconds: float = KEEP_S, who: str | None = None) -> None:
        """The user is still around: keep the GPU that long (for a first task; see keep_until)."""
        note(self.home, "keep", who, why="touch")
        def go():
            try:
                w = current(self.home)
                if w is not None:
                    keep_alive(self.backend.store, self._bucket(), w["wid"], seconds)
            except Exception as e:
                self.log(f"warm GPU: keep-alive failed: {e}")
        threading.Thread(target=go, daemon=True).start()

    def _bucket(self) -> str:
        return f"{self.backend.prefix}-cache"

    def record_end(self, w: dict, state: str) -> None:
        """Its cost, from its start to its last heartbeat (or now), once."""
        from .backends.nebius import USD_PER_HOUR
        if current(self.home) != w:
            return
        hb = None
        try:
            hb = heartbeat(self.backend.store, self._bucket(), w["wid"])
        except Exception as e:  # its cost then runs until now, an overestimate
            self.log(f"warm GPU {w['wid']}: no last heartbeat ({type(e).__name__}: {e})")
        end = min(time.time(), hb["t"] + 60) if hb and hb.get("t") else time.time()
        hours = max(0.0, end - w["started"]) / 3600
        rec = {"t": round(time.time()), "job_id": w["job_id"], "state": state, "hours": round(hours, 4),
               "usd_estimate": round(hours * USD_PER_HOUR, 4)}
        usage_file(self.home).parent.mkdir(parents=True, exist_ok=True)
        with open(usage_file(self.home), "a") as f:
            f.write(json.dumps(rec) + "\n")
        worker_file(self.home).unlink(missing_ok=True)

    def watch(self, every: float = 60) -> None:
        """Notice when the GPU ends (to note its cost), in one background thread."""
        with self._lock:
            if self._watching:
                return
            self._watching = True

        def loop():
            while True:
                time.sleep(every)
                try:
                    with self._lock:
                        if self._alive() is None:
                            self._watching = False
                            return
                except Exception as e:
                    self.log(f"warm GPU: watch failed: {e}")
        threading.Thread(target=loop, daemon=True).start()

