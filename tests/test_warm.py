"""The warm GPU (warm.py, warm_worker.py) with an in-memory object store: no cloud calls."""
import io
import json
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from sim2chip import warm
from sim2chip.backends import nebius
from sim2chip.backends.nebius import NebiusBackend
from sim2chip.workspace import DONE, Cache, Job


class _S3:
    def __init__(self, store):
        self.store = store

    def get_object(self, Bucket, Key):
        with self.store.lock:
            if (Bucket, Key) not in self.store.objects:
                raise RuntimeError("NoSuchKey")
            return {"Body": io.BytesIO(self.store.objects[(Bucket, Key)])}

    def upload_file(self, path, bucket, key):
        self.store.put(bucket, key, Path(path).read_bytes())

    def download_file(self, bucket, key, path):
        with self.store.lock:
            if (bucket, key) not in self.store.objects:
                raise RuntimeError("An error occurred (404): Not Found")
            Path(path).write_bytes(self.store.objects[(bucket, key)])


class Store:
    """Thread-safe: the GPU's loop and the run that waits for it share it."""

    def __init__(self):
        self.objects, self.buckets, self.lock = {}, set(), threading.Lock()
        self.s3 = _S3(self)

    def put(self, bucket, key, data):
        with self.lock:
            self.objects[(bucket, key)] = data

    def delete(self, bucket, key):
        with self.lock:
            self.objects.pop((bucket, key), None)

    def listing(self, bucket, prefix=""):
        with self.lock:
            return {k: len(v) for (b, k), v in self.objects.items() if b == bucket and k.startswith(prefix)}

    def pull(self, bucket, keys, root, strip=""):
        n = 0
        for key in keys:
            dst = root / key[len(strip):].lstrip("/")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(bucket, key, str(dst))
                n += 1
        return n

    def url(self, bucket, key, hours=12):
        return f"https://signed/{bucket}/{key}"

    def upload_dir(self, bucket, src, prefix, changed_only=True):
        import os
        files = [Path(d) / n for d, _, names in os.walk(src, followlinks=True) for n in names]
        for f in files:
            self.put(bucket, f"{prefix}/{f.relative_to(src).as_posix()}" if prefix else f.relative_to(src).as_posix(),
                     f.read_bytes())
        return len(files)

    def download(self, bucket, prefix, dst, patterns=("*",), overwrite=True):
        import fnmatch
        n = 0
        for key in self.listing(bucket, prefix + "/"):
            rel = key[len(prefix) + 1:]
            if any(fnmatch.fnmatch(rel, p) for p in patterns) and (overwrite or not (dst / rel).exists()):
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(bucket, key, str(dst / rel))
                n += 1
        return n


def stub_code(tmp_path: Path, body: str) -> str:
    """A run's code archive whose cloud_worker is `body`, served from a file:// URL."""
    src = tmp_path / "stub" / "sim2chip"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "cloud_worker.py").write_text(body)
    tgz = tmp_path / "stub.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(src, arcname="sim2chip")
    return tgz.as_uri()


def test_the_warm_gpu_runs_tasks_cancels_them_and_stops_when_idle(tmp_path, monkeypatch):
    from sim2chip import teacher, warm_worker

    store = Store()
    monkeypatch.setattr(warm_worker, "S3Store", lambda: store)
    monkeypatch.setattr(warm_worker, "ROOT", tmp_path / "s2c")
    monkeypatch.setattr(warm_worker, "CACHE", tmp_path / "s2c" / "home" / "cache")
    monkeypatch.setattr(teacher, "weights", lambda repo, log=print, timeline=None: repo)
    monkeypatch.setattr(warm, "POLL_S", 0.05)
    monkeypatch.setattr(warm, "IDLE_S", 0.5)
    monkeypatch.setenv("S2C_CACHE_BUCKET", "cache")
    monkeypatch.setenv("SIM2CHIP_PYTHON", sys.executable)
    monkeypatch.setattr(sys, "argv", ["warm_worker", "--id", "w1"])
    code = stub_code(tmp_path, "import os, sys, time\n"
                               "print('args', sys.argv[1:], os.environ['S2C_JOB_BUCKET'], os.environ.get('SAMPLE'))\n"
                               "if '--slow' in sys.argv: time.sleep(30)\n")
    store.put("cache", warm.key("w1", "keepalive.json"), json.dumps({"until": time.time() + 1}).encode())
    gpu = threading.Thread(target=warm_worker.main, daemon=True)
    gpu.start()
    warm.submit(store, "cache", "w1", {"id": "t1", "job": "a", "bucket": "job-a", "code_url": code,
                                       "args": ["--from", "train"], "env": {"SAMPLE": "6"}})
    warm.submit(store, "cache", "w1", {"id": "t2", "job": "b", "bucket": "job-b", "code_url": code,
                                       "args": ["--slow"], "log": "prefetch.log"})

    def state(tid):
        return warm.task_state(store, "cache", "w1", tid, False, time.time())

    for _ in range(400):
        if state("t1") == "COMPLETED" and state("t2") == "RUNNING":
            break
        time.sleep(0.05)
    assert state("t1") == "COMPLETED" and state("t2") == "RUNNING"
    # The task's output goes to its own job bucket, and the task's run used its env and bucket.
    out = store.objects[("job-a", "logs/worker.job.log")].decode()
    assert "'--job', 'a', '--from', 'train'] job-a 6" in out
    assert warm.heartbeat(store, "cache", "w1")["state"] == "ready"
    warm.cancel(store, "cache", "w1", "t2")
    for _ in range(400):
        if state("t2") != "RUNNING":
            break
        time.sleep(0.05)
    assert state("t2") == "FAILED" and ("job-b", "logs/prefetch.log") in store.objects
    gpu.join(10)  # nothing left to do and the keep-alive passed: it stops by itself
    assert not gpu.is_alive() and warm.heartbeat(store, "cache", "w1")["state"] == "stopped"
    assert not store.listing("cache", warm.key("w1", "tasks"))


class CLI:
    def __init__(self, worker_state="RUNNING"):
        self.jobs, self.worker_state = [], worker_state

    def ensure_bucket(self, name):
        pass

    def create_job(self, name, runtime, env, script, timeout_h, env_secrets, where=None):
        self.jobs.append({"name": name, "image": runtime.image, "env": env, "where": where})
        return f"aijob-{len(self.jobs)}"

    def job_state(self, job_id):
        return self.worker_state if job_id == "aijob-warm" else "COMPLETED"

    def job_logs(self, job_id):
        return ""

    def cancel(self, job_id):
        self.cancelled = [*getattr(self, "cancelled", []), job_id]


def run_setup(tmp_path, monkeypatch, store):
    from sim2chip.pipeline import Pipeline
    from sim2chip.spec import from_dict

    monkeypatch.setattr(nebius, "REGISTRY", "cr.example/abc")
    monkeypatch.setattr(nebius, "IMAGE_TAG", "t1")
    for k in nebius.KITS:
        store.put("sim2chip-cache", f"envs/t1/{k}.tar.zst", b"kit")
    spec = from_dict({"name": "j", "task": "count", "classes": [{"id": "cup", "size_m": {"measure": "length",
                                                                                       "range": [0.1, 0.2]}}],
                      "user_data": {"raw": "photos"}, "generation": {"frames": 0}, "hardware": "generic"})
    job = Job(tmp_path / "jobs" / "j")
    (job.input / "photos").mkdir(parents=True)
    (job.input / "photos" / "a.jpg").write_bytes(b"jpg")
    (job.root / "cloud_id").write_text("")
    pipe = Pipeline(spec, job, Cache(tmp_path / "cache"))
    warm.worker_file(tmp_path).parent.mkdir(parents=True)
    warm.worker_file(tmp_path).write_text(json.dumps({"wid": "w1", "job_id": "aijob-warm", "started": time.time()}))
    return pipe


def test_a_run_goes_to_the_warm_gpu_as_a_task(tmp_path, monkeypatch):
    store = Store()
    pipe = run_setup(tmp_path, monkeypatch, store)
    fps = pipe.fingerprints()
    store.put("sim2chip-cache", warm.key("w1", "state.json"), json.dumps({"state": "ready", "t": time.time()}).encode())
    real_put = store.put

    def gpu(bucket, key, data):  # the GPU takes the task at once and runs every stage
        real_put(bucket, key, data)
        if key.startswith(warm.key("w1", "tasks")):
            task = json.loads(data)
            assert task["picodet"] and task["code_url"].endswith("_code/sim2chip.tar.gz")
            for st, fp in fps.items():
                real_put("sim2chip-j", f"{st}/{DONE}", json.dumps({"fingerprint": fp}).encode())
            real_put("sim2chip-j", "logs/cloud_pipeline.log", b"> train (x)\n")
            real_put("sim2chip-cache", warm.key("w1", "running", f"{task['id']}.json"), b"{}")
            real_put("sim2chip-cache", warm.key("w1", "done", f"{task['id']}.json"), b'{"code": 0}')
    store.put = gpu
    be = NebiusBackend(max_usd=10.0, cli=CLI(), store=store, poll_s=0)
    lines = []
    be.run_pipeline(pipe, log=lines.append)
    assert not be.cli.jobs  # no GPU of its own
    assert any("warm GPU aijob-warm" in line for line in lines) and "> train (x)" in lines
    rec = json.loads((pipe.job.logs / "cloud_usage.jsonl").read_text().splitlines()[-1])
    assert rec["warm"] and rec["job_id"] == "aijob-warm"
    assert not (pipe.job.logs / nebius.CLOUD_JOB).exists() and all(pipe.job.done(st) for st in fps)


def test_a_run_gets_its_own_gpu_when_the_warm_one_is_gone(tmp_path, monkeypatch):
    store = Store()
    pipe = run_setup(tmp_path, monkeypatch, store)
    be = NebiusBackend(max_usd=10.0, cli=CLI(worker_state="COMPLETED"), store=store, poll_s=0)
    with pytest.raises(Exception):  # the fake cold job does nothing: the stages are missing afterwards
        be.run_pipeline(pipe, log=lambda line: None)
    assert [j["name"] for j in be.cli.jobs] == ["s2c-j-run"]


def test_a_task_the_gpu_never_takes_goes_back(tmp_path, monkeypatch):
    store = Store()
    pipe = run_setup(tmp_path, monkeypatch, store)
    # It said ready when the run looked, then stopped without taking the task.
    store.put("sim2chip-cache", warm.key("w1", "state.json"), json.dumps({"state": "ready", "t": time.time()}).encode())
    real_put = store.put

    def gpu(bucket, key, data):
        real_put(bucket, key, data)
        if key.startswith(warm.key("w1", "tasks")):
            real_put("sim2chip-cache", warm.key("w1", "state.json"),
                     json.dumps({"state": "stopped", "t": time.time()}).encode())
    store.put = gpu
    be = NebiusBackend(max_usd=10.0, cli=CLI(), store=store, poll_s=0)
    with pytest.raises(Exception):
        be.run_pipeline(pipe, log=lambda line: None)
    assert [j["name"] for j in be.cli.jobs] == ["s2c-j-run"]
    assert not store.listing("sim2chip-cache", warm.key("w1", "tasks"))  # withdrawn


def test_a_warm_gpu_that_never_comes_up_is_given_up(tmp_path, monkeypatch):
    # 2026-10-13: Nebius kept the warm GPU in PROVISIONING for 12 min and the run waited on it with no end.
    store = Store()
    pipe = run_setup(tmp_path, monkeypatch, store)
    w = warm.current(tmp_path)
    assert warm.usable(store, "sim2chip-cache", "w1", w["started"])  # just asked for: starting
    stale = time.time() - warm.SETUP_S - 1
    assert not warm.usable(store, "sim2chip-cache", "w1", stale)
    assert warm.task_state(store, "sim2chip-cache", "w1", "t1", False, stale) == "LOST"
    warm.worker_file(tmp_path).write_text(json.dumps({**w, "started": stale}))
    be = NebiusBackend(max_usd=10.0, cli=CLI(), store=store, poll_s=0)
    with pytest.raises(Exception):  # the fake cold job does nothing
        be.run_pipeline(pipe, log=lambda line: None)
    assert [j["name"] for j in be.cli.jobs] == ["s2c-j-run"]  # its own GPU, not the dead one
    gpu = warm.WarmGPU(tmp_path, backend=be, log=lambda line: None)
    gpu.give_up(warm.current(tmp_path))
    assert be.cli.cancelled == ["aijob-warm"] and warm.current(tmp_path) is None


def test_a_job_nebius_finds_no_gpu_for_is_cancelled(tmp_path, monkeypatch):
    store = Store()
    pipe = run_setup(tmp_path, monkeypatch, store)
    warm.worker_file(tmp_path).unlink()
    monkeypatch.setattr(nebius, "PROVISION_S", 0)
    cli = CLI()
    cli.job_state = lambda job_id: "PROVISIONING"
    be = NebiusBackend(max_usd=10.0, cli=cli, store=store, poll_s=0)
    with pytest.raises(Exception, match="no free GPU"):
        be.run_pipeline(pipe, log=lambda line: None)
    # Asked on the Intel L40S, then the AMD one; both cancelled, and the next run asks in that order no more.
    assert [j["where"][0] for j in cli.jobs] == ["gpu-l40s-a", "gpu-l40s-d"]
    assert cli.cancelled == ["aijob-1", "aijob-2"]
    nebius.note_no_gpu(tmp_path, "gpu-l40s-d")
    nebius.note_no_gpu(tmp_path, "gpu-l40s-a")
    assert [p for p, _ in nebius.platforms(tmp_path)] == ["gpu-l40s-a", "gpu-l40s-d"]  # both sick: as listed
    (tmp_path / "logs" / "no_gpu.json").write_text(json.dumps({"gpu-l40s-a": time.time()}))
    assert [p for p, _ in nebius.platforms(tmp_path)] == ["gpu-l40s-d", "gpu-l40s-a"]


def test_start_writes_the_worker_and_its_script(tmp_path, monkeypatch):
    store = Store()
    monkeypatch.setattr(nebius, "REGISTRY", "cr.example/abc")
    monkeypatch.setattr(nebius, "IMAGE_TAG", "t1")
    for k in nebius.KITS:
        store.put("sim2chip-cache", f"envs/t1/{k}.tar.zst", b"kit")
    be = NebiusBackend(max_usd=10.0, cli=CLI(), store=store, poll_s=0)
    w = be.start_warm(tmp_path)
    assert warm.current(tmp_path) == w and be.cli.jobs[0]["name"] == f"s2c-warm-{w['wid']}"
    env = be.cli.jobs[0]["env"]
    assert env["S2C_ENVS"].count("https://") == 3 and "picodet" in env["S2C_ENVS_LATE"]
    script = (tmp_path / "warm" / "warm.nebius.sh").read_text()
    assert "sim2chip.warm_worker" in script and warm.LATE_READY in script and "TF_FORCE_GPU_ALLOW_GROWTH" in script
    # The light image (starts 2.2 min sooner): runs that render start their own Isaac Sim job.
    assert be.cli.jobs[0]["image"].endswith("eyeforge-light:t1") and "SIM2CHIP_ISAAC_PYTHON" not in script
    warm.worker_file(tmp_path).write_text(json.dumps({**w, "job_id": "aijob-warm"}))  # the fake CLI's running job
    assert be.warm_for(Cache(tmp_path / "cache"), nebius.worker_runtime(True)) is None
    assert be.warm_for(Cache(tmp_path / "cache"), nebius.worker_runtime(False)) is not None
    # A GPU started in Isaac Sim's image (2026-10-09 to 10-10) takes those runs too.
    isaac = nebius.worker_runtime(True).image
    warm.worker_file(tmp_path).write_text(json.dumps({**w, "job_id": "aijob-warm", "image": isaac}))
    assert be.warm_for(Cache(tmp_path / "cache"), nebius.worker_runtime(True)) is not None
    assert ("sim2chip-cache", warm.key(w["wid"], "code.tgz")) in store.objects


def test_the_daily_budget_counts_the_warm_gpu_once(tmp_path):
    from sim2chip.job_service import cloud_spent

    logs = tmp_path / "jobs" / "a" / "logs"
    logs.mkdir(parents=True)
    now = time.time()
    (logs / "cloud_usage.jsonl").write_text(json.dumps({"t": now, "usd_estimate": 0.5}) + "\n"
                                            + json.dumps({"t": now, "usd_estimate": 0.2, "warm": True}) + "\n")
    warm.usage_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    warm.usage_file(tmp_path).write_text(json.dumps({"t": now, "usd_estimate": 1.0}) + "\n")
    assert cloud_spent(tmp_path) == pytest.approx(1.5)


def test_gaps_follow_a_user_from_session_to_job(tmp_path, monkeypatch):
    clock = iter([100.0, 160.0, 170.0, 400.0, 500.0, 900.0])
    monkeypatch.setattr(warm.time, "time", lambda: next(clock))
    warm.note(tmp_path, "keep", "session:a", why="photos uploaded")
    warm.note(tmp_path, "keep", "session:a", why="photos uploaded")  # a second batch: counts from here
    warm.note(tmp_path, "job", "table", session="session:a")
    warm.note(tmp_path, "task", "table", kind="label")
    warm.note(tmp_path, "keep", "session:b", why="photos uploaded")  # never trains
    warm.note(tmp_path, "keep", "table", why="touch")
    assert warm.gaps(tmp_path) == {"gaps": [240.0], "abandoned": 2}


def test_a_keep_alive_is_spent_by_the_task_it_waited_for():
    """The GPU stops right after its work (user, 2026-10-10): a keep-alive asked for before the last task started
    was for that task; one asked for after it (another user uploading photos) still holds the GPU."""
    assert warm.keep_until({"until": 500.0, "t": 100.0}, None) == 500.0  # no task yet: wait for the first
    assert warm.keep_until({"until": 500.0, "t": 100.0}, 200.0) == 0.0   # spent by the task started at 200
    assert warm.keep_until({"until": 500.0, "t": 300.0}, 200.0) == 500.0  # asked for after it
    assert warm.keep_until({"until": 500.0}, 200.0) == 0.0               # from before this rule: spent
    assert warm.keep_until(None, None) == 0.0


def test_the_scene_gpu_takes_as_many_runs_as_its_memory_holds(tmp_path, monkeypatch):
    monkeypatch.delenv("SIM2CHIP_SCENE_PARALLEL", raising=False)
    assert warm.parallel_for("light", "1gpu-16vcpu-96gb") == warm.PARALLEL
    assert warm.parallel_for("isaac", "1gpu-8vcpu-32gb") == 1    # one build took 27.2 GB of 32 (2026-10-14)
    assert warm.parallel_for("isaac", "1gpu-16vcpu-96gb") == 3
    monkeypatch.setenv("SIM2CHIP_SCENE_PARALLEL", "2")
    assert warm.parallel_for("isaac", "1gpu-8vcpu-32gb") == 2
    store = Store()
    monkeypatch.setattr(nebius, "REGISTRY", "cr.example/abc")
    monkeypatch.setattr(nebius, "IMAGE_TAG", "t1")
    monkeypatch.setenv("SIM2CHIP_SCENE_GPU", "gpu-l40s-d/1gpu-16vcpu-96gb")
    monkeypatch.delenv("SIM2CHIP_SCENE_PARALLEL")
    for k in nebius.KITS:
        store.put("sim2chip-cache", f"envs/t1/{k}.tar.zst", b"kit")
    be = NebiusBackend(max_usd=10.0, cli=CLI(), store=store, poll_s=0)
    monkeypatch.setattr(be, "_secrets", lambda llm: ["llm"] if llm else [])
    w = be.start_warm(tmp_path, "isaac")
    job = be.cli.jobs[0]
    assert job["image"].endswith("eyeforge-isaac:t1") and job["where"] == ("gpu-l40s-d", "1gpu-16vcpu-96gb")
    assert job["env"]["S2C_WARM_PARALLEL"] == "3" and float(job["env"]["S2C_WARM_HOURS"]) > warm.HOURS
    assert warm.current(tmp_path, "isaac") == w and warm.current(tmp_path) is None
    script = (tmp_path / "warm" / "isaac.nebius.sh").read_text()
    assert "SIM2CHIP_ISAAC_PYTHON" in script and "SIM2CHIP_ERASE_PYTHON" in script and "SIM2CHIP_SCENE_PYTHON" in script
    # A run that renders goes to it; one that does not, too, while no light GPU is up.
    warm.worker_file(tmp_path, "isaac").write_text(json.dumps({**w, "job_id": "aijob-warm"}))
    cache = Cache(tmp_path / "cache")
    assert be.warm_for(cache, nebius.worker_runtime(True))["kind"] == "isaac"
    assert be.warm_for(cache, nebius.worker_runtime(False))["kind"] == "isaac"


def test_a_task_waiting_on_a_full_gpu_is_queued_not_lost():
    store = Store()
    old = time.time() - warm.PICK_S - 1

    def beat(running, parallel):
        store.put("cache", warm.key("w1", "state.json"), json.dumps(
            {"state": "ready", "t": time.time(), "running": running, "parallel": parallel}).encode())
    beat(["a", "b"], 2)
    assert warm.task_state(store, "cache", "w1", "t3", False, old) == "QUEUED"
    beat(["a"], 2)  # a free place and still not taken: the GPU does not see it
    assert warm.task_state(store, "cache", "w1", "t3", False, old) == "LOST"


def test_the_server_asks_for_the_scene_gpu_when_a_run_renders(tmp_path, monkeypatch):
    from sim2chip.job_service import JobService

    class Proc:
        def poll(self):
            return None  # still running

    class Scenes:
        asked = []

        def start(self, reason, who=None, runs=1):
            self.asked.append((who, runs))
    jobs = JobService(tmp_path, launcher=lambda *a, **k: Proc(), scenes=Scenes())
    for name, renders in (("a", True), ("plain", False), ("b", True)):
        job = Job(tmp_path / "jobs" / name)
        job.logs.mkdir(parents=True)
        monkeypatch.setattr(jobs, "_renders", lambda job, request, start, r=renders: r)
        jobs.start(job, tmp_path / "request.json")
    assert Scenes.asked == [("a", 1), ("b", 2)]  # the second sees the first still rendering


def test_a_warm_gpu_that_goes_silent_is_cancelled(tmp_path, monkeypatch):
    # 2026-10-14: two scene builds filled 32 GB of RAM, the GPU stopped beating and its job billed on.
    store = Store()
    run_setup(tmp_path, monkeypatch, store)
    be = NebiusBackend(max_usd=10.0, cli=CLI(), store=store, poll_s=0)
    gpu = warm.WarmGPU(tmp_path, backend=be, log=lambda line: None)
    w = warm.current(tmp_path)
    beat = lambda t: store.put("sim2chip-cache", warm.key("w1", "state.json"),  # noqa: E731
                               json.dumps({"state": "ready", "t": t}).encode())
    beat(time.time() - warm.STALE_S - 1)  # late, not yet given up
    assert not gpu.silent(w)
    beat(time.time() - warm.SILENT_S - 1)
    assert gpu.silent(w)
    gpu.give_up(w)
    assert be.cli.cancelled == ["aijob-warm"] and warm.current(tmp_path) is None


def test_the_scene_gpu_machine_follows_the_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("SIM2CHIP_SCENE_GPU", raising=False)
    monkeypatch.setattr(nebius, "SCENE_BIG", "gpu-l40s-d/1gpu-16vcpu-96gb")
    monkeypatch.setattr(nebius, "PLATFORM", "gpu-l40s-a")
    monkeypatch.setattr(nebius, "PRESET", "1gpu-8vcpu-32gb")
    monkeypatch.setattr(nebius, "FALLBACK", "gpu-l40s-d/1gpu-16vcpu-96gb")
    assert nebius.scene_platforms(tmp_path, 1)[0] == ("gpu-l40s-a", "1gpu-8vcpu-32gb")
    big, small = ("gpu-l40s-d", "1gpu-16vcpu-96gb"), ("gpu-l40s-a", "1gpu-8vcpu-32gb")
    assert nebius.scene_platforms(tmp_path, 2) == [big, small]
    nebius.note_no_gpu(tmp_path, "gpu-l40s-d")  # no AMD machine lately: Intel first
    assert nebius.scene_platforms(tmp_path, 3)[0] == ("gpu-l40s-a", "1gpu-8vcpu-32gb")


def test_a_full_scene_gpu_sends_the_run_to_its_own_gpu(tmp_path, monkeypatch):
    store = Store()
    w = {"wid": "w1", "job_id": "aijob-warm", "started": time.time(), "parallel": 1}
    assert warm.room(store, "cache", w) == 1  # starting: its places, as asked
    warm.submit(store, "cache", "w1", {"id": "t1"})
    assert warm.room(store, "cache", w) == 0  # one waiting takes the place
    store.put("cache", warm.key("w1", "state.json"), json.dumps(
        {"state": "ready", "t": time.time(), "running": ["t1"], "parallel": 3}).encode())
    assert warm.room(store, "cache", w) == 2  # running, not counted twice
    monkeypatch.setattr(nebius, "REGISTRY", "cr.example/abc")
    monkeypatch.setattr(nebius, "IMAGE_TAG", "t1")
    be = NebiusBackend(max_usd=10.0, cli=CLI(), store=store, poll_s=0)
    (tmp_path / "warm").mkdir()
    image = nebius.worker_runtime(True).image
    warm.worker_file(tmp_path, "isaac").write_text(json.dumps({**w, "image": image, "kind": "isaac"}))
    store.put("sim2chip-cache", warm.key("w1", "state.json"), json.dumps(
        {"state": "ready", "t": time.time(), "running": ["t1"], "parallel": 1}).encode())
    assert be.warm_for(Cache(tmp_path / "cache"), nebius.worker_runtime(True)) is None
    store.put("sim2chip-cache", warm.key("w1", "state.json"), json.dumps(
        {"state": "ready", "t": time.time(), "running": ["t1"], "parallel": 3}).encode())
    assert be.warm_for(Cache(tmp_path / "cache"), nebius.worker_runtime(True)) is not None
