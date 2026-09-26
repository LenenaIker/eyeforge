"""Nebius backend with a fake CLI and a fake object store: no cloud calls."""
import io
import json
import tarfile

import pytest

from sim2chip.backends import nebius
from sim2chip.backends.nebius import NebiusBackend, bootstrap, code_archive
from sim2chip.stages import BY_NAME
from sim2chip.workspace import DONE, Cache, Job


class FakeCLI:
    def __init__(self, store, final="COMPLETED"):
        self.store, self.final, self.jobs = store, final, []

    def ensure_bucket(self, name):
        self.store.buckets.add(name)

    def create_job(self, name, runtime, env, script, timeout_h, env_secrets):
        self.jobs.append({"name": name, "image": runtime.image, "env": env, "timeout_h": timeout_h,
                          "secrets": env_secrets, "script": script.read_text()})
        if self.final == "COMPLETED":  # the remote stage writes its marker into the job bucket
            stage = name.rsplit("-", 1)[-1]
            self.store.objects[("sim2chip-j", f"{stage}/{DONE}")] = json.dumps({"fingerprint": "fp"}).encode()
            self.store.objects[("sim2chip-j", f"{stage}/labels.json")] = b"{}"
            self.store.objects[("sim2chip-j", f"{stage}/frames/0.png")] = b"png"
        return "aijob-1"

    def job_state(self, job_id):
        return self.final

    def job_logs(self, job_id):
        return "noise\n[sim2chip] rendered 10 frames\n"


class FakeStore:
    def __init__(self):
        self.objects, self.buckets = {}, set()

    def url(self, bucket, key):
        return f"https://signed/{bucket}/{key}"

    def put(self, bucket, key, data):
        self.objects[(bucket, key)] = data

    def upload_dir(self, bucket, src, prefix):
        files = [f for f in src.rglob("*") if f.is_file()]
        for f in files:
            self.objects[(bucket, f"{prefix}/{f.relative_to(src).as_posix()}")] = f.read_bytes()
        return len(files)

    def delete(self, bucket, key):
        self.objects.pop((bucket, key), None)

    def download(self, bucket, prefix, dst, patterns, overwrite=True):
        import fnmatch
        n = 0
        for (b, key), data in list(self.objects.items()):
            rel = key[len(prefix):].lstrip("/")
            if b == bucket and key.startswith(prefix + "/") and any(fnmatch.fnmatch(rel, p) for p in patterns):
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                (dst / rel).write_bytes(data)
                n += 1
        return n


@pytest.fixture
def job(tmp_path):
    j = Job(tmp_path / "j")
    (j.input / "photos").mkdir(parents=True)
    (j.input / "photos" / "a.jpg").write_bytes(b"jpg")
    j.spec_path.write_text("{}")
    return j


def backend(store, final="COMPLETED", max_usd=10.0):
    return NebiusBackend(dry_run=False, max_usd=max_usd, cli=FakeCLI(store, final), store=store, poll_s=0)


def test_stage_runs_as_job_and_brings_back_small_results(job, tmp_path):
    store = FakeStore()
    b = backend(store)
    b.run_stage(BY_NAME["render"], job, Cache(job.root.parent / "cache"), "fp")
    j = b.cli.jobs[0]
    assert j["image"].startswith("nvcr.io/nvidia/isaac-sim")
    assert store.buckets == {"sim2chip-j", "sim2chip-cache"}
    assert j["env"] == {"S2C_JOB_BUCKET": "sim2chip-j", "S2C_CACHE_BUCKET": "sim2chip-cache",
                        "S2C_CODE_URL": "https://signed/sim2chip-j/_code/sim2chip.tar.gz"}
    assert j["timeout_h"] == 2 and all(s.startswith("AWS_") for s in j["secrets"])
    script = j["script"]
    assert script.index("cloud_sync pull render") < script.index("run_stage render") < script.index(
        "cloud_sync push render")
    assert ("sim2chip-j", "input/photos/a.jpg") in store.objects
    assert ("sim2chip-j", "_code/sim2chip.tar.gz") in store.objects
    assert job.done("render") == {"fingerprint": "fp"}
    assert (job.stage_dir("render") / "labels.json").exists()
    assert not (job.stage_dir("render") / "frames").exists()  # heavy output stays in the bucket
    usage = json.loads((job.logs / "cloud_usage.jsonl").read_text())
    assert usage["stage"] == "render" and usage["state"] == "COMPLETED"


def test_stale_marker_is_removed_before_the_job(job):
    store = FakeStore()
    store.objects[("sim2chip-j", f"render/{DONE}")] = b'{"fingerprint": "fp"}'
    backend(store, final="FAILED").run_stage(BY_NAME["render"], job, Cache(job.root.parent / "cache"), "fp")
    assert job.done("render") is None


def test_spend_cap(job):
    with pytest.raises(RuntimeError, match="over the cap"):
        backend(FakeStore(), max_usd=1.0).run_stage(BY_NAME["render"], job, Cache(job.root.parent / "cache"), "fp")


def test_llm_stage_needs_the_secret(job, monkeypatch):
    monkeypatch.delenv("SIM2CHIP_NEBIUS_SECRET", raising=False)
    monkeypatch.setattr(nebius, "REPO", job.root)  # no .env there
    with pytest.raises(RuntimeError, match="SIM2CHIP_NEBIUS_SECRET"):
        backend(FakeStore()).run_stage(BY_NAME["qa_assets"], job, Cache(job.root.parent / "cache"), "fp")
    monkeypatch.setenv("SIM2CHIP_NEBIUS_SECRET", "s2c-token")
    store = FakeStore()
    b = backend(store, final="FAILED")
    b.run_stage(BY_NAME["qa_assets"], job, Cache(job.root.parent / "cache"), "fp")
    assert "NEBIUS_API_KEY=s2c-token" in b.cli.jobs[0]["secrets"]


def test_code_archive_and_bootstrap():
    names = tarfile.open(fileobj=io.BytesIO(code_archive())).getnames()
    assert "sim2chip/run_stage.py" in names and not any("__pycache__" in n for n in names)
    script = bootstrap(BY_NAME["train"], "abc")
    assert "pip install -q pillow" in script and "python3 -m sim2chip.run_stage train" in script
    assert "--target /tmp/s2c/deps boto3" in script


def test_sync_pulls_what_the_stage_reads(tmp_path):
    from sim2chip import cloud_sync

    class Store(cloud_sync.S3Store):
        def __init__(self, objects):
            self.objects = objects

        def listing(self, bucket, prefix=""):
            return {k: len(v) for (b, k), v in self.objects.items() if b == bucket and k.startswith(prefix)}

        def pull(self, bucket, keys, root, strip=""):
            for k in keys:
                (root / k).parent.mkdir(parents=True, exist_ok=True)
                (root / k).write_bytes(self.objects[(bucket, k)])
            return len(keys)

    manifest = json.dumps([{"uid": "u1"}]).encode()
    objects = {("j", "spec.json"): b"{}", ("j", "input/a.jpg"): b"x", ("j", "fetch_assets/manifest.json"): manifest,
               ("j", "convert_assets/assets.json"): b"[]", ("j", "render/0.png"): b"p",
               ("c", "usd/norm/u1_v2.usd"): b"u", ("c", "usd/raw/textures/u1_texture0.png"): b"t",
               ("c", "usd/raw/u2.usd"): b"other job", ("c", "hdri/sky.hdr"): b"h", ("c", "hdri/other.hdr"): b"o",
               ("j", "fetch_assets/scene.json"): json.dumps({"skies": ["hdri/sky.hdr"], "ground": []}).encode()}
    job, cache = tmp_path / "job", tmp_path / "cache"
    cloud_sync.pull(Store(objects), BY_NAME["preview_assets"], job, cache, "j", "c")
    got = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert got == ["cache/hdri/sky.hdr", "cache/usd/norm/u1_v2.usd", "cache/usd/raw/textures/u1_texture0.png",
                   "job/convert_assets/assets.json", "job/fetch_assets/manifest.json", "job/fetch_assets/scene.json",
                   "job/spec.json"]


class _S3:
    def __init__(self, store):
        self.store = store

    def download_file(self, bucket, key, path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.store.objects[(bucket, key)])


class ListingStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.s3 = _S3(self)

    def listing(self, bucket, prefix=""):
        return {k: len(v) for (b, k), v in self.objects.items() if b == bucket and k.startswith(prefix)}


def test_worker_runs_the_whole_pipeline_in_one_job(tmp_path, monkeypatch):
    from sim2chip.pipeline import Pipeline
    from sim2chip.spec import from_dict

    monkeypatch.setenv("SIM2CHIP_NEBIUS_SECRET", "tf-key")
    spec = from_dict({"name": "j", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                                "assets": {"categories": ["cow"]}}], "user_data": {"raw": "photos"}})
    job = Job(tmp_path / "jobs" / "j")
    (job.input / "photos").mkdir(parents=True)
    (job.input / "photos" / "a.jpg").write_bytes(b"jpg")
    pipe = Pipeline(spec, job, Cache(tmp_path / "cache"))
    fps = pipe.fingerprints()
    store = ListingStore()

    class WorkerCLI(FakeCLI):
        def create_job(self, name, runtime, env, script, timeout_h, env_secrets):
            self.jobs.append({"name": name, "image": runtime.image, "script": script.read_text(), "secrets": env_secrets})
            for st, fp in fps.items():  # the worker ran every stage and pushed its marker
                store.objects[("sim2chip-j", f"{st}/{DONE}")] = json.dumps({"fingerprint": fp}).encode()
            store.objects[("sim2chip-j", "logs/cloud_pipeline.log")] = b"> label (x)\n  label: done\nnoise\n"
            store.objects[("sim2chip-j", "export/model.tflite")] = b"tflite"
            store.objects[("sim2chip-j", "render/rgb_0001.png")] = b"png"
            store.objects[("sim2chip-j", "render/bounding_box_2d_tight_labels_0001.json")] = b"{}"
            store.objects[("sim2chip-j", "render/_done.json")] = json.dumps({"fingerprint": fps["render"]}).encode()
            return "aijob-9"

    be = NebiusBackend(dry_run=False, max_usd=10.0, cli=WorkerCLI(store), store=store, poll_s=0)
    lines = []
    be._follow = lambda job_id, job, bucket, log: (log("> label (x)"), "COMPLETED")[1]  # markers come only at the end
    be.run_pipeline(pipe, log=lines.append)
    assert len(be.cli.jobs) == 1 and be.cli.jobs[0]["image"].startswith("nvcr.io/nvidia/isaac-sim")
    script = be.cli.jobs[0]["script"]
    assert "sim2chip.cloud_worker" in script and "tensorflow[and-cuda]" in script and "SIM2CHIP_ISAAC_PYTHON" in script
    assert "NEBIUS_API_KEY=tf-key" in be.cli.jobs[0]["secrets"]  # label calls Token Factory
    assert "> label (x)" in lines and "noise" not in lines
    assert all(job.done(st) for st in fps) and (job.root / "export" / "model.tflite").exists()
    assert (job.root / "render" / "rgb_0001.png").exists()
    assert not (job.root / "render" / "bounding_box_2d_tight_labels_0001.json").exists()
    assert ("sim2chip-j", "input/photos/a.jpg") in store.objects  # the photos went up


def test_upload_follows_symlinked_input_folders(tmp_path):
    """--input links the user's folder into the job; the upload must go inside it."""
    import os

    from sim2chip.cloud_sync import S3Store

    class Store(S3Store):
        def __init__(self):
            self.keys = []
            outer = self

            class S3:
                def upload_file(self, f, bucket, key):
                    outer.keys.append(key)
            self.s3 = S3()

        def listing(self, bucket, prefix=""):
            return {}

    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "a.jpg").write_bytes(b"x")
    (tmp_path / "input").mkdir()
    os.symlink(tmp_path / "photos", tmp_path / "input" / "maceta")
    s = Store()
    assert s.upload_dir("b", tmp_path / "input", "input") == 1 and s.keys == ["input/maceta/a.jpg"]


def test_timeline_breakdown(tmp_path):
    j = Job(tmp_path / "j")
    j.timeline("render", 300, "nebius")
    j.timeline("train", 120, "nebius")
    lines = []
    nebius.print_timeline(j, lines.append)
    assert "render (nebius)" in lines[1] and "train" in lines[2]
