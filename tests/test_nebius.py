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
