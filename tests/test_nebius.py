"""Nebius backend with a fake CLI and a fake object store: no cloud calls."""
import io
import json
import tarfile

import pytest

from sim2chip.backends import nebius
from sim2chip.backends.nebius import NebiusBackend, bootstrap, code_archive
from sim2chip.stages import BY_NAME
from sim2chip.workspace import DONE, Job


class FakeCLI:
    def __init__(self, store, final="COMPLETED"):
        self.store, self.final, self.jobs = store, final, []

    def bucket_id(self, name, create=True):
        return f"id-{name}"

    def create_job(self, name, runtime, volumes, script, timeout_h, env_secrets):
        self.jobs.append({"name": name, "image": runtime.image, "volumes": volumes, "timeout_h": timeout_h,
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
        self.objects = {}

    def put(self, bucket, key, data):
        self.objects[(bucket, key)] = data

    def upload_dir(self, bucket, src, prefix):
        files = [f for f in src.rglob("*") if f.is_file()]
        for f in files:
            self.objects[(bucket, f"{prefix}/{f.relative_to(src).as_posix()}")] = f.read_bytes()
        return len(files)

    def delete(self, bucket, key):
        self.objects.pop((bucket, key), None)

    def download(self, bucket, prefix, dst, patterns):
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
    b.run_stage(BY_NAME["render"], job, None, "fp")
    j = b.cli.jobs[0]
    assert j["image"].startswith("nvcr.io/nvidia/isaac-sim")
    assert j["volumes"] == ["id-sim2chip-j:/s2c/job", "id-sim2chip-cache:/s2c/cache"]
    assert j["timeout_h"] == 2 and j["secrets"] == []
    assert "run_stage render --job /s2c/job --cache /s2c/cache --fingerprint fp" in j["script"]
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
    backend(store, final="FAILED").run_stage(BY_NAME["render"], job, None, "fp")
    assert job.done("render") is None


def test_spend_cap(job):
    with pytest.raises(RuntimeError, match="over the cap"):
        backend(FakeStore(), max_usd=1.0).run_stage(BY_NAME["render"], job, None, "fp")


def test_llm_stage_needs_the_secret(job, monkeypatch):
    monkeypatch.delenv("SIM2CHIP_NEBIUS_SECRET", raising=False)
    monkeypatch.setattr(nebius, "REPO", job.root)  # no .env there
    with pytest.raises(RuntimeError, match="SIM2CHIP_NEBIUS_SECRET"):
        backend(FakeStore()).run_stage(BY_NAME["qa_assets"], job, None, "fp")
    monkeypatch.setenv("SIM2CHIP_NEBIUS_SECRET", "s2c-token")
    store = FakeStore()
    b = backend(store, final="FAILED")
    b.run_stage(BY_NAME["qa_assets"], job, None, "fp")
    assert b.cli.jobs[0]["secrets"] == ["NEBIUS_API_KEY=s2c-token"]


def test_code_archive_and_bootstrap():
    names = tarfile.open(fileobj=io.BytesIO(code_archive())).getnames()
    assert "sim2chip/run_stage.py" in names and not any("__pycache__" in n for n in names)
    script = bootstrap(BY_NAME["train"], "abc")
    assert "pip install" in script and "python3 -m sim2chip.run_stage train" in script
