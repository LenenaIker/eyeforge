"""photo2scene in a Nebius job: the steps' Pythons and weights come from the cache bucket once, when a step needs them."""
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from sim2chip import cloud_sync
from sim2chip import photo2scene as p2s
from sim2chip.backends import nebius
from sim2chip.photo2scene import cloud


class Store:
    def __init__(self):
        self.objects, self.gets = {}, []
        self.s3 = self

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, path, Config=None):
        self.gets.append(key)
        Path(path).write_bytes(self.objects[(bucket, key)])

    def listing(self, bucket, prefix=""):
        return {k: len(v) for (b, k), v in self.objects.items() if b == bucket and k.startswith(prefix)}

    pull = cloud_sync.S3Store.pull


def packed(tmp_path, kit) -> bytes:
    src = tmp_path / "src" / kit / "bin"
    src.mkdir(parents=True)
    (src / "python").write_text("#!/bin/sh\n")
    tar = tmp_path / f"{kit}.tar"
    with tarfile.open(tar, "w") as t:
        t.add(src.parent, kit)
    subprocess.run(["zstd", "-q", str(tar)], check=True)
    return Path(f"{tar}.zst").read_bytes()


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    store = Store()
    monkeypatch.setattr(cloud_sync, "S3Store", lambda: store)
    monkeypatch.setattr(cloud, "ENVS", str(tmp_path / "envs"))
    monkeypatch.setattr(cloud, "WEIGHTS", str(tmp_path / "weights"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("S2C_CACHE_BUCKET", "c")
    monkeypatch.setenv("SIM2CHIP_RUNS_ON", "nebius")
    return store


def test_a_step_pulls_its_python_and_weights_once(tmp_path, bucket):
    bucket.objects[("c", "envs/scene/current.json")] = json.dumps({"mesh": "envs/scene/mesh-1.tar.zst"}).encode()
    bucket.objects[("c", "envs/scene/mesh-1.tar.zst")] = packed(tmp_path, "mesh")
    bucket.objects[("c", "weights/scene/sam3d/pipeline.yaml")] = b"yaml"
    bucket.objects[("c", "weights/scene/torch/hub/checkpoints/dino.pth")] = b"dino"
    bucket.objects[("c", "hf/models--Ruicheng--moge-vitl/snapshots/r1/model.pt")] = b"moge"  # SAM 3D's own depth
    env = cloud.ready("mesh", "flow_sam3d.py", log=lambda m: None)
    envs = Path(cloud.ENVS)
    assert (envs / "mesh" / "bin" / "python").exists()
    assert (envs / "sam3d-src" / "checkpoints" / "hf" / "pipeline.yaml").read_bytes() == b"yaml"
    assert env["SIM2CHIP_SAM3D_DIR"] == f"{envs}/sam3d-src" and env["TORCH_HOME"] == f"{cloud.WEIGHTS}/torch"
    assert (Path(cloud.WEIGHTS) / "torch" / "hub" / "checkpoints" / "dino.pth").exists()
    assert env["HF_HUB_OFFLINE"] == "1" and (tmp_path / "hf" / "models--Ruicheng--moge-vitl").is_dir()
    n = len(bucket.gets)
    cloud.ready("mesh", "flow_sam3d.py", log=lambda m: None)
    cloud.ready("mesh", "bg_build.py", log=lambda m: None)  # no weights of its own
    assert len(bucket.gets) == n


def test_composing_never_pulls_the_eraser(tmp_path, bucket):
    bucket.objects[("c", "hf/models--Qwen--Qwen-Image-Edit-2511/refs/main")] = b"r1"
    env = cloud.ready("sam3", "bg_review.py", log=lambda m: None)
    assert env == {} and not bucket.gets
    with pytest.raises(RuntimeError, match="scene_kits.py"):
        cloud.ready("erase", "erase_qwen.py", log=lambda m: None)  # no erase kit there yet


def test_hf_repos_land_in_the_hub_cache_and_load_offline(tmp_path, bucket):
    for repo in cloud.SAM3:
        top = f"hf/models--{repo.replace('/', '--')}"
        bucket.objects[("c", f"{top}/refs/main")] = b"r1"
        bucket.objects[("c", f"{top}/snapshots/r1/config.json")] = b"{}"
    bucket.objects[("c", "weights/scene/lama/big-lama.pt")] = b"lama"
    env = cloud.ready("sam3", "bg_prepare.py", log=lambda m: None)
    assert (tmp_path / "hf" / "models--facebook--sam3" / "snapshots" / "r1" / "config.json").exists()
    assert env == {"SIM2CHIP_LAMA": f"{cloud.WEIGHTS}/lama/big-lama.pt", "HF_HUB_OFFLINE": "1"}


def test_in_the_cloud_the_scene_pythons_are_not_missing(monkeypatch, bucket):
    monkeypatch.setenv("SIM2CHIP_SCENE_PYTHON", "/nowhere/python")
    monkeypatch.setenv("SIM2CHIP_ERASE_PYTHON", "/nowhere/python")
    assert p2s.missing_pythons(("mesh", "erase")) == []
    monkeypatch.delenv("SIM2CHIP_RUNS_ON")
    assert len(p2s.missing_pythons(("mesh", "erase"))) == 2


def test_the_worker_names_the_scene_pythons(monkeypatch):
    monkeypatch.setattr(nebius, "REGISTRY", "cr.example/reg")
    monkeypatch.setattr(nebius, "IMAGE_TAG", "abc")
    script = nebius.worker_script("j", [], teacher=True, runtime=nebius.worker_runtime(True))
    assert f"SIM2CHIP_SCENE_PYTHON={cloud.KITS['mesh']}" in script
    assert f"SIM2CHIP_ERASE_PYTHON={cloud.KITS['erase']}" in script
