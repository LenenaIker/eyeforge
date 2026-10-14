"""photo2scene in a Nebius job: the steps' Pythons and weights come from the cache bucket the first time a step needs
them (scripts/scene_kits.py puts them there). A job that only composes a scene from the library never pulls the
erase model, the biggest of them (Qwen-Image-Edit, 17 GB in 4 bits).

    envs/scene/current.json          which kit of each Python a job takes ({"mesh": key, "erase": key})
    envs/scene/<kit>-<hash>.tar.zst  a conda env packed for /tmp/s2c/envs/<kit> (the mesh one with SAM 3D's and
                                     MoGe's code next to it)
    weights/scene/<name>/            SAM 3D Objects, MoGe-2, LaMa, DINOv2 (torch hub), Qwen-Image-Edit in 4 bits
    hf/models--<org>--<name>/        Hugging Face repos as a hub cache, pulled into HF_HUB_CACHE
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ENVS = "/tmp/s2c/envs"
ENVS_PREFIX = "envs/scene"
WEIGHTS_PREFIX = "weights/scene"
HF_PREFIX = "hf"
WEIGHTS = "/tmp/s2c/weights/scene"
KITS = {"mesh": f"{ENVS}/mesh/bin/python", "erase": f"{ENVS}/erase/bin/python"}
# The eraser already in 4 bits (erase_qwen.py --save-nf4): 17 GB instead of the 58 of the bf16 repo, and loaded in
# seconds (from the bf16 repo, 7-8 min on a Nebius disk, 2026-10-12). The LoRAs from their repos.
QWEN = ["qwen-nf4", "lightx2v/Qwen-Image-Edit-2511-Lightning", "prithivMLmods/Qwen-Image-Edit-2511-Object-Remover"]
SAM3 = ["facebook/sam3"]
# The weights each step script loads (the rest load none, or call the vision model through the API).
SCRIPT_WEIGHTS = {"flow_sam3d.py": ["sam3d", "torch", "Ruicheng/moge-vitl"], "bg_depth.py": ["moge"], "surf_views.py": ["moge"],
                  "bg_prepare.py": [*SAM3, "lama"], "flow_segment.py": SAM3, "flow_things.py": SAM3,
                  "flow_check_masks.py": SAM3, "flow_assets.py": SAM3, "piece_surfaces.py": SAM3, "erase_qwen.py": QWEN}


def on() -> bool:
    """In a cloud job with the cache bucket at hand."""
    return os.environ.get("SIM2CHIP_RUNS_ON") == "nebius" and bool(os.environ.get("S2C_CACHE_BUCKET"))


def current(store, bucket: str) -> dict | None:
    try:
        return json.loads(store.s3.get_object(Bucket=bucket, Key=f"{ENVS_PREFIX}/current.json")["Body"].read())
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            return None
        raise


def _env(kit: str | None, names: list[str]) -> dict[str, str]:
    """Where the step finds its code and weights."""
    env = {}
    if kit == "mesh":
        env |= {"SIM2CHIP_SAM3D_DIR": f"{ENVS}/sam3d-src", "MOGE_ROOT": f"{ENVS}/moge-src",
                "MOGE_DEPS": f"{ENVS}/moge-deps"}
    if "moge" in names:
        env["MOGE_CKPT"] = f"{WEIGHTS}/moge/model.pt"
    if "lama" in names:
        env["SIM2CHIP_LAMA"] = f"{WEIGHTS}/lama/big-lama.pt"
    if "qwen-nf4" in names:
        env["SIM2CHIP_QWEN_NF4"] = f"{WEIGHTS}/qwen-nf4"
    if "torch" in names:
        env["TORCH_HOME"] = f"{WEIGHTS}/torch"
    if any("/" in n for n in names):
        env["HF_HUB_OFFLINE"] = "1"  # every repo it loads is in the mirror: no call to Hugging Face
    return env


def ready(python: str, script: str, log=print) -> dict[str, str]:
    """The kit of `python` and the weights of `script`, pulled once. Returns the variables the step needs. Runs on
    the scene GPU's machine build at once (warm.py): one pulls, the others wait and find it there."""
    import fcntl
    Path(ENVS).mkdir(parents=True, exist_ok=True)
    with open(Path(ENVS, ".pull.lock"), "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        return _ready(python, script, log)


def _ready(python: str, script: str, log) -> dict[str, str]:
    from ..cloud_sync import S3Store
    kit = python if python in KITS else None
    names = SCRIPT_WEIGHTS.get(script, [])
    store, bucket = None, os.environ["S2C_CACHE_BUCKET"]
    if kit and not Path(ENVS, kit, ".ready").exists():
        store = S3Store()
        _kit(store, bucket, kit, log)
    for name in names:
        if "/" in name:  # a Hugging Face repo
            hub = Path(os.environ.get("HF_HUB_CACHE", "/tmp/s2c/hf"))
            prefix, dst, strip = f"{HF_PREFIX}/models--{name.replace('/', '--')}", hub, HF_PREFIX
        elif name == "sam3d":
            prefix, dst, strip = f"{WEIGHTS_PREFIX}/sam3d", Path(ENVS, "sam3d-src", "checkpoints", "hf"), \
                f"{WEIGHTS_PREFIX}/sam3d"
        else:
            prefix, dst, strip = f"{WEIGHTS_PREFIX}/{name}", Path(WEIGHTS), WEIGHTS_PREFIX
        marker = Path(WEIGHTS, ".pulled", name.replace("/", "--"))
        if marker.exists():
            continue
        store = store or S3Store()
        t0 = time.time()
        keys = store.listing(bucket, prefix + "/")
        if not keys:
            raise RuntimeError(f"no {name} in s3://{bucket}/{prefix}/: run scripts/scene_kits.py")
        store.pull(bucket, list(keys), dst, strip=strip)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        log(f"  {name}: {sum(keys.values()) / 1e9:.1f} GB of weights in {time.time() - t0:.0f} s")
    return _env(kit, names)


def _kit(store, bucket: str, kit: str, log) -> None:
    """One packed env: downloaded in parallel parts (one stream gets about 60 MB/s, docs/imagenes.md), then
    unpacked."""
    from boto3.s3.transfer import TransferConfig
    key = (current(store, bucket) or {}).get(kit)
    if not key:
        raise RuntimeError(f"no {kit} kit in s3://{bucket}/{ENVS_PREFIX}/current.json: run scripts/scene_kits.py envs")
    t0 = time.time()
    tmp = Path(ENVS).parent / "kits" / Path(key).name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    store.s3.download_file(bucket, key, str(tmp), Config=TransferConfig(max_concurrency=32,
                                                                       multipart_chunksize=64 << 20))
    t1 = time.time()
    Path(ENVS).mkdir(parents=True, exist_ok=True)
    subprocess.run(f"zstd -T0 -dcq {tmp} | tar xf - -C {ENVS}", shell=True, check=True)
    size = tmp.stat().st_size
    tmp.unlink()
    Path(ENVS, kit, ".ready").touch()
    log(f"  {kit} kit: {size / 1e9:.1f} GB, download {t1 - t0:.0f} s, unpack {time.time() - t1:.0f} s")
