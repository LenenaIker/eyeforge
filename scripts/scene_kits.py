"""Puts photo2scene's Pythons and weights in the cache bucket, for Nebius jobs (sim2chip.photo2scene.cloud).

    python scripts/scene_kits.py envs        # the mesh and erase Pythons, packed from this machine's conda envs
    python scripts/scene_kits.py weights     # SAM 3D Objects, MoGe-2, LaMa, DINOv2 and the 4-bit eraser from here
    python scripts/scene_kits.py hf REPO...  # Hugging Face repos, mirrored as a hub cache (hf/models--.../)

The envs are conda envs, not built from pip lists like the worker's kits (images/build.sh): SAM 3D Objects compiles
its own CUDA extensions, which took a day to get right here. conda-pack rewrites their paths for /tmp/s2c/envs/<kit>,
and the mesh kit also carries SAM 3D's and MoGe's code. Each kit is keyed by a hash of what is installed in it, and
envs/scene/current.json says which ones a job takes. `hf` downloads what this machine does not have (run it on a
Nebius job for the big ones: the bucket is far from here, 16 MB/s up, 2026-10-12). Needs conda-pack (PYTHONPATH),
zstd, the Object Storage key, and the paths below (or their variables) for `envs` and `weights`.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim2chip import config  # noqa: E402
from sim2chip.cloud_sync import S3Store  # noqa: E402
from sim2chip.photo2scene import cloud  # noqa: E402

BUCKET = f"{config.env('SIM2CHIP_BUCKET_PREFIX', 'sim2chip')}-cache"
TOOLS = Path(config.env("SIM2CHIP_TOOLS", "/media/lenena-iker/Crucial/tools"))
SAM3D = Path(config.env("SIM2CHIP_SAM3D_DIR") or TOOLS / "sam-3d-objects")
MOGE = Path(config.env("MOGE_ROOT") or TOOLS / "MoGe")
MOGE_DEPS = Path(config.env("MOGE_DEPS") or TOOLS / "moge_deps")
MOGE_CKPT = Path(config.env("MOGE_CKPT") or TOOLS / "moge_ckpt/moge-2-vitl/model.pt")
LAMA = Path(config.env("SIM2CHIP_LAMA") or TOOLS / "lama/big-lama.pt")
TORCH_HUB = Path(config.env("TORCH_HOME") or TOOLS / "torch_cache") / "hub"
QWEN_NF4 = Path(config.env("SIM2CHIP_QWEN_NF4") or TOOLS / "qwen_edit_nf4")  # erase_qwen.py --save-nf4
ENVS = {"mesh": "SIM2CHIP_SCENE_PYTHON", "erase": "SIM2CHIP_ERASE_PYTHON"}


def env_hash(prefix: Path) -> str:
    """What is installed: conda's records and the pip packages' names and versions."""
    names = sorted(p.name for p in (prefix / "conda-meta").glob("*.json"))
    names += sorted(p.name for p in prefix.glob("lib/python3*/site-packages/*.dist-info"))
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:12]


def pack_envs(store: S3Store) -> None:
    import conda_pack
    current = cloud.current(store, BUCKET) or {}
    for kit, var in ENVS.items():
        prefix = Path(config.env(var)).resolve().parents[1]
        key = f"{cloud.ENVS_PREFIX}/{kit}-{env_hash(prefix)}.tar.zst"
        if key in store.listing(BUCKET, key):
            print(f"{kit}: {key} already there")
        else:
            with tempfile.TemporaryDirectory(dir=TOOLS) as tmp:
                tar = Path(tmp) / f"{kit}.tar"
                conda_pack.pack(prefix=str(prefix), output=str(tar), format="tar", arcroot=kit,
                                dest_prefix=f"{cloud.ENVS}/{kit}", ignore_editable_packages=True,
                                ignore_missing_files=True, n_threads=-1, verbose=False)
                if kit == "mesh":  # SAM 3D's code (an editable install here) and MoGe's, without their weights
                    with tarfile.open(tar, "a") as t:
                        t.add(SAM3D, "sam3d-src", filter=lambda i: None if "/checkpoints" in i.name else i)
                        t.add(MOGE, "moge-src")
                        t.add(MOGE_DEPS, "moge-deps")
                        pth = Path(tmp) / "_editable_impl_sam3d_objects.pth"
                        pth.write_text(f"{cloud.ENVS}/sam3d-src\n")
                        site = next(prefix.glob("lib/python3*/site-packages")).relative_to(prefix)
                        t.add(pth, f"{kit}/{site}/{pth.name}")
                subprocess.run(["zstd", "-T0", "-3", "-q", "--rm", str(tar), "-o", f"{tar}.zst"], check=True)
                size = Path(f"{tar}.zst").stat().st_size
                print(f"{kit}: uploading {size / 1e9:.1f} GB to {key}", flush=True)
                store.s3.upload_file(f"{tar}.zst", BUCKET, key)
        current[kit] = key
    store.put(BUCKET, f"{cloud.ENVS_PREFIX}/current.json", json.dumps(current, indent=1).encode())
    print(f"s3://{BUCKET}/{cloud.ENVS_PREFIX}/current.json: {current}")


def upload_tree(store: S3Store, src: Path, prefix: str) -> None:
    there = store.listing(BUCKET, prefix + "/")
    files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    for f in files:
        key = f"{prefix}/{f.name if src.is_file() else f.relative_to(src).as_posix()}"
        if there.get(key) != f.stat().st_size:
            print(f"{key}: {f.stat().st_size / 1e6:.0f} MB", flush=True)
            store.s3.upload_file(str(f.resolve()), BUCKET, key)
    print(f"{src} -> s3://{BUCKET}/{prefix}/")


def upload_weights(store: S3Store) -> None:
    w = cloud.WEIGHTS_PREFIX
    upload_tree(store, SAM3D / "checkpoints" / "hf", f"{w}/sam3d")
    upload_tree(store, MOGE_CKPT, f"{w}/moge")
    upload_tree(store, LAMA, f"{w}/lama")
    upload_tree(store, TORCH_HUB, f"{w}/torch/hub")
    upload_tree(store, QWEN_NF4, f"{w}/qwen-nf4")


def snapshot(repo: str) -> Path:
    """The repo's current snapshot in this machine's hub cache, downloaded first when it is not there."""
    hub = Path(config.env("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    ref = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    if ref.exists():
        return ref.parents[1] / "snapshots" / ref.read_text().strip()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))


def mirror_hf(store: S3Store, repos: list[str]) -> None:
    """Each repo's current snapshot as real files under hf/models--<org>--<name>/snapshots/<rev>/, with refs/main:
    a Hugging Face cache once pulled into HF_HUB_CACHE, so the steps load the repo id offline."""
    for repo in repos:
        snap = snapshot(repo)
        top = f"{cloud.HF_PREFIX}/models--{repo.replace('/', '--')}"
        store.put(BUCKET, f"{top}/refs/main", snap.name.encode())
        upload_tree(store, snap, f"{top}/snapshots/{snap.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["envs", "weights", "hf"])
    ap.add_argument("repos", nargs="*")
    args = ap.parse_args()
    store = S3Store()
    if args.what == "envs":
        pack_envs(store)
    elif args.what == "weights":
        upload_weights(store)
    else:
        mirror_hf(store, args.repos)


if __name__ == "__main__":
    main()
