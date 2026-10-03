"""Copies the teachers' weights (SAM 3, SigLIP 2) from this machine's Hugging Face cache to the cache bucket, where
Nebius jobs read them without a Hugging Face token (sim2chip.teacher.weights). Run it once, and again when
teacher.MODEL or teacher.SIGLIP_MODEL change. Needs the weights downloaded here first (SAM 3 is gated: accept its
license and log in to Hugging Face), HF_HUB_CACHE if they are not in the default cache, and the Object Storage
key (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).

    python scripts/upload_weights.py
"""
import os
from pathlib import Path

from sim2chip import config, teacher
from sim2chip.cloud_sync import S3Store

BUCKET = f"{config.env('SIM2CHIP_BUCKET_PREFIX', 'sim2chip')}-cache"


def snapshot(repo: str) -> Path:
    hub = Path(config.env("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    snaps = sorted((hub / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/config.json"),
                   key=lambda f: f.stat().st_mtime)
    if not snaps:
        raise SystemExit(f"{repo} is not in {hub}: download it first (for instance by running the teacher here)")
    return snaps[-1].parent


def main() -> None:
    store = S3Store()
    for repo in (teacher.MODEL, teacher.SIGLIP_MODEL):
        src = snapshot(repo)
        prefix = f"{teacher.WEIGHTS_PREFIX}/{repo}"
        there = store.listing(BUCKET, prefix + "/")
        for f in sorted(src.iterdir()):  # files are symlinks into the cache's blobs: upload what they point to
            key = f"{prefix}/{f.name}"
            size = os.path.getsize(f)
            if there.get(key) == size:
                continue
            print(f"{key}: {size / 1e6:.0f} MB", flush=True)
            store.s3.upload_file(str(f.resolve()), BUCKET, key)
        print(f"{repo}: s3://{BUCKET}/{prefix}/ up to date")


if __name__ == "__main__":
    main()
