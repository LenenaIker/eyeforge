"""Moves a stage's data between Object Storage and the job container's local disk.

Serverless jobs could mount the buckets, but the mount is root-only (Isaac Sim's image runs as a normal
user) and does not allow rewriting files. So each job pulls what its stage reads, runs the stage on
local disk and pushes what it wrote:

    python -m sim2chip.cloud_sync pull <stage> --job DIR --cache DIR
    python -m sim2chip.cloud_sync push <stage> --job DIR --cache DIR

Buckets come from S2C_JOB_BUCKET and S2C_CACHE_BUCKET, credentials from AWS_ACCESS_KEY_ID and
AWS_SECRET_ACCESS_KEY. What a stage reads: spec.json, input/ if `reads_input`, the folders of the
stages in `after`, and, if `uses_cache`, every cache object of this job's assets (keys contain the
asset uid) plus the skies and ground textures listed in fetch_assets/scene.json. Cache files already present (same size) are not transferred again.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

REGION = config.env("SIM2CHIP_REGION", "eu-north1")  # the project's region; the worker gets it from the job
S3_ENDPOINT = f"https://storage.{REGION}.nebius.cloud"


class S3Store:
    def __init__(self, key: str | None = None, secret: str | None = None):
        try:
            import boto3
        except ImportError:
            raise RuntimeError("Object Storage access needs boto3: pip install -e .[cloud]") from None
        key = key or os.environ.get("AWS_ACCESS_KEY_ID")
        secret = secret or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not key or not secret:
            raise RuntimeError("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (Object Storage key) are not set")
        self.s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name=REGION, aws_access_key_id=key,
                               aws_secret_access_key=secret)

    def listing(self, bucket: str, prefix: str = "") -> dict[str, int]:
        out = {}
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                out[o["Key"]] = o["Size"]
        return out

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=bucket, Key=key, Body=data)

    def delete(self, bucket: str, key: str) -> None:
        self.s3.delete_object(Bucket=bucket, Key=key)

    def url(self, bucket: str, key: str, hours: float = 12) -> str:
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                              ExpiresIn=int(hours * 3600))

    def pull(self, bucket: str, keys: list[str], root: Path, strip: str = "") -> int:
        """Downloads `keys` under root (minus the `strip` prefix), skipping files already there."""
        todo = []
        for key in keys:
            dst = root / key[len(strip):].lstrip("/")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                todo.append((key, dst))
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda kd: self.s3.download_file(bucket, kd[0], str(kd[1])), todo))
        return len(todo)

    def download(self, bucket: str, prefix: str, dst: Path, patterns: tuple[str, ...] = ("*",),
                 overwrite: bool = True) -> int:
        """Small results; overwrite=False for immutable files (asset reviews) already present."""
        keys = [k for k in self.listing(bucket, prefix + "/")
                if any(fnmatch.fnmatch(k[len(prefix) + 1:], p) for p in patterns)
                and (overwrite or not (dst / k[len(prefix) + 1:]).exists())]
        for key in keys:
            out = dst / key[len(prefix) + 1:]
            out.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(bucket, key, str(out))
        return len(keys)

    def upload_dir(self, bucket: str, src: Path, prefix: str, changed_only: bool = True) -> int:
        """Uploads files (symlinks followed); with changed_only, only those missing or of another size."""
        if not src.exists():
            return 0
        have = self.listing(bucket, prefix + "/" if prefix else "") if changed_only else {}
        todo = []
        # os.walk with followlinks: job inputs are symlinks to the user's folders, which rglob does not enter.
        files = sorted(Path(d) / n for d, _, names in os.walk(src, followlinks=True) for n in names)
        for f in files:
            if f.is_file():
                key = f"{prefix}/{f.relative_to(src).as_posix()}" if prefix else f.relative_to(src).as_posix()
                if have.get(key) != f.stat().st_size:
                    todo.append((f, key))
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda fk: self.s3.upload_file(str(fk[0]), bucket, fk[1]), todo))
        return len(todo)


def asset_uids(job: Path) -> set[str]:
    manifest = job / "fetch_assets" / "manifest.json"
    return {m["uid"] for m in json.loads(manifest.read_text())} if manifest.exists() else set()


def asset_cache_keys(listing: dict[str, int], job: Path) -> list[str]:
    """Cache objects of the job's assets and scene: downloads, USD conversions, skies and textures. A download
    whose USD conversion is already in the cache is left out: nothing reads it again."""
    uids = asset_uids(job)
    scene_file = job / "fetch_assets" / "scene.json"
    scene = json.loads(scene_file.read_text()) if scene_file.exists() else {}
    listed = {k for k in scene.values() if isinstance(k, str)} | {
        k for v in scene.values() if isinstance(v, list) for k in v if isinstance(k, str)}
    converted = {u for u in uids if f"usd/glb2usd/{u}.usd" in listing}
    keys = []
    for k in listing:
        if k in listed:
            keys.append(k)
            continue
        owner = next((u for u in uids if u in k), None)
        if owner and (k.startswith("usd/") or owner not in converted):
            keys.append(k)
    return keys


def pull(store: S3Store, stage, job: Path, cache: Path, job_bucket: str, cache_bucket: str) -> None:
    listing = store.listing(job_bucket)
    wanted = ["spec.json"] + (["input/"] if stage.reads_input else [])
    wanted += [f"{a}/" for a in stage.after]
    if stage.uses_cache:
        wanted.append("fetch_assets/")
    keys = [k for k in listing if any(k == w or (w.endswith("/") and k.startswith(w)) for w in wanted)]
    n = store.pull(job_bucket, keys, job)
    m = 0
    if stage.uses_cache:
        m = store.pull(cache_bucket, asset_cache_keys(store.listing(cache_bucket), job), cache)
    if stage.uses_db:
        m += store.pull(cache_bucket, list(store.listing(cache_bucket, "db/")), cache)
    print(f"[sim2chip] pulled {n} job files and {m} cache files", flush=True)


def push(store: S3Store, stage, job: Path, cache: Path, job_bucket: str, cache_bucket: str) -> None:
    # Stage outputs are rewritten on every run: upload them all. Cache files never change once written.
    n = store.upload_dir(job_bucket, job / stage.name, stage.name, changed_only=False)
    n += store.upload_dir(job_bucket, job / "logs", "logs", changed_only=False)
    m = store.upload_dir(cache_bucket, cache, "") if stage.uses_cache else 0
    if stage.uses_db and not stage.uses_cache:  # the knowledge base is append-only: new files only
        m += store.upload_dir(cache_bucket, cache / "db", "db")
    print(f"[sim2chip] pushed {n} job files and {m} cache files", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["pull", "push"])
    p.add_argument("stage")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    from .stages import BY_NAME
    stage = BY_NAME[args.stage]
    args.job.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    fn = pull if args.action == "pull" else push
    fn(S3Store(), stage, args.job, args.cache, os.environ["S2C_JOB_BUCKET"], os.environ["S2C_CACHE_BUCKET"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
