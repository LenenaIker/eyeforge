"""Upload bench: how long a dataset of phone photos takes to go up, and how much of it is the server.

    python experiments/upload_bench/bench.py [--photos data/real/grass] [--mbit 10] [--url URL]

Without --url it starts a local server (an empty home in a temporary folder) and uploads there; with --url it
uploads to that server (a public one needs its session cookie in EYEFORGE_COOKIE, see .env). Each case makes a
new library dataset: the first batch creates it and the rest join it, like the web (web/src/api.ts).

Cases: originals or shrunk (Pillow, 1600 px long side, JPEG 0.9, as images.ts does in the browser), in 40 MB
batches one after another (the web until 2026-10-13) or 8 MB batches 3 at a time (now). The uplink is shared:
with 3 at a time each connection gets a third of --mbit (curl --limit-rate), so the cases compare fairly.
Prints a table of seconds per case and the server's own time (Server-Timing: reading the form, saving,
the library) and writes it to results.md next to this file.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LONG_SIDE = 1600


def shrunk(photos: list[Path], out: Path) -> list[Path]:
    from PIL import Image, ImageOps
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for p in photos:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((LONG_SIDE, LONG_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        (out / f"{p.stem}.jpg").write_bytes(buf.getvalue())
        done.append(out / f"{p.stem}.jpg")
    return done


def batches(files: list[Path], cap: int) -> list[list[Path]]:
    out, size = [[]], 0
    for f in files:
        if size + f.stat().st_size > cap and out[-1]:
            out.append([])
            size = 0
        out[-1].append(f)
        size += f.stat().st_size
    return out


def post(url: str, files: list[Path], fields: dict, rate: int, cookie: str | None) -> tuple[float, dict, dict]:
    """One batch with curl: (seconds, JSON answer, Server-Timing parts in seconds)."""
    cmd = ["curl", "-s", "-S", "-H", "Expect:", "--limit-rate", str(rate), "-D", "-", "-w", "\nTIME %{time_total}", url]
    for k, v in fields.items():
        cmd += ["-F", f"{k}={v}"]
    for f in files:
        cmd += ["-F", f"photos=@{f};type=image/jpeg"]
    if cookie:
        cmd += ["-H", f"Cookie: {cookie}"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    out, _, secs = out.rpartition("\nTIME ")
    head, _, body = out.rpartition("\n\n")  # text mode turned \r\n into \n
    timing = {}
    m = re.search(r"(?im)^server-timing:\s*(.+)$", head)
    for part in (m.group(1).split(",") if m else []):
        name, _, dur = part.strip().partition(";dur=")
        timing[name] = float(dur) / 1000
    status = int(head.split("\n\n")[-1].split()[1])  # the last header block (curl may show an interim one)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {body[:200]}")
    return float(secs), json.loads(body), timing


def upload(base: str, files: list[Path], cap: int, parallel: int, mbit: float, cookie: str | None, name: str) -> dict:
    rate = int(mbit * 1e6 / 8 / parallel)
    parts = batches(files, cap)
    t0 = time.perf_counter()
    # The first batch goes alone (it makes the dataset), with the whole uplink.
    secs, made, timing = post(f"{base}/api/library/datasets", parts[0], {"name": name}, rate * parallel, cookie)
    timings = [timing]
    with ThreadPoolExecutor(parallel) as pool:
        for _, _, t in pool.map(lambda b: post(f"{base}/api/library/datasets/{made['id']}/photos", b, {}, rate, cookie),
                                parts[1:]):
            timings.append(t)
    total = time.perf_counter() - t0
    server = {k: sum(t.get(k, 0) for t in timings) for k in ("read", "save", "library", "total")}
    return {"seconds": total, "batches": len(parts), "mb": sum(f.stat().st_size for f in files) / 1e6,
            "server": server, "dataset": made["id"]}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "real" / "grass")
    ap.add_argument("--mbit", type=float, default=10, help="the uplink, shared by the parallel batches")
    ap.add_argument("--url", help="a running server; default: a local one in a temporary folder")
    ap.add_argument("--cases", default="orig-40x1,orig-8x3,small-40x1,small-8x3")
    args = ap.parse_args()

    photos = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    tmp = Path(tempfile.mkdtemp(prefix="upload_bench_", dir=ROOT / "data"))  # the Crucial disk: ~1.5 GB written
    small = shrunk(photos, tmp / "small")
    server = None
    base = args.url
    if base is None:
        port = free_port()
        server = subprocess.Popen([sys.executable, "-m", "sim2chip.cli", "--home", str(tmp / "home"), "serve",
                                   "--port", str(port)], cwd=ROOT,
                                  stdout=open(tmp / "server.log", "w"), stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                subprocess.run(["curl", "-sf", f"{base}/api/config"], check=True, capture_output=True)
                break
            except subprocess.CalledProcessError:
                time.sleep(0.5)
    rows = []
    try:
        for case in args.cases.split(","):
            kind, shape = case.split("-")
            cap_mb, parallel = (int(x) for x in shape.split("x"))
            files = photos if kind == "orig" else small
            r = upload(base, files, cap_mb * 1_000_000, parallel, args.mbit, os.environ.get("EYEFORGE_COOKIE"),
                       f"upload bench {case}")
            rows.append((case, r))
            print(f"{case}: {r['seconds']:.1f} s, {r['mb']:.0f} MB in {r['batches']} batches, server "
                  + ", ".join(f"{k} {v:.2f}" for k, v in r["server"].items()), flush=True)
    finally:
        if server is not None:
            server.terminate()
            server.wait()
        shutil.rmtree(tmp, ignore_errors=True)
    lines = [f"Upload bench, {len(photos)} photos from {args.photos.name}, {args.mbit:g} Mbit/s uplink, "
             f"{'local server' if args.url is None else args.url} ({time.strftime('%Y-%m-%d')})", "",
             "| case | MB | batches | seconds | server read | save | library | server total |",
             "|---|---|---|---|---|---|---|---|"]
    for case, r in rows:
        s = r["server"]
        lines.append(f"| {case} | {r['mb']:.0f} | {r['batches']} | {r['seconds']:.1f} | {s['read']:.2f} | "
                     f"{s['save']:.2f} | {s['library']:.2f} | {s['total']:.2f} |")
    out = Path(__file__).with_name("results.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
