"""Walk test arm D, Matrix-3D (github.com/SkyworkAI/Matrix-3D, MIT): its panoramic video model (PanoVideoGen 720p
5B on Wan2.2-TI2V-5B) walks the camera straight out of our panorama, and every frame is a panorama from a moved
point, so the grid's views are cut from the frames, no 3D needed. Four rails per place (headings 0, 90, 180, 270, to
13 m), which hold every move of those four headings' views. Run in the matrix3d env
(/media/lenena-iker/Crucial/envs/matrix3d):

    python experiments/sphere_walk/matrix3d_walk.py prep --dir data/sphere_test/<place>
    python experiments/sphere_walk/matrix3d_walk.py run --dir data/sphere_test/<place>   # torchrun, per rail
    python experiments/sphere_walk/matrix3d_walk.py cut --dir data/sphere_test/<place>

Input: the levelled panorama (our levelling, so a rail on Matrix-3D's horizontal plane is on our ground) with
the never-seen ground painted (B_layers' front texture), and our metric distance for its scale (the local copy of
Matrix-3D is patched to read SIM2CHIP_METRIC_RAIL and to keep the rail as given instead of capping it at half the
depth ahead). Writes <dir>/walk/D_m3d/rail_HHH/ (Matrix-3D's folder) and <dir>/walk/D/NAME.png."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2chip.photo2scene.sphere import capture  # noqa: E402

M3D = Path("/media/lenena-iker/Crucial/tools/Matrix-3D")
ENV = Path("/media/lenena-iker/Crucial/envs/matrix3d")
RAILS = (0, 90, 180, 270)
FAR_M = 13.0
FRAMES = 81
OFFSET = {"f": 0, "r": 90, "b": 180, "l": 270}
PROMPT = "A street on a sunny day, a 360 degree panorama photographed at eye height."


def levelled(img: np.ndarray, R: np.ndarray, size) -> np.ndarray:
    """An equirectangular image in the panorama's frame resampled into the levelled frame (d_pano = R^T d_level)."""
    W, H = size
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d_level = np.stack([np.cos(LA) * np.sin(LO), -np.sin(LA), np.cos(LA) * np.cos(LO)], -1)
    return capture.sample(img, d_level @ R)


def rail(theta_deg: float) -> list:
    """Matrix-3D's camera list (world to camera, OpenCV axes, y down): straight along heading theta, FRAMES poses
    from 0 to FAR_M."""
    t = math.radians(theta_deg)
    Rm = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])
    out = []
    for k in range(FRAMES):
        c = Rm[2] * FAR_M * k / (FRAMES - 1)  # the camera centre, along its own z
        m = np.eye(4)
        m[:3, :3] = Rm
        m[:3, 3] = -Rm @ c
        out.append(m.tolist())
    return out


def prep(d: Path):
    w = d / "walk"
    geo = json.loads((d / "geometry.json").read_text())
    R = np.array(geo["levelling"])
    tex = np.asarray(Image.open(w / "BL_scene" / "tex_front.png").convert("RGB"))
    pano = np.clip(levelled(tex, R, (2048, 1024)) + 0.5, 0, 255).astype(np.uint8)
    pd = np.load(d / "proxy_distance.npz")
    dist = np.where(pd["valid"], pd["distance"], 0).astype(np.float32)
    dist_l = levelled(dist[..., None], R, (1024, 512))[..., 0]
    for hd in RAILS:
        r = w / "D_m3d" / f"rail_{hd:03d}"
        r.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pano).save(r / "pano_img.png")
        (r / "prompt.txt").write_text(PROMPT)
        (r / "rail.json").write_text(json.dumps(rail(hd)))
        np.save(r / "metric.npy", dist_l)
    print("prepared", d.name)


def run(d: Path):
    for hd in RAILS:
        r = (d / "walk" / "D_m3d" / f"rail_{hd:03d}").resolve()
        if (r / "pano_video.mp4").exists():
            continue
        t = time.time()
        # nvdiffrast builds its CUDA plugin on first use: the conda nvcc plus the pip wheels' CUDA headers.
        nv = ENV / "lib" / "python3.10" / "site-packages" / "nvidia"
        env = {**os.environ, "SIM2CHIP_METRIC_RAIL": str(r / "metric.npy"), "OPENCV_IO_ENABLE_OPENEXR": "1",
               "CUDA_HOME": str(ENV), "PATH": f"{ENV}/bin:" + os.environ["PATH"],
               "CPATH": ":".join([str(ENV / "targets/x86_64-linux/include")] + [str(p) for p in nv.glob("*/include")]),
               "LIBRARY_PATH": ":".join([str(ENV / "targets/x86_64-linux/lib"), str(ENV / "lib")]
                                        + [str(p) for p in nv.glob("*/lib")])}
        cmd = [str(ENV / "bin" / "torchrun"), "--nproc_per_node", "1", "code/panoramic_image_to_video.py",
               "--inout_dir", str(r), "--resolution", "720", "--use_5b_model", "--enable_vram_management",
               "--json_path", str(r / "rail.json")]
        with open(r / "run.log", "w") as log:
            code = subprocess.call(cmd, cwd=M3D, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(d.name, hd, "exit", code, f"{time.time() - t:.0f} s", flush=True)


def frames(video: Path) -> list:
    import imageio.v3 as iio
    return [np.asarray(f) for f in iio.imiter(video)]


def cut(d: Path):
    w = d / "walk"
    out = w / "D"
    out.mkdir(exist_ok=True)
    views = json.loads((w / "views.json").read_text())["views"]
    vids = {}
    for v in views:
        if v["heading"] not in RAILS:
            continue
        m = v["move"]
        theta = RAILS[0] if m == "0" else (v["heading"] + OFFSET[m[0]]) % 360
        dist = 0.0 if m == "0" else float(m[1:])
        if theta not in RAILS:
            continue
        r = w / "D_m3d" / f"rail_{theta:03d}"
        if not (r / "pano_video.mp4").exists():
            continue
        if theta not in vids:
            vids[theta] = frames(r / "pano_video.mp4")
        k = int(round(dist / FAR_M * (FRAMES - 1)))
        pano = vids[theta][k]
        # The frames are rolled so the rail's heading is the centre column: the view turns by the difference.
        rays = capture.rays(*v["size"], v["fov_deg"], v["yaw_deg"] - theta, v["pitch_deg"])
        Image.fromarray(np.clip(capture.sample(pano, rays) + 0.5, 0, 255).astype(np.uint8)).save(out / f"{v['name']}.png")
    print("cut", d.name, len(list(out.iterdir())), "views")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "run", "cut"])
    ap.add_argument("--dir", required=True, action="append")
    args = ap.parse_args()
    for d in args.dir:
        {"prep": prep, "run": run, "cut": cut}[args.step](Path(d))


if __name__ == "__main__":
    main()
