"""Walk test W.1, arm A (base), in edge-sdg: Isaac's linear renders to sRGB views. Where no mesh was hit, the
panorama's sky if that direction is sky there (the sky is at infinity, so it is exact from anywhere), else a hole;
the mesh where the sphere never saw (magenta in tex_proxy.png) is a hole too.

    python experiments/sphere_walk/post.py --dir data/sphere_test/<place>

Writes <dir>/walk/A/NAME.png (holes in magenta), <dir>/walk/holes/NAME.png (hole mask) and
<dir>/walk/depth/NAME.npy (float16, holes 0)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

MAGENTA = (255, 0, 255)


def srgb_to_linear(c):
    c = np.asarray(c, np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", default="A", help="A writes A/, holes/ and depth/; another arm <arm>/, <arm>_holes/ "
                    "and <arm>_depth/, from <arm>_raw/")
    args = ap.parse_args()
    d = Path(args.dir)
    w = d / "walk"
    a = args.arm
    raw, out_img = w / ("raw" if a == "A" else f"{a}_raw"), w / a
    out_hole, out_dep = (w / "holes", w / "depth") if a == "A" else (w / f"{a}_holes", w / f"{a}_depth")
    for sub in (out_img, out_hole, out_dep):
        sub.mkdir(exist_ok=True)
    plan = json.loads((w / "views.json").read_text())
    geo = json.loads((d / "geometry.json").read_text())
    R = np.array(geo["levelling"])
    bg = np.asarray(Image.open(d / "background.jpg").convert("RGB"))
    seen = np.asarray(Image.open(d / "seen.png")) > 127
    pd = np.load(d / "proxy_distance.npz")
    ph, pw = pd["valid"].shape
    sky = (np.asarray(Image.fromarray(seen).resize((pw, ph))) > 0) & ~pd["valid"]

    centre = [v for v in plan["views"] if v["move"] == "0" and (raw / f"lin_{v['name']}.npy").exists()]
    ratios = []
    for v in centre:
        real = srgb_to_linear(np.asarray(Image.open(w / f"real_h{v['heading']:03d}.png").convert("RGB")))
        lin = np.load(raw / f"lin_{v['name']}.npy").astype(np.float32)
        ok = (real.mean(-1) > 0.02) & (lin.mean(-1) > 1e-4)
        ratios.append(np.median(real[ok] / np.maximum(lin[ok], 1e-6), 0))
    gain = np.median(ratios, 0)
    report = {"gain": np.round(gain, 4).tolist(), "views": {}}
    for v in plan["views"]:
        if not (raw / f"lin_{v['name']}.npy").exists():  # render_walk.py --only
            continue
        lin = np.load(raw / f"lin_{v['name']}.npy").astype(np.float32)
        dep = np.load(raw / f"depth_{v['name']}.npy").astype(np.float32)
        img = linear_to_srgb(lin * gain).astype(np.uint8)
        empty = dep <= 0
        rays = render_plan.pano_rays(capture.rays(*v["size"], v["fov_deg"], v["yaw_deg"], v["pitch_deg"]), R)
        x, y = capture.to_pixel(rays, pw, ph)
        is_sky = sky[np.clip(np.round(y).astype(int), 0, ph - 1), np.round(x).astype(int) % pw]
        col = np.clip(capture.sample(bg, rays) + 0.5, 0, 255).astype(np.uint8)
        img[empty & is_sky] = col[empty & is_sky]
        unseen = (img[..., 0] > 200) & (img[..., 1] < 90) & (img[..., 2] > 200) & ~empty  # tex_proxy's magenta
        hole = (empty & ~is_sky) | unseen
        img[hole] = MAGENTA
        Image.fromarray(img).save(out_img / f"{v['name']}.png")
        Image.fromarray((hole * 255).astype(np.uint8)).save(out_hole / f"{v['name']}.png")
        np.save(out_dep / f"{v['name']}.npy", dep.astype(np.float16))
        report["views"][v["name"]] = round(float(hole.mean()), 4)
    (w / f"post_{a}.json").write_text(json.dumps(report, indent=1))
    print("gain", report["gain"])


if __name__ == "__main__":
    main()
