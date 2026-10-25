"""Measure 2 of docs/plan_gs_gen3c_2026-10-20.md: how well each arm reproduces what GEN3C showed at the frames
gs_train.py left out. Each arm's Isaac render of those poses (render_walk.py --views <GS arm>/holdout.json --raw
<dir>/walk/<arm>_hold_raw) against the GEN3C frame, only where its cache had a hole (what GEN3C invented); one gain
per arm, fitted where the panorama saw, turns Isaac's linear light into the frame's sRGB. Run in the gsplat env:

    python experiments/sphere_walk/gs_eval.py --dir data/sphere_test/<place> --holdout GS1 --arm BL_scene --arm GS1

Writes <dir>/walk/hold_eval.json and hold_sheet.jpg (rows: frames; columns: GEN3C, then each arm)."""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import lpips
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from gs_train import BLACK, Site

SHEET_ROWS = 6


def linear_to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--holdout", default="GS1", help="the GS arm whose holdout.json was rendered")
    ap.add_argument("--clips", nargs="+", default=["N01"])
    ap.add_argument("--arm", action="append", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    w = d / "walk"
    site = Site(d)
    views = json.loads((w / args.holdout / "holdout.json").read_text())["views"]
    clips = {}
    for folder in args.clips:
        for p in json.loads((w / folder / "paths.json").read_text())["paths"]:
            clips[p["name"]] = w / folder / f"{p['name']}.mp4"
    net = lpips.LPIPS(net="alex", verbose=False).cuda()
    cache = {}
    res = {a: {"psnr": [], "lpips": []} for a in args.arm}
    rows = []
    for i, v in enumerate(views):
        clip, k = v["name"].rsplit("_f", 1)
        k = int(k)
        if clip not in cache:
            cache = {clip: (iio.imread(clips[clip]), iio.imread(clips[clip].with_name(clip + "_buffer.mp4")))}
        frames, buf = cache[clip]
        gt = frames[k].astype(np.float32) / 255
        W, H = v["size"]
        f = W / 2 / np.tan(np.radians(v["fov_deg"]) / 2)
        hole = ndimage.binary_dilation(buf[k].max(-1) < BLACK, iterations=2)
        sky = site.pano_lookup(site.sky, site.world_rays_level(v["pose"], W, H, f)) > 0
        seen, hole = ~hole & ~sky, hole & ~sky
        row = [gt]
        for a in args.arm:
            lin = np.load(w / f"{a}_hold_raw" / f"lin_{v['name']}.npy").astype(np.float32)
            ok = seen & (lin.mean(-1) > 1e-4)
            g = np.median((gt[ok] ** 2.2) / np.maximum(lin[ok], 1e-6), 0) if ok.sum() > 500 else np.ones(3)
            img = linear_to_srgb(lin * g).astype(np.float32)
            row.append(img)
            if hole.sum() < 2000:
                continue
            mse = ((img - gt) ** 2)[hole].mean()
            res[a]["psnr"].append(float(-10 * np.log10(max(mse, 1e-10))))
            t = lambda x: torch.as_tensor(np.where(hole[..., None], x, 0)).permute(2, 0, 1)[None].cuda() * 2 - 1  # noqa: E731
            with torch.no_grad():
                res[a]["lpips"].append(float(net(t(img), t(gt)).item() / hole.mean()))
        if i % max(1, len(views) // SHEET_ROWS) == 0 and len(rows) < SHEET_ROWS:
            rows.append(np.concatenate([np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((480, 264)))
                                        for x in row], 1))
    out = {a: {"psnr_hole": round(float(np.mean(r["psnr"])), 2), "lpips_hole": round(float(np.mean(r["lpips"])), 3),
               "frames": len(r["psnr"])} for a, r in res.items()}
    (w / "hold_eval.json").write_text(json.dumps(out, indent=1))
    Image.fromarray(np.concatenate(rows, 0)).save(w / "hold_sheet.jpg", quality=88)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
