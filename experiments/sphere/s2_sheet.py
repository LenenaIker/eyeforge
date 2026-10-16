"""S.2 check (docs/roadmap_photosphere.md): each photosphere through sphere.build_geometry (cuts, MoGe-2 360 depth,
SAM 3 ground, ground plane, scale and proxy), then one JPG sheet per sphere:
    the panorama with SAM 3's ground (green) and sky (blue);
    the 360 distance (log colour, 1 to 60 m; black where nothing was measured);
    the proxy from above (height over the ground in colour, 5 m grid, the camera in the centre, heading 0 up) next to
    the numbers: the ground's tilt, inliers and residual, MoGe's camera height against the capture height.

    python experiments/sphere/s2_sheet.py data/real/photospheres-scenes/*.PHOTOSPHERE.jpg [--height 1.75]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim2chip.photo2scene import Flow
from sim2chip.photo2scene.sphere import build_geometry

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 2048
TOP = 1024           # side of the view from above, px
TOP_HALF_M = 25.0    # it shows 50 x 50 m around the camera
# Colour ramps as a few stops, linearly interpolated (near turbo and viridis; no matplotlib in the env).
TURBO = np.array([[48, 18, 59], [70, 134, 251], [27, 229, 181], [164, 252, 60], [251, 185, 56], [227, 68, 10],
                  [122, 4, 3]], np.float32)
VIRIDIS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float32)


def ramp(stops: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def captured_rows(d: Path, size) -> slice:
    rows = np.flatnonzero(np.asarray(Image.open(d / "seen.png").resize(size)).any(1))
    return slice(int(rows[0]), int(rows[-1]) + 1)


def overlay(d: Path) -> Image.Image:
    pano = np.asarray(Image.open(d / "pano.jpg").resize((WIDTH, WIDTH // 2))).astype(np.float32)
    stuff = np.load(d / "stuff.npz")
    for name, colour in (("ground", (40, 220, 60)), ("sky", (60, 120, 255))):
        m = np.asarray(Image.fromarray(stuff[name].astype(np.float32)).resize((WIDTH, WIDTH // 2))) > 0.5
        pano[m] = pano[m] * 0.55 + np.array(colour) * 0.45
    return Image.fromarray(pano.astype(np.uint8))


def distance_picture(d: Path) -> Image.Image:
    depth = np.load(d / "depth.npz")
    dist, valid = depth["distance"], depth["valid"]
    t = np.clip(np.log(np.maximum(dist, 1e-3)) / np.log(60), 0, 1)
    rgb = ramp(TURBO, t)
    rgb[~valid] = 0
    return Image.fromarray(rgb).resize((WIDTH, WIDTH // 2), Image.Resampling.NEAREST)


def from_above(d: Path, height: float) -> Image.Image:
    proxy = np.load(d / "proxy.npz")
    v = proxy["vertices"][~proxy["ground"]]  # without the ground, which covers everything
    img = np.full((TOP, TOP, 3), 245, np.uint8)
    s = TOP / (2 * TOP_HALF_M)
    x, z = (v[:, 0] * s + TOP / 2).astype(int), (TOP / 2 - v[:, 2] * s).astype(int)
    over = np.clip((height - v[:, 1]) / 8.0, 0, 1)  # height over the ground, 0 to 8 m
    ok = (x >= 0) & (x < TOP) & (z >= 0) & (z < TOP)
    order = np.argsort(over[ok])  # the tall ones drawn last
    img[z[ok][order], x[ok][order]] = ramp(VIRIDIS, over[ok][order])
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    for m in np.arange(-TOP_HALF_M, TOP_HALF_M + 1, 5):
        p = m * s + TOP / 2
        draw.line([(p, 0), (p, TOP)], fill=(200, 200, 200))
        draw.line([(0, p), (TOP, p)], fill=(200, 200, 200))
    c = TOP / 2
    draw.ellipse([c - 6, c - 6, c + 6, c + 6], fill=(255, 0, 0))
    draw.line([(c, c), (c, c - 40)], fill=(255, 0, 0), width=3)
    draw.text((8, 8), "from above, 5 m grid; colour = height over ground (0-8 m)", fill=(0, 0, 0), font_size=20)
    return out


def sheet(d: Path, info: dict) -> Image.Image:
    rows = captured_rows(d, (WIDTH, WIDTH // 2))
    pano = overlay(d).crop((0, rows.start, WIDTH, rows.stop))
    dist = distance_picture(d).crop((0, rows.start, WIDTH, rows.stop))
    above = from_above(d, info["geometry"]["camera_height_m"])
    out = Image.new("RGB", (WIDTH, 40 + pano.height + dist.height + TOP + 20), "white")
    g, geo = info["geometry"]["ground"], info["geometry"]
    rise = "-" if g["rise_10_20_m"] is None else f"{g['rise_10_20_m'] * geo['scale'] * 100:.0f} cm"
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), f"{d.name}: ground (green), sky (blue); distance 1-60 m (log)", fill=(0, 0, 0), font_size=22)
    out.paste(pano, (0, 40))
    out.paste(dist, (0, 40 + pano.height))
    y = 40 + pano.height + dist.height + 10
    out.paste(above, (0, y))
    lines = [f"camera over the ground, MoGe-2: {g['height_m']:.2f} m",
             f"height used: {geo['camera_height_m']:.2f} m ({geo['height_from']})",
             f"scale applied: {geo['scale']:.3f}",
             f"ground tilt from the stitcher's vertical: {g['tilt_deg']:.2f} deg",
             f"ground points on the plane: {g['inliers']:.0%} of {g['points']}",
             f"median distance to the plane: {g['residual_m'] * 100:.1f} cm (MoGe scale)",
             f"measured ground over the plane at 10-20 m: {rise}",
             f"proxy: {geo['proxy']['vertices']} vertices, {geo['proxy']['faces']} faces",
             f"captured: {info['captured']['up_deg']} up, {info['captured']['down_deg']} down"]
    for k, line in enumerate(lines):
        draw.text((TOP + 40, y + 20 + 40 * k), line, fill=(0, 0, 0), font_size=28)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spheres", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sphere_test")
    ap.add_argument("--height", type=float, default=None,
                    help="height the spheres were taken at, metres (default: MoGe-2's measure)")
    args = ap.parse_args()
    summary = []
    for p in args.spheres:
        d = args.out / p.name.split(".")[0]
        t0 = time.time()
        info = build_geometry(Flow(d), p, args.height)
        path = args.out / f"sheet_s2_{d.name}.jpg"
        sheet(d, info).save(path, quality=88)
        g = info["geometry"]["ground"]
        summary.append({"place": d.name, "seconds": round(time.time() - t0), **{k: g[k] for k in (
            "height_m", "tilt_deg", "inliers", "residual_m")}})
        print(json.dumps(summary[-1]), "->", path, flush=True)
    (args.out / "s2_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
