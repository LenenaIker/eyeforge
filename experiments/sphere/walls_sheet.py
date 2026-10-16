"""Wall straightening check (user, 2026-10-17): each place's proxy rebuilt with its walls straightened
(geometry.solve), against the copy kept before (proxy_before_walls.npz). One JPG: per place, the proxy from above
before and after (5 m grid, height over the ground in colour, the found walls in red on the after side) and per
wall how far its points stray from a straight line, before and after.

    python experiments/sphere/walls_sheet.py data/sphere_test/PXL_*
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from s2_sheet import TOP, TOP_HALF_M, VIRIDIS, ramp  # noqa: E402

from sim2chip.photo2scene.sphere import geometry  # noqa: E402

BAND_M = 0.6  # points this close to a wall's line count as the wall's when measuring its straightness


def above(v: np.ndarray, height: float, walls=()) -> Image.Image:
    img = np.full((TOP, TOP, 3), 245, np.uint8)
    s = TOP / (2 * TOP_HALF_M)
    x, z = (v[:, 0] * s + TOP / 2).astype(int), (TOP / 2 - v[:, 2] * s).astype(int)
    over = np.clip((height - v[:, 1]) / 8.0, 0, 1)
    ok = (x >= 0) & (x < TOP) & (z >= 0) & (z < TOP)
    order = np.argsort(over[ok])
    img[z[ok][order], x[ok][order]] = ramp(VIRIDIS, over[ok][order])
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    for m in np.arange(-TOP_HALF_M, TOP_HALF_M + 1, 5):
        p = m * s + TOP / 2
        draw.line([(p, 0), (p, TOP)], fill=(205, 205, 205))
        draw.line([(0, p), (TOP, p)], fill=(205, 205, 205))
    c = TOP / 2
    draw.ellipse([c - 6, c - 6, c + 6, c + 6], fill=(255, 0, 0))
    for w in walls:  # its line across the view, n.(x, z) + c = 0
        n = np.array([w["normal"][0], w["normal"][2]])
        t = np.array([-n[1], n[0]])
        p0 = -w["offset"] * n
        a, b = p0 - 60 * t, p0 + 60 * t
        draw.line([(a[0] * s + c, c - a[1] * s), (b[0] * s + c, c - b[1] * s)], fill=(230, 30, 30), width=1)
    return out


def stray(v: np.ndarray, ground: np.ndarray, height: float, w: dict) -> float:
    """RMS distance (cm) from the wall's line of the proxy points in its band, above the ground."""
    n = np.array([w["normal"][0], w["normal"][2]])
    q = v[:, [0, 2]]
    off = q @ n + w["offset"]
    sel = (~ground) & (np.abs(off) < BAND_M) & (height - v[:, 1] > geometry.WALL_MIN_HEIGHT_M)
    return float(np.sqrt(np.mean(off[sel] ** 2)) * 100) if sel.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("places", nargs="+", type=Path)
    args = ap.parse_args()
    panels = []
    for d in args.places:
        old = json.loads((d / "geometry.json").read_text())
        geo = geometry.solve(d, old.get("height_given_m"), proxy_depth=old.get("proxy_from"),
                             proxy_stuff=old.get("proxy_ground_from"))
        before, after = np.load(d / "proxy_before_walls.npz"), np.load(d / "proxy.npz")
        h = geo["camera_height_m"]
        lines = [f"{d.name}: {len(geo['walls'])} walls straightened"]
        for k, w in enumerate(geo["walls"]):
            lines.append(f"wall {k}: {w['width_m']:.1f} x {w['height_m']:.1f} m, its points off a plane "
                         f"{w['stray_before_m'] * 100:.0f} -> {w['stray_after_m'] * 100:.0f} cm rms "
                         f"({w['moved']:.0%} moved)")
        print("\n  ".join(lines), flush=True)
        left = above(before["vertices"][~before["ground"]], h)
        right = above(after["vertices"][~after["ground"]], h, geo["walls"])
        panel = Image.new("RGB", (2 * TOP + 20, TOP + 34 + 24 * len(lines)), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((8, 6), "before", fill=(0, 0, 0), font_size=22)
        draw.text((TOP + 28, 6), "after (walls in red)", fill=(0, 0, 0), font_size=22)
        panel.paste(left, (0, 34))
        panel.paste(right, (TOP + 20, 34))
        for k, line in enumerate(lines):
            draw.text((8, TOP + 40 + 24 * k), line, fill=(0, 0, 0), font_size=20)
        panels.append(panel)
    out = Image.new("RGB", (panels[0].width, sum(p.height + 20 for p in panels)), "white")
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height + 20
    path = args.places[0].parent / "sheet_walls.jpg"
    out.save(path, quality=88)
    print(path)


if __name__ == "__main__":
    main()
