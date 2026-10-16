"""S.1 check (docs/roadmap_photosphere.md): each photosphere through sphere.capture.prepare, and one JPG sheet per
sphere with the working panorama (each cut's outline drawn on it, cyan level, orange looking down) above the cuts.

    python experiments/sphere/s1_sheet.py data/real/photospheres-scenes/*.PHOTOSPHERE.jpg [--out data/sphere_test]
"""
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim2chip.photo2scene.sphere import capture

ROOT = Path(__file__).resolve().parents[2]
SHEET_WIDTH = 2048
THUMB = (340, 255)
COLOURS = {0.0: (0, 220, 255), 35.0: (255, 150, 0)}


def outline(view: dict, width: int) -> list[list[tuple[float, float]]]:
    """A cut's border in panorama pixels, split where it crosses the seam."""
    w, h = view["size"]
    d = capture.view_rays(view)
    border = np.concatenate([d[0], d[:, -1], d[-1, ::-1], d[::-1, 0]])[::8]
    x, y = capture.to_pixel(border, width, width // 2)
    lines, cur = [], []
    for k in range(len(x)):
        if cur and abs(x[k] - cur[-1][0]) > width / 2:
            lines.append(cur)
            cur = []
        cur.append((float(x[k]), float(y[k])))
    return lines + [cur]


def sheet(d: Path, info: dict) -> Image.Image:
    pano = Image.open(d / "pano.jpg").resize((SHEET_WIDTH, SHEET_WIDTH // 2), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(pano)
    for v in info["cuts"]:
        for line in outline(v, SHEET_WIDTH):
            if len(line) > 1:
                draw.line(line, fill=COLOURS.get(v["pitch_deg"], (255, 0, 255)), width=2)
    rows = np.flatnonzero(np.asarray(Image.open(d / "seen.png").resize(pano.size)).any(1))
    if len(rows):  # only the captured rows, with a margin for the outlines that reach past them
        pano = pano.crop((0, max(rows[0] - 24, 0), SHEET_WIDTH, min(rows[-1] + 24, pano.height)))
    cols = SHEET_WIDTH // THUMB[0]
    rows = -(-len(info["cuts"]) // cols)
    head = 40
    out = Image.new("RGB", (SHEET_WIDTH, head + pano.height + rows * (THUMB[1] + 22)), (255, 255, 255))
    c = info["captured"]
    ImageDraw.Draw(out).text((8, 8), f"{info['source']}: captured {c['up_deg']} up, {c['down_deg']} down, "
                             f"{c['around_share']:.0%} around; {len(info['cuts'])} cuts; heading "
                             f"{info.get('heading_deg')}, {info.get('taken_at')}", fill=(0, 0, 0), font_size=22)
    out.paste(pano, (0, head))
    for k, v in enumerate(info["cuts"]):
        x, y = (k % cols) * THUMB[0], head + pano.height + (k // cols) * (THUMB[1] + 22)
        out.paste(Image.open(d / "cuts" / v["file"]).resize(THUMB, Image.Resampling.LANCZOS), (x, y + 20))
        ImageDraw.Draw(out).text((x + 4, y + 2), f"{v['file']}  {v['captured']:.0%}", fill=(0, 0, 0), font_size=16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spheres", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sphere_test")
    args = ap.parse_args()
    for p in args.spheres:
        d = args.out / p.name.split(".")[0]
        t0 = time.time()
        info = capture.prepare(p, d)
        path = args.out / f"sheet_s1_{d.name}.jpg"
        sheet(d, info).save(path, quality=88)
        print(f"{d.name}: {len(info['cuts'])} cuts in {time.time() - t0:.1f} s, {info['captured']} -> {path}")


if __name__ == "__main__":
    main()
