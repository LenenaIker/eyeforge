"""S.3 check (docs/roadmap_photosphere.md): each photosphere through sphere.build_geometry and
sphere.build_background (SAM 3 objects across the cuts, each erased once with Qwen), then one JPG sheet per sphere:
    the panorama with the objects outlined (targets red, movable things yellow, truncated ones orange) and ids;
    the erased panorama (the background);
    the ground kinds SAM 3 voted (road grey-blue, parking violet, sidewalk sand, grass green, floor brown);
    each erase view before and after.

    python experiments/sphere/s3_sheet.py data/real/photospheres-scenes/*.PHOTOSPHERE.jpg --target car \
        --movable person --movable bicycle
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim2chip.photo2scene import Flow
from sim2chip.photo2scene.sphere import GROUND_KINDS, build_background, build_geometry

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 2048
THUMB_H = 220
ROLE_COLOURS = {"target": (255, 40, 40), "movable": (255, 220, 0)}
TRUNCATED = (255, 140, 0)
KIND_COLOURS = {"road": (90, 110, 160), "parking": (150, 80, 200), "sidewalk": (230, 200, 130),
                "grass": (60, 200, 60), "floor": (150, 100, 60)}


def captured_rows(d: Path) -> slice:
    rows = np.flatnonzero(np.asarray(Image.open(d / "seen.png").resize((WIDTH, WIDTH // 2))).any(1))
    return slice(int(rows[0]), int(rows[-1]) + 1)


def outlined(d: Path) -> Image.Image:
    pano = np.asarray(Image.open(d / "pano.jpg").resize((WIDTH, WIDTH // 2))).copy()
    inst = json.loads((d / "instances.json").read_text())
    z = np.load(d / "instances.npz")
    MW, MH = (int(v) for v in z["size"])
    labels = []
    for i, o in enumerate(inst["objects"]):
        m = np.unpackbits(z["masks"][i])[:MW * MH].reshape(MH, MW)
        m = np.asarray(Image.fromarray(m * 255).resize((WIDTH, WIDTH // 2), Image.Resampling.NEAREST)) > 127
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        colour = TRUNCATED if o["truncated"] else ROLE_COLOURS[o["role"]]
        for dy in (0, 1):
            pano[np.roll(edge, dy, 0)] = colour
        ys, xs = np.nonzero(m)
        labels.append((int(np.median(xs)), int(ys.min()) - 18, str(o["id"]), colour))
    out = Image.fromarray(pano)
    draw = ImageDraw.Draw(out)
    for x, y, text, colour in labels:
        draw.text((x, max(y, 0)), text, fill=colour, font_size=16, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def kinds(d: Path) -> Image.Image:
    pano = np.asarray(Image.open(d / "background.jpg").resize((WIDTH, WIDTH // 2))).astype(np.float32) * 0.45
    stuff = np.load(d / "stuff.npz")
    share = np.stack([np.asarray(Image.fromarray(stuff[k].astype(np.float32)).resize((WIDTH, WIDTH // 2)))
                      for k in GROUND_KINDS])
    best = share.argmax(0)
    for k, kind in enumerate(GROUND_KINDS):
        m = (best == k) & (share[k] > 0.5)
        pano[m] += np.array(KIND_COLOURS[kind]) * 0.55
    return Image.fromarray(np.clip(pano, 0, 255).astype(np.uint8))


def erase_strip(d: Path, names: list[str]) -> list[Image.Image]:
    rows, row, x = [], Image.new("RGB", (WIDTH, THUMB_H + 24), "white"), 0
    for name in names:
        pair = [Image.open(d / "erase" / f"{name}_{s}.jpg") for s in ("before", "after")]
        pair = [p.resize((round(p.width * THUMB_H / p.height), THUMB_H)) for p in pair]
        w = pair[0].width * 2 + 8
        if x + w > WIDTH:
            rows.append(row)
            row, x = Image.new("RGB", (WIDTH, THUMB_H + 24), "white"), 0
        ImageDraw.Draw(row).text((x + 4, 2), f"{name}: before | after", fill=(0, 0, 0), font_size=16)
        row.paste(pair[0], (x, 24))
        row.paste(pair[1], (x + pair[0].width + 8, 24))
        x += w + 16
    return rows + [row]


def sheet(d: Path) -> Image.Image:
    rows = captured_rows(d)
    crop = lambda im: im.crop((0, rows.start, WIDTH, rows.stop))  # noqa: E731
    inst, erase = json.loads((d / "instances.json").read_text()), json.loads((d / "erase.json").read_text())
    parts = [crop(outlined(d)), crop(Image.open(d / "background.jpg").resize((WIDTH, WIDTH // 2))), crop(kinds(d)),
             *erase_strip(d, [g["name"] for g in erase["groups"]])]
    left = erase["left"]
    roles = {r: sum(o["role"] == r for o in inst["objects"]) for r in ROLE_COLOURS}
    head = (f"{d.name}: {roles['target']} targets ({', '.join(inst['targets'])}), {roles['movable']} movable; "
            f"{sum(o['truncated'] for o in inst['objects'])} truncated; {len(erase['groups'])} erase groups in "
            f"{erase['seconds']:.0f} s; left after the eraser: {left or 'none'}")
    out = Image.new("RGB", (WIDTH, 40 + sum(p.height + 6 for p in parts)), "white")
    ImageDraw.Draw(out).text((8, 8), head, fill=(0, 0, 0), font_size=20)
    y = 40
    for p in parts:
        out.paste(p, (0, y))
        y += p.height + 6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spheres", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sphere_test")
    ap.add_argument("--target", action="append", required=True)
    ap.add_argument("--movable", action="append", default=[])
    ap.add_argument("--height", type=float, default=None)
    args = ap.parse_args()
    for p in args.spheres:
        d = args.out / p.name.split(".")[0]
        t0 = time.time()
        flow = Flow(d)
        build_geometry(flow, p, args.height)
        build_background(flow, args.target, args.movable, args.height)
        path = args.out / f"sheet_s3_{d.name}.jpg"
        sheet(d).save(path, quality=88)
        print(f"{d.name}: {time.time() - t0:.0f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
