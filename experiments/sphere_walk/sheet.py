"""Walk test W.3, in edge-sdg: comparison sheets for the user, JPG for the phone.

    python experiments/sphere_walk/sheet.py --dir data/sphere_test/<place> --arm A --arm B_lama ...

Per arm, data/sphere_test/sheet_walk_<arm>_<place>.jpg: one row per heading (0, 90, 180, 270), columns the
panorama's cut and the camera at 0, forward 2 and 5 m, back 5 m, right 3 m and left 5 m. And
sheet_walk_arms_<place>.jpg: one row per arm, headings 90 and 180, the camera at 0, forward 5 m and right 3 m."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORDS = {"f": "forward", "b": "back", "r": "right", "l": "left"}


def name_of(c):
    return "at the capture point" if c == "0" else f"{WORDS[c[0]]} {c[1:]} m"
TW, TH, BAR = 400, 300, 26


def tile(sheet, draw, font, img, label, x, y):
    sheet.paste(img.resize((TW, TH)), (x, y + BAR))
    draw.text((x + 4, y + 3), label, fill="black", font=font)


def holes(img):
    a = np.asarray(img)
    return ((a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] > 200)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--cols", help="comma-separated moves of the per-arm sheets (default by the grid's distances)")
    args = ap.parse_args()
    d = Path(args.dir)
    w = d / "walk"
    font = ImageFont.load_default(size=18)
    views = {v["name"]: v for v in json.loads((w / "views.json").read_text())["views"]}
    moves = {v["move"] for v in views.values()}
    far = "13" if "f13" in moves else "5"
    COLS = ["0", "f3", "f8", "f13", "b8", "r5", "l13"] if far == "13" else ["0", "f2", "f5", "b5", "r3", "l5"]
    if args.cols:
        COLS = args.cols.split(",")
        far = "5"
    for arm in args.arm:
        heads = (0, 90, 180, 270)
        sheet = Image.new("RGB", ((len(COLS) + 1) * TW, len(heads) * (TH + BAR)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, hd in enumerate(heads):
            y = i * (TH + BAR)
            tile(sheet, draw, font, Image.open(w / f"real_h{hd:03d}.png"), f"{hd} deg: real photo", 0, y)
            for j, c in enumerate(COLS):
                img = Image.open(w / arm / f"h{hd:03d}_{c}.png").convert("RGB")
                blocked = "" if views[f"h{hd:03d}_{c}"].get("reachable", True) else " BLOCKED"
                tile(sheet, draw, font, img, f"{arm}: {name_of(c)} ({holes(img):.0%} holes){blocked}", (j + 1) * TW, y)
        out = d.parent / f"sheet_walk_{arm}_{d.name}.jpg"
        sheet.save(out, quality=85)
        print(out)
    cols, heads = (["0", "f5", "f13", "r8"] if far == "13" else ["0", "f5", "r3"]), (90, 180)
    sheet = Image.new("RGB", (len(heads) * (len(cols) + 1) * TW, len(args.arm) * (TH + BAR)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, arm in enumerate(args.arm):
        y = i * (TH + BAR)
        x = 0
        for hd in heads:
            tile(sheet, draw, font, Image.open(w / f"real_h{hd:03d}.png"), f"{hd} deg: real photo", x, y)
            x += TW
            for c in cols:
                tile(sheet, draw, font, Image.open(w / arm / f"h{hd:03d}_{c}.png").convert("RGB"),
                     f"{arm}: {hd} deg, {name_of(c)}", x, y)
                x += TW
    out = d.parent / f"sheet_walk_arms_{d.name}.jpg"
    sheet.save(out, quality=85)
    print(out)


if __name__ == "__main__":
    main()
