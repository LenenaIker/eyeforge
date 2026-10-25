"""The sheet for the user's eye (docs/plan_gs_gen3c_2026-10-20.md, measure 4): post.py's Isaac views of each arm
side by side, at the capture point and moved 3 and 5 m, a few headings.

    python experiments/sphere_walk/gs_sheet.py --dir data/sphere_test/<place> --arm B_layers:capas \\
        --arm N01v:TSDF --arm GS1:splat [--moves 0,f3,f5] [--headings 0,90,180,270]

Writes <dir>/walk/gs_sheet.jpg."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TILE = (512, 384)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", action="append", required=True, help="<post.py folder>:<label>")
    ap.add_argument("--moves", default="0,f3,f5")
    ap.add_argument("--headings", default="0,90,180,270")
    ap.add_argument("--out", default="gs_sheet.jpg")
    args = ap.parse_args()
    w = Path(args.dir) / "walk"
    arms = [a.split(":") for a in args.arm]
    rows = []
    head = Image.new("RGB", (TILE[0] * (len(arms) + 1), 40), "white")
    dr = ImageDraw.Draw(head)
    for j, (_, label) in enumerate(arms):
        dr.text((TILE[0] * (j + 1) + 10, 10), label, fill="black")
    rows.append(np.asarray(head))
    for h in args.headings.split(","):
        for m in args.moves.split(","):
            name = f"h{int(h):03d}_{m}"
            tag = Image.new("RGB", TILE, "white")
            ImageDraw.Draw(tag).text((10, 10), f"rumbo {h}, {'centro' if m == '0' else m}", fill="black")
            tiles = [np.asarray(tag)]
            for folder, _ in arms:
                p = w / folder / f"{name}.png"
                tiles.append(np.asarray(Image.open(p).convert("RGB").resize(TILE)) if p.exists()
                             else np.full((TILE[1], TILE[0], 3), 128, np.uint8))
            rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(w / args.out, quality=85)
    print(w / args.out)


if __name__ == "__main__":
    main()
