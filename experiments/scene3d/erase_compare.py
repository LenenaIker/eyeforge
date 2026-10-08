"""Side-by-side sheet of object removal: the photo, LaMa (bg_prepare.py background.png) and Qwen-Image-Edit
(erase_qwen.py background_qwen.png), one row per photo, the erase mask outlined on the photo. Any env with PIL:

    python experiments/scene3d/erase_compare.py --out data/scene3d_test/erase_compare \
        --dir data/scene3d_test/flow/home_office --dir data/street_test/flow/basque_street

Writes <out>/<name>.jpg per photo and <out>/sheet.jpg with every row.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

H = 420


def outlined(photo, mask):
    a = np.asarray(photo).copy()
    m = mask
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            a[np.roll(np.roll(edge, dy, 0), dx, 1)] = [255, 40, 200]
    return Image.fromarray(a)


def row(d):
    photo = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
    mask = np.asarray(Image.open(os.path.join(d, 'erased.png'))) > 0
    tiles = [('original', outlined(photo, mask)),
             ('remover_actual (LaMa)', Image.open(os.path.join(d, 'background.png'))),
             ('qwen', Image.open(os.path.join(d, 'background_qwen.png')))]
    tiles = [(n, t.convert('RGB').resize((round(t.width * H / t.height), H))) for n, t in tiles]
    w = sum(t.width for _, t in tiles) + 8 * (len(tiles) - 1)
    out = Image.new('RGB', (w, H + 26), 'white')
    x = 0
    dr = ImageDraw.Draw(out)
    for n, t in tiles:
        out.paste(t, (x, 26))
        dr.text((x + 6, 6), f'{os.path.basename(d.rstrip("/"))}: {n}', fill=(0, 0, 0))
        x += t.width + 8
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', action='append', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for d in args.dir:
        if not os.path.exists(os.path.join(d, 'background_qwen.png')):
            continue
        r = row(d)
        r.save(os.path.join(args.out, os.path.basename(d.rstrip('/')) + '.jpg'), quality=92)
        rows.append(r)
    sheet = Image.new('RGB', (max(r.width for r in rows), sum(r.height + 8 for r in rows)), 'white')
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 8
    sheet.save(os.path.join(args.out, 'sheet.jpg'), quality=90)


if __name__ == '__main__':
    main()
