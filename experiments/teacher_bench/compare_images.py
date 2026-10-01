"""One comparison image per photo: the user's marks and each model's boxes side by side.

    python experiments/teacher_bench/compare_images.py   -> runs/teacher_bench/compare/<photo>.jpg
Green dot: user's pot found; orange: missed. Blue box: found a pot; red: false; grey: rejected by the verifier.
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

import common
from report import match

MODELS = ["glm", "locate", "locate_tiles", "sam3", "sam3_0.5", "locate_tiles+qwen"]  # qwen alone: far behind


def main():
    # compare_images.py [folder model...]: another set of models in another folder
    out = common.OUT / (sys.argv[1] if len(sys.argv) > 1 else "compare")
    models = sys.argv[2:] or MODELS
    runs = {m: json.loads((common.OUT / f"{m}.json").read_text()) for m in models if (common.OUT / f"{m}.json").exists()}
    font = ImageFont.load_default(size=28)
    for photo, pts in common.truth().items():
        panels = []
        for m, d in runs.items():
            im = common.load(photo)
            w, h = im.size
            dr = ImageDraw.Draw(im)
            boxes = d["preds"].get(photo, [])
            for b in boxes:
                (x0, x1), (y0, y1) = sorted(b["box"][::2]), sorted(b["box"][1::2])
                b["box"] = [x0, y0, x1, y1]
            found, used = match(boxes, pts)
            for b in d["meta"].get("rejected", {}).get(photo, []):
                (x0, x1), (y0, y1) = sorted(b["box"][::2]), sorted(b["box"][1::2])
                dr.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=(150, 150, 150), width=2)
            for b, ok in zip(boxes, used):
                x0, y0, x1, y1 = b["box"]
                dr.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=(40, 160, 255) if ok else (255, 40, 40), width=3)
            for (x, y), f in zip(pts, found):
                dr.ellipse((x * w - 7, y * h - 7, x * w + 7, y * h + 7),
                           fill=(40, 230, 40) if f is not None else (255, 160, 0), outline=(0, 0, 0))
            n_ok = sum(f is not None for f in found)
            label = f"{m}: {n_ok}/{len(pts)} found, {used.count(False)} false"
            dr.rectangle((0, 0, dr.textlength(label, font=font) + 16, 40), fill=(0, 0, 0))
            dr.text((8, 4), label, fill=(255, 255, 0), font=font)
            panels.append(im)
        cols = 3
        rows = (len(panels) + cols - 1) // cols
        pw, ph = panels[0].size
        sheet = Image.new("RGB", (pw * cols, ph * rows), (20, 20, 20))
        for i, p in enumerate(panels):
            sheet.paste(p, ((i % cols) * pw, (i // cols) * ph))
        dst = out / photo.replace("/", "__")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(dst, quality=85)
    print(out)


if __name__ == "__main__":
    main()
