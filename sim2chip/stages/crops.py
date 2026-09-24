"""Crops of the region of interest at the model's input size, from the synthetic renders.

Targets: crop around the tight bbox with a random margin. Background frames: random boxes of similar
size and aspect in the lower part of the image (where targets usually are).
Output: crops/<class>/*.png
"""
from __future__ import annotations

import json
import random
import shutil

from .base import Context, Stage

MIN_SIDE = 20  # px in the render; smaller targets carry no useful detail


class Crops(Stage):
    name = "crops"
    runtime = "cpu"
    uses = ("hardware", "background_class")
    after = ("render",)

    def enabled(self, spec) -> bool:
        return any(c.assets for c in spec.classes)

    def run(self, ctx: Context) -> dict:
        import numpy as np
        from PIL import Image

        src = ctx.job.stage_dir("render")
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        size = ctx.hardware.input_size
        bg = ctx.spec.background_class
        rng = random.Random(0)
        counts: dict[str, int] = {}

        def save(img, box, cls, name):
            d = out / cls
            d.mkdir(parents=True, exist_ok=True)
            img.crop(tuple(int(round(v)) for v in box)).resize((size, size), Image.BILINEAR).save(d / name)
            counts[cls] = counts.get(cls, 0) + 1

        for lab in json.loads((src / "labels.json").read_text()):
            f, cls = lab["frame"], lab["class"]
            img = Image.open(src / f"rgb_{f:04d}.png").convert("RGB")
            W, H = img.size
            if cls == bg:
                for k in range(2):
                    w = rng.uniform(40, min(260, W))
                    h = w / rng.uniform(0.7, 1.8)
                    x = rng.uniform(0, W - w)
                    y = rng.uniform(H * 0.35, max(H * 0.35, H - h))
                    save(img, (x, y, x + w, min(H, y + h)), cls, f"{f:05d}_{k}.png")
                continue
            bb = np.load(src / f"bounding_box_2d_tight_{f:04d}.npy")
            names = json.loads((src / f"bounding_box_2d_tight_labels_{f:04d}.json").read_text())
            ids = [int(k) for k, v in names.items() if v.get("class") == cls]
            rows = [r for r in bb if int(r["semanticId"]) in ids]
            if not rows:
                continue  # target out of frame
            r = rows[0]
            x0, y0, x1, y1 = float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])
            bw, bh = x1 - x0, y1 - y0
            if min(bw, bh) < MIN_SIDE:
                continue
            if (x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2) and rng.random() < 0.7:
                continue  # mostly cut by the image border
            m = [rng.uniform(0.03, 0.25) for _ in range(4)]
            save(img, (max(0, x0 - m[0] * bw), max(0, y0 - m[1] * bh), min(W, x1 + m[2] * bw), min(H, y1 + m[3] * bh)),
                 cls, f"{f:05d}.png")
        ctx.log(f"crops {counts}")
        return {"crops": counts}
