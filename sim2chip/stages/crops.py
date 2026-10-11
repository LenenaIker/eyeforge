"""Crops of the region of interest at the model's input size, from the synthetic renders.

Targets: crop around the tight bbox with a random margin. Background frames: boxes drawn from the target
boxes of this same render (size, aspect and height in the image, jittered), so where and how big a crop
is tells nothing about its class, whatever the scene (a road, a balcony, a table).
With spec.look, every crop gets the color, sharpness and noise of the user's camera (look.py).
Output: crops/<class>/*.png
"""
from __future__ import annotations

import json
import random
import shutil

from ..spec import to_dict
from .base import Context, Stage

MIN_SIDE = 0.2  # of the model's input size: a target smaller than this in the render carries no useful detail


class Crops(Stage):
    name = "crops"
    version = "2"  # background boxes from the target boxes, not a road-sized band
    runtime = "cpu"
    uses = ("hardware", "background_class", "generation", "look")
    after = ("render",)
    skippable = "with photos"

    def enabled(self, spec) -> bool:
        from .render import Render
        return spec.task == "roi_classification" and Render().enabled(spec)

    def run(self, ctx: Context) -> dict:
        import numpy as np
        from PIL import Image

        src = ctx.job.stage_dir("render")
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        size = ctx.hardware.input_size
        bg = ctx.spec.background_class
        rng = random.Random(ctx.spec.generation.seed)
        counts: dict[str, int] = {}
        labels = json.loads((src / "labels.json").read_text())
        tf, look_rng = None, random.Random(f"look-{ctx.spec.generation.seed}")  # own rng: same crops with or without
        if ctx.spec.look:
            from .. import look
            tf = look.for_renders(to_dict(ctx.spec.look), [src / f"rgb_{lab['frame']:04d}.png" for lab in labels], size)
            ctx.log(f"camera look: {tf}")

        def save(img, box, cls, name):
            d = out / cls
            d.mkdir(parents=True, exist_ok=True)
            crop = img.crop(tuple(int(round(v)) for v in box)).resize((size, size), Image.BILINEAR)
            if tf:
                from ..look import apply
                crop = apply(crop, tf, look_rng)
            crop.save(d / name)
            counts[cls] = counts.get(cls, 0) + 1

        boxes = []  # (w, h, center y), as fractions of the image, of every target crop
        empty = []
        for lab in labels:
            f, cls = lab["frame"], lab["class"]
            if cls == bg:
                empty.append(f)
                continue
            with Image.open(src / f"rgb_{f:04d}.png") as im:
                img = im.convert("RGB")
            W, H = img.size
            bb = np.load(src / f"bounding_box_2d_tight_{f:04d}.npy")
            names = json.loads((src / f"bounding_box_2d_tight_labels_{f:04d}.json").read_text())
            ids = [int(k) for k, v in names.items() if v.get("class") == cls]
            rows = [r for r in bb if int(r["semanticId"]) in ids]
            if not rows:
                continue  # target out of frame
            r = rows[0]
            x0, y0, x1, y1 = float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])
            bw, bh = x1 - x0, y1 - y0
            if min(bw, bh) < MIN_SIDE * size:
                continue
            if (x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2) and rng.random() < 0.7:
                continue  # mostly cut by the image border
            m = [rng.uniform(0.03, 0.25) for _ in range(4)]
            box = (max(0, x0 - m[0] * bw), max(0, y0 - m[1] * bh), min(W, x1 + m[2] * bw), min(H, y1 + m[3] * bh))
            save(img, box, cls, f"{f:05d}.png")
            boxes.append(((box[2] - box[0]) / W, (box[3] - box[1]) / H, (box[1] + box[3]) / 2 / H))
        for f in empty:
            with Image.open(src / f"rgb_{f:04d}.png") as im:
                img = im.convert("RGB")
            W, H = img.size
            for k in range(2):
                if boxes:
                    bw, bh, cy = rng.choice(boxes)
                    j = rng.uniform(0.8, 1.25)
                    w, h = min(W, bw * W * j), min(H, bh * H * j)
                    y = min(max(0.0, cy * H - h / 2 + rng.uniform(-0.1, 0.1) * H), H - h)
                else:  # no target crops at all: anywhere, any size
                    w = rng.uniform(MIN_SIDE * size, W)
                    h = min(H, w / rng.uniform(0.7, 1.8))
                    y = rng.uniform(0, H - h)
                x = rng.uniform(0, W - w)
                save(img, (x, y, x + w, y + h), cls=bg, name=f"{f:05d}_{k}.png")
        ctx.log(f"crops {counts}")
        return {"crops": counts}
