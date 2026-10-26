"""The synthetic frames at the model's input size, for classifying.

The frames of the walk scenes (scene_render) are whole views with one class each (or the background class), like
the user's photos: each is taken whole.
With spec.look, every crop gets the color, sharpness and noise of the user's camera (look.py).
Output: crops/<class>/*.png
"""
from __future__ import annotations

import json
import random
import shutil

from ..spec import to_dict
from .base import Context, Stage


class Crops(Stage):
    name = "crops"
    version = "4"  # 4: the walk scenes' frames only, whole (the scene template's crops are gone); 3: whole views
    runtime = "cpu"
    uses = ("hardware", "background_class", "generation", "look")
    after = ("render",)
    skippable = "with photos"

    def enabled(self, spec) -> bool:
        from .render import Render
        return spec.task == "roi_classification" and Render().enabled(spec)

    def run(self, ctx: Context) -> dict:
        from PIL import Image

        src = ctx.job.stage_dir("render")
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        size = ctx.hardware.input_size
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

        for lab in labels:
            f, cls = lab["frame"], lab["class"]
            if cls:
                with Image.open(src / f"rgb_{f:04d}.png") as im:
                    img = im.convert("RGB")
                save(img, (0, 0, *img.size), cls, f"{f:05d}.png")
        ctx.log(f"crops {counts}")
        return {"crops": counts}
