"""Count task: whole synthetic frames at the model's input size, with the centre point of every visible target.

The centre is that of the tight (visible part) bounding box, in 0..1 of the image, so it stays right after
the frame is resized to the square model input. Targets hidden almost completely have no box (scene_render):
the camera could not count them either.
With spec.look, every frame gets the color, sharpness and noise of the user's camera (look.py).
Output: points/images/*.png, points/points.json {image: [[x, y, class, w, h], ...]} (w, h: box size in 0..1)
"""
from __future__ import annotations

import json
import random
import shutil

from ..spec import to_dict
from .base import Context, Stage

MIN_SIDE = 3          # px in the render: smaller than this is not an object anyone could count


class Points(Stage):
    name = "points"
    version = "5"  # v2: box size with every point; v3: frames at the count model's input size (a detector's own)
    # v4: frames of the user's place with their boxes in the label; v5: the walk scenes' frames only
    runtime = "cpu"
    uses = ("hardware", "classes", "task", "look")
    after = ("render",)
    skippable = "with photos"

    def enabled(self, spec) -> bool:
        from .render import Render
        return spec.task == "count" and Render().enabled(spec)

    def run(self, ctx: Context) -> dict:
        from PIL import Image

        src = ctx.job.stage_dir("render")
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        (out / "images").mkdir(parents=True)
        size = ctx.hardware.count_input()
        classes = {c.id for c in ctx.spec.classes}
        points: dict[str, list] = {}
        labels = json.loads((src / "labels.json").read_text())
        tf, rng = None, random.Random(ctx.spec.generation.seed)
        if ctx.spec.look:
            from .. import look
            tf = look.for_renders(to_dict(ctx.spec.look), [src / f"rgb_{lab['frame']:04d}.png" for lab in labels], size)
            ctx.log(f"camera look: {tf}")
        for lab in labels:
            f = lab["frame"]
            img = Image.open(src / f"rgb_{f:04d}.png").convert("RGB")
            W, H = img.size
            pts = []
            for cls, x0, y0, x1, y1 in lab.get("boxes", []):  # the visible part's box (scene_render)
                if cls in classes and min(x1 - x0, y1 - y0) >= MIN_SIDE:
                    pts.append([round((x0 + x1) / 2 / W, 4), round((y0 + y1) / 2 / H, 4), cls,
                                round((x1 - x0) / W, 4), round((y1 - y0) / H, 4)])
            name = f"{f:05d}.png"
            small = img.resize((size, size), Image.BILINEAR)
            if tf:
                from ..look import apply
                small = apply(small, tf, rng)
            small.save(out / "images" / name)
            points[name] = pts
        (out / "points.json").write_text(json.dumps(points))
        n = [len(p) for p in points.values()]
        ctx.log(f"{len(n)} frames, {sum(n)} objects, up to {max(n, default=0)} per frame")
        return {"frames": len(n), "objects": sum(n)}
