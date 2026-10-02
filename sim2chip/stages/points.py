"""Count task: whole synthetic frames at the model's input size, with the centre point of every visible target.

The centre is that of the tight (visible part) bounding box, in 0..1 of the image, so it stays right after
the frame is resized to the square model input. Targets hidden almost completely are left out: the camera
could not count them either.
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
MAX_OCCLUSION = 0.85  # fraction hidden by other things


class Points(Stage):
    name = "points"
    version = "3"  # v2: box size with every point; v3: frames at the count model's input size (a detector's own)
    runtime = "cpu"
    uses = ("hardware", "classes", "task", "look")
    after = ("render",)

    def enabled(self, spec) -> bool:
        return spec.task == "count" and spec.synthetic

    def run(self, ctx: Context) -> dict:
        import numpy as np
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
            bb_file = src / f"bounding_box_2d_tight_{f:04d}.npy"
            if bb_file.exists():
                names = json.loads((src / f"bounding_box_2d_tight_labels_{f:04d}.json").read_text())
                bb = np.load(bb_file)
                for r in bb:
                    cls = names.get(str(int(r["semanticId"])), {}).get("class")
                    x0, y0, x1, y1 = float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])
                    hidden = float(r["occlusionRatio"]) if "occlusionRatio" in (bb.dtype.names or ()) else 0.0
                    if cls in classes and min(x1 - x0, y1 - y0) >= MIN_SIDE and hidden <= MAX_OCCLUSION:
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
