"""User photos: labeled by the user and/or by the teacher (label stage). Map their classes to the spec,
resize to the model input and split train/test.

Output: ingest/{train,test}/<class>/*.png
"""
from __future__ import annotations

import random
import shutil

from .base import Context, Stage

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Ingest(Stage):
    name = "ingest"
    runtime = "cpu"
    uses = ("user_data", "hardware", "classes", "background_class")
    after = ("label",)

    def enabled(self, spec) -> bool:
        return bool(spec.user_data.labeled or spec.user_data.raw)

    def run(self, ctx: Context) -> dict:
        from PIL import Image

        ud = ctx.spec.user_data
        sources = []  # (folder, class map)
        if ud.labeled:
            src = ctx.job.resolve_input(ud.labeled)
            if not src.is_dir():
                raise FileNotFoundError(f"user_data.labeled: {src} is not a folder")
            sources.append((src, ud.class_map))
        if ud.raw:
            sources.append((ctx.job.stage_dir("label"), {}))  # the teacher already uses spec class names
        size = ctx.hardware.input_size
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        rng = random.Random(0)
        need = ud.test_per_class + ud.train_per_class
        counts = {}
        for cls in ctx.spec.labels:
            files = sorted(f for src, cmap in sources for d in cmap.get(cls, [cls]) for f in (src / d).glob("*")
                           if f.suffix.lower() in IMAGE_EXT)
            rng.shuffle(files)
            # Take images in shuffled order until enough pass the size filter: opening every image of a
            # large dataset just to read its size would take minutes.
            picked = []
            for f in files:
                with Image.open(f) as im:
                    if min(im.size) >= ud.min_side_px:
                        picked.append(f)
                if len(picked) == need:
                    break
            n_test = min(ud.test_per_class, len(picked) // 2) if len(picked) < need else ud.test_per_class
            splits = {"test": picked[:n_test], "train": picked[n_test:n_test + ud.train_per_class]}
            for split, sel in splits.items():
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for f in sel:
                    with Image.open(f) as im:
                        im.convert("RGB").resize((size, size), Image.BILINEAR).save(d / f"{f.stem}.png")
            counts[cls] = {k: len(v) for k, v in splits.items()}
            ctx.log(f"{cls:16s} {len(files):7d} photos, test {counts[cls]['test']}, train {counts[cls]['train']}")
        return {"counts": counts}
