"""User photos: labeled by the user and/or by the teacher (label stage). Map their classes to the spec,
resize to the model input and split train/test. With `user_data.test`, the test photos come only from
that folder and all the others go to training.

Output: ingest/{train,test}/<class>/*.png
Count task: the teacher's (or the user's) points come along; the train_per_class and test_per_class limits
then count photos. Output: ingest/{train,test}/images/*.png and ingest/{train,test}/points.json
{image: {"points": [[x, y, class], ...], "key": photo path in the raw folder, "source": its label/ copy}}
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .base import Context, Stage

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Ingest(Stage):
    name = "ingest"
    version = "3"
    runtime = "cpu"
    reads_input = True
    uses = ("user_data", "hardware", "classes", "background_class", "task")
    after = ("label",)

    def enabled(self, spec) -> bool:
        ud = spec.user_data
        return bool(ud.labeled or ud.raw or ud.test)

    def run(self, ctx: Context) -> dict:
        from PIL import Image, ImageOps

        ud = ctx.spec.user_data
        if ctx.spec.task == "count":
            return self._count(ctx)
        sources = []  # (folder, class map)
        if ud.labeled:
            sources.append((self._folder(ctx, "labeled", ud.labeled), ud.class_map))
        if ud.raw:
            sources.append((ctx.job.stage_dir("label"), {}))  # the teacher already uses spec class names
        # A separate test folder (other day, other place) gives an honest score; without it the test
        # photos are drawn from the same pool as the training ones.
        test_src = [(self._folder(ctx, "test", ud.test), ud.class_map)] if ud.test else []
        size = ctx.hardware.input_size
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        rng = random.Random(0)

        def pick(srcs, cls, n):
            files = sorted(f for src, cmap in srcs for d in cmap.get(cls, [cls]) for f in (src / d).glob("*")
                           if f.suffix.lower() in IMAGE_EXT)
            rng.shuffle(files)
            # Take images in shuffled order until enough pass the size filter: opening every image of a
            # large dataset just to read its size would take minutes.
            picked = []
            for f in files:
                if len(picked) == n:
                    break
                with Image.open(f) as im:
                    if min(im.size) >= ud.min_side_px:
                        picked.append(f)
            return files, picked

        counts = {}
        for cls in ctx.spec.labels:
            if test_src:
                files, train = pick(sources, cls, ud.train_per_class)
                _, test = pick(test_src, cls, ud.test_per_class)
            else:
                need = ud.test_per_class + ud.train_per_class
                files, picked = pick(sources, cls, need)
                n_test = min(ud.test_per_class, len(picked) // 2) if len(picked) < need else ud.test_per_class
                test, train = picked[:n_test], picked[n_test:n_test + ud.train_per_class]
            for split, sel in (("test", test), ("train", train)):
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for f in sel:
                    with Image.open(f) as im:
                        # Phones store the orientation in EXIF instead of rotating the pixels.
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        im.resize((size, size), Image.BILINEAR).save(d / f"{f.stem}.png")
            counts[cls] = {"test": len(test), "train": len(train)}
            ctx.log(f"{cls:16s} {len(files):7d} photos, test {len(test)}, train {len(train)}")
        return {"counts": counts}

    def _count(self, ctx: Context) -> dict:
        from PIL import Image, ImageOps

        from .label import copy_name

        ud, size = ctx.spec.user_data, ctx.hardware.input_size
        label = ctx.job.stage_dir("label")
        verdicts = json.loads((label / "verdicts.json").read_text())
        keys = sorted(k for k, v in verdicts.items() if v["assigned"] == "images")
        random.Random(0).shuffle(keys)
        need = ud.test_per_class + ud.train_per_class
        n_test = min(ud.test_per_class, len(keys) // 2) if len(keys) < need else ud.test_per_class
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        counts = {}
        for split, sel in (("test", keys[:n_test]), ("train", keys[n_test:n_test + ud.train_per_class])):
            d = out / split / "images"
            d.mkdir(parents=True)
            points = {}
            for key in sel:
                src = label / "images" / copy_name(key)
                name = f"{Path(copy_name(key)).stem}.png"
                with Image.open(src) as im:
                    ImageOps.exif_transpose(im).convert("RGB").resize((size, size), Image.BILINEAR).save(d / name)
                points[name] = {"points": [[o["x"], o["y"], o["class"]] for o in verdicts[key]["objects"]],
                                "key": key, "source": str(src.relative_to(ctx.job.root))}
            (out / split / "points.json").write_text(json.dumps(points, indent=1))
            counts[split] = {"photos": len(sel), "objects": sum(len(p["points"]) for p in points.values())}
        ctx.log(f"{len(keys)} marked photos: test {counts['test']}, train {counts['train']}")
        return {"counts": counts}

    @staticmethod
    def _folder(ctx: Context, key: str, rel: str):
        src = ctx.job.resolve_input(rel)
        if not src.is_dir():
            raise FileNotFoundError(f"user_data.{key}: {src} is not a folder")
        return src
