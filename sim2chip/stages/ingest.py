"""User photos: labeled by the user and/or by the teacher (label stage). Map their classes to the spec,
resize to the model input and split train/test. With `user_data.test`, the test photos come only from
that folder and all the others go to training. Photos that 3D models were made from never go to test (the
model has seen them as 3D). With fewer than MIN_TEST test photos the score is noisy: the log says so.

Output: ingest/{train,test}/<class>/*.png
Count task: the teacher's (or the user's) points come along; the train_per_class and test_per_class limits
then count photos. Output: ingest/{train,test}/images/*.png and ingest/{train,test}/points.json
{image: {"points": [[x, y, class(, w, h)], ...], "key": photo in the raw folder, "source": its label/ copy}}
(w, h: the teacher's box size; the user's own marks have none)
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

from ..workspace import IMAGE_EXT, content_key
from .base import Context, Stage

MIN_TEST = 20  # below this many test photos (per class) one photo moves the score by 5 points or more


class Ingest(Stage):
    name = "ingest"
    version = "5"  # v4: 3D source photos out of test; count split by photo content. v5: box sizes
    runtime = "cpu"
    reads_input = True
    uses = ("user_data", "hardware", "classes", "background_class", "task")
    after = ("label", "generate_assets")

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

        held = held_out(ctx)
        counts = {}
        for cls in ctx.spec.labels:
            if test_src:
                files, train = pick(sources, cls, ud.train_per_class)
                _, test = pick(test_src, cls, ud.test_per_class)
            else:
                need = ud.test_per_class + ud.train_per_class
                files, picked = pick(sources, cls, need)
                n_test = min(ud.test_per_class, len(picked) // 2) if len(picked) < need else ud.test_per_class
                if held:
                    picked = [f for f in picked if content_key(f) not in held] + \
                             [f for f in picked if content_key(f) in held]
                    n_test = min(n_test, sum(content_key(f) not in held for f in picked))
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
            if 0 < len(test) < MIN_TEST:
                ctx.log(f"WARNING: only {len(test)} test photos of {cls}: each one moves its score by "
                        f"{100 / len(test):.0f} points")
        return {"counts": counts}

    def _count(self, ctx: Context) -> dict:
        from PIL import Image, ImageOps

        from .label import copy_name

        ud, size = ctx.spec.user_data, ctx.hardware.input_size
        label = ctx.job.stage_dir("label")
        verdicts = json.loads((label / "verdicts.json").read_text())
        src_dir = ctx.job.resolve_input(ud.raw)
        # Ordered by photo content, not by name or folder: a photo lands on the same side in every job.
        ck = {k: content_key(src_dir / k) for k, v in verdicts.items() if v["assigned"] == "images"}
        held = held_out(ctx)
        keys = sorted(ck, key=lambda k: (ck[k] in held, hashlib.sha256(f"split:{ck[k]}".encode()).hexdigest()))
        need = ud.test_per_class + ud.train_per_class
        n_test = min(ud.test_per_class, len(keys) // 2) if len(keys) < need else ud.test_per_class
        n_test = min(n_test, sum(ck[k] not in held for k in keys))
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
                points[name] = {"points": [point(o) for o in verdicts[key]["objects"]],
                                "key": key, "source": str(src.relative_to(ctx.job.root))}
            (out / split / "points.json").write_text(json.dumps(points, indent=1))
            counts[split] = {"photos": len(sel), "objects": sum(len(p["points"]) for p in points.values())}
        ctx.log(f"{len(keys)} marked photos ({len(held & set(ck.values()))} used for 3D, kept out of test): "
                f"test {counts['test']}, train {counts['train']}")
        if 0 < n_test < MIN_TEST:
            ctx.log(f"WARNING: only {n_test} test photos: the score is noisy, add photos for a steadier one")
        return {"counts": counts}

    @staticmethod
    def _folder(ctx: Context, key: str, rel: str):
        src = ctx.job.resolve_input(rel)
        if not src.is_dir():
            raise FileNotFoundError(f"user_data.{key}: {src} is not a folder")
        return src


def held_out(ctx: Context) -> set[str]:
    """Content keys of the photos 3D models were made from: a class's own photo folder, and the user's
    photos that generate_assets picked (assets.photos = "user_data")."""
    keys = set()
    for c in ctx.spec.classes:
        if c.assets and c.assets.photos and c.assets.photos != "user_data":
            folder = ctx.job.resolve_input(c.assets.photos)
            if folder.is_dir():
                keys |= {content_key(f) for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXT}
    manifest = ctx.job.stage_dir("generate_assets") / "manifest.json"
    if manifest.exists():
        keys |= {m["photo_key"] for m in json.loads(manifest.read_text()) if m.get("photo_key")}
    return keys


def point(o: dict) -> list:
    """[x, y, class], plus the box size when the mark has a box."""
    b = o.get("box")
    return [o["x"], o["y"], o["class"]] + ([round(b[2] - b[0], 4), round(b[3] - b[1], 4)] if b else [])
