"""User photos: labeled by the user and/or by the teacher (label stage). Map their classes to the spec,
resize to the model input and split train/test. With `user_data.test`, the test photos come only from
that folder and all the others go to training. Photos that 3D models were made from never go to test (the
model has seen them as 3D). With fewer than MIN_TEST test photos the score is noisy: the log says so.

The split is kept (split.json, by photo content): photos added to a trained job never move the old ones, so a
new version is scored on the same test photos as the old one. Half of the added photos go to test (the case
the user found missing gets measured too) and the rest all go to training, past the train_per_class cap.
Exact copies of a photo count once (they would otherwise land on both sides).

Output: ingest/{train,test}/<class>/*.png, ingest/split.json {photo content key: "test" | "train" | "unused"}
Count task: the teacher's (or the user's) points come along; the train_per_class and test_per_class limits
then count photos. Output: ingest/{train,test}/images/*.png and ingest/{train,test}/points.json
{image: {"points": [[x, y, class(, w, h)], ...], "key": photo in the raw folder, "source": its label/ copy}}
(w, h: the teacher's box size; the user's own marks have none)
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ..workspace import IMAGE_EXT, content_key, unique_name
from .base import Context, Stage

SPLIT = "split.json"
MIN_TEST = 20  # below this many test photos (per class) one photo moves the score by 5 points or more


class Ingest(Stage):
    name = "ingest"
    version = "6"  # v4: 3D source photos out of test; count split by photo content. v5: box sizes.
    # v6: the whole test folder of the user (half went unused), photos of the same name no longer overwritten
    runtime = "cpu"
    reads_input = True
    rereads = (SPLIT,)  # earlier sides of each photo
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
        prev, since = read_split(out), made_at(out)
        # A job ingested before split.json existed: its sides are in the output's folders, by photo name.
        old = {} if prev else {(f.parent.name, f.stem): f.parent.parent.name
                               for f in out.glob("*/*/*.png") if f.parent.parent.name in ("test", "train")}
        shutil.rmtree(out, ignore_errors=True)

        def big_enough(f) -> bool:
            with Image.open(f) as im:
                return min(im.size) >= ud.min_side_px

        def files_of(srcs, cls) -> dict:
            fs = sorted(f for src, cmap in srcs for d in cmap.get(cls, [cls]) for f in (src / d).glob("*")
                        if f.suffix.lower() in IMAGE_EXT)
            return {content_key(f): f for f in fs}

        held = held_out(ctx)
        counts, sides = {}, {}
        for cls in ctx.spec.labels:
            by_key = files_of(sources, cls)
            files = list(by_key.values())
            if old:
                prev = {k: old.get((cls, f.stem), "unused") for k, f in by_key.items() if f.stat().st_mtime <= since}
            if test_src:  # the user's own test folder: only a training side to keep
                s = split(list(by_key), set(), prev, 0, ud.train_per_class, lambda k: big_enough(by_key[k]))
                t_key = files_of(test_src, cls)
                test = test_photos(t_key, ud.test_per_class, big_enough)
            else:
                s = split(list(by_key), held, prev, ud.test_per_class, ud.train_per_class,
                          lambda k: big_enough(by_key[k]))
                test = [by_key[k] for k, v in s.items() if v == "test"]
            train = [by_key[k] for k, v in s.items() if v == "train"]
            sides |= s
            for side, sel in (("test", test), ("train", train)):
                d = out / side / cls
                d.mkdir(parents=True, exist_ok=True)
                used: set[str] = set()  # a.jpg and a.png, or the same name in the user's and the teacher's folders
                for f in sel:
                    with Image.open(f) as im:
                        # Phones store the orientation in EXIF instead of rotating the pixels.
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        im.resize((size, size), Image.BILINEAR).save(d / unique_name(f"{f.stem}.png", used, str(f)))
            counts[cls] = {"test": len(test), "train": len(train)}
            ctx.log(f"{cls:16s} {len(files):7d} photos, test {len(test)}, train {len(train)}")
            if 0 < len(test) < MIN_TEST:
                ctx.log(f"WARNING: only {len(test)} test photos of {cls}: each one moves its score by "
                        f"{100 / len(test):.0f} points")
        write_split(out, sides, prev)
        return {"counts": counts}

    def _count(self, ctx: Context) -> dict:
        from PIL import Image, ImageOps

        from ..workspace import copy_name

        ud, size = ctx.spec.user_data, ctx.hardware.input_size
        label = ctx.job.stage_dir("label")
        verdicts = json.loads((label / "verdicts.json").read_text())
        src_dir = ctx.job.resolve_input(ud.raw)
        # Ordered by photo content, not by name or folder: a photo lands on the same side in every job.
        ck = {k: content_key(src_dir / k) for k, v in verdicts.items() if v["assigned"] == "images"}
        held = held_out(ctx)
        out = ctx.job.stage_dir(self.name)
        prev = read_split(out)
        if not prev and (out / "test" / "points.json").exists():  # ingested before split.json existed
            since = made_at(out)
            old = {v["key"]: side for side in ("test", "train")
                   for v in json.loads((out / side / "points.json").read_text()).values()}
            prev = {c: old.get(k, "unused") for k, c in ck.items() if (src_dir / k).stat().st_mtime <= since}
        shutil.rmtree(out, ignore_errors=True)
        sides = split(list(ck.values()), held, prev, ud.test_per_class, ud.train_per_class)
        keys = sorted(ck, key=lambda k: order(ck[k]))
        n_test = sum(sides[ck[k]] == "test" for k in keys)
        counts = {}
        for side in ("test", "train"):
            sel = [k for k in keys if sides[ck[k]] == side]
            d = out / side / "images"
            d.mkdir(parents=True)
            points, used = {}, set()
            for key in sel:
                copy = verdicts[key].get("copy") or copy_name(key)  # older jobs: no "copy"
                src = label / "images" / copy
                name = unique_name(f"{Path(copy).stem}.png", used, key)
                with Image.open(src) as im:
                    ImageOps.exif_transpose(im).convert("RGB").resize((size, size), Image.BILINEAR).save(d / name)
                points[name] = {"points": [point(o) for o in verdicts[key]["objects"]],
                                "key": key, "source": str(src.relative_to(ctx.job.root))}
            (out / side / "points.json").write_text(json.dumps(points, indent=1))
            counts[side] = {"photos": len(sel), "objects": sum(len(p["points"]) for p in points.values())}
        ctx.log(f"{len(keys)} marked photos ({len(held & set(ck.values()))} used for 3D, kept out of test): "
                f"test {counts['test']}, train {counts['train']}")
        write_split(out, sides, prev)
        if 0 < n_test < MIN_TEST:
            ctx.log(f"WARNING: only {n_test} test photos: the score is noisy, add photos for a steadier one")
        return {"counts": counts}

    @staticmethod
    def _folder(ctx: Context, key: str, rel: str):
        src = ctx.job.resolve_input(rel)
        if not src.is_dir():
            raise FileNotFoundError(f"user_data.{key}: {src} is not a folder")
        return src


def order(key: str) -> str:
    """A photo's place in the split order: by content, so the same photo lands on the same side in every job."""
    return hashlib.sha256(f"split:{key}".encode()).hexdigest()


def split(keys: list[str], held: set[str], prev: dict[str, str], test_cap: int, train_cap: int,
          usable=lambda k: True) -> dict[str, str]:
    """Each photo's side: "test", "train" or "unused" (over the caps, or too small).

    First split: half to test when there are fewer photos than both caps, else test_cap to test and train_cap
    to train. Later splits (prev, the earlier sides) keep every known photo where it was and split only the
    new ones: half of them to test while test_cap allows, the rest all to training. Photos in `held` (3D
    models were made from them) never go to test. `usable` is checked lazily: opening every photo of a large
    dataset just to read its size would take minutes."""
    ordered = sorted(keys, key=lambda k: (k in held, order(k)))
    sides = {k: ("train" if prev[k] == "test" and k in held else prev[k]) for k in ordered if k in prev}
    new = [k for k in ordered if k not in sides]
    if not prev:
        need, picked = test_cap + train_cap, []
        for k in new:
            if len(picked) == need:
                break
            if usable(k):
                picked.append(k)
        n_test = min(test_cap, len(picked) // 2) if len(picked) < need else test_cap
        n_test = min(n_test, sum(k not in held for k in picked))
        sides |= {k: "test" for k in picked[:n_test]} | {k: "train" for k in picked[n_test:n_test + train_cap]}
    else:
        fresh = [k for k in new if usable(k)]
        room = max(0, test_cap - sum(v == "test" for v in sides.values()))
        n_test = min(room, sum(k not in held for k in fresh) // 2)
        sides |= {k: "test" for k in fresh[:n_test]} | {k: "train" for k in fresh[n_test:]}
    return {k: sides.get(k, "unused") for k in keys}


def test_photos(by_key: dict, cap: int, usable) -> list:
    """The user's own test folder: every usable photo, up to cap (split() would send half of a folder smaller
    than the caps to training, and a test folder has no training side: that half went unused)."""
    out = []
    for k in sorted(by_key, key=order):
        if len(out) == cap:
            break
        if usable(by_key[k]):
            out.append(by_key[k])
    return out


def made_at(out) -> float:
    """When the previous output was written: photos older than that were there, newer ones were added since.
    (Not _done.json: the pipeline removes it before running the stage.)"""
    times = [f.stat().st_mtime for f in out.rglob("*") if f.is_file()]
    return max(times) if times else float("inf")


def read_split(out) -> dict[str, str]:
    f = out / SPLIT
    return json.loads(f.read_text()) if f.exists() else {}


def write_split(out, sides: dict[str, str], prev: dict[str, str]) -> None:
    """Photos no longer in the job (deleted, relabeled as unsure) keep their side, should they come back."""
    out.mkdir(parents=True, exist_ok=True)
    (out / SPLIT).write_text(json.dumps(prev | sides, indent=1, sort_keys=True))


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
