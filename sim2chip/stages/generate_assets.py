"""3D models generated from photos, for classes with `assets.photos` (see AssetQuery).

Every photo is scored by an open-vocabulary detector for the class; the best ones become 3D models (one per
photo, cycling seeds when there are fewer photos than `generate`). Generated models are cached by photo
content, query and seed, so rerunning a job or reusing the photos in another job costs nothing.

Cache:  generated/<uid>.glb, generated/<photo hash>.json  (detection per query)
Output: generate_assets/manifest.json  entries shaped like fetch_assets' manifest, plus "photo"
        generate_assets/<class>/<uid>.png  the cut-out image the model was built from (to review by eye)
"""
from __future__ import annotations

import hashlib
import json
import shutil

from ..workspace import IMAGE_EXT, content_key
from .base import Context, Stage

GEN_VERSION = 1  # bump when the generation changes: cached models get new uids
# These are photos OF the object: a box under 2% of the image is something else (a detector false positive).
MIN_BOX = 0.02


def box_share(box, path) -> float:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        w, h = ImageOps.exif_transpose(im).size
    return (box[2] - box[0]) * (box[3] - box[1]) / (w * h)


def query_of(c) -> str:
    """Short English noun phrase for the detector (the spec requires one: class ids may be in any language)."""
    a = c.assets
    return (a.queries or a.categories)[0]


class GenerateAssets(Stage):
    name = "generate_assets"
    runtime = "trellis"
    gpu = "any"
    max_hours = 2
    reads_input = True
    uses_cache = True
    uses = ("classes", "user_data")
    after = ("label",)
    skippable = "with photos"
    # New or corrected photos (--from label) must not remake the 3D models, and with them the whole synthetic
    # branch: the models are cached by photo content and later versions of a job compare on the same data.
    follows_upstream = False
    results = ("*.json", "*/*.png")

    def enabled(self, spec) -> bool:
        return spec.synthetic and any(c.assets and c.assets.photos for c in spec.classes)

    def photos_of(self, ctx: Context, c) -> list:
        """The class's photos: its own folder, or the user's photos of that class."""
        ud = ctx.spec.user_data
        if c.assets.photos != "user_data":
            dirs = [ctx.job.resolve_input(c.assets.photos)]
        else:
            dirs = [ctx.job.resolve_input(ud.labeled) / d for d in ud.class_map.get(c.id, [c.id])] if ud.labeled \
                else []
            if ud.raw:
                dirs.append(ctx.job.stage_dir("label") / c.id)
        return sorted(f for d in dirs if d.is_dir() for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXT)

    def run(self, ctx: Context) -> dict:
        from PIL import Image, ImageOps

        from ..img2mesh import MIN_SCORE, Detection, ImageTo3D

        cache = ctx.cache.root / "generated"
        cache.mkdir(parents=True, exist_ok=True)
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        model = None
        manifest = []
        for c in [c for c in ctx.spec.classes if c.assets and c.assets.photos]:
            query = query_of(c)
            files = self.photos_of(ctx, c)
            if not files:
                raise FileNotFoundError(f"class {c.id}: no photos found for assets.photos={c.assets.photos}")
            model = model or ImageTo3D()
            scored = []
            for f in files:
                key = content_key(f)
                det_file = cache / f"{key}.json"
                dets = json.loads(det_file.read_text()) if det_file.exists() else {}
                if query not in dets:
                    with Image.open(f) as im:
                        d = model.detect(ImageOps.exif_transpose(im).convert("RGB"), query)
                    dets[query] = {"score": d.score, "box": d.box}
                    det_file.write_text(json.dumps(dets))
                scored.append((dets[query]["score"], f, key, Detection(**dets[query])))
            scored.sort(key=lambda s: -s[0])
            good = [s for s in scored if s[0] >= MIN_SCORE and s[3].box and box_share(s[3].box, s[1]) >= MIN_BOX]
            ctx.log(f"{c.id:16s} {len(files)} photos, {len(good)} show '{query}' (best score {scored[0][0]:.2f})")
            if not good:
                ctx.log(f"WARNING: no photo of class {c.id} shows '{query}', no model generated")
                continue
            (out / c.id).mkdir()
            for k in range(c.assets.n_generate):
                score, f, key, det = good[k % len(good)]
                seed = 1 + k // len(good)
                # The crop depends on the detection, so the query is part of the key (not only photo and seed).
                uid = f"gen-{hashlib.sha256(f'{key}:{query}:{seed}:{GEN_VERSION}'.encode()).hexdigest()[:16]}"
                glb = cache / f"{uid}.glb"
                seen = out / c.id / f"{uid}.png"
                if not glb.exists():
                    with Image.open(f) as im:
                        prep = model.generate(ImageOps.exif_transpose(im).convert("RGB"), det, seed, str(glb))
                    prep.save(cache / f"{uid}.png")
                    ctx.log(f"{c.id:16s} {f.name} seed {seed} -> {uid}")
                if (cache / f"{uid}.png").exists():
                    shutil.copy2(cache / f"{uid}.png", seen)
                manifest.append({"uid": uid, "class": c.id, "category": query, "name": f"{c.id} from {f.name}",
                                 "license": "user photo", "author": "user", "source": "TRELLIS v1 (image to 3D)",
                                 "glb": str(glb.relative_to(ctx.cache.root)), "photo": f.name, "photo_key": key,
                                 "detection": round(score, 3), "generated": True})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        per_class = {}
        for m in manifest:
            per_class[m["class"]] = per_class.get(m["class"], 0) + 1
        ctx.log(f"{len(manifest)} generated models {per_class}")
        return {"models": per_class}
