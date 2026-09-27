"""Scene profile: what the user's photos show, turned into the synthetic scene settings.

A vision model (Token Factory) describes a sample of the photos one by one: which kind of place (building
front from the street, balcony, room, open ground), how far the camera is, where the objects stand, the wall
material and the railing style. The answers are counted into spec fields (scene.contexts with their shares,
wall_tags, railings) that the user confirms or corrects in the interview, before anything is rendered.

    sim2chip describe-scene data/real/maceta --classes "flower pots"
"""
from __future__ import annotations

import random
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .spec import CONTEXTS, RAILINGS, WALLS

SAMPLE = 16         # photos looked at
MIN_SHARE = 0.1     # contexts seen in fewer photos than this are left out

PROMPT = """This photo comes from a user who wants a small camera model for: {classes}.
We will rebuild places like this one in 3D to make training images, so describe where those objects are.

Answer only with JSON:
{{"context": "facade" (a building front seen from the street) | "balcony" (on a balcony or terrace, seen from
  close) | "indoor" (inside a room) | "ground" (outdoors on open ground: street, garden, yard, field),
 "distance_m": <typical camera distance to the objects, meters>,
 "supports": [where the objects stand, from "ground", "balcony_floor", "railing", "window_sill", "floor",
  "furniture"],
 "wall": "brick" | "plaster" | "stone" | "concrete" | "wood" | "none" (the main wall material in view),
 "railing": "bars" | "glass" | "solid" | "none"}}"""


def describe(photos: list[Path], classes: str, client, model: str, seed: int = 0) -> list[dict]:
    """One description per sampled photo (photos without a valid answer are skipped)."""
    from PIL import Image, ImageOps

    from .llm import ask_json, image_part

    sample = sorted(photos)
    random.Random(seed).shuffle(sample)
    sample = sample[:SAMPLE]
    tmp = Path(tempfile.mkdtemp())

    def one(item):
        i, f = item
        small = tmp / f"{i}.jpg"
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((768, 768))
            im.save(small, quality=88)
        msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(classes=classes)},
                                            image_part(small)]}]
        v = ask_json(client, msg, model, "scene_profile", lambda v: v.get("context") in CONTEXTS,
                     temperature=0.0, max_tokens=4096)
        return {**v, "photo": f.name} if v else None

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return [d for d in pool.map(one, enumerate(sample)) if d]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def aggregate(descriptions: list[dict]) -> dict:
    """Scene fields from the descriptions: contexts with their share of photos (rare ones dropped), camera
    distances per context, supports seen, the wall materials and railing styles seen."""
    n = len(descriptions)
    if not n:
        return {}
    kinds = Counter(d["context"] for d in descriptions)
    contexts = []
    for kind, k in kinds.most_common():
        if k / n < MIN_SHARE:
            continue
        mine = [d for d in descriptions if d["context"] == kind]
        dist = sorted(float(d["distance_m"]) for d in mine if _number(d.get("distance_m")))
        sup = Counter(s for d in mine for s in d.get("supports") or [] if s in CONTEXTS[kind])
        c = {"kind": kind, "share": round(k / n, 2)}
        if dist:
            c["distance_m"] = [round(max(0.5, dist[0] * 0.7), 1), round(max(1.0, dist[-1] * 1.3), 1)]
        if sup:
            c["supports"] = [s for s, _ in sup.most_common()]
        contexts.append(c)
    walls = [w for w, _ in Counter(d.get("wall") for d in descriptions).most_common() if w in WALLS][:3]
    rails = Counter({"glass": "bars"}.get(d.get("railing"), d.get("railing")) for d in descriptions)
    out = {"contexts": contexts}
    if walls:
        out["wall_tags"] = walls
    railings = [r for r, _ in rails.most_common() if r in RAILINGS]
    if railings:
        out["railings"] = railings
    return out


def summary(profile: dict, n_photos: int) -> str:
    """One paragraph a user can check."""
    names = {"facade": "building fronts from the street", "balcony": "balconies seen from close",
             "indoor": "rooms", "ground": "open ground"}
    parts = [f"{round(c['share'] * 100)}% {names[c['kind']]}" +
             (f" ({c['distance_m'][0]}-{c['distance_m'][1]} m)" if c.get("distance_m") else "") +
             (f", on {', '.join(s.replace('_', ' ') for s in c['supports'])}" if c.get("supports") else "")
             for c in profile.get("contexts", [])]
    text = f"In {n_photos} of your photos I see: " + "; ".join(parts) + "."
    if profile.get("wall_tags"):
        text += f" Walls: {', '.join(profile['wall_tags'])}."
    if profile.get("railings"):
        text += f" Railings: {', '.join(profile['railings'])}."
    return text


def profile_folder(folder: Path, classes: str, client, model: str) -> tuple[dict, str]:
    from .workspace import IMAGE_EXT
    photos = [f for f in Path(folder).rglob("*") if f.suffix.lower() in IMAGE_EXT]
    if not photos:
        raise FileNotFoundError(f"no photos in {folder}")
    d = describe(photos, classes, client, model)
    prof = aggregate(d)
    return prof, summary(prof, len(d))


def _number(v) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False
