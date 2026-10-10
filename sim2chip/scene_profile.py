"""Scene profile: what the user's photos show, turned into the synthetic scene settings.

A vision model (Token Factory) describes a sample of the photos one by one: which kind of place (building
front from the street, balcony, room, open ground), how far the camera is, where the objects stand, the wall
material and the railing style. The answers are counted into spec fields (scene.contexts with their shares,
wall_tags, railings) that the user confirms or corrects in the interview, before anything is rendered.

    sim2chip describe-scene data/real/maceta --classes "flower pots"
"""
from __future__ import annotations

import math
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .spec import CONTEXTS, RAILINGS, WALLS

SAMPLE = 16         # photos looked at, drawn at random
DEADLINE_S = 90     # the user waits: describe what answered by then and go on with it
MIN_SHARE = 0.1     # contexts seen in fewer photos than this are left out

# What each context of spec.CONTEXTS looks like in a photo (a new context needs its words here).
CONTEXT_WORDS = {"facade": "a building front seen from the street", "balcony": "on a balcony or terrace, seen from close",
                 "indoor": "inside a room", "ground": "outdoors on open ground: street, garden, yard, field"}


def _quoted(items) -> str:
    return ", ".join(f'"{i}"' for i in items)


PROMPT = """This photo comes from a user who wants a small camera model for: {classes}.
We will rebuild places like this one in 3D to make training images, so describe the photo in detail.

Answer only with JSON:
{{"description": "<two or three sentences: what the photo shows, where those objects are, the light, where the
  camera stands>",
 "count": <how many of those objects are visible, 0 if none>,
 "size_m": <their typical size in meters, the longest side seen from above; judge from things of known size:
  doors are about 2 m high, railings about 1 m, chair seats about 0.45 m, bricks 0.25 m>,
 "height_m": <their typical height in meters>,
 "camera_height_m": <height of the camera above the floor it stands on, meters>,
 "camera_pitch_deg": <how far down the camera looks, degrees: 0 straight ahead, 90 straight down at the surface>,
 "wall_behind": <true when a wall, fence or building stands right behind the objects, within about 2 m>,
 "others": [up to 6 other movable things standing near them on the same surfaces, English nouns: "chair",
  "watering can", "bicycle"; not parts of the building (shutters, awnings, lamps, windows) and not vehicles],
 "objects": [every kind of object that stands out in the photo, those above included, up to 8:
  {{"name": "<English noun, singular>", "count": <how many>}}],
 "background": "<what is under and behind the objects, a few English words: dark wooden table, white wall>",
 "floor": "<what those objects stand on, one or two English words: tiles, concrete, grass, wood, asphalt>",
 "light": "sun" | "overcast" | "evening" | "night" | "indoor",
 "context": """ + " | ".join(f'"{k}" ({CONTEXT_WORDS[k]})' for k in CONTEXTS) + """,
 "distance_m": <typical camera distance to the objects, meters>,
 "supports": [where the objects stand, from """ + _quoted(dict.fromkeys(s for v in CONTEXTS.values() for s in v)) + """],
 "wall": """ + " | ".join(f'"{w}"' for w in (*WALLS, "none")) + """ (the main wall material in view),
 "railing": """ + " | ".join(f'"{r}"' for r in (*RAILINGS, "none")) + """}}"""

LIGHT_WEATHER = {"sun": "clear", "overcast": "cloudy", "evening": "evening", "night": "night"}
MIN_OTHER = 0.15  # things seen near the objects in fewer photos than this are not proposed as props
# The classes of a dataset's own profile, made before anyone says what the camera should look for.
GENERAL = "nothing chosen yet; the objects are whatever stands out in the photo"


def sample(photos: list, seed: int = 0) -> list:
    """The photos looked at: SAMPLE of them, drawn at random but always the same for the same photos."""
    out = sorted(photos)
    random.Random(seed).shuffle(out)
    return out[:SAMPLE]


def describe(photos: list[Path], classes: str, client, model: str, seed: int = 0, progress=None) -> list[dict]:
    """One description per sampled photo (photos without a valid answer are skipped). progress(done, total)
    is called as photos finish."""
    from PIL import Image, ImageOps

    from .llm import PHOTO_CALL, REASONING_TOKENS, ask_json, image_part, map_within

    picked = sample(photos, seed)
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
                     tries=1, temperature=0.0, max_tokens=REASONING_TOKENS, **PHOTO_CALL)
        return {**v, "photo": f.name} if v else None

    try:
        out = map_within(one, list(enumerate(picked)), DEADLINE_S, progress=progress)
        return [d for d in out if d]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # late threads then fail on their own, harmlessly


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
    camera = _camera(descriptions)
    if camera.get("top_down"):
        # Photos that look down at a surface (a table, a lawn, a conveyor) are that surface seen from above,
        # whatever room or street it is in: open ground with the surface's own texture, the camera looking down
        # from the measured distance (water_container_counter, 2026-10-11: rendered as side views of a room).
        dist = sorted(float(d["distance_m"]) for d in descriptions if _number(d.get("distance_m")))
        contexts = [{"kind": "ground", "share": 1.0, "supports": ["ground"],
                     **({"distance_m": [round(max(0.2, dist[0] * 0.8), 2), round(dist[-1] * 1.25, 2)]} if dist else {})}]
    else:
        behind = [d for d in descriptions if d["context"] == "ground"]
        for c in contexts:
            if c["kind"] == "ground" and behind:
                share = sum(d.get("wall_behind") is True for d in behind) / len(behind)
                if share >= MIN_SHARE:
                    c["backdrop"] = round(share, 2)
    walls = [w for w, _ in Counter(d.get("wall") for d in descriptions).most_common() if w in WALLS][:3]
    rails = Counter({"glass": "bars"}.get(d.get("railing"), d.get("railing")) for d in descriptions)
    out = {"contexts": contexts}
    if camera.get("pitch_deg"):
        out["camera"] = {k: v for k, v in camera.items() if k != "top_down"}
    out |= _objects_and_light(descriptions)
    if walls:
        out["wall_tags"] = walls
    railings = [r for r, _ in rails.most_common() if r in RAILINGS]
    if railings:
        out["railings"] = railings
    return out


TOP_DOWN_DEG = 45  # photos looking down more steeply than this, most of them: a surface seen from above


def _camera(descriptions: list[dict]) -> dict:
    """The camera of photos that look down at what they show: {"pitch_deg": [min, max], "height_m": [...] (above
    the surface, from the distance), "top_down": True}; nothing when most photos look ahead (the camera then
    frames each object by its size)."""
    pairs = [(float(d["camera_pitch_deg"]), float(d["distance_m"]) if _number(d.get("distance_m")) else None)
             for d in descriptions if _number(d.get("camera_pitch_deg"))]
    steep = [(p, dist) for p, dist in pairs if p >= TOP_DOWN_DEG]
    if not pairs or len(steep) / len(descriptions) < 0.5:
        return {}
    pitch = sorted(min(89.0, p) for p, _ in steep)
    out = {"pitch_deg": [round(max(TOP_DOWN_DEG, pitch[0] - 5)), round(min(89, pitch[-1] + 5))], "top_down": True}
    heights = sorted(dist * math.sin(math.radians(p)) for p, dist in steep if dist)
    if heights:
        out["height_m"] = [round(max(0.1, heights[0] * 0.8), 2), round(heights[-1] * 1.25, 2)]
    return out


def _objects_and_light(descriptions: list[dict]) -> dict:
    """Scene fields for the floor and the weather, and under "suggest" what belongs elsewhere in the spec:
    the objects' size (the class), the camera height, props, objects per photo and the photo descriptions."""
    n, out, sug = len(descriptions), {}, {}

    def spread(key, lo=0.8, hi=1.25):
        v = sorted(float(d[key]) for d in descriptions if _number(d.get(key)))
        if not v:
            return None
        a, b = v[int(0.1 * (len(v) - 1))], v[int(round(0.9 * (len(v) - 1)))]
        return [round(a * lo, 2), round(b * hi, 2)]

    floors = Counter(str(d.get("floor", "")).strip().lower() for d in descriptions if d.get("floor"))
    if ground := [f for f, k in floors.most_common(2) if k / n >= MIN_OTHER]:
        out["ground_tags"] = ground
    light = Counter(d.get("light") for d in descriptions)
    if weather := [LIGHT_WEATHER[w] for w, k in light.most_common() if w in LIGHT_WEATHER and k / n >= MIN_SHARE]:
        out["weather"] = weather
    if size := spread("size_m"):
        sug["size_m"] = {"measure": "length", "range": size}
    if height := spread("height_m"):
        sug["height_m"] = height
    if cam := spread("camera_height_m", 0.9, 1.1):
        sug["camera_height_m"] = cam
    counts = sorted(int(float(d["count"])) for d in descriptions if _number(d.get("count")))
    if counts:
        sug["objects_per_photo"] = {"median": counts[len(counts) // 2], "max": counts[-1]}
    others = Counter(o.strip().lower() for d in descriptions for o in set(d.get("others") or []) if isinstance(o, str))
    sug["props"] = [o for o, k in others.most_common(6) if k / n >= MIN_OTHER]
    seen = Counter(o["name"].strip().lower() for d in descriptions for o in _objects(d))
    sug["seen"] = [{"name": name, "share": round(k / n, 2),
                    "count": sorted(o["count"] for d in descriptions for o in _objects(d)
                                    if o["name"].strip().lower() == name)[(k - 1) // 2]}
                   for name, k in seen.most_common(10) if k / n >= MIN_OTHER]
    backs = Counter(str(d["background"]).strip().lower() for d in descriptions if d.get("background"))
    sug["backgrounds"] = [b for b, _ in backs.most_common(4)]
    if light.get("indoor", 0) / n >= 0.5:
        sug["indoor"] = True
    sug["photos"] = [f"{d.get('photo', '?')}: {d['description']}" for d in descriptions if d.get("description")]
    return {**out, "suggest": sug}


def summary(profile: dict, n_photos: int) -> str:
    """One paragraph a user can check."""
    names = {"facade": "building fronts from the street", "balcony": "balconies seen from close",
             "indoor": "rooms", "ground": "open ground"}
    parts = [f"{round(c['share'] * 100)}% {names[c['kind']]}" +
             (f" ({c['distance_m'][0]}-{c['distance_m'][1]} m)" if c.get("distance_m") else "") +
             (f", on {', '.join(s.replace('_', ' ') for s in c['supports'])}" if c.get("supports") else "") +
             (f", {round(c['backdrop'] * 100)}% with a wall right behind" if c.get("backdrop") else "")
             for c in profile.get("contexts", [])]
    text = f"In {n_photos} of your photos I see: " + "; ".join(parts) + "."
    if cam := profile.get("camera"):
        text += f" The camera looks down at {cam['pitch_deg'][0]}-{cam['pitch_deg'][1]} degrees."
    if profile.get("wall_tags"):
        text += f" Walls: {', '.join(profile['wall_tags'])}."
    if profile.get("railings"):
        text += f" Railings: {', '.join(profile['railings'])}."
    sug = profile.get("suggest", {})
    if sug.get("size_m"):
        lo, hi = sug["size_m"]["range"]
        text += f" The objects measure about {lo}-{hi} m."
    if sug.get("objects_per_photo"):
        text += f" Usually {sug['objects_per_photo']['median']} per photo, up to {sug['objects_per_photo']['max']}."
    if sug.get("props"):
        text += f" Around them: {', '.join(sug['props'])}."
    return text


def dataset_summary(profile: dict, n_photos: int) -> str:
    """One paragraph on a dataset before any goal: what objects show up and where."""
    sug = profile.get("suggest", {})
    parts = []
    if sug.get("seen"):
        parts.append("Objects: " + ", ".join(f"{o['name']} (in {round(o['share'] * 100)}% of photos, about "
                                             f"{o['count']} each)" for o in sug["seen"]) + ".")
    if sug.get("backgrounds"):
        parts.append("Backgrounds: " + "; ".join(sug["backgrounds"]) + ".")
    names = {"facade": "building fronts", "balcony": "balconies", "indoor": "indoors", "ground": "open ground"}
    if profile.get("contexts"):
        parts.append("Places: " + ", ".join(f"{names[c['kind']]} {round(c['share'] * 100)}%"
                                            + (f" at {c['distance_m'][0]}-{c['distance_m'][1]} m" if c.get("distance_m")
                                               else "") for c in profile["contexts"]) + ".")
    if profile.get("weather"):
        parts.append(f"Light: {', '.join(profile['weather'])}.")
    if sug.get("camera_height_m"):
        parts.append(f"Camera {sug['camera_height_m'][0]}-{sug['camera_height_m'][1]} m above the floor.")
    return f"From {n_photos} photo{'s' if n_photos != 1 else ''}. " + " ".join(parts) if parts else f"Nothing could be read from {n_photos} photos."


def profile_folder(folder: Path, classes: str, client, model: str, progress=None, notes=None) -> tuple[dict, str]:
    """notes(photos, classes, run): the descriptions kept from earlier (library.Library.describe), calling
    run(photos) for those it lacks."""
    from .workspace import IMAGE_EXT
    photos = [f for f in Path(folder).rglob("*") if f.suffix.lower() in IMAGE_EXT]
    if not photos:
        raise FileNotFoundError(f"no photos in {folder}")
    run = lambda some: describe(some, classes, client, model, progress=progress)  # noqa: E731
    d = None
    if notes:
        # Photos described before (a library dataset's profile) answer at once, whichever they are: the sample
        # follows file names, which differ where a dataset is put as a folder, and one photo more to describe
        # took the vision model 35 to 45 s (library bench, 2026-10-10).
        kept = notes(sorted(photos), classes, None)
        if len(kept) >= max(1, min(SAMPLE, len(photos)) * 3 // 4):
            d = kept[:SAMPLE]
    if d is None:
        d = notes(sample(photos), classes, run) if notes else run(photos)
    prof = aggregate(d)
    return prof, summary(prof, len(d))


def _objects(d: dict) -> list[dict]:
    """A description's objects, the malformed ones left out."""
    return [o for o in d.get("objects") or [] if isinstance(o, dict) and isinstance(o.get("name"), str)
            and o["name"].strip() and isinstance(o.get("count"), (int, float)) and o["count"] >= 0]


def _number(v) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False
