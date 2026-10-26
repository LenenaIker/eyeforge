"""Scene profile: what the user's photos show (the photo analysis).

A vision model (Token Factory) describes a sample of the photos one by one: which kind of place (building
front from the street, balcony, room, open ground), the objects and their size, the light and where the camera
stands. The answers are counted into what the interview tells the chat model (camera height and pitch, sizes)
and the dataset's profile; the walk scene proposals read the descriptions themselves (scene_pick.py).

    sim2chip describe-scene data/real/maceta --classes "flower pots"
"""
from __future__ import annotations

import math
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

SAMPLE = 16         # photos looked at, drawn at random
DEADLINE_S = 90     # the user waits: describe what answered by then and go on with it
MIN_SHARE = 0.1     # places seen in fewer photos than this are left out

# The kinds of place a photo is taken in ("context" of a description) and how each looks.
PLACES = {"facade": "a building front seen from the street", "balcony": "on a balcony or terrace, seen from close",
                 "indoor": "inside a room", "ground": "outdoors on open ground: street, garden, yard, field"}


PROMPT = """This photo comes from a user who wants a small camera model for: {classes}.
We will rebuild places like this one in 3D to make training images, so describe the photo in detail.

Answer only with JSON:
{{"description": "<two or three sentences: what the photo shows, where those objects are, the light, where the
  camera stands>",
 "count": <how many of those objects are visible, 0 if none; null when they are surfaces or areas (grass, road,
  floor) rather than separate things>,
 "size_m": <their typical size in meters, the longest side seen from above; judge from things of known size:
  doors are about 2 m high, railings about 1 m, chair seats about 0.45 m, bricks 0.25 m; null for surfaces or areas>,
 "height_m": <their typical height in meters; null for surfaces or areas>,
 "camera_height_m": <height of the camera above the floor it stands on, meters>,
 "camera_pitch_deg": <how far down the camera looks, degrees: 0 straight ahead, 90 straight down at the surface>,
 "others": [up to 6 other movable things standing near them on the same surfaces, English nouns: "chair",
  "watering can", "bicycle"; not parts of the building (shutters, awnings, lamps, windows) and not vehicles],
 "objects": [every kind of object that stands out in the photo, those above included, up to 8:
  {{"name": "<English noun, singular>", "count": <how many>, "size_m": <typical longest side, meters>}}],
 "background": "<what is under and behind the objects, a few English words: dark wooden table, white wall>",
 "light": "sun" | "overcast" | "evening" | "night" | "indoor",
 "context": """ + " | ".join(f'"{k}" ({w})' for k, w in PLACES.items()) + """,
 "distance_m": <typical camera distance to the objects, meters>}}"""

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
        v = ask_json(client, msg, model, "scene_profile", lambda v: v.get("context") in PLACES,
                     tries=1, temperature=0.0, max_tokens=REASONING_TOKENS, **PHOTO_CALL)
        return {**v, "photo": f.name} if v else None

    try:
        out = map_within(one, list(enumerate(picked)), DEADLINE_S, progress=progress)
        return [d for d in out if d]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # late threads then fail on their own, harmlessly


def aggregate(descriptions: list[dict], general: bool = False) -> dict:
    """What the descriptions say together: the places with their share of photos (rare ones dropped), the camera
    of photos looking down, the objects and the light. general: descriptions made before anyone chose the classes
    (GENERAL), whose count and size are of whatever stood out."""
    n = len(descriptions)
    if not n:
        return {}
    kinds = Counter(d["context"] for d in descriptions)
    out = {"places": [{"kind": kind, "share": round(k / n, 2)} for kind, k in kinds.most_common() if k / n >= MIN_SHARE]}
    camera = _camera(descriptions)
    if camera.get("pitch_deg"):
        out["camera"] = {k: v for k, v in camera.items() if k not in ("top_down", "steady")}
    return out | _objects_and_light(descriptions, general)


TOP_DOWN_DEG = 45  # photos looking down more steeply than this, most of them: a surface seen from above
# A camera that hardly moves between photos (mounted) gets a narrow range around the middle photo instead of the
# whole spread plus margins, so no angle or distance it never has is made up (roadmap B.2). Steady: most photos
# close to the median, so one odd estimate does not count. The vision model answers in 5 degree steps; the lawn,
# taken by hand, had 56 % of its photos within 5 degrees and 44 % within 15 % of the distance (2026-10-11).
STEADY_SHARE = 0.8
STEADY_DEG = 5        # pitch this close to the median: the range is the median +-5 degrees
STEADY_RATIO = 0.15   # distance (height) this close to the median: the range is the median +-10 %


def _median(v: list[float]) -> float:
    v = sorted(v)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2


def _steady(v: list[float], near) -> bool:
    """Are most values near (value, median) the median?"""
    mid = _median(v)
    return sum(near(x, mid) for x in v) >= STEADY_SHARE * len(v)


def _spread(v: list[float], floor: float, steady: bool) -> list[float]:
    """A range of distances or heights in meters: the median +-10 % for a steady camera, else the whole spread
    with margins (0.8 and 1.25 times), never under floor."""
    if steady:
        mid = _median(v)
        return [round(max(floor, mid * 0.9), 2), round(mid * 1.1, 2)]
    return [round(max(floor, min(v) * 0.8), 2), round(max(v) * 1.25, 2)]


def _camera(descriptions: list[dict]) -> dict:
    """The camera of photos that look down at what they show: {"pitch_deg": [min, max], "height_m": [...] (above
    the surface, from the distance), "top_down": True, "steady": <a mounted camera: narrow ranges>}; nothing when
    most photos look ahead (the camera then frames each object by its size)."""
    pairs = [(float(d["camera_pitch_deg"]), float(d["distance_m"]) if _number(d.get("distance_m")) else None)
             for d in descriptions if _number(d.get("camera_pitch_deg"))]
    steep = [(p, dist) for p, dist in pairs if p >= TOP_DOWN_DEG]
    if not pairs or len(steep) / len(descriptions) < 0.5:
        return {}
    pitch = sorted(min(89.0, p) for p, _ in steep)
    heights = [dist * math.sin(math.radians(p)) for p, dist in steep if dist]
    # Steady only when both hold: photos taken by hand from about the same height still tilt the phone.
    steady = _steady(pitch, lambda x, mid: abs(x - mid) <= STEADY_DEG) and \
        (not heights or _steady(heights, lambda x, mid: abs(x - mid) <= STEADY_RATIO * mid))
    if steady:
        mid = _median(pitch)
        out = {"pitch_deg": [round(max(TOP_DOWN_DEG, mid - 5)), round(min(89, mid + 5))]}
    else:
        out = {"pitch_deg": [round(max(TOP_DOWN_DEG, pitch[0] - 5)), round(min(89, pitch[-1] + 5))]}
    if heights:
        out["height_m"] = _spread(heights, 0.1, steady)
    return {**out, "top_down": True, "steady": steady}


def _objects_and_light(descriptions: list[dict], general: bool = False) -> dict:
    """The light seen ("weather"), and under "suggest" what belongs elsewhere in the spec:
    the objects' size (the class), the camera height, props, objects per photo and the photo descriptions.
    general: no size or count for the class, only per thing seen (a lawn's most numerous thing is its leaves:
    35 per photo of 0.08-0.75 m were proposed for the class grass, 2026-10-11)."""
    n, out, sug = len(descriptions), {}, {}

    def spread(key, lo=0.8, hi=1.25):
        v = sorted(float(d[key]) for d in descriptions if _number(d.get(key)))
        if not v:
            return None
        a, b = v[int(0.1 * (len(v) - 1))], v[int(round(0.9 * (len(v) - 1)))]
        return [round(a * lo, 2), round(b * hi, 2)]

    light = Counter(d.get("light") for d in descriptions)
    if weather := [LIGHT_WEATHER[w] for w, k in light.most_common() if w in LIGHT_WEATHER and k / n >= MIN_SHARE]:
        out["weather"] = weather
    if not general and (size := spread("size_m")):
        sug["size_m"] = {"measure": "length", "range": size}
    if not general and (height := spread("height_m")):
        sug["height_m"] = height
    if cam := spread("camera_height_m", 0.9, 1.1):
        sug["camera_height_m"] = cam
    counts = sorted(int(float(d["count"])) for d in descriptions if _number(d.get("count")))
    if counts and not general:
        sug["objects_per_photo"] = {"median": counts[len(counts) // 2], "max": counts[-1]}
    others = Counter(o.strip().lower() for d in descriptions for o in set(d.get("others") or []) if isinstance(o, str))
    sug["props"] = [o for o, k in others.most_common(6) if k / n >= MIN_OTHER]
    seen = Counter(o["name"].strip().lower() for d in descriptions for o in _objects(d))
    sug["seen"] = []
    for name, k in seen.most_common(10):
        if k / n < MIN_OTHER:
            continue
        mine = [o for d in descriptions for o in _objects(d) if o["name"].strip().lower() == name]
        thing = {"name": name, "share": round(k / n, 2), "count": sorted(o["count"] for o in mine)[(k - 1) // 2]}
        if sizes := sorted(float(o["size_m"]) for o in mine if _number(o.get("size_m"))):
            thing["size_m"] = sizes[(len(sizes) - 1) // 2]
        sug["seen"].append(thing)
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
    parts = [f"{round(c['share'] * 100)}% {names[c['kind']]}" for c in profile.get("places", [])]
    text = f"In {n_photos} of your photos I see: " + "; ".join(parts) + "."
    if cam := profile.get("camera"):
        text += f" The camera looks down at {cam['pitch_deg'][0]}-{cam['pitch_deg'][1]} degrees."
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
    if profile.get("places"):
        parts.append("Places: " + ", ".join(f"{names[c['kind']]} {round(c['share'] * 100)}%"
                                            for c in profile["places"]) + ".")
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
    prof = aggregate(d, general=classes == GENERAL)
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
