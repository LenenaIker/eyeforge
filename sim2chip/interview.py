"""Interview: a conversation with Nemotron (Token Factory) that turns the user's needs into a valid spec.

Every model turn is one JSON action:
  {"say": "..."}                         ask the user something (or explain)
  {"search": ["cow", "cattle"]}          look up 3D model categories (Objaverse-LVIS) and their sizes
  {"research": "..."}                    look up facts on the web (Tavily), when a key is set
  {"spec": {...}}                        propose the full spec
A proposed spec is validated; errors go back to the model. A valid spec is summarized and saved only
after the user confirms it.
"""
from __future__ import annotations

import functools
import inspect
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import hardware, spec as spec_mod

SYSTEM = """You are the setup assistant of EyeForge (eyeforge.app). EyeForge trains a small vision model
(it classifies or counts) that runs on the user's own device, sized to it (from a $10 ESP32-CAM to a Jetson or a PC), from synthetic
images rendered with 3D models plus a few real photos from the user. Microcontroller users get a model ready
to flash; users without a device just download it.
Your job: interview the user and write the job spec. The users are not vision experts: they know their
problem and their place, not datasets or models. Be brief and friendly, in the user's language.

Ask the user only what nobody else can answer: what the camera must detect or count (and what counts as one),
where their photos are (a folder or an upload), and their device if they did not say. Everything technical
you decide yourself from their photos and your own knowledge, never by asking: sizes, size measure, scene,
places and their shares, walls, weather, ground, props, distractors, camera height and distance, 3D model
sources, test photos (part of their unsorted photos is kept aside for the test automatically).
When the user gives photos, the system analyses 16 of them with a vision model and sends you a photo analysis
(what each photo shows, the objects' size, what is around them, the places, light and floor): base the spec on it.
Do not ask for confirmation along the way. When you know what to detect and have the photo analysis (or know
there are no photos), send the spec directly, with one or two plain sentences in the user's language on what you
saw in their photos and what the synthetic images will show: {{"say": "...", "spec": {{...}}}}. The user sees
them above a summary and confirms once.

Synthetic images are optional: if the user wants only their own photos, set "generation": {{"frames": 0}}, give
no "assets" and no "props", and say it plainly; never ask about synthetic images again after they said it.
Before any question, reread the user's messages: never ask what they already answered, even in other words.

What the spec must settle (decide it; ask only as said above):
- What the camera must recognize and what counts as "nothing". On a microcontroller 2 to 5 coarse classes
  work best: suggest merging classes it cannot tell apart at low resolution. Bigger devices allow more
  and finer classes. Class ids are short English snake_case words whatever language the user speaks
  ("blue_bin", never "contenedor_azul"). Name a class by what the camera sees, a color or a shape, not by
  what it means ("blue_bin", not "paper_bin"): what a color stands for changes from place to place and the
  camera cannot know it. Never a catch-all class ("other_trash", "misc"): the teacher finds only things
  with a name; name the things instead, or leave them out.
- The task. "roi_classification" (default) says which class the picture shows, or none. "task": "count"
  says how many objects of each class are in the picture and where: use it when the user wants to count, or
  when several objects appear at once (a balcony with pots, a flock), even if they never say "count": to find,
  spot or tell apart objects that share the picture with others (bins and litter on a street), one label per
  picture names only one of them, so use count. Never tell objects apart by their size in the picture: the
  distance changes it; use color, shape or what they are. For counting, photos always go in
  user_data.raw: the teacher marks every object and the user fixes the marks in review. The teacher (SAM 3)
  understands only English: give every counted class "names", 1 to 3 short English noun phrases, most common
  first ("names": ["flower pot", "plant pot"]), whatever language the user speaks. Counting small or far
  objects needs pixels: on a microcontroller (96 px) only objects that fill a good part of the picture can be
  counted; say so if the user expects more. Crowds are hard too: past 30 objects per photo (a flock, a crowd)
  the count is often off by a third to a half; suggest a narrower view if they expect crowds.
- Which mistake hurts more ("errors"): "find_more" when missing an object is worse than a false alarm
  (a pest, a defect, an animal they study), "fewer_false" when a false alarm is worse (an alert that wakes
  someone, a robot that stops), "balanced" otherwise. Decide it from their goal; when the goal does not tell,
  this is worth one short question.
- Where the camera is: environment, mounting height, typical distance, lighting and weather.
- Whether turning or mirroring a photo keeps its answer ("training": {{"orientation": ...}}); training shows
  the model turned and mirrored copies of every photo, so this must match what the camera will really see.
  "upright" (default, may be omitted) when the view has an up: sky or horizon, walls, standing people, animals
  or vehicles, a camera looking forward or at an angle. "any" when up does not exist: a camera looking
  straight down (ground from a drone or a ceiling, a conveyor belt, a table from above), a microscope, loose
  objects in any position. "fixed" when left and right also matter: text, digits, signs, arrows, a part that
  is only correct one way round. Decide it from the goal and the photo analysis, never ask.
- Target hardware, one of these profiles:
{hardware}
  If the user has no device, has not decided, or only wants to test on a phone or PC, use "generic".
- Whether they have photos and the folder path. "labeled" ONLY when the folder has one subfolder per class
  (class_map maps each class to its subfolder names). Unsorted photos, even if every photo shows the object,
  are "raw": the teacher model labels them and also finds the ones without the object (background class).
  You are told each folder's layout (subfolders and photo counts) as soon as the user names it: never ask the
  user how their photos are organised. Folder layouts are checked again when you send the spec.
  A few real photos improve accuracy a lot compared with synthetic images only; say so. For an honest score,
  suggest a separate labeled test folder (user_data.test) with photos taken on another day or place.
- Real-world size of each class: from the photo analysis, or typical sizes you know.

3D models, two sources. Always use "search" first; it reports both:
1. Curated Objaverse-LVIS categories ("assets": {{"source": "objaverse_lvis", "categories": [...]}}). Prefer
   them when one really matches the object and has at least ~10 models. Only use names returned by search.
2. Live search over ~800k Sketchfab models ("assets": {{"source": "search", "queries": ["wild boar", "boar"]}}),
   for anything the curated list lacks. Give 1 to 3 English queries, most specific first; aim for 15+ models
   in total, since quality control will discard some (skulls, cartoons, fragments).
If an object still has too few models, or you would substitute it with something similar or merge it into
another class, say so plainly and ask the user whether that matters for their goal (for some goals it is
critical, for others irrelevant); this is one of the few things worth a question. Offer the options: merge classes, use a substitute, rely on their own
real photos for that class, or build 3D models from their photos ("photos": "user_data", or a folder of
photos of the object; add "source": "photos" when nothing else is found; "queries" must then hold the
object's English name, which the detector uses to find it in the photos). Photo-made models work best for
rigid, compact objects shot whole on a plain background. Never merge or substitute silently.
Scene: template "road" is a hand-built road with verges (traffic). For anything else use "generic": a ground
plane textured from Poly Haven by "ground_tags" and sky maps (which also give the background) chosen by
"sky_tags" plus "weather". Tags must all match, so pick 1 or 2 specific ones: sky categories include
{sky_tags}; ground categories include {ground_tags}; free tags also work, e.g. sky "forest", "farm", "city", "parking", "warehouse", ground
"grass", "leaves", "dirt", "gravel", "mud", "asphalt", "tiles". Indoors, put "indoor" in sky_tags (weather is
then ignored). Tags are checked when you send the spec.
Weather options: {weather}. Distractors: "tree", "person" (nothing else); keep them only if they appear
where the camera is, else set "distractors": [].
ground_tags describe the floor under the objects (asphalt, tiles, grass...), never walls or roofs.
Places: by default synthetic scenes stand the objects on open ground, seen from camera.height_m. The generic
template can also build "scene.contexts": building fronts seen from the street ("facade": objects on balcony
floors, railings, window sills and the pavement), balconies seen from close ("balcony"), rooms ("indoor":
floor, furniture, window sill) and open ground ("ground"), each with a share of the images, plus "wall_tags"
(brick, plaster, stone, concrete, wood) and "railings" (bars, solid). Exact format: "contexts": [{{"kind":
"facade", "share": 0.7, "distance_m": [3, 20], "supports": ["railing", "balcony_floor"]}}, {{"kind": "balcony",
"share": 0.3}}] (distance_m and supports are optional; supports by kind: {supports}). As soon as you have a folder of the
user's photos, describe them: {{"describe_scene": {{"folder": "<folder>", "classes": "<what to find, in
English>"}}}} (done by itself when the user gives photos). You get back what the photos show and the
proposed fields: use them. Objects on walls or ceilings are not simulated: say so if that is what they need.
Props: scenes with only the target look empty and the model then learns "anything in the picture is a target".
Always propose 2 to 5 background props typical of the place ("props": a building site gets traffic cones,
pallets, barriers, dirt piles; a forest gets rocks, logs, bushes), each with live search queries and a
real size. They are found live like a class and never labeled. They must not look like any class (no cars as
props for a vehicle camera). Take them from the things the photo analysis saw around the objects.

Spec format (JSON, defaults shown; omit a field to keep its default):
{schema}
Notes: size_m.measure is "length" (long side from above), "width" (short side from above) or "height";
choose the one that varies least across the class. assets.count is the number of models per class.
classes lists only the objects to find: the background class ("empty", no object) is added by itself, never
list it as a class or ask the user about it. When every photo is one of the classes (surfaces, scenes or
states that cover the whole view, like lawn / border / no grass, or day / night), set "background_class": null
so no extra class is added: the model must always pick one of the user's classes.
repaint (0-1) is how often a model is painted a random solid color: set it (e.g. 0.35) only when real
instances of the class come in many colors (cars), with metallic [0.2, 0.8] for shiny paint; omit both otherwise.
user_data.labeled / user_data.raw / user_data.test are folder paths (raw may list several, merged); class_map maps each class id to the subfolder names of
the labeled folder. Keep qa.mode "auto".

How to talk: at most one short question per turn, only the ones allowed above, and skip what the user
already told you. Two moments need photos, and you ask for them right then with the "photos" action (the
user gets an upload box in the chat; they may also decline in words):
- After a search, when a class has few usable models (under ~10), the matches are not really the object,
  or it is the user's own specific object: ask whether some of their photos show it clearly, to build 3D
  models from them. Say what makes a good one: the whole object, big in the frame, on a background that
  does not distract (plain wall or floor), ideally from 2 or 3 angles; 1 to 5 photos are enough.
  {{"photos": {{"for": "3d", "class": "<class id>", "say": "<your question>"}}}}
- Early, if the user has not given photos yet: photos from the real camera or place, unsorted, to train and
  test (they also let you see the place). {{"photos": {{"for": "training", "say": "<your question>"}}}}
Before asking for 3D photos, check the photos the user already gave (uploaded or a folder path they wrote):
some may already be good. {{"pick_photos": {{"class": "<class id>", "query": "<English name>", "folder":
"<that folder>", "say": "<short intro>"}}}} ranks them with a vision model and shows the user the best ones
to confirm; you are told if none fit, and then you ask for new photos.
When photos arrive or are chosen you are told their folder: use it in the spec ("3d": that class's
assets.photos plus an English "queries" name; "training": user_data.raw).

Reply ONLY with one JSON object per turn: {{"say": "..."}}, {{"search": ["..."]}}, {{"photos": {{...}}}},
{{"pick_photos": {{...}}}}, {{"describe_scene": {{...}}}} or {{"say": "...", "spec": {{...}}}}."""

EXAMPLE = {"name": "example", "classes": [{"id": "cow", "description": "cows seen from the side or front",
                                           "size_m": {"measure": "length", "range": [2.0, 2.5]},
                                           "assets": {"source": "objaverse_lvis", "categories": ["cow"],
                                                      "count": 20}}],
           "props": [{"id": "hay_bale", "queries": ["hay bale"], "size_m": {"measure": "length", "range": [1.2, 2.4]},
                      "count": 4, "max_per_frame": 4}],
           "qa": {"mode": "auto"}}


FALLBACK_TAGS = {"hdris": ["outdoor", "nature", "urban", "indoor", "night", "sunrise-sunset", "overcast",
                            "partly cloudy", "clear"],
                 "textures": ["floor", "terrain", "natural", "rock", "concrete", "sand", "road", "asphalt",
                              "cobblestone", "wood", "brick"]}


def scene_tags(kind: str) -> list[str]:
    """Poly Haven categories, live when the API answers."""
    from . import polyhaven
    try:
        return polyhaven.categories(kind)[:25]
    except OSError:
        return FALLBACK_TAGS[kind]


def system_prompt() -> str:
    schema = json.dumps(spec_mod.to_dict(spec_mod.from_dict(EXAMPLE)), indent=1)
    profiles = "\n".join(f"  - {h.name}: {h.description}" for h in hardware.PROFILES.values())
    return SYSTEM.format(hardware=profiles, weather=", ".join(spec_mod.WEATHER),
                         sky_tags=", ".join(scene_tags("hdris")), ground_tags=", ".join(scene_tags("textures")),
                         schema=schema, supports="; ".join(f"{k}: {', '.join(v)}"
                                                           for k, v in spec_mod.CONTEXTS.items()))


class CategoryIndex:
    """Objaverse-LVIS category names with their model counts, searched by substring."""

    def __init__(self, counts: dict[str, int] | None = None, cache_dir: Path | None = None, live=None):
        if counts is None:
            if cache_dir:
                from .workspace import objaverse_at
                objaverse = objaverse_at(cache_dir)
            else:
                import objaverse
            counts = {k: len(v) for k, v in objaverse.load_lvis_annotations().items()}
        self.counts = counts
        self.live = live  # query -> list of asset_search.Hit, or None for curated categories only

    def search(self, queries: list[str], limit: int = 12) -> dict:
        curated = self.curated(queries, limit)
        if self.live is None:
            return curated
        live = {}
        for q in queries:
            hits = self.live(q)
            live[q] = {"models": len(hits), "examples": [h.name for h in hits[:6]]}
        return {"curated_categories": curated, "live_search": live}

    def curated(self, queries: list[str], limit: int = 12) -> dict[str, int]:
        hits: dict[str, int] = {}
        for q in queries:
            q = q.lower().strip().replace(" ", "_")
            # Plural forms too: "cows" -> "cow", "buses" -> "bus".
            variants = {q, q[:-1] if q.endswith("s") else q, q[:-2] if q.endswith("es") else q} - {""}
            for name, n in self.counts.items():
                base = name.split("_(")[0]
                if any(v in name or (len(base) >= 4 and base in v) for v in variants):
                    hits[name] = n
        return dict(sorted(hits.items(), key=lambda kv: -kv[1])[:limit])


NOT_GROUND = {"wall", "walls", "roof", "roofs", "ceiling", "facade", "window", "windows"}


def check_scene(scene) -> str | None:
    """Do the scene's Poly Haven tags find skies and ground textures? Checked live when a spec is proposed."""
    from . import polyhaven
    wrong = [t for t in scene.ground_tags if t.lower() in NOT_GROUND]
    if wrong:
        return f"ground_tags {wrong} are not a floor: the ground texture goes under the objects"
    if scene.template == "road" and not scene.sky_tags:
        return None
    try:
        if not polyhaven.pick_skies(polyhaven.catalog("hdris"), scene.sky_tags, scene.weather, 1, 0):
            return f"no Poly Haven sky matches sky_tags {scene.sky_tags} with weather {scene.weather}; use fewer tags"
        if scene.ground_tags and not polyhaven.pick_textures(polyhaven.catalog("textures"), scene.ground_tags, 1, 0):
            return f"no Poly Haven ground texture matches ground_tags {scene.ground_tags}; use fewer or other tags"
    except OSError:
        return None  # offline: the render stage will report it
    return None


PICK_PROMPT = """Could this photo be used to build a 3D model of "{query}"? A good photo shows one {query}
whole (not cut by the frame), big in the frame, sharp, on a background that does not distract.
Answer only with JSON: {{"shows": true|false, "whole": true|false, "big": true|false, "clean_background":
true|false, "score": <0 to 1, how good for a 3D model>, "reason": "<short>"}}"""
PICK_MIN_SCORE = 0.6
MAX_INVALID_SPECS = 4  # rejected proposals in a row before the model must ask the user instead
# Answers that are not one JSON object, in a row, before the interview gives up. Seen 2026-10-03 (bench, reasoning
# off): the same malformed JSON 59 times, 60 turns and 1.4 M tokens.
MAX_BAD_FORMAT = 3
PICK_MAX = 8


NO_VERDICT = ("no valid answer", "no answer in time")  # the vision model failed, the photo was not judged
PICK_DEADLINE_S = 120  # ranking photos for 3D models: the user waits, photos not answered by then score 0


def photo_ranker(client, model: str, limit: int = 80, workers: int | None = None):
    """Vision-model ranking of a folder's photos for building 3D models of a query (Token Factory)."""
    import tempfile

    from PIL import Image, ImageOps

    from .llm import PHOTO_CALL, REASONING_TOKENS, image_part, map_within, parse_json
    from .workspace import IMAGE_EXT

    def rank(folder: str, query: str, progress=None) -> list[dict]:
        files = sorted(f for f in Path(folder).rglob("*") if f.suffix.lower() in IMAGE_EXT)
        files = sorted(random.sample(files, limit)) if len(files) > limit else files  # a random sample, not the first
        prompt = PICK_PROMPT.format(query=query)
        with tempfile.TemporaryDirectory() as tmp:
            def one(item):
                i, f = item
                small = Path(tmp) / f"{i}.jpg"
                with Image.open(f) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((512, 512))
                    im.save(small, quality=85)
                msg = [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(small)]}]
                try:
                    # Kimi-K2.6 reasons first: nothing within 1024 (2026-10-01), often cut at 4096
                    v = parse_json(client.chat(msg, model, purpose="pick_photos", temperature=0.0, max_tokens=REASONING_TOKENS,
                                               **PHOTO_CALL))
                    ok = v.get("shows") and v.get("whole")
                    return {"file": str(f.resolve()), "score": float(v.get("score", 0)) if ok else 0.0,
                            "reason": str(v.get("reason", ""))}
                except (ValueError, TypeError, AttributeError):
                    return {"file": str(f.resolve()), "score": 0.0, "reason": NO_VERDICT[0]}
            got = map_within(one, list(enumerate(files)), PICK_DEADLINE_S, workers, progress)
        out = [c or {"file": str(f.resolve()), "score": 0.0, "reason": NO_VERDICT[1]} for c, f in zip(got, files)]
        return sorted(out, key=lambda c: -c["score"])
    return rank


def count_photos(folder: Path) -> int:
    from .workspace import IMAGE_EXT
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXT)


def folder_layout(folder: Path, limit: int = 20) -> str:
    """How a photo folder is organised, for the model: photos loose in it and per subfolder (with everything
    below). The interview asked users whether their photos were sorted by class; the disk already says."""
    from .workspace import IMAGE_EXT
    loose = sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT)
    subs = {d.name: count_photos(d) for d in sorted(folder.iterdir()) if d.is_dir() and not d.name.startswith(".")}
    subs = {k: v for k, v in subs.items() if v}
    listed = ", ".join(f"{k} ({v})" for k, v in list(subs.items())[:limit]) + (" ..." if len(subs) > limit else "")
    if not subs:
        return (f"{folder}: {loose} photos, no subfolders: unsorted, so user_data.raw (the teacher labels them).")
    how = ("If the subfolder names are the classes, this is user_data.labeled with class_map; if they are places, "
           "days or anything else, it is user_data.raw (the teacher labels them).")
    if loose:
        return f"{folder}: {loose} photos loose and subfolders {listed}: mixed, so user_data.raw."
    return f"{folder}: no loose photos, subfolders {listed}. {how}"


def sam3_names_problem(s: spec_mod.Spec) -> str | None:
    """Counting with the user's raw photos: SAM 3 marks them and understands only English, so every class needs
    English names. Without them it got "maceta" and found nothing in 25 photos (2026-09-30)."""
    from . import teacher
    if s.task != "count" or not s.user_data.raw:
        return None
    missing = [c.id for c in s.classes if not teacher.names(c)]
    if not missing:
        return None
    return (f"classes {missing} have no \"names\". The teacher that marks the objects in the user's photos (SAM 3) "
            "understands only English: give each counted class \"names\": 1 to 3 short English noun phrases, most "
            "common first, e.g. \"names\": [\"flower pot\", \"plant pot\"]")


def check_photos(s: spec_mod.Spec) -> str | None:
    """Do the photo folders the spec names exist and have the layout it assumes? Only absolute paths are
    checked (relative ones are uploads that live in the job folder)."""
    from .workspace import IMAGE_EXT
    ud = s.user_data

    def photos_in(d: Path) -> int:
        return sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT) if d.is_dir() else 0

    for key, rel in [("labeled", ud.labeled), ("test", ud.test)] + [("raw", r) for r in sources(ud.raw)]:
        if not rel or not Path(rel).is_absolute():
            continue
        d = Path(rel)
        if not d.is_dir():
            return (f"user_data.{key}: folder {d} does not exist on this machine. Tell the user this exact path "
                    f"was not found (it may have been moved or renamed) and ask for the right one; do not resend it")
        if key in ("labeled", "test"):
            subs = sorted(x.name for x in d.iterdir() if x.is_dir())
            missing = [f for c in s.labels for f in ud.class_map.get(c, [c]) if not photos_in(d / f)]
            if missing and photos_in(d):
                return (f"user_data.{key}: {d} has {photos_in(d)} unsorted photos and subfolders {subs}; it is "
                        f"not one subfolder per class (no photos for {missing}). Unsorted photos go in user_data.raw")
            if len(missing) == len(s.labels):
                return f"user_data.{key}: no photos for any class in {d} (subfolders {subs})"
        elif not photos_in(d) and not any(x.is_dir() for x in d.iterdir()):
            return f"user_data.raw: no photos in {d}"
    for c in s.classes:
        p = c.assets.photos if c.assets else None
        if p and p != "user_data" and Path(p).is_absolute() and not photos_in(Path(p)):
            return f"class {c.id}: assets.photos folder {p} has no photos"
    return None


def live_searcher(objaverse_dir: Path, pages: int = 2):
    """Live search for the interview: open-license Sketchfab hits the Objaverse mirror can serve."""
    from . import asset_search
    mirror = None

    def run(query: str):
        nonlocal mirror
        if mirror is None:
            mirror = asset_search.Mirror(objaverse_dir)
        return [h for h in asset_search.search(query, pages) if h.uid in mirror]
    return run


YES = {"y", "yes", "yeah", "yep", "ok", "okay", "sure", "correct", "right", "s", "si", "sí", "vale", "claro",
       "correcto", "perfecto", "adelante", "bien", "de acuerdo"}
NO = {"no", "not", "but", "pero", "cambia", "change", "falta", "missing"}


UPLOADS_ONLY = """
Photos: the user can only upload photos in this chat (the photos action); there is no access to folders on
any disk, so never ask for a folder path and use only the folders you are told after an upload.
"""


def _drop_background_class(spec: dict) -> None:
    """The model sometimes lists the background ("empty") as one more class (2026-09-27); the pipeline adds it by
    itself, so drop it instead of rejecting the whole spec."""
    bg = spec.get("background_class", "empty")
    if bg is None:
        return
    if isinstance(spec.get("classes"), list):
        spec["classes"] = [c for c in spec["classes"] if not (isinstance(c, dict) and c.get("id") == bg)]


def sources(raw) -> list[str]:
    """user_data.raw as a list: one folder, or several to merge."""
    return [raw] if isinstance(raw, str) else list(raw or [])


def spec_paths(spec: spec_mod.Spec) -> list[str]:
    """Every folder or file path a spec names (user photos, 3D photos, QA lists)."""
    ud, qa = spec.user_data, spec.qa
    paths = [*sources(ud.raw), ud.labeled, ud.test, qa.exclude, qa.class_override, qa.no_color]
    paths += [c.assets.photos for c in spec.classes if c.assets]
    paths += [m for c in spec.classes if c.assets for m in c.assets.models] + [m for p in spec.props for m in p.models]
    return [p for p in paths if p]


NO_PHOTO_MODELS = """
On this server 3D models cannot be made from the user's photos yet: never set assets.photos, never use
"source": "photos" or pick_photos; find models with "search" (English queries). The user's photos still
train and test the model (user_data)."""

LIBRARY = """
The user's library: things they already have, kept from earlier models or uploaded. A kit is a named list of
them. When the user names a kit (or a dataset or object listed below), use those instead of searching for 3D
models or asking for new photos:
- a dataset as the user's photos: user_data "raw": "library:<id>" (the teacher labels them), or "labeled":
  "library:<id>" when the dataset's photos carry labels that match the classes (class_map maps each class to
  those labels);
- objects as 3D models of a class: "assets": {{"source": "library", "models": ["library:<id>", ...]}} (with
  source "search" and queries instead if more models should be searched too, keeping "models");
- objects that are not one of the classes go into props ({{"id", "queries": [], "models": ["library:<id>"],
  "size_m"}}): look-alikes and background things the camera must learn to ignore.
Each dataset comes with what was seen in it (objects, backgrounds, places). Before proposing a spec that uses
a dataset, measure it for this goal: {{"describe_scene": {{"folder": "library:<id>", "classes": "<what to find>"}}}}
(sizes, camera, scene fields). Photos the user attaches in this chat count too: when a dataset is also used,
"raw" lists both, ["library:<id>", "<attached folder>"].
A dataset the teacher labeled before is still used as "raw" (only a dataset listed with labels of its own is
"labeled"), and it keeps the teacher's labels: when what the user wants now fits, use the same task and exactly
the same class ids (and background class); the job then reuses the labels and the user is not asked to check the
photos again. Other classes or another task mean the teacher labels every photo anew and the user checks them:
say so before proposing them, and propose the earlier classes as the faster option. The 3D models kept for a
class are found again by the same search queries: reuse them for the same kind of object with "assets":
{{"source": "search", "queries": [those queries]}} (a dataset id is never a 3D model; a class or a prop with 0 kept
had no usable model; try other queries). When the user names a dataset, never ask for photos again.
Decide for each object whether it is a class or a prop from what the user wants now: the same object can be the
target of one model and a distractor of another. Every class or prop with 3D models still needs size_m. Use only
the ids listed here.
{items}
"""

RELABEL = """[Labels of dataset {ds}] The teacher already labeled this dataset ({task}) for: {old}. The proposed
spec also needs {new}, never labeled in it: the teacher would mark {them} on every photo and the user would check
a few photos. Do not propose the spec yet. In the user's language, tell them which classes the dataset is
already labeled with and ask whether to keep those or to add or use the new ones (the labeled ones are reused at
no cost, the same English names find them again). Then propose what they choose."""

RESEARCH = """
You can look facts up on the web: {{"research": "<one precise question>"}} returns a short answer and its sources.
Use it instead of guessing when a number matters and you are not sure of it: the real size of an object or a
part (it sets size_m and whether the camera can see it at that distance and resolution), its English or technical
name (for the 3D model search), or what it looks like. At most {n} per interview; the user never sees them.
"""
MAX_RESEARCH = 6

ENGLISH_PROMPT = """These are search terms for 3D models and for an English-only object detector. Translate
each term into the English words a 3D model site would use, keeping every detail the term gives (kind, place,
size) and never a shorter, more generic word: "maceta" -> "flower pot", "silla de terraza" -> "patio chair",
"mesa pequeña de terraza" -> "small patio table". A term already in English stays as it is. Reply with one
JSON object {"<term>": "<English>"} and nothing else. Terms: """


def is_yes(answer: str) -> bool:
    """A short affirmative answer ("Sí, adelante.", "ok!", "yes please") without a change request."""
    import re
    words = re.findall(r"[a-záéíóúñ]+", answer.lower())
    if not words or len(words) > 6 or any(w in NO for w in words):
        return False
    text = " ".join(words)
    return words[0] in YES or text in YES


def summary(s: spec_mod.Spec) -> str:
    what = "counts and locates" if s.task == "count" else "classifies"
    lines = [f"Spec '{s.name}' for {s.hardware}, {what}:",
             f"  classes: {', '.join(s.labels)} "
             + (f"(background: {s.background_class})" if s.background_class else "(no background class)")]
    for c in s.classes:
        size = f"{c.size_m.measure} {c.size_m.range[0]}-{c.size_m.range[1]} m" if c.size_m else "no 3D size"
        a = c.assets
        cats = "no 3D models" if not a else ", ".join(a.categories) if a.source == "objaverse_lvis" else \
            f"search {a.queries}" if a.source == "search" else \
            f"{len(a.models)} from your library" if a.source == "library" else \
            f"made from photos ({a.queries[0] if a.queries else '?'})"
        if a and a.models and a.source != "library":
            cats += f" + {len(a.models)} from your library"
        if a and a.photos and a.source != "photos":
            cats += f" + made from photos ({a.queries[0]})"
        lines.append(f"  - {c.id}: {c.description} | {size} | models: {cats}")
    sc = s.scene
    lines.append(f"  camera: {s.camera.height_m[0]}-{s.camera.height_m[1]} m high, scene {sc.template}, "
                 f"weather {', '.join(sc.weather)}")
    if sc.contexts:
        lines.append("  places: " + ", ".join(f"{c.kind} {round(c.share * 100)}%" for c in sc.contexts) +
                     (f"; walls {', '.join(sc.wall_tags)}" if sc.wall_tags else "") +
                     (f"; railings {', '.join(sc.railings)}" if sc.railings else ""))
    if sc.sky_tags or sc.ground_tags:
        lines.append(f"  place: sky {', '.join(sc.sky_tags) or 'any'}, ground {', '.join(sc.ground_tags) or 'plain'}")
    extra = [f"distractors {', '.join(sc.distractors)}" if sc.distractors else "",
             f"props {', '.join(p.id for p in s.props)}" if s.props else ""]
    if any(extra):
        lines.append("  also in the scene: " + "; ".join(e for e in extra if e))
    ud = s.user_data
    photos = [f"sorted by class {ud.labeled}" if ud.labeled else "",
              f"unsorted {ud.raw} (the teacher model labels them)" if ud.raw else "",
              f"test {ud.test}" if ud.test else ""]
    turns = {"any": "turned any way and mirrored (no up in the view)",
             "fixed": "slightly turned and tilted, never mirrored (left and right matter)"}
    lines.append(f"  training copies: {turns.get(s.training.orientation, 'slightly turned, tilted and mirrored left-right')}")
    lines.append(f"  photos: {', '.join(p for p in photos if p) or 'none (synthetic only)'}")
    if not s.synthetic:
        lines.append("  synthetic images: none, the model learns from your photos only")
    return "\n".join(lines)


FIRST_QUESTION = "Describe what your camera should see / Describe qué debe ver tu cámara:"
CONFIRM = "Is this right? / ¿Es correcto? (yes/sí, or tell me what to change / o dime qué cambiar)"


def _rename_classes(spec: dict, renamed: dict[str, str]) -> None:
    """Class ids renamed everywhere a spec names them: the classes, the background class and the class map."""
    for c in spec.get("classes") or []:
        if isinstance(c, dict) and c.get("id") in renamed:
            c["id"] = renamed[c["id"]]
    if spec.get("background_class") in renamed:
        spec["background_class"] = renamed[spec["background_class"]]
    ud = spec.get("user_data")
    if isinstance(ud, dict) and isinstance(ud.get("class_map"), dict):
        ud["class_map"] = {renamed.get(k, k): v for k, v in ud["class_map"].items()}


def _unknown_categories(session, s: spec_mod.Spec) -> str | None:
    unknown = [cat for c in s.classes if c.assets and c.assets.source == "objaverse_lvis"
               for cat in c.assets.categories if cat not in session.categories.counts]
    return f"categories not found in Objaverse-LVIS: {unknown}, use search" if unknown else None


def _foreign_folders(session, s: spec_mod.Spec) -> str | None:
    foreign = [f for f in spec_paths(s) if Path(f).is_absolute() and session._foreign(f)]
    return (f"these folders are not photos uploaded in this chat: {foreign}. " + UPLOADS_ONLY.strip()) \
        if foreign else None


def _library_refs(session, s: spec_mod.Spec) -> str | None:
    # A model often writes an object's id without the prefix: a bare library-style id counts as a reference too.
    refs = {p[len("library:"):] for p in spec_paths(s) if p.startswith("library:")} | \
        {p for p in spec_paths(s) if re.fullmatch(r"[0-9a-f]{16}", p)}
    unknown = sorted(refs - getattr(session, "library_ids", set()))
    return f"these library ids are not in the user's library: {unknown}; use only the listed ones" if unknown else None


def _dataset_as_model(session, s: spec_mod.Spec) -> str | None:
    """A dataset id given as a class's 3D models (the chat read "3D models kept" in a dataset's line, 2026-10-09)."""
    items = [(c.id, c.assets.models) for c in s.classes if c.assets] + [(p.id, p.models) for p in s.props]
    bad = sorted({cid for cid, models in items for m in models or []
                  if re.sub(r"^library:", "", str(m)) in getattr(session, "dataset_ids", set())})
    if not bad:
        return None
    return (f"{', '.join(bad)}: a dataset id is photos, not 3D models. To reuse the 3D models kept for that dataset, "
            'give "assets": {"source": "search", "queries": [the same search queries]} and "models": []')


def _unlabeled_dataset(session, s: spec_mod.Spec) -> str | None:
    ds = re.fullmatch(r"(?:library:)?([0-9a-f]{16})", str(s.user_data.labeled or ""))
    if not ds or ds.group(1) in getattr(session, "labeled_ids", set()):
        return None
    return (f"dataset {ds.group(1)} has no labels of its own (the teacher's earlier labels are not that): give it as "
            f'user_data "raw": "library:{ds.group(1)}", with "labeled": null and no class_map')


def _scene_problem(session, s: spec_mod.Spec) -> str | None:
    return session.scene_check(s.scene) if session.scene_check else None


def _photo_models_problem(session, s: spec_mod.Spec) -> str | None:
    if not session.photo_models and any(c.assets and c.assets.photos for c in s.classes):
        return NO_PHOTO_MODELS.strip()
    return None


# What a proposed spec must pass beyond spec.validate, in order: the first problem goes back to the model.
SPEC_CHECKS = (_unknown_categories, _foreign_folders, _library_refs, _dataset_as_model, _unlabeled_dataset, _scene_problem, lambda _, s: check_photos(s),
               lambda _, s: sam3_names_problem(s), _photo_models_problem)


@dataclass
class Turn:
    kind: str                       # "say" (a question), "photos" (asks for photos), "pick" (proposes some of
    text: str                       # the user's photos), "confirm" (spec proposed), "done" or "failed"
    spec: spec_mod.Spec | None = None
    request: dict | None = None     # photos: {"for", "class"}; pick: {"for", "class", "query", "candidates"}
    intro: str | None = None        # confirm: the model's own words, without the spec summary (a front end
                                    # that shows the spec its own way uses this instead of text)


class Session:
    """One interview, driven one user message at a time (terminal, web or any other front end)."""

    def __init__(self, client, model: str, categories: CategoryIndex, max_turns: int = 60, scene_check=None,
                 photo_ranker=None, uploads: Path | None = None, scene_profiler=None, photo_models: bool = True,
                 uploads_only: bool = False, researcher=None):
        self.client, self.model, self.categories, self.max_turns = client, model, categories, max_turns
        # False where 3D models cannot be made from photos (the Nebius backend has no TRELLIS runtime yet).
        self.photo_models = photo_models
        # True on a public server: only photos uploaded in this chat, never a folder of the server's disk.
        self.uploads_only = uploads_only
        self.researcher = researcher  # question -> {"answer", "sources"} (research.searcher), or None
        self.researched = 0
        self.scene_check = scene_check  # scene -> error message or None (live Poly Haven check)
        self.photo_ranker = photo_ranker  # (folder, query) -> [{"file", "score", "reason"}], best first
        self.uploads = uploads            # where chosen photos are gathered
        self.scene_profiler = scene_profiler  # (folder, classes) -> (scene fields, summary)
        self.candidates: list[dict] = []  # the last photos proposed to the user
        self.messages = [{"role": "system", "content": system_prompt() + ("" if photo_models else NO_PHOTO_MODELS)
                          + (UPLOADS_ONLY if uploads_only else "")
                          + (RESEARCH.format(n=MAX_RESEARCH) if researcher else "")}]
        self.proposed: spec_mod.Spec | None = None
        self.transcript: Path | None = None  # set by the server: logs/interviews/<session>.json
        self.finished = False
        self.translation_error: str | None = None  # why the last English translation of queries failed
        self.goal = ""  # the user's first message: what the photo analysis looks for
        self.analysed: set[str] = set()  # folders already analysed
        self.settled: list[list[str]] = []  # query lists of the job being revised (see revise)
        self.settled_ids: set[str] = set()  # its class ids, kept as they are
        # What the current turn is doing, step by step, for a front end to show while it waits (see _step).
        # `activity_id` is the caller's id for the turn, so a poll never shows the steps of the previous one.
        self.activity: list[dict] = []
        self.activity_id: str | None = None
        self.library_ids: set[str] = set()  # library items this chat may reference (see use_library)
        self.photo_notes = None  # (photos, classes, run) -> descriptions, the kept ones reused (library)
        self.open_dataset = None  # library dataset id -> a folder of its photos this session may read
        self.labelings: dict[str, list[dict]] = {}  # library dataset id -> library.labelings
        self.replies = 0  # the user's messages so far
        self.relabel_asked: dict[str, int] = {}  # dataset id -> self.replies when the user was to be asked

    def use_library(self, items: str, ids: set[str], kit: str | None = None, notes=None, open_dataset=None,
                    labeled: set[str] | None = None, datasets: set[str] | None = None,
                    labelings: dict[str, list[dict]] | None = None) -> None:
        """Tell the model what the user's library holds (kits and datasets, see LIBRARY); kit: the name of the kit
        the user chose to start from. notes: keeps and reuses photo descriptions (Library.describe);
        open_dataset(id): a folder with a dataset's photos, so describe_scene can measure it for this goal;
        labeled: the datasets with labels of their own, the only ones a spec may give as "labeled"; datasets: every
        dataset id, never a 3D model; labelings: per dataset, how the teacher labeled it before
        (library.labelings), so the user is asked before a spec labels it for other classes (relabel_question)."""
        self.library_ids = set(ids)
        self.labelings = labelings or {}
        self.dataset_ids = set(datasets or ())
        self.labeled_ids = set(labeled or ())
        self.photo_notes, self.open_dataset = notes, open_dataset
        self.messages[0]["content"] += LIBRARY.format(items=items)
        if kit:
            self._note(f"The user starts this model from their kit \"{kit}\": use its datasets and objects.")

    def restore(self, saved: list[dict]) -> None:
        """Go on with a conversation from its transcript (_save), after the server restarted (a deploy cut a
        user's chat, 2026-10-09): the model sees it as it was, the folders already analysed are not analysed
        again, and a proposal still waiting for the user's yes is waiting again."""
        from .llm import parse_json
        self.messages = [m for m in saved if m.get("role") in ("system", "user", "assistant", "tool")]
        notes = [m["content"] for m in self.messages if m["role"] == "user" and isinstance(m.get("content"), str)]
        self.goal = next((n[:400] for n in notes if not n.startswith("[")), "")
        for n in notes:
            for folder in re.findall(r"^\[(?:Folder layout[^\]]*|Photo analysis of [^\]]*)\] (/[^\s:]+)", n):
                self.analysed.add(folder.rstrip(":"))
            for folder in re.findall(r"^\[Photo analysis of [^\]]* in (/[^\]\s]+)\]", n):
                self.analysed.add(folder)
        last = self.messages[-1] if self.messages else {}
        if last.get("role") == "assistant":
            try:
                action = parse_json(last["content"])
                if isinstance(action, dict) and isinstance(action.get("spec"), dict):
                    self.proposed = spec_mod.from_dict(action["spec"])
            except (ValueError, spec_mod.SpecError, TypeError):
                pass

    def begin(self, activity_id: str | None = None) -> None:
        """Start a new turn's activity list; call before reply, add_photos or choose."""
        self.activity, self.activity_id = [], activity_id

    def _step(self, text: str, total: int | None = None) -> dict:
        """A step the user sees while the turn runs: running until _end marks it done or failed."""
        step = {"text": text, "state": "running", "started": time.time()}
        if total:
            step |= {"done": 0, "total": total}
        self.activity.append(step)
        return step

    @staticmethod
    def _end(step: dict, state: str = "done", detail: str | None = None) -> None:
        step["state"], step["ended"] = state, time.time()
        if detail:
            step["detail"] = detail

    @staticmethod
    def _counting(fn, step: dict, *args):
        """Call a photo helper, letting it count finished photos into the step when it can."""
        if "progress" not in inspect.signature(fn).parameters:
            return fn(*args)
        def progress(done: int, total: int) -> None:
            step["done"], step["total"] = done, total
        return fn(*args, progress=progress)

    def reply(self, user_text: str) -> Turn:
        if self.finished:
            raise RuntimeError("the interview is over")
        self.goal = self.goal or user_text[:400]
        self.replies += 1
        if self.proposed is not None:
            if is_yes(user_text):
                self.finished = True
                return Turn("done", "", self.proposed)
            self.proposed = None
            user_text = f"The user wants changes: {user_text}"
        self.messages.append({"role": "user", "content": user_text})
        for folder in re.findall(r"(?:~|/)[^\s,;:'\"()]+", user_text):  # folders written in the message
            self._analyse(folder.rstrip("."))
        return self._advance()

    def revise(self, spec: spec_mod.Spec) -> Turn:
        """Start from a trained job's spec instead of from scratch: the model sees it as its own accepted
        proposal, and the user's next message says what to change."""
        self.goal = "; ".join(f"{c.id}: {c.description}" for c in spec.classes)[:400]
        self.settled = [c.assets.queries for c in spec.classes if c.assets and c.assets.queries] + \
            [p.queries for p in spec.props if p.queries]
        self.settled_ids = {c.id for c in spec.classes} | ({spec.background_class} if spec.background_class else set())
        self.messages.append({"role": "user", "content": (
            "[The user already trained a model with the spec below and wants to change something. Ask nothing "
            "that the spec already answers. When they say what to change, change only that, keep everything "
            "else exactly as it is (user_data paths included) and propose the full spec again.]")})
        self.messages.append({"role": "assistant", "content": json.dumps({"spec": json.loads(spec.to_json())})})
        self._save()
        return Turn("say", "What would you like to change? For example the device, the camera height, the "
                           "places it will see, or a class to add or merge.")

    def _analyse(self, folder: str) -> None:
        """Layout and photo analysis of a folder the user gave, once per folder, before the model answers."""
        path = Path(folder).expanduser()
        if not path.is_dir() or self._foreign(path):
            return
        key = str(path.resolve())
        if key in self.analysed or any(key.startswith(a + "/") for a in self.analysed):
            return
        self.analysed.add(key)
        try:
            self._note(f"[Folder layout, read from the disk; never ask the user how it is organised] "
                       f"{folder_layout(path)}")
        except OSError as e:
            self._note(f"[Folder layout] could not read {key}: {e}")
        if self.scene_profiler is not None:
            self._describe({"folder": key, "classes": self.goal or "the objects the user wants"})

    def add_photos(self, folder: str, count: int, purpose: str, cls: str | None = None) -> Turn:
        """The user uploaded (or pointed to) photos, asked for or not: tell the model and continue."""
        if self.finished:
            raise RuntimeError("the interview is over")
        self.proposed = None
        what = f"to build 3D models of class {cls}" if purpose == "3d" else "from their camera, unsorted"
        use = f"classes[{cls}].assets.photos" if purpose == "3d" else \
            "user_data.raw (with the dataset or folder already there, if any: a list of both)"
        self.messages.append({"role": "user", "content": f"[The user provided {count} photos {what}, in folder "
                                                         f"{folder}. Use it as {use}.]"})
        if purpose != "3d":
            self._analyse(folder)
        return self._advance()

    def choose(self, files: list[str], cls: str) -> Turn:
        """The user confirmed some of the proposed photos for 3D models: gather them in one folder."""
        import shutil

        allowed = {c["file"] for c in self.candidates}
        chosen = [f for f in files if f in allowed]
        if not chosen:
            return self.reply("None of those photos, thanks.")
        dest = (self.uploads or Path("uploads")) / f"3d_{re.sub(r'[^a-z0-9_]+', '_', cls.lower())}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in chosen:
            shutil.copy2(f, dest / Path(f).name)
        self.candidates = []
        return self.add_photos(str(dest.resolve()), len(chosen), "3d", cls)

    def _foreign(self, folder: Path | str | None) -> bool:
        """A folder this session may not read: on a public server, anything outside its own uploads."""
        if not self.uploads_only or not folder:
            return False
        root = (self.uploads or Path("uploads")).resolve()
        return not Path(folder).expanduser().resolve().is_relative_to(root)

    def _pick(self, req: dict) -> Turn | None:
        """Rank the user's photos for 3D; a Turn for the user, or None after telling the model why not."""
        if not self.photo_models:
            self._note(NO_PHOTO_MODELS.strip())
            return None
        folder, query, cls = Path(str(req.get("folder", ""))).expanduser(), req.get("query"), req.get("class")
        if self.photo_ranker is None or not query or not cls:
            self._note("pick_photos needs class, query and folder, and a photo ranker; ask for photos instead.")
            return None
        if self._foreign(folder):
            self._note(UPLOADS_ONLY.strip())
            return None
        if not folder.is_dir():
            self._note(f"pick_photos: folder {folder} does not exist. Tell the user this exact path was not found "
                       "and ask where the photos are.")
            return None
        step = self._step(f"Looking for clear photos of {query} for 3D models")
        ranked = self._counting(self.photo_ranker, step, str(folder), query)
        self.candidates = [c for c in ranked if c["score"] >= PICK_MIN_SCORE][:PICK_MAX]
        if ranked and all(c["reason"] in NO_VERDICT for c in ranked):
            # Seen 2026-09-30: GLM timed out on all 6 photos and the model was told none showed a pot, so it asked
            # the user for photos they had already given. A failed check is not a verdict: the user chooses.
            self._end(step, "failed", "the photo check did not answer; you choose")
            self.candidates = ranked[:PICK_MAX]
            return Turn("pick", "I could not check your photos automatically. Choose the ones that show the "
                        f"{query} whole and big in the frame:",
                        request={"for": "3d", "class": cls, "query": query, "candidates": self.candidates})
        self._end(step, detail=f"{len(self.candidates)} of {len(ranked)} photos could work")
        if not self.candidates:
            self._note(f"pick_photos: none of the {len(ranked)} photos in {folder} shows a clear {query} "
                       "(whole, big, clean background). Ask the user for such photos with the photos action.")
            return None
        return Turn("pick", str(req.get("say") or f"Some of your photos could work for 3D models of {cls}:"),
                    request={"for": "3d", "class": cls, "query": query, "candidates": self.candidates})

    def _describe(self, req: dict) -> None:
        """Profile the user's photos and hand the proposed scene fields to the model."""
        folder = Path(str(req.get("folder", ""))).expanduser()
        ref = re.fullmatch(r"(?:library:)?([0-9a-f]{16})", str(req.get("folder", "")))
        if ref and self.open_dataset is not None:
            if ref.group(1) not in self.library_ids:
                self._note(f"describe_scene: {ref.group(1)} is not one of the listed datasets.")
                return
            folder = self.open_dataset(ref.group(1)) or folder
        if self.scene_profiler is None:
            self._note("describe_scene is not available here; keep the default open ground scene.")
        elif self._foreign(folder):
            self._note(UPLOADS_ONLY.strip())
        elif not folder.is_dir():
            self._note(f"describe_scene: folder {folder} does not exist.")
        else:
            step = self._step("Looking at your photos: place, distance, where the objects stand")
            try:
                profiler = self.scene_profiler
                if self.photo_notes is not None and "notes" in inspect.signature(profiler).parameters:
                    profiler = functools.partial(profiler, notes=self.photo_notes)
                fields, text = self._counting(profiler, step, str(folder),
                                              str(req.get("classes") or "the objects"))
            except (OSError, ValueError, RuntimeError) as e:
                self._end(step, "failed", str(e))
                self._note(f"describe_scene failed ({e}); keep the default open ground scene.")
                return
            if not fields.get("contexts"):
                self._end(step, "failed", "no photo could be described")
                self._note("describe_scene: no photo could be described; keep the default open ground scene.")
                return
            self._end(step)
            sug = fields.pop("suggest", {})
            photos = "\n".join(sug.pop("photos", []))
            self._note(f"[Photo analysis of the user's photos in {folder}] {text}\nWhat each photo shows:\n{photos}\n"
                       f"Scene fields to use (template generic): {json.dumps(fields)}\nAlso measured: "
                       f"{json.dumps(sug)} (size_m for the class, camera_height_m for camera.height_m, props: things "
                       "around the objects, to search as props when they cannot be mistaken for the class). Use all "
                       "this in the spec without asking the user about it; tell them in one plain sentence what you saw.")

    # The model's actions, in the order they are looked for in its answer, and the method that handles each. A
    # handler returns the Turn that ends the user's wait, or None to ask the model again. Text sent with a spec
    # gives way to it: seen 2026-09-27, {"say": "check it", "spec": {...}} was shown as plain text, so the user
    # never got the spec to confirm and the model kept saying it was ready.
    ACTIONS = (("say", "_on_say"), ("pick_photos", "_on_pick"), ("describe_scene", "_on_describe"),
               ("photos", "_on_photos"), ("research", "_on_research"), ("search", "_on_search"),
               ("spec", "_on_spec"))
    OBJECT_ACTIONS = {"pick_photos", "describe_scene", "photos"}  # taken only with an object as value

    def _advance(self) -> Turn:
        from .llm import parse_json

        self._invalid = 0  # spec proposals rejected in a row
        bad_format = 0  # answers in a row that were not one JSON object
        for i in range(self.max_turns):
            step = self._step("Thinking" if i == 0 else "Thinking about what it found")
            try:
                # Full reasoning, T 0.3: in experiments/interview_bench (2026-10-03, 20 interviews each) low
                # effort, no reasoning and T 1.0 all gave more rejected specs or wrong fields. No answer
                # reached 4096 tokens; 8192 is headroom, as reasoning counts inside max_tokens.
                text = self.client.chat(self.messages, self.model, purpose="interview", temperature=0.3,
                                        max_tokens=8192)
            except Exception as e:
                self._end(step, "failed", str(e))
                raise
            self.messages.append({"role": "assistant", "content": text})
            self._save()
            try:
                action, problem = parse_json(text), "it is not a JSON object"
            except ValueError as e:
                action, problem = None, str(e)
            if not isinstance(action, dict):
                bad_format += 1
                if bad_format >= MAX_BAD_FORMAT:
                    self._end(step, "failed", "the answer was not in the expected format again")
                    self.finished = True
                    return Turn("failed", "The interview could not go on: the model kept answering in a broken "
                                          "format. Please start again.")
                self._end(step, "failed", "the answer was not in the expected format, asking again")
                self._note(f"Your answer is not valid JSON ({problem}). Reply with exactly one JSON object as "
                           "instructed.")
                continue
            bad_format = 0
            self._end(step)
            handler = self._handler(action)
            if handler is None:
                self._note("Unknown action. Use say, search or spec.")
                continue
            turn = handler(action)
            if turn is not None:
                return turn
        self.finished = True
        return Turn("failed", "The interview did not converge; nothing was saved.")

    def _handler(self, action: dict):
        """The method for the first action of ACTIONS the answer holds, or None."""
        for key, method in self.ACTIONS:
            if key not in action or key in self.OBJECT_ACTIONS and not isinstance(action[key], dict):
                continue
            if key == "say" and "spec" in action:
                continue
            return getattr(self, method)
        return None

    def _on_say(self, action: dict) -> Turn | None:
        if not str(action["say"]).strip():
            self._note("Your say message was empty. Ask the user something or propose the spec.")
            return None
        return Turn("say", str(action["say"]))

    def _on_pick(self, action: dict) -> Turn | None:
        return self._pick(action["pick_photos"])

    def _on_describe(self, action: dict) -> None:
        self._describe(action["describe_scene"])

    def _on_photos(self, action: dict) -> Turn:
        req = action["photos"]
        purpose = req.get("for") if req.get("for") in ("3d", "training") else "training"
        return Turn("photos", str(req.get("say") or "Do you have photos for this?"),
                    request={"for": purpose, "class": req.get("class")})

    def _on_research(self, action: dict) -> None:
        self._research(str(action["research"]))

    def _on_search(self, action: dict) -> None:
        q = action["search"] if isinstance(action["search"], list) else [action["search"]]
        step = self._step("Searching the 3D model catalog: " + ", ".join(map(str, q)))
        found = self.categories.search(q)
        self._end(step, detail=f"{len(found)} categories found" if found else "nothing found")
        self._note("search results (category: number of models): "
                   + json.dumps(found or "nothing found, try other words"))

    def _on_spec(self, action: dict) -> Turn | None:
        if isinstance(action["spec"], dict):
            self._english_queries(action["spec"])
            _drop_background_class(action["spec"])
        step = self._step("Checking the proposed setup")
        try:
            s = spec_mod.from_dict(action["spec"])
            problem = next((p for check in SPEC_CHECKS if (p := check(self, s))), None)
            if problem:
                raise spec_mod.SpecError(problem)
        except spec_mod.SpecError as e:
            # Seen 2026-09-27: 25 rejected proposals in a row, 4 minutes of "thinking" for the user.
            self._end(step, "failed", f"{e} (fixing it)")
            self._invalid += 1
            if self._invalid >= MAX_INVALID_SPECS * 2:
                self.finished = True
                return Turn("failed", f"The interview could not produce a valid setup: {e}")
            if self._invalid >= MAX_INVALID_SPECS:
                self._note(f"The spec is invalid again: {e}. Stop proposing it: tell the user, in their "
                           "language and in plain words, what is missing or wrong and ask how to fix it "
                           "(say action).")
            else:
                self._note(f"The spec is invalid: {e}. Fix it and send it again.")
            return None
        self._end(step)
        if ask := self.relabel_question(s):  # not a mistake to fix: the user decides, so it is no invalid spec
            self._note(ask)
            return None
        self.proposed = s
        intro = str(action.get("say") or "").strip()
        return Turn("confirm", (intro + "\n\n" if intro else "") + summary(s), s, intro=intro)

    def relabel_question(self, s: spec_mod.Spec) -> str | None:
        """A library dataset the teacher labeled before, now with classes it was never labeled for (another English
        name): the model must tell the user which classes it is labeled with and ask, once, before proposing.
        In code, as Nemotron ignored the same request in LIBRARY (basura_counter, 2026-10-10: other class ids
        went to a full check of every photo without a word)."""
        from . import teacher
        raw = s.user_data.raw if isinstance(s.user_data.raw, list) else [s.user_data.raw]
        task = "counts objects" if s.task == "count" else "one label per photo"
        now = {c: p[0] for c, p in teacher.phrases(s).items()} if s.task == "count" else {c: c for c in s.labels}
        for src in raw:
            ds = re.fullmatch(r"(?:library:)?([0-9a-f]{16})", str(src or ""))
            done = [l for l in self.labelings.get(ds.group(1), []) if l["task"] == s.task] if ds else []
            if not done:
                continue
            old = {p for l in done for p in (l.get("phrases") or {}).values()} if s.task == "count" else \
                {c for l in done for c in l["classes"] + ([l["background"]] if l.get("background") else [])}
            new = [c for c, p in now.items() if p not in old]
            asked = self.relabel_asked.get(ds.group(1))
            if not new or (asked is not None and self.replies > asked):
                continue
            self.relabel_asked[ds.group(1)] = self.replies
            return RELABEL.format(ds=ds.group(1), task=task, old=", ".join(sorted(old)),
                                  new=", ".join(f"{c} ({now[c]})" if now[c] != c else c for c in new),
                                  them="it" if len(new) == 1 else "them")
        return None

    def _english_queries(self, spec: dict) -> None:
        """Put an English name first in every search query list. The prompt asks for English, but the model
        sometimes keeps the user's language ("maceta", "silla de terraza", 2026-09-27): the 3D model search
        then finds little and the photo detector (OWLv2, English only) finds nothing. The original stays after.
        Class ids go into English too, in the same call (user, 2026-10-10: ids always in English)."""
        from .llm import parse_json

        classes = [c for c in spec.get("classes") or [] if isinstance(c, dict) and isinstance(c.get("id"), str)]
        ids = [c["id"] for c in classes] + ([spec["background_class"]] if isinstance(spec.get("background_class"), str)
                                            else [])

        def own(i: str) -> set[str]:  # the class's English names: an id that is one of them is English already
            c = next((c for c in classes if c["id"] == i), {})
            a = c.get("assets") if isinstance(c.get("assets"), dict) else {}
            return {str(t).lower() for k in ("queries", "categories") for t in a.get(k) or []} | \
                {str(t).lower() for t in c.get("names") or []}
        id_terms = {i: i.replace("_", " ") for i in ids
                    if i not in self.settled_ids and i.replace("_", " ").lower() not in own(i)}

        lists = [a["queries"] for c in spec.get("classes") or [] if isinstance(a := c.get("assets"), dict)
                 and isinstance(a.get("queries"), list)]
        lists += [p["queries"] for p in spec.get("props") or [] if isinstance(p, dict)
                  and isinstance(p.get("queries"), list)]
        names_lists = [c["names"] for c in spec.get("classes") or [] if isinstance(c, dict)
                       and isinstance(c.get("names"), list)]
        lists += names_lists
        # Revising a job: lists kept as they were are already settled. Rewording them (railings -> railing)
        # would change the spec and send the job back to searching 3D models and rendering for nothing.
        lists = [qs for qs in lists if qs not in self.settled]
        terms = sorted({q for qs in lists for q in qs if isinstance(q, str) and q.strip()} | set(id_terms.values()))
        if not terms:
            return
        msg = [{"role": "user", "content": ENGLISH_PROMPT + json.dumps(terms, ensure_ascii=False)}]
        step = self._step("Putting the search words in English")
        try:
            names = parse_json(self.client.chat(msg, self.model, purpose="english_queries", temperature=0.0,
                                                max_tokens=1024, thinking=False))
        except Exception as e:  # a failed translation must not stop the interview: keep the queries as they are
            self.translation_error = f"{type(e).__name__}: {e}"  # kept in the transcript
            self._end(step, "failed", "kept the words as they are")
            return
        self._end(step)
        if not isinstance(names, dict):
            return
        for qs in lists:
            out = []
            for q in qs:
                en = names.get(q)
                en = [en] if isinstance(en, str) and en.strip() else []
                # names go to SAM 3, which understands only English: the original word is dropped, not kept after
                # (seen 2026-09-30: "names": ["maceta", "flower pot"] despite the prompt)
                keep = [] if en and any(qs is n for n in names_lists) else [q]
                for t in en + keep:
                    if t.strip().lower() not in {o.lower() for o in out}:
                        out.append(t.strip())
            qs[:] = out
        renamed = {}
        for i, term in id_terms.items():
            en = names.get(term)
            new = re.sub(r"[^a-z0-9]+", "_", en.lower()).strip("_") if isinstance(en, str) else ""
            if new and new != i and new not in ids and new not in renamed.values():
                renamed[i] = new
        if renamed:
            _rename_classes(spec, renamed)
            self._note("Class ids are always in English: " + ", ".join(f"{a} -> {b}" for a, b in renamed.items())
                       + ". Use these ids from now on.")

    def _research(self, question: str) -> None:
        """A web search for the model; logged with the model calls so the job's usage shows it."""
        from .research import ResearchError
        if self.researcher is None:
            self._note("research is not available here: use what you know and say when a number is a guess.")
            return
        if self.researched >= MAX_RESEARCH:
            self._note(f"No more research in this interview (limit {MAX_RESEARCH}): use what you found.")
            return
        self.researched += 1
        step = self._step(f"Searching the web: {question}")
        try:
            found = self.researcher(question)
        except ResearchError as e:
            self._end(step, "failed", str(e))
            self._note(f"research failed ({e}): use what you know and say when a number is a guess.")
            return
        self._end(step, detail=f"{len(found['sources'])} sources")
        log = getattr(self.client, "usage_log", None)
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as f:
                f.write(json.dumps({"model": "tavily/search", "purpose": "research", "prompt_tokens": 0,
                                    "completion_tokens": 0}) + "\n")
        sources = "\n".join(f"- {x['title']} ({x['url']}): {x['content']}" for x in found["sources"])
        self._note(f"[Web research: {question}]\nAnswer: {found['answer'] or 'none'}\nSources:\n{sources}\n"
                   "Trust numbers the sources agree on; the user did not see this.")

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._save()

    def _save(self) -> None:
        """The whole conversation with the model, rejections included, to see afterwards why a turn took long."""
        if self.transcript is not None:
            self.transcript.parent.mkdir(parents=True, exist_ok=True)
            extra = [{"role": "log", "translation_error": self.translation_error}] if self.translation_error else []
            self.transcript.write_text(json.dumps(self.messages + extra, ensure_ascii=False, indent=1))


def run(client, model: str, categories: CategoryIndex, ask: Callable[[str], str], tell: Callable[[str], None],
        max_turns: int = 60, scene_check=None, photo_ranker=None, uploads: Path | None = None,
        scene_profiler=None) -> spec_mod.Spec | None:
    """Terminal front end: ask() gets the user's answer to a prompt, tell() shows text."""
    session = Session(client, model, categories, max_turns, scene_check, photo_ranker, uploads, scene_profiler)
    turn = session.reply(ask(FIRST_QUESTION + "\n> "))
    while True:
        if turn.kind == "say":
            turn = session.reply(ask(turn.text + "\n> "))
        elif turn.kind == "pick":
            tell(turn.text)
            cands = turn.request["candidates"]
            for i, c in enumerate(cands, 1):
                tell(f"  {i}. {Path(c['file']).name} ({c['score']:.2f}) {c['reason']}")
            answer = ask("Numbers to use (Enter = all), or answer in words > ").strip()
            nums = re.findall(r"\d+", answer)
            if not answer or (nums and not re.sub(r"[\d\s,]", "", answer)):
                picked = [cands[int(n) - 1]["file"] for n in nums if 0 < int(n) <= len(cands)] if nums \
                    else [c["file"] for c in cands]
                turn = session.choose(picked, turn.request["class"])
            else:
                turn = session.reply(answer)
        elif turn.kind == "photos":
            answer = ask(turn.text + "\n(folder with the photos, or answer in words) > ").strip()
            folder = Path(answer).expanduser()
            n = count_photos(folder) if answer and folder.is_dir() else 0
            turn = session.add_photos(str(folder.resolve()), n, turn.request["for"], turn.request.get("class")) \
                if n else session.reply(answer)
        elif turn.kind == "confirm":
            tell(turn.text)
            turn = session.reply(ask(CONFIRM + " > "))
        elif turn.kind == "done":
            return turn.spec
        else:
            tell(turn.text)
            return None


def scene_profiler(client, model: str):
    """(folder, classes) -> (proposed scene fields, summary) with a vision model."""
    from . import scene_profile

    def profile(folder: str, classes: str, progress=None, notes=None):
        return scene_profile.profile_folder(Path(folder), classes, client, model, progress=progress, notes=notes)
    return profile
