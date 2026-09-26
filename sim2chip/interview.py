"""Interview: a conversation with Nemotron (Token Factory) that turns the user's needs into a valid spec.

Every model turn is one JSON action:
  {"say": "..."}                         ask the user something (or explain)
  {"search": ["cow", "cattle"]}          look up 3D model categories (Objaverse-LVIS) and their sizes
  {"spec": {...}}                        propose the full spec
A proposed spec is validated; errors go back to the model. A valid spec is summarized and saved only
after the user confirms it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import hardware, spec as spec_mod

SYSTEM = """You are the setup assistant of EyeForge (eyeforge.app). EyeForge trains a small vision model
(it classifies or counts) that runs on the user's own device, sized to it (from a $10 ESP32-CAM to a Jetson or a PC), from synthetic
images rendered with 3D models plus a few real photos from the user. Microcontroller users get a model ready
to flash; users without a device just download it.
Your job: interview the user and write the job spec. Be brief and friendly, ask one or two questions at a time,
in the user's language.

Find out:
- What the camera must recognize and what counts as "nothing". On a microcontroller 2 to 5 coarse classes
  work best: suggest merging classes it cannot tell apart at low resolution. Bigger devices allow more
  and finer classes.
- The task. "roi_classification" (default) says which class the picture shows, or none. "task": "count"
  says how many objects of each class are in the picture and where: use it when the user wants to count, or
  when several objects appear at once (a balcony with pots, a flock). For counting, photos always go in
  user_data.raw: the teacher marks every object and the user fixes the marks in review. Counting small or far
  objects needs pixels: on a microcontroller (96 px) only objects that fill a good part of the picture can be
  counted; say so if the user expects more.
- Where the camera is: environment, mounting height, typical distance, lighting and weather.
- Target hardware, one of these profiles:
{hardware}
  If the user has no device, has not decided, or only wants to test on a phone or PC, use "generic".
- Whether they have photos and the folder path. "labeled" ONLY when the folder has one subfolder per class
  (class_map maps each class to its subfolder names). Unsorted photos, even if every photo shows the object,
  are "raw": the teacher model labels them and also finds the ones without the object (background class).
  Folder layouts are checked when you send the spec.
  A few real photos improve accuracy a lot compared with synthetic images only; say so. For an honest score,
  suggest a separate labeled test folder (user_data.test) with photos taken on another day or place.
- Real-world size of each class (you propose typical sizes, the user confirms).

3D models, two sources. Always use "search" first; it reports both:
1. Curated Objaverse-LVIS categories ("assets": {{"source": "objaverse_lvis", "categories": [...]}}). Prefer
   them when one really matches the object and has at least ~10 models. Only use names returned by search.
2. Live search over ~800k Sketchfab models ("assets": {{"source": "search", "queries": ["wild boar", "boar"]}}),
   for anything the curated list lacks. Give 1 to 3 English queries, most specific first; aim for 15+ models
   in total, since quality control will discard some (skulls, cartoons, fragments).
If an object still has too few models, or you would substitute it with something similar or merge it into
another class, say so plainly and ASK the user whether that matters for their goal (for some goals it is
critical, for others irrelevant). Offer the options: merge classes, use a substitute, rely on their own
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
(brick, plaster, stone, concrete, wood) and "railings" (bars, solid). As soon as you have a folder of the
user's photos, describe them: {{"describe_scene": {{"folder": "<folder>", "classes": "<what to find, in
English>"}}}}. You get back what the photos show and the proposed fields: tell the user in one or two plain
sentences what you saw and ask whether the images should look like that; then use the fields (the user may
change the shares). Objects on walls or ceilings are not simulated: say so if that is what they need.
Props: scenes with only the target look empty and the model then learns "anything in the picture is a target".
Always propose 2 to 5 background props typical of the place ("props": a building site gets traffic cones,
pallets, barriers, dirt piles; a forest gets rocks, logs, bushes), each with live search queries and a
real size. They are found live like a class and never labeled. They must not look like any class (no cars as
props for a vehicle camera); ask the user if a prop could be confused with what they want to detect.

Spec format (JSON, defaults shown; omit a field to keep its default):
{schema}
Notes: size_m.measure is "length" (long side from above), "width" (short side from above) or "height";
choose the one that varies least across the class. assets.count is the number of models per class.
user_data.labeled / user_data.raw / user_data.test are folder paths; class_map maps each class id to the subfolder names of
the labeled folder. Keep qa.mode "auto".

How to talk: ask for things when they become relevant, not all at the start, and skip what the user
already told you. Two moments need photos, and you ask for them right then with the "photos" action (the
user gets an upload box in the chat; they may also decline in words):
- After a search, when a class has few usable models (under ~10), the matches are not really the object,
  or it is the user's own specific object: ask whether some of their photos show it clearly, to build 3D
  models from them. Say what makes a good one: the whole object, big in the frame, on a background that
  does not distract (plain wall or floor), ideally from 2 or 3 angles; 1 to 5 photos are enough.
  {{"photos": {{"for": "3d", "class": "<class id>", "say": "<your question>"}}}}
- When you talk about the camera: photos from the real camera or place, unsorted, to train and test.
  {{"photos": {{"for": "training", "say": "<your question>"}}}}
Before asking for 3D photos, check the photos the user already gave (uploaded or a folder path they wrote):
some may already be good. {{"pick_photos": {{"class": "<class id>", "query": "<English name>", "folder":
"<that folder>", "say": "<short intro>"}}}} ranks them with a vision model and shows the user the best ones
to confirm; you are told if none fit, and then you ask for new photos.
When photos arrive or are chosen you are told their folder: use it in the spec ("3d": that class's
assets.photos plus an English "queries" name; "training": user_data.raw).

Reply ONLY with one JSON object per turn: {{"say": "..."}}, {{"search": ["..."]}}, {{"photos": {{...}}}},
{{"pick_photos": {{...}}}}, {{"describe_scene": {{...}}}} or {{"spec": {{...}}}}."""

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
                         schema=schema)


class CategoryIndex:
    """Objaverse-LVIS category names with their model counts, searched by substring."""

    def __init__(self, counts: dict[str, int] | None = None, cache_dir: Path | None = None, live=None):
        if counts is None:
            import objaverse
            if cache_dir:
                objaverse.BASE_PATH = str(cache_dir)
                objaverse._VERSIONED_PATH = str(cache_dir / "hf-objaverse-v1")
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
PICK_MAX = 8


def photo_ranker(client, model: str, limit: int = 80, workers: int = 8):
    """Vision-model ranking of a folder's photos for building 3D models of a query (Token Factory)."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image, ImageOps

    from .llm import image_part, parse_json
    from .stages.ingest import IMAGE_EXT

    def rank(folder: str, query: str) -> list[dict]:
        files = sorted(f for f in Path(folder).rglob("*") if f.suffix.lower() in IMAGE_EXT)[:limit]
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
                    v = parse_json(client.chat(msg, model, purpose="pick_photos", temperature=0.0, max_tokens=1024))
                    ok = v.get("shows") and v.get("whole")
                    return {"file": str(f.resolve()), "score": float(v.get("score", 0)) if ok else 0.0,
                            "reason": str(v.get("reason", ""))}
                except (ValueError, TypeError, AttributeError):
                    return {"file": str(f.resolve()), "score": 0.0, "reason": "no valid answer"}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                out = list(pool.map(one, enumerate(files)))
        return sorted(out, key=lambda c: -c["score"])
    return rank


def count_photos(folder: Path) -> int:
    from .stages.ingest import IMAGE_EXT
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXT)


def check_photos(s: spec_mod.Spec) -> str | None:
    """Do the photo folders the spec names exist and have the layout it assumes? Only absolute paths are
    checked (relative ones are uploads that live in the job folder)."""
    from .stages.ingest import IMAGE_EXT
    ud = s.user_data

    def photos_in(d: Path) -> int:
        return sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT) if d.is_dir() else 0

    for key in ("labeled", "raw", "test"):
        rel = getattr(ud, key)
        if not rel or not Path(rel).is_absolute():
            continue
        d = Path(rel)
        if not d.is_dir():
            return f"user_data.{key}: folder {d} does not exist; ask the user for the right path"
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
             f"  classes: {', '.join(s.labels)} (background: {s.background_class})"]
    for c in s.classes:
        size = f"{c.size_m.measure} {c.size_m.range[0]}-{c.size_m.range[1]} m" if c.size_m else "no 3D size"
        a = c.assets
        cats = "no 3D models" if not a else ", ".join(a.categories) if a.source == "objaverse_lvis" else \
            f"search {a.queries}" if a.source == "search" else f"made from photos ({a.queries[0] if a.queries else '?'})"
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
    lines.append(f"  photos: {', '.join(p for p in photos if p) or 'none (synthetic only)'}")
    return "\n".join(lines)


FIRST_QUESTION = "Describe what your camera should see / Describe qué debe ver tu cámara:"
CONFIRM = "Is this right? / ¿Es correcto? (yes/sí, or tell me what to change / o dime qué cambiar)"


@dataclass
class Turn:
    kind: str                       # "say" (a question), "photos" (asks for photos), "pick" (proposes some of
    text: str                       # the user's photos), "confirm" (spec proposed), "done" or "failed"
    spec: spec_mod.Spec | None = None
    request: dict | None = None     # photos: {"for", "class"}; pick: {"for", "class", "query", "candidates"}


class Session:
    """One interview, driven one user message at a time (terminal, web or any other front end)."""

    def __init__(self, client, model: str, categories: CategoryIndex, max_turns: int = 60, scene_check=None,
                 photo_ranker=None, uploads: Path | None = None, scene_profiler=None):
        self.client, self.model, self.categories, self.max_turns = client, model, categories, max_turns
        self.scene_check = scene_check  # scene -> error message or None (live Poly Haven check)
        self.photo_ranker = photo_ranker  # (folder, query) -> [{"file", "score", "reason"}], best first
        self.uploads = uploads            # where chosen photos are gathered
        self.scene_profiler = scene_profiler  # (folder, classes) -> (scene fields, summary)
        self.candidates: list[dict] = []  # the last photos proposed to the user
        self.messages = [{"role": "system", "content": system_prompt()}]
        self.proposed: spec_mod.Spec | None = None
        self.finished = False

    def reply(self, user_text: str) -> Turn:
        if self.finished:
            raise RuntimeError("the interview is over")
        if self.proposed is not None:
            if is_yes(user_text):
                self.finished = True
                return Turn("done", "", self.proposed)
            self.proposed = None
            user_text = f"The user wants changes: {user_text}"
        self.messages.append({"role": "user", "content": user_text})
        return self._advance()

    def add_photos(self, folder: str, count: int, purpose: str, cls: str | None = None) -> Turn:
        """The user uploaded (or pointed to) photos, asked for or not: tell the model and continue."""
        if self.finished:
            raise RuntimeError("the interview is over")
        self.proposed = None
        what = f"to build 3D models of class {cls}" if purpose == "3d" else "from their camera, unsorted"
        use = f"classes[{cls}].assets.photos" if purpose == "3d" else "user_data.raw"
        self.messages.append({"role": "user", "content": f"[The user provided {count} photos {what}, in folder "
                                                         f"{folder}. Use it as {use}.]"})
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

    def _pick(self, req: dict) -> Turn | None:
        """Rank the user's photos for 3D; a Turn for the user, or None after telling the model why not."""
        folder, query, cls = Path(str(req.get("folder", ""))).expanduser(), req.get("query"), req.get("class")
        if self.photo_ranker is None or not query or not cls:
            self._note("pick_photos needs class, query and folder, and a photo ranker; ask for photos instead.")
            return None
        if not folder.is_dir():
            self._note(f"pick_photos: folder {folder} does not exist. Ask the user where the photos are.")
            return None
        ranked = self.photo_ranker(str(folder), query)
        self.candidates = [c for c in ranked if c["score"] >= PICK_MIN_SCORE][:PICK_MAX]
        if not self.candidates:
            self._note(f"pick_photos: none of the {len(ranked)} photos in {folder} shows a clear {query} "
                       "(whole, big, clean background). Ask the user for such photos with the photos action.")
            return None
        return Turn("pick", str(req.get("say") or f"Some of your photos could work for 3D models of {cls}:"),
                    request={"for": "3d", "class": cls, "query": query, "candidates": self.candidates})

    def _describe(self, req: dict) -> None:
        """Profile the user's photos and hand the proposed scene fields to the model."""
        folder = Path(str(req.get("folder", ""))).expanduser()
        if self.scene_profiler is None:
            self._note("describe_scene is not available here; keep the default open ground scene.")
        elif not folder.is_dir():
            self._note(f"describe_scene: folder {folder} does not exist.")
        else:
            try:
                fields, text = self.scene_profiler(str(folder), str(req.get("classes") or "the objects"))
            except (OSError, ValueError, RuntimeError) as e:
                self._note(f"describe_scene failed ({e}); keep the default open ground scene.")
                return
            if not fields.get("contexts"):
                self._note("describe_scene: no photo could be described; keep the default open ground scene.")
                return
            self._note(f"[Scene profile of the user's photos] {text}\nProposed scene fields (template generic): "
                       f"{json.dumps(fields)}. Tell the user what you saw and ask if the images should look like "
                       "that before you use them.")

    def _advance(self) -> Turn:
        from .llm import parse_json

        for _ in range(self.max_turns):
            text = self.client.chat(self.messages, self.model, purpose="interview", temperature=0.3, max_tokens=4096)
            self.messages.append({"role": "assistant", "content": text})
            try:
                action = parse_json(text)
            except ValueError:
                self._note("Reply with exactly one JSON object as instructed.")
                continue
            if not isinstance(action, dict):
                self._note("Reply with exactly one JSON object as instructed.")
                continue
            if "say" in action:
                if not str(action["say"]).strip():
                    self._note("Your say message was empty. Ask the user something or propose the spec.")
                    continue
                return Turn("say", str(action["say"]))
            if isinstance(action.get("pick_photos"), dict):
                turn = self._pick(action["pick_photos"])
                if turn:
                    return turn
                continue
            if isinstance(action.get("describe_scene"), dict):
                self._describe(action["describe_scene"])
                continue
            if isinstance(action.get("photos"), dict):
                req = action["photos"]
                purpose = req.get("for") if req.get("for") in ("3d", "training") else "training"
                return Turn("photos", str(req.get("say") or "Do you have photos for this?"),
                            request={"for": purpose, "class": req.get("class")})
            if "search" in action:
                q = action["search"] if isinstance(action["search"], list) else [action["search"]]
                found = self.categories.search(q)
                self._note("search results (category: number of models): "
                           + json.dumps(found or "nothing found, try other words"))
            elif "spec" in action:
                try:
                    s = spec_mod.from_dict(action["spec"])
                    unknown = [cat for c in s.classes if c.assets and c.assets.source == "objaverse_lvis"
                               for cat in c.assets.categories if cat not in self.categories.counts]
                    if unknown:
                        raise spec_mod.SpecError(f"categories not found in Objaverse-LVIS: {unknown}, use search")
                    problem = (self.scene_check(s.scene) if self.scene_check else None) or check_photos(s)
                    if problem:
                        raise spec_mod.SpecError(problem)
                except spec_mod.SpecError as e:
                    self._note(f"The spec is invalid: {e}. Fix it and send it again.")
                    continue
                self.proposed = s
                return Turn("confirm", summary(s), s)
            else:
                self._note("Unknown action. Use say, search or spec.")
        self.finished = True
        return Turn("failed", "The interview did not converge; nothing was saved.")

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


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

    def profile(folder: str, classes: str):
        return scene_profile.profile_folder(Path(folder), classes, client, model)
    return profile
