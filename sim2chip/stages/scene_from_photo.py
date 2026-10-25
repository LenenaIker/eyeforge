"""Synthetic frames from the user's own place (photo2scene, docs/plan_photo2scene.md): the dataset's scenes built
by earlier jobs (input/scenes.json, from the library) are composed again with this job's seed; with none, one is
built from a training photo first, and the library keeps it for the next jobs. The photo's route (roadmap C.3): the
vision model's observation and rules in code (photo2scene.place.observe) send one working surface seen from above to
the support route, a street or a room to the place routes (photo2scene.place), and anything else (night, a square
without a template) nowhere; a photo that does not fit is noted as failed with why, not tried again, and the next
one is tried. Classes that are surfaces, not things to move around, take the surface route (the vision model says
so when it rates the photos). Counting views hold any of the classes; classifying views hold one class each (or
none, for the background class), and every class gets its own scene, from a photo of it, on every route.

Render collects the frames (the scene template is off, base.TEMPLATE). The scene photos stay out of the test
(ingest.scene_photos), and the new one is picked among the photos ingest put in training.

Output: scene_from_photo/scenes.json  the scenes built here, for the library (library.take_scenes)
        scene_from_photo/frames.json  [{"file", "scene", "boxes": [[class, x0, y0, x1, y1], ...]}] (pixels),
                                      classifying also "class": the view's one class
        scene_from_photo/<scene>/     each scene's folder, its views in isaac/render_NN.png
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import statistics
from pathlib import Path

from ..workspace import cache_key, content_key
from .base import Context, Stage

VIEWS = int(os.environ.get("SIM2CHIP_SCENE_VIEWS", 20))  # new views per job (classifying, per class), shared by its
                     # scenes (the variable: to measure doses, D.1)
SCENES_PER_JOB = int(os.environ.get("SIM2CHIP_SCENES", 1))  # scenes composed per job (classifying, per class)
# The things of every training photo shuffled into every scene, not only the scene photo's own (one photo shows one
# or two bottles; roads 2026-10-14: "cars, not street, are what is missing"). Builds a scene from every training
# photo to take its things. Support route only. Off until measured (2026-10-16): SIM2CHIP_SCENE_POOL=1.
POOL = os.environ.get("SIM2CHIP_SCENE_POOL") == "1"
BUILD_TRIES = 2      # a scene from up to this many photos: the next one when a photo makes none. The vision model rated
                     # 9/10 a photo the teacher called sun cream that showed only a figurine (2026-10-11)
PICK_AMONG = 6       # typical photos the vision model rates for the scene; the cleanest is built (roadmap B.3)
PICK_S = 120         # the rating's deadline: photos without an answer by then keep their place after the rated
PICK_PROMPT = """We will rebuild the place in this photo in 3D and move the {targets} around in it to make training
images, so we want the cleanest photo of the place. Judge only this: every {targets} whole in the frame (not cut by
the photo's edges, not hidden behind something), none overlapping or touching another, few other things around them,
and the surface they stand on clearly visible. Answer only with JSON:
{{"count": <how many {targets} you see, 0 if none>, "score": <0 to 10, 10 the cleanest; 0 with none>,
 "why": "<one short sentence>",
 "things": <true when the {targets} are separate things one could pick up and move (a bottle, a cup), false when
  they are surfaces or areas (grass, a road, a floor)>}}"""
SURFACES = "your classes are surfaces or areas, not things to move around; EyeForge cannot rebuild those yet"
NO_SURFACE_VIEWS = "no new view of your surfaces kept its class"
# The surface route (photo2scene.surface) is off until it helps: the lawn with re-framed views and new borders scored
# 0.771 against 0.814 with the photos alone (roadmap B.5; the user's decision, 2026-10-11). Off, surfaces get no
# scene and say why.
# SIM2CHIP_SURFACE_ROUTE=1 turns it on to measure it.
SURFACE_ROUTE = os.environ.get("SIM2CHIP_SURFACE_ROUTE") == "1"
# The street route is on (the user's decision, 2026-10-13) although on the user's roads (counting cars) the likeness
# filter kept none of the street's views, 0.662 against 0.651 with the photos alone (roadmap C.6): the filter keeps
# bad views out. SIM2CHIP_STREET_ROUTE=0 turns it off; off, a street photo makes no scene and says why.
STREET_ROUTE = os.environ.get("SIM2CHIP_STREET_ROUTE", "1") == "1"
NO_STREETS = "streets do not make training images good enough yet"


ROUTES = {"support": "support", "village_street_day": "street", "indoor_floor": "room"}  # flow_route.py's routes


def _not_kept(folder: str, names: list[str]) -> set[str]:
    """What the library does not keep of a scene: its logs and one job's views (a place's populated/, the support
    route's isaac/; a place's own isaac/ folders are the place)."""
    out = {n for n in names if n in ("logs", "populated")}
    if (Path(folder) / "plan.json").exists():  # the support route's scene folder
        out |= {"isaac"} & set(names)
    return out


def _link(src, dst):
    """A scene's file into the job: a hard link when it can (the files are hundreds of MB), else a copy. Every
    step that changes a file writes a new one, so the source never changes under the link."""
    import os
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@contextlib.contextmanager
def _building(work: Path):
    """One build of a scene folder at a time on this machine (a file lock next to it)."""
    import fcntl
    work.parent.mkdir(parents=True, exist_ok=True)
    with open(work.with_name(work.name + ".lock"), "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def no_route(spec) -> str | None:
    """Why no photo2scene route fits this spec, in the user's words, or None when one does. The photos tell
    the route (a surface from above, a street or a room: SceneFromPhoto.run), and whether the classes are things or
    surfaces (SceneFromPhoto._pick)."""
    if not spec.synthetic:
        return "the setup asks for your photos only"
    if not spec.generation.photo_scenes:
        return "rebuilding your place is turned off in the setup"
    if not spec.user_data.raw:
        return "rebuilding your place needs unsorted photos of it"
    return None


class SceneFromPhoto(Stage):
    name = "scene_from_photo"
    version = "10"  # 2: the vision model picks the cleanest photo; 3: classifying too, one class per view;
    # 4: surfaces re-framed in 3D; 5: streets and rooms; 6: classifying in them, every view of a place kept;
    # 7: the street route off (C.6); 8: on again (2026-10-13); 9: NVIDIA trees, lots, the photo's ground, light
    # per object, skies fetched where data/ is not (2026-10-14); 10: certificates for the steps, grass texture, lit
    # template ground, matte ground (2026-10-14)
    runtime = "cpu"   # it runs each step in the Python that has its models (photo2scene.PYTHONS)
    gpu = "rtx"
    max_hours = 2
    reads_input = True
    uses_cache = True
    needs_llm = True
    uses = ("classes", "camera", "generation", "task", "user_data")
    after = ("ingest", "label")
    skippable = "with photos"
    results = ("*.json", "*/**/*")  # the scenes built here go to the library whole

    def enabled(self, spec) -> bool:
        return no_route(spec) is None

    def run(self, ctx: Context) -> dict:
        from .. import photo2scene as p2s
        from ..library import SCENES
        from ..photo2scene import place
        from ..tasks import task_of
        from .label import phrases_of

        out = ctx.job.stage_dir(self.name)
        out.mkdir(parents=True, exist_ok=True)
        missing = p2s.missing_pythons()
        if missing:
            raise RuntimeError(f"photo2scene needs {', '.join(missing)} on this machine")
        given = json.loads((ctx.job.input / SCENES).read_text()) if (ctx.job.input / SCENES).exists() else {}
        phrases = phrases_of(ctx.spec, ctx.cache, ctx.log)
        targets = [ps[0] for ps in phrases.values() if ps]
        usage = ctx.job.logs / "llm_usage.jsonl"
        built, frames = [], []
        # (scene id, flow, group): every ready scene of this job; the first SCENES_PER_JOB of a group are composed.
        # Counting, one group; classifying, a library scene is its class's own.
        ready = []
        points = task_of(ctx.spec).points
        for s in given.get("ready", []):
            d = out / s["id"]
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(ctx.job.input / s["dir"], d, copy_function=_link)
            ctx.log(f"scene {s['id']} (photo {s['photo_id']}) from the library")
            ready.append((s["id"], p2s.Flow(d, usage, ctx.log), 0 if points else ("library", s["id"])))
        if not ready or points and (POOL or len(ready) < SCENES_PER_JOB):
            tried = set(given.get("failed_photos", [])) | {s.get("photo_id") for s in given.get("ready", [])}
            picks, votes = [], []
            for cls, group in self._candidates(ctx, tried):
                # Classifying, a class's photos are rated for that class: asked about every class, the model gave
                # 10/10 to a sun cream photo (by the teacher) with only a figurine in it (2026-10-11).
                ranked, things = self._pick(ctx, group, phrases[cls][:1] if cls in phrases else targets)
                picks.append(ranked)
                votes += things
            if votes and sum(votes) * 2 < len(votes):
                ctx.log(f"the vision model sees surfaces, not things, in {len(votes) - sum(votes)} of {len(votes)} "
                        "photos")
                picks = []
                if task_of(ctx.spec).points or not SURFACE_ROUTE:
                    built.append({"photo_id": None, "route": "support", "status": "failed", "why": SURFACES})
                else:
                    views = self._surfaces(ctx, out, usage)
                    frames += views
                    built.append({"photo_id": None, "route": "surface", "status": "ready" if views else "failed",
                                  **({} if views else {"why": NO_SURFACE_VIEWS})})
            want = None if POOL else SCENES_PER_JOB - (len(ready) if points else 0)
            for gi, (pid, photo) in self._tries(picks, built, want):
                # Built in the cache (by photo and classes): a run that fails later, or is tried again, does not
                # pay the erase and the meshes twice.
                work = ctx.cache.root / "scenes" / f"{content_key(photo)}-{cache_key(targets)}"
                ctx.log(f"building a scene from photo {pid} ({photo.name})")
                route = "none"  # out of scope until the observation says otherwise
                try:
                    flow = p2s.Flow(work, usage, ctx.log)
                    with _building(work):  # another run on this machine (the scene GPU) may build it too: it waits,
                        # then finds every step's outputs and goes on
                        route = ROUTES.get(place.observe(flow, photo)["route"], "support")
                        ctx.log(f"photo {pid} takes the {route} route")
                        if route == "street" and not STREET_ROUTE:
                            raise p2s.Stop(NO_STREETS)
                        summary = (p2s.build if route == "support" else place.build)(flow, photo, targets)
                except p2s.Stop as e:
                    ctx.log(f"photo {pid} makes no scene: {e}")
                    built.append({"photo_id": pid, "route": route, "status": "failed", "why": str(e)})
                    continue
                d = out / f"photo_{pid}"
                shutil.rmtree(d, ignore_errors=True)
                shutil.copytree(work, d, ignore=_not_kept, copy_function=_link)
                flow = p2s.Flow(d, usage, ctx.log)
                cards = self._cards(d)
                objects = [{**a, "concept": self._class_of(a["name"], phrases) or a["name"],
                            **({"card": cards[a["index"]]} if a["index"] in cards else {})} for a in summary["objects"]]
                where = summary["plan"]["support"] if route == "support" else route
                built.append({"photo_id": pid, "route": route, "status": "ready", "dir": d.name,
                              "name": f"{where.capitalize()} from photo {pid}", "thumb": f"{d.name}/photo.png",
                              "info": {"plan": summary["plan"]} if route == "support" else
                              {k: summary[k] for k in ("template", "hdri_group")}, "objects": objects})
                ready.append((d.name, flow, gi))
        pool = p2s.pool_of([f.d for _, f, _ in ready]) if POOL else None
        taken: dict = {}
        for sid, flow, group in ready:
            taken.setdefault(group, [])
            if len(taken[group]) < SCENES_PER_JOB:
                taken[group].append((sid, flow))
        for scenes in taken.values():
            for k, (sid, flow) in enumerate(scenes):  # the group's views shared by its scenes
                views = VIEWS // len(scenes) + (k < VIEWS % len(scenes))
                if views:
                    frames += self._compose(ctx, flow, sid, phrases, views, pool)
        (out / SCENES).write_text(json.dumps(built, indent=1))
        (out / "frames.json").write_text(json.dumps(frames, indent=1))
        boxes = sum(len(f["boxes"]) for f in frames)
        ctx.log(f"{len(frames)} frames from the user's place, {boxes} objects")
        return {"frames": len(frames), "objects": boxes, "built": sum(b["status"] == "ready" for b in built)}

    @staticmethod
    def _surfaces(ctx: Context, out: Path, usage: Path) -> list[dict]:
        """Classifying surfaces (photo2scene.surface): the training photos re-framed in 3D and joined in pairs, each
        view labelled by the teacher with what it learnt from the user's corrections. Built in the cache by photos
        and settings; the views are linked into out/surface."""
        from .. import likeness, photo2scene as p2s, teacher
        from ..photo2scene import surface
        from .ingest import train_photos
        photos = [(f, f.parent.name) for f in train_photos(ctx)]
        learned = ctx.job.stage_dir("label") / "teacher.json"
        centre = json.loads(learned.read_text()).get("centres") if learned.exists() else None
        if not photos or not centre:
            ctx.log("no training photos with the teacher's learnt classes: no views of your surfaces")
            return []

        def classify(files):
            emb = likeness.embeddings(files, ctx.cache.root, ctx.log)
            return [teacher.nearest(emb[f], centre) if f in emb else (None, 0.0) for f in files]
        cam, gen = ctx.spec.camera, ctx.spec.generation
        key = cache_key([[content_key(f), c] for f, c in photos] + [cam.pitch_deg, cam.height_m, gen.seed, gen.frames])
        work = ctx.cache.root / "surfaces" / key
        ctx.log(f"re-framing {len(photos)} training photos of your surfaces in 3D")
        views = surface.build(p2s.Flow(work, usage, ctx.log), photos, gen.frames, cam.pitch_deg, cam.height_m,
                              gen.seed, classify, ctx.spec.user_data.label_min_confidence, ctx.log)
        d = out / "surface"
        shutil.rmtree(d, ignore_errors=True)
        (d / "views").mkdir(parents=True)
        (d / "mix").mkdir()
        for v in views:
            _link(work / v["file"], d / v["file"])
        return [{"file": f"surface/{v['file']}", "scene": "surface", "boxes": [], "class": v["class"]} for v in views]

    @staticmethod
    def _tries(picks: list[list[tuple[int, Path]]], built: list[dict], want: int | None = 1):
        """The photos to build from, group by group, with the group's index: until `want` of a group's photos made
        a scene (all of them when None), trying at most BUILD_TRIES - 1 more photos than that (each try is recorded
        in built, failed ones too, so the library does not try them again)."""
        for gi, group in enumerate(picks):
            made = 0
            for item in group[:len(group) if want is None else max(want + BUILD_TRIES - 1, 0)]:
                if want is not None and made >= want:
                    break
                n = len(built)
                yield gi, item
                if len(built) > n and built[-1]["status"] == "ready":
                    made += 1

    def _compose(self, ctx: Context, flow, scene: str, phrases, views: int = VIEWS,
                 pool: list[dict] | None = None) -> list[dict]:
        """This job's views of one scene, with the boxes of the user's classes only. Classifying, each view shows
        one class (its things of other classes left out), or none in a share of views when there is a background
        class; a view whose one class is out of sight is left out. pool: things of the job's other photos
        (photo2scene.pool_of), on the support route."""
        from ..photo2scene import compose, place
        from ..tasks import task_of
        groups, empty = None, 0.0
        if not task_of(ctx.spec).points:
            cat = flow.d / "assets.json"
            names = [a["name"] for a in json.loads(cat.read_text())["assets"]] if cat.exists() else []
            groups = {n: cls for n in names if (cls := self._class_of(n, phrases))}
            empty = ctx.spec.generation.empty_ratio if ctx.spec.background_class else 0.0
        out = []
        seed = ctx.spec.generation.seed
        got = (place.compose(flow, seed, groups, empty) if (flow.d / "extended").is_dir()  # a street or a room
               else compose(flow, views, seed, groups, empty, pool))
        for v in got:
            boxes = [[cls, *b["box"]] for b in v["boxes"] if b.get("role") == "target"
                     and (cls := self._class_of(b["name"], phrases))]
            frame = {"file": f"{scene}/{v['file']}", "scene": scene, "boxes": boxes}
            if groups is not None:
                seen = {b[0] for b in boxes}
                frame["class"] = seen.pop() if len(seen) == 1 else None if seen else ctx.spec.background_class
                if frame["class"] is None:
                    continue
            out.append(frame)
        return out

    @staticmethod
    def _cards(d: Path) -> dict[int, str]:
        """A picture per erased thing for its library card: the photo cropped around its mask, on white."""
        import numpy as np
        from PIL import Image
        photo = np.asarray(Image.open(d / "photo.png").convert("RGB"))
        masks = np.load(d / "assets.npz")["masks"]
        (d / "cards").mkdir(exist_ok=True)
        out = {}
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            pad = int(0.08 * max(np.ptp(xs), np.ptp(ys))) + 2
            y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
            crop = np.where(m[y0:y1, x0:x1, None], photo[y0:y1, x0:x1], 255).astype(np.uint8)
            Image.fromarray(crop).save(d / "cards" / f"{i:03d}.png")
            out[i] = f"cards/{i:03d}.png"
        return out

    @staticmethod
    def _class_of(name: str, phrases: dict[str, list[str]]) -> str | None:
        """The class an erased thing belongs to: its name is one of the class's phrases, or ends with the same
        noun and has the phrase's other words (photo2scene.steps.flow_assets.is_target)."""
        def words(t):
            return [w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w
                    for w in re.findall(r"[a-z0-9]+", t.lower())]
        n = words(name)
        for cls, ps in phrases.items():
            for p in ps:
                t = words(p)
                if n and t and n[-1] == t[-1] and set(t) <= set(n):
                    return cls
        return None

    @staticmethod
    def _pick(ctx: Context, candidates: list[tuple[int, Path]],
              targets: list[str]) -> tuple[list[tuple[int, Path]], list[bool]]:
        """The first PICK_AMONG candidates rated by the vision model for a clean scene (objects whole, apart, little
        clutter), the best first; the order of _candidates breaks ties and places the photos it could not rate. The
        typical count alone picked photos with cut or stacked bottles (2026-10-11). Also the model's word on each
        rated photo: are the classes things (True) or surfaces? Ratings are kept in the cache by photo and
        classes."""
        from ..llm import LOCAL_VLM, PHOTO_CALL, REASONING_TOKENS, Client, LLMError, ask_json, image_part, map_within
        from PIL import Image, ImageOps

        head = candidates[:PICK_AMONG]
        if not head:
            return candidates, []
        store = ctx.cache.root / "scene_pick"
        store.mkdir(parents=True, exist_ok=True)
        what = " or ".join(targets) or "objects"
        client = None

        def rate(item):
            nonlocal client
            pid, photo = item
            kept = store / f"{content_key(photo)}-{cache_key(targets)}.json"
            if kept.exists():
                return json.loads(kept.read_text())
            client = client or Client(usage_log=ctx.job.logs / "llm_usage.jsonl")
            small = store / f"{content_key(photo)}-{pid}.jpg"  # its own: the same photo twice rates in two threads
            with Image.open(photo) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((768, 768))
                im.save(small, quality=88)
            msg = [{"role": "user", "content": [{"type": "text", "text": PICK_PROMPT.format(targets=what)},
                                                image_part(small)]}]
            try:
                v = ask_json(client, msg, LOCAL_VLM, "scene_pick",  # on this GPU, free (Kimi-K2.6 before 2026-10-15)
                             lambda v: isinstance(v.get("score"), (int, float)), tries=1, temperature=0.0,
                             max_tokens=REASONING_TOKENS, **PHOTO_CALL)
            except LLMError:
                return None
            finally:
                small.unlink(missing_ok=True)
            if v:
                none = v.get("count") == 0  # a photo without the class (a teacher's label) cannot make its scene
                v = {"score": -1.0 if none else float(v["score"]), "why": str(v.get("why") or ""),
                     "things": v.get("things") is not False}
                kept.write_text(json.dumps(v))
            return v

        rated = map_within(rate, head, PICK_S)
        if not any(rated):
            ctx.log("the photo check did not answer: the scene photo is the first in the usual order")
            return candidates, []
        order = sorted(range(len(head)), key=lambda i: (-(rated[i] or {"score": -0.5})["score"], i))
        for rank, i in enumerate(order):
            pid, photo = head[i]
            r = rated[i]
            ctx.log(f"{'picked' if rank == 0 else 'also rated'} photo {pid} ({photo.name}): "
                    + (("none of the class seen, " if r["score"] < 0 else f"{r['score']:g}/10, ") + r["why"]
                       if r else "no answer"))
        return [head[i] for i in order] + candidates[PICK_AMONG:], [r.get("things", True) for r in rated if r]

    @staticmethod
    def _candidates(ctx: Context, tried: set[int]) -> list[tuple[str | None, list[tuple[int, Path]]]]:
        """Training photos (ingest) to build scenes from, in groups that each get a scene. Counting, one group of
        the photos with objects, a typical count first: the one closest to the median count, reviewed photos before
        the teacher's. Classifying, a group per class (the background class left out) of the photos labelled with
        it, reviewed first: one-class views of a class need a scene with its things. Each group with its class
        (None counting)."""
        from ..library import KEEP
        from ..tasks import task_of
        snap = json.loads((ctx.job.input / KEEP).read_text()) if (ctx.job.input / KEEP).exists() else None
        if snap is None:
            return []
        raw = ctx.job.resolve_input(ctx.spec.user_data.raw)
        if not task_of(ctx.spec).points:
            from .ingest import train_photos
            groups: dict[str, list] = {}
            for f in train_photos(ctx):
                info = snap["photos"].get(f.name)
                cls = f.parent.name
                if info and cls != ctx.spec.background_class and info["photo_id"] not in tried and (raw / f.name).exists():
                    groups.setdefault(cls, []).append((not info["reviewed"], info["photo_id"], raw / f.name))
            return [(cls, [(pid, f) for _, pid, f in sorted(rows)]) for cls, rows in sorted(groups.items())]
        train = ctx.job.stage_dir("ingest") / "train" / "points.json"
        if not train.exists():
            return []
        rows = []
        for v in json.loads(train.read_text()).values():
            info = snap["photos"].get(Path(v["key"]).name)
            if info and v["points"] and info["photo_id"] not in tried and (raw / v["key"]).exists():
                rows.append((info["photo_id"], raw / v["key"], len(v["points"]), info["reviewed"]))
        if not rows:
            return []
        mid = statistics.median(r[2] for r in rows)
        rows.sort(key=lambda r: (not r[3], abs(r[2] - mid), r[0]))
        return [(None, [(r[0], r[1]) for r in rows])]
