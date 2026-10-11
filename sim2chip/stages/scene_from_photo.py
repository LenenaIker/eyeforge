"""Synthetic frames from the user's own place (photo2scene, docs/plan_photo2scene.md): the dataset's scenes built
by earlier jobs (input/scenes.json, from the library) are composed again with this job's seed; with none, one is
built from a training photo first, and the library keeps it for the next jobs. One working surface seen from above
only (the support route), with classes that are things to move around, not surfaces (the vision model says so when
it rates the photos); a photo that does not fit is noted as failed and not tried again. Counting views hold any of
the classes; classifying views hold one class each (or none, for the background class), and every class gets its
own scene, from a photo of it.

Render collects the frames (the scene template is off, base.TEMPLATE). The scene photos stay out of the test
(ingest.scene_photos), and the new one is picked among the photos ingest put in training.

Output: scene_from_photo/scenes.json  the scenes built here, for the library (library.take_scenes)
        scene_from_photo/frames.json  [{"file", "scene", "boxes": [[class, x0, y0, x1, y1], ...]}] (pixels),
                                      classifying also "class": the view's one class
        scene_from_photo/<scene>/     each scene's folder, its views in isaac/render_NN.png
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
from pathlib import Path

from ..workspace import cache_key, content_key
from .base import Context, Stage

VIEWS = 20           # new views per scene and job
BUILD_TRIES = 2      # one scene per job (classifying, per class), from up to this many photos: the next one when a
                     # photo makes none. The vision model rated 9/10 a photo the teacher called sun cream that showed
                     # only a figurine (2026-10-11)
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


def _link(src, dst):
    """A scene's file into the job: a hard link when it can (the files are hundreds of MB), else a copy. Every
    step that changes a file writes a new one, so the source never changes under the link."""
    import os
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def no_route(spec) -> str | None:
    """Why no photo2scene route fits this spec, in the user's words, or None when one does. So far only the support
    route: things on one surface seen from above (the photo analysis sets the camera's pitch then). Whether the
    classes are things or surfaces only the photos tell (SceneFromPhoto._pick)."""
    if not spec.synthetic:
        return "the setup asks for your photos only"
    if not spec.generation.photo_scenes:
        return "rebuilding your place is turned off in the setup"
    if not spec.user_data.raw:
        return "rebuilding your place needs unsorted photos of it"
    if not spec.camera.pitch_deg:
        return "your photos do not look down on one surface, the only kind of place EyeForge rebuilds so far"
    return None


class SceneFromPhoto(Stage):
    name = "scene_from_photo"
    version = "3"  # 2: the vision model picks the cleanest photo; 3: classifying too, one class per view
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
        for s in given.get("ready", []):
            d = out / s["id"]
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(ctx.job.input / s["dir"], d, copy_function=_link)
            ctx.log(f"scene {s['id']} (photo {s['photo_id']}) from the library")
            frames += self._compose(ctx, p2s.Flow(d, usage, ctx.log), s["id"], phrases)
        if not given.get("ready"):
            tried = set(given.get("failed_photos", []))
            picks, votes = [], []
            for cls, group in self._candidates(ctx, tried):
                # Classifying, a class's photos are rated for that class: asked about every class, the model gave
                # 10/10 to a sun cream photo (by the teacher) with only a figurine in it (2026-10-11).
                ranked, things = self._pick(ctx, group, phrases[cls][:1] if cls in phrases else targets)
                picks.append(ranked[:BUILD_TRIES])
                votes += things
            if votes and sum(votes) * 2 < len(votes):
                ctx.log(f"the vision model sees surfaces, not things, in {len(votes) - sum(votes)} of {len(votes)} "
                        "photos: no scene")
                built.append({"photo_id": None, "route": "support", "status": "failed", "why": SURFACES})
                picks = []
            for pid, photo in self._tries(picks, built):
                # Built in the cache (by photo and classes): a run that fails later, or is tried again, does not
                # pay the erase and the meshes twice.
                work = ctx.cache.root / "scenes" / f"{content_key(photo)}-{cache_key(targets)}"
                ctx.log(f"building a scene from photo {pid} ({photo.name})")
                try:
                    summary = p2s.build(p2s.Flow(work, usage, ctx.log), photo, targets)
                except p2s.Stop as e:
                    ctx.log(f"photo {pid} makes no scene: {e}")
                    built.append({"photo_id": pid, "route": "support", "status": "failed", "why": str(e)})
                    continue
                d = out / f"photo_{pid}"
                shutil.rmtree(d, ignore_errors=True)
                shutil.copytree(work, d, ignore=shutil.ignore_patterns("logs", "isaac"), copy_function=_link)
                flow = p2s.Flow(d, usage, ctx.log)
                cards = self._cards(d)
                objects = [{**a, "concept": self._class_of(a["name"], phrases) or a["name"],
                            **({"card": cards[a["index"]]} if a["index"] in cards else {})} for a in summary["objects"]]
                built.append({"photo_id": pid, "route": "support", "status": "ready", "dir": d.name,
                              "name": f"{summary['plan']['support'].capitalize()} from photo {pid}",
                              "thumb": f"{d.name}/photo.png", "info": {"plan": summary["plan"]},
                              "objects": objects})
                frames += self._compose(ctx, flow, d.name, phrases)
        (out / SCENES).write_text(json.dumps(built, indent=1))
        (out / "frames.json").write_text(json.dumps(frames, indent=1))
        boxes = sum(len(f["boxes"]) for f in frames)
        ctx.log(f"{len(frames)} frames from the user's place, {boxes} objects")
        return {"frames": len(frames), "objects": boxes, "built": sum(b["status"] == "ready" for b in built)}

    @staticmethod
    def _tries(picks: list[list[tuple[int, Path]]], built: list[dict]):
        """The photos to build from, group by group: the next of a group only while its last one made no scene
        (each try is recorded in built, failed ones too, so the library does not try them again)."""
        for group in picks:
            for item in group[:BUILD_TRIES]:
                n = len(built)
                yield item
                if len(built) > n and built[-1]["status"] == "ready":
                    break

    def _compose(self, ctx: Context, flow, scene: str, phrases) -> list[dict]:
        """This job's views of one scene, with the boxes of the user's classes only. Classifying, each view shows
        one class (its things of other classes left out), or none in a share of views when there is a background
        class; a view whose one class is out of sight is left out."""
        from ..photo2scene import compose
        from ..tasks import task_of
        groups, empty = None, 0.0
        if not task_of(ctx.spec).points:
            cat = flow.d / "assets.json"
            names = [a["name"] for a in json.loads(cat.read_text())["assets"]] if cat.exists() else []
            groups = {n: cls for n in names if (cls := self._class_of(n, phrases))}
            empty = ctx.spec.generation.empty_ratio if ctx.spec.background_class else 0.0
        out = []
        for v in compose(flow, VIEWS, ctx.spec.generation.seed, groups, empty):
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
        from ..llm import PHOTO_CALL, REASONING_TOKENS, Client, LLMError, ask_json, image_part, map_within
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
                v = ask_json(client, msg, client.model_for("scene"), "scene_pick",
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
