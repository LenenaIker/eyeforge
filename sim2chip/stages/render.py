"""Synthetic dataset: one target (or none) per frame, placed by the scene template from the spec, with the
spec's background props scattered around it. With scene.contexts, some frames are instead a building front,
a balcony or a room (scenes/places.py) with the targets on its floors, railings and sills. For the count
task, 0 to generation.max_targets targets per frame, spread around the point the camera looks at.

The camera distance is derived from the (first) target's real size and the spec's target_fill, so it adapts
to any object. Output: render/rgb_*.png, bounding_box_2d_tight_*.{npy,json}, render/labels.json
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field

from ..spec import MEASURES, camera_distance
from ..tasks import task_of
from .base import Context, Stage


# Auto exposure after the render, as a phone sets it before the shot: dark frames are brightened in linear light
# until their median reaches EXPOSE_MEDIAN (sRGB 0..1), by at most EXPOSE_MAX_GAIN, never darkened. A frame still
# mostly black after that (a room corner, a facade against the sun) is left out: it teaches nothing.
EXPOSE_MEDIAN = 0.42
EXPOSE_MAX_GAIN = 4.0
BLACK_LEVEL = 30 / 255   # sRGB value counted as nearly black
BLACK_SHARE = 0.4        # frames with more of it than this are dropped


def expose(path) -> bool:
    """Brighten a render in place; False when it stays mostly black."""
    import numpy as np
    from PIL import Image

    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    lin = rgb ** 2.2
    median = float(np.median(lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    gain = min(EXPOSE_MAX_GAIN, max(1.0, EXPOSE_MEDIAN ** 2.2 / max(median, 1e-6)))
    if gain > 1.01:
        rgb = np.clip(lin * gain, 0, 1) ** (1 / 2.2)
        out = (rgb * 255 + 0.5).astype(np.uint8)
        if im.mode == "RGBA":  # keep the writer's alpha
            out = np.dstack([out, np.asarray(im)[..., 3]])
        Image.fromarray(out).save(path)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float((luma < BLACK_LEVEL).mean()) <= BLACK_SHARE


class Render(Stage):
    name = "render"
    version = "5"  # v5: fewer frames when some classes have no 3D model; v2: paint, empty-frame distance and camera aim from the spec; v3: masonry parapets;
    # v4: sun behind the camera in most frames, room lamp, auto exposure and nearly black frames dropped
    runtime = "isaac"
    uses_cache = True
    gpu = "rtx"
    max_hours = 2
    uses = ("scene", "camera", "generation", "classes", "props", "background_class", "task")
    after = ("qa_assets", "fetch_assets")  # fetch_assets: scene.json with skies and ground textures
    skippable = "with photos"

    def enabled(self, spec) -> bool:
        return spec.synthetic

    def run(self, ctx: Context) -> dict:
        import shutil

        import omni.replicator.core as rep
        from omni.kit.app import get_app

        from ..scenes.common import Paint, Props, set_visible

        spec, gen = ctx.spec, ctx.spec.generation
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        rng = random.Random(gen.seed)
        app = get_app()
        stage = _new_stage()
        props = Props(stage, app)
        scene = _build_scene(ctx, stage, props, rng)
        paint = Paint(stage)
        cam = _camera(stage, spec.camera)
        scatter, targets = _load_models(ctx, props)
        for _ in range(10):
            app.update()
        props.recenter([t[1] for ts in targets.values() for t in ts] + scatter.prims)
        scene.after_load()
        for _ in range(30):
            app.update()

        rep.orchestrator.set_capture_on_play(False)
        rp = rep.create.render_product(str(cam.GetPath()), tuple(spec.camera.resolution))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(out), rgb=True, bounding_box_2d_tight=True)
        writer.attach([rp])

        shots = _Shots(spec, rng, scene, props, paint, cam, scatter, targets, real_counts(ctx))
        places = bool(spec.scene.contexts)
        visible, labels = [], []
        frames = synthetic_frames(gen.frames, [c.id for c in spec.classes], targets)
        if frames < gen.frames:
            ctx.log(f"only {len(shots.classes)} of {len(spec.classes)} classes have 3D models: {frames} frames "
                    f"instead of {gen.frames}, so the real photos of the others are not drowned")
        for f in range(frames):
            scene.randomize(change_sky=f % 20 == 0)
            kind = "ground"
            if places:  # a new building and room every 20 frames, a context per frame by its share
                if f and f % 20 == 0:
                    scene.new_layout()
                elif f % 5 == 0:  # new colors more often than new layouts: 200 frames had only 10 rooms
                    scene.recolor()
                kind = scene.pick_context()
                scene.show(None if kind == "ground" else kind)
            cls, group = shots.pick()
            shot = shots.on_ground(group) if kind == "ground" else shots.in_context(kind, group)
            for pr in visible:
                set_visible(pr, False)
            for pr in shot.shown:
                set_visible(pr, True)
            visible = shot.shown
            rep.orchestrator.step(rt_subframes=4)
            shown_uids = [t[0]["uid"] for t in group if t[1] in shot.shown]
            labels.append({"frame": f, "class": cls, "uid": shot.asset["uid"] if shot.asset else None,
                           **({"targets": shown_uids} if shots.counting else {}),
                           **({"context": kind} if places else {}),
                           "cam_dist": round(shot.dist, 2), "cam_h": round(shot.height, 2), "painted": shot.painted,
                           "props": len(shot.props), "hidden": round(shot.hidden, 2)})
            if f % 200 == 0:
                ctx.log(f"[{f}/{frames}] {cls}")
        rep.orchestrator.wait_until_complete()
        kept = [lab for lab in labels if expose(out / f"rgb_{lab['frame']:04d}.png")]
        if len(kept) < len(labels):
            ctx.log(f"{len(labels) - len(kept)} nearly black frames left out")
        labels = kept
        (out / "labels.json").write_text(json.dumps(labels, indent=1))
        counts = {}
        for lab in labels:
            counts[lab["class"]] = counts.get(lab["class"], 0) + 1
        ctx.log(f"{len(labels)} frames {counts}")
        return {"frames": counts}


def _new_stage():
    """An empty USD stage, z up and in meters, with the renderer waiting for what it loads."""
    import carb
    import omni.usd
    from pxr import UsdGeom

    settings = carb.settings.get_settings()
    # Remote textures and HDRIs load asynchronously: captures come out black or white otherwise.
    settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
    settings.set("/omni/replicator/asyncRendering", False)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def real_counts(ctx: Context) -> list[int]:
    """Count: how many objects each of the user's photos shows (the label stage's marks), none without them."""
    f = ctx.job.stage_dir("label") / "verdicts.json"
    if not task_of(ctx.spec).points or not f.exists():
        return []
    return [len(v["objects"]) for v in json.loads(f.read_text()).values() if v.get("objects") is not None]


def _build_scene(ctx: Context, stage, props, rng):
    """The spec's scene template, with the skies, ground and walls fetch_assets chose."""
    from .. import scenes

    spec = ctx.spec
    chosen = json.loads((ctx.job.stage_dir("fetch_assets") / "scene.json").read_text())
    make = scenes.load(scenes.template_for(spec.scene))
    kw = {"walls": [str(ctx.cache.root / p) for p in chosen.get("walls", [])], "contexts": spec.scene.contexts,
          "railings": spec.scene.railings} if spec.scene.contexts else {}
    if chosen.get("ground_tile_m"):  # mixed ground layouts (ground_mix.py) cover more than one texture
        kw["ground_tile"] = chosen["ground_tile_m"]
    scene = make(stage, props, rng, skies=[str(ctx.cache.root / p) for p in chosen["skies"]],
                 distractors=spec.scene.distractors, ground=[str(ctx.cache.root / p) for p in chosen["ground"]],
                 **kw)
    scene.build()
    return scene


def _camera(stage, cam_spec):
    """A pinhole camera with the spec's field of view and aspect."""
    from pxr import Gf, UsdGeom

    w, h = cam_spec.resolution
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    aperture = 20.955
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000))
    cam.CreateFocalLengthAttr(aperture / 2 / math.tan(math.radians(cam_spec.hfov_deg) / 2))
    return cam


def synthetic_frames(frames: int, classes: list[str], targets) -> int:
    """The frames to render: the spec's, shrunk by the share of classes that have 3D models. Frames that show only
    some classes outweigh the real photos of the others, which nothing else teaches."""
    covered = len(set(targets) & set(classes)) / max(1, len(classes))
    return max(1, round(frames * covered)) if covered else frames


def _load_models(ctx: Context, props):
    """The kept 3D models in the stage: props in their scatter, targets by class as (asset, prim, scale, size)."""
    from ..scenes.scatter import PropScatter

    spec, gen = ctx.spec, ctx.spec.generation
    assets = json.loads((ctx.job.stage_dir("qa_assets") / "assets.json").read_text())
    scatter = PropScatter(props, spec.camera.hfov_deg)
    for p in spec.props:
        models = []
        for a in (a for a in assets if a.get("role") == "prop" and a["class"] == p.id):
            dim = a["dims"][MEASURES.index(p.size_m.measure)]
            if dim <= 0:  # flat: no size to scale by
                continue
            models.append((str(ctx.cache.root / a["usd"]), (math.hypot(*a["dims"][:2]) / 2, a["dims"][2]),
                           (p.size_m.range[0] / dim, p.size_m.range[1] / dim)))
        if models:
            scatter.add_kind(p.max_per_frame, models, f"/World/Props/{p.id}")
    if spec.props:
        ctx.log(f"{len(scatter.kinds)} kinds of props, {len(scatter.prims)} instances")
    targets: dict[str, list] = {}
    for i, a in enumerate(a for a in assets if a.get("role") != "prop"
                          and a["dims"][MEASURES.index(spec.cls(a["class"]).size_m.measure)] > 0):
        c = spec.cls(a["class"])
        prim = props.add(f"/World/Targets/t{i:03d}", str(ctx.cache.root / a["usd"]), label=a["class"])
        # Real-world scale from the spec, drawn once per asset (stable across runs and jobs).
        dim = a["dims"][MEASURES.index(c.size_m.measure)]
        scale = random.Random(f"{gen.seed}-{a['uid']}").uniform(*c.size_m.range) / dim
        size = max(a["dims"]) * scale
        targets.setdefault(a["class"], []).append((a, prim, scale, size))
    return scatter, targets


@dataclass
class _Shot:
    """What one frame shows and where the camera stood."""
    shown: list            # prims visible in the frame
    asset: dict | None     # the first target's asset, None in an empty frame
    dist: float            # camera to the point it looks at
    height: float          # camera height
    painted: bool = False
    props: list = field(default_factory=list)
    hidden: float = 0.0    # share of the target the props hide


class _Shots:
    """The frame loop's shared state and one method per kind of shot. The random draws happen in the order the
    loop always made them, so a seed gives the same frames as before this was split out."""

    def __init__(self, spec, rng, scene, props, paint, cam, scatter, targets, real: list[int] | None = None):
        self.spec, self.cam_spec, self.gen, self.rng = spec, spec.camera, spec.generation, rng
        # Count: how many objects the user's photos show (the teacher's marks), so frames hold as many.
        self.real = [n for n in real or [] if n > 0]
        self.empty = (sum(n == 0 for n in real) / len(real)) if real else None
        self.scene, self.props, self.paint, self.cam, self.scatter = scene, props, paint, cam, scatter
        self.targets = targets
        self.classes = [c.id for c in spec.classes if c.id in targets]
        self.pool = [t for ts in targets.values() for t in ts]
        self.counting = task_of(spec).points
        self.by_id = {c.id: c for c in spec.classes}

    def pick(self) -> tuple[str, list]:
        """The frame's class and its targets: none in an empty frame, several at once when counting (each prim at
        most once per frame)."""
        rng, bg = self.rng, self.spec.background_class  # None: every frame shows a class
        empty = self.gen.empty_ratio if self.empty is None or not self.counting else min(0.5, max(0.05, self.empty))
        cls = bg if bg and rng.random() < empty else rng.choice(self.classes)
        if cls == bg:
            return cls, []
        if self.counting:
            first = rng.choice(self.targets[cls])
            rest = [t for t in self.pool if t is not first]
            more = rng.choice(self.real) - 1 if self.real else rng.randint(0, self.gen.n_targets - 1)
            return cls, [first] + rng.sample(rest, min(len(rest), more))
        return cls, [rng.choice(self.targets[cls])]

    def paint_as(self, a, prim) -> bool:
        c = self.by_id[a["class"]]
        return self.paint.apply(prim, self.rng, force=a.get("no_color", False), p=c.repaint_p,
                                metallic=c.metallic_range)

    def ground_view(self, size: float):
        """(distance to the target, its horizontal part, camera height or None to draw it later). With
        camera.pitch_deg the camera looks down at that angle from its height (a phone over a lawn); else the
        distance comes from the target's size and target_fill."""
        rng, cam = self.rng, self.cam_spec
        if cam.pitch_deg:
            h = rng.uniform(*cam.height_m)
            p = math.radians(min(rng.uniform(*cam.pitch_deg), 89.0))  # 90: look_at has no up vector
            return h / math.sin(p), h / math.tan(p), h
        d = camera_distance(size, rng.uniform(*cam.target_fill), cam.hfov_deg, cam.distance_m)
        return d, d, None

    def in_context(self, kind: str, group: list) -> _Shot:
        """A building front, a balcony or a room: the place decides the camera and the spots (floors, railings,
        sills)."""
        from ..scenes.common import look_at

        w, h = self.cam_spec.resolution
        items = [(math.hypot(*t[0]["dims"][:2]) / 2 * t[2], t[0]["dims"][2] * t[2]) for t in group]
        spots, eye, look = self.scene.shot(kind, items, self.cam_spec.hfov_deg, w / h)
        shown, painted = [], False
        for (ga, gprim, gscale, _), pos in zip(group, spots):
            if pos is not None:
                self.props.set_pose(gprim, pos, self.rng.uniform(0, 360), gscale)
                painted = self.paint_as(ga, gprim) or painted
                shown.append(gprim)
        look_at(self.cam.GetPrim(), eye, look)
        self.scene.aim_sun(eye, look)
        return _Shot(shown, group[0][0] if group else None, math.dist(eye, look), eye[2], painted)

    def on_ground(self, group: list) -> _Shot:
        """Open ground: the camera placed from the first target's size (or the profile's pitch), the rest of the
        group spread around it, then distractors and props."""
        from ..scenes.common import look_at
        from ..scenes.scatter import cluster

        rng, cam = self.rng, self.cam_spec
        shown, painted, asset, radius, height = [], False, None, 0.0, 0.0
        if group:
            asset, prim, scale, size = group[0]
            target = self.scene.place_target(prim, scale)
            painted = self.paint_as(asset, prim)
            shown.append(prim)
            radius, height = math.hypot(*asset["dims"][:2]) / 2 * scale, asset["dims"][2] * scale
            aim_h = height
            dist, hdist, ch = self.ground_view(size)
            if len(group) > 1:
                # Spread over the ground seen around the first one; the props then keep out of the group.
                spread = min(0.9 * dist * math.tan(math.radians(cam.hfov_deg) / 2), 0.8 * dist)
                radii = [math.hypot(*t[0]["dims"][:2]) / 2 * t[2] for t in group]
                for (ga, gprim, gscale, _), pos in zip(group[1:], cluster(target, radii, spread, rng)[1:]):
                    if pos is not None:
                        self.props.set_pose(gprim, pos, rng.uniform(0, 360), gscale)
                        self.paint_as(ga, gprim)
                        shown.append(gprim)
                radius = spread + max(radii)
        else:
            # Background views from where a target frame would be shot, so distance tells nothing.
            target = (0.0, 0.0, 0.0)
            ea, _, escale, size = rng.choice(self.pool)
            aim_h = ea["dims"][2] * escale
            dist, hdist, ch = self.ground_view(size)
        az = math.radians(rng.uniform(0, 360))
        if ch is None:
            ch = rng.uniform(*cam.height_m)
        eye = (target[0] + hdist * math.cos(az), target[1] + hdist * math.sin(az), ch)
        j = dist * 0.12
        # Aim at the body of the object (a car's doors, a pot's middle), not a fixed height.
        look_at(self.cam.GetPrim(), eye, (target[0] + rng.uniform(-j, j), target[1] + rng.uniform(-j, j),
                                          rng.uniform(0.2, 0.8) * aim_h))
        self.scene.aim_sun(eye, target)
        wall = getattr(self.scene, "backdrop_for", lambda *a: None)(eye, target, radius)
        shown += self.scene.place_distractors(eye) + ([wall] if wall is not None else [])
        placed, hidden = self.scatter.place(eye, target, radius, height, rng)
        return _Shot(shown + placed, asset, dist, ch, painted, placed, hidden)
