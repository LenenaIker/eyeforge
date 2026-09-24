"""Synthetic dataset: one target (or none) per frame, placed by the scene template from the spec, with the
spec's background props scattered around it.

The camera distance is derived from the target's real size and the spec's target_fill, so it adapts to
any object. Output: render/rgb_*.png, bounding_box_2d_tight_*.{npy,json}, render/labels.json
"""
from __future__ import annotations

import json
import math
import random

from ..spec import MEASURES, camera_distance
from .base import Context, Stage


class Render(Stage):
    name = "render"
    runtime = "isaac"
    uses_cache = True
    gpu = "rtx"
    max_hours = 2
    uses = ("scene", "camera", "generation", "classes", "props", "background_class")
    after = ("qa_assets", "fetch_assets")  # fetch_assets: scene.json with skies and ground textures

    def enabled(self, spec) -> bool:
        return any(c.assets for c in spec.classes)

    def run(self, ctx: Context) -> dict:
        import shutil

        import carb
        import omni.replicator.core as rep
        import omni.usd
        from omni.kit.app import get_app
        from pxr import Gf, UsdGeom

        from .. import scenes
        from ..scenes.common import Paint, Props, look_at, set_visible
        from ..scenes.scatter import PropScatter

        spec, cam_spec, gen = ctx.spec, ctx.spec.camera, ctx.spec.generation
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        rng = random.Random(gen.seed)
        app = get_app()

        settings = carb.settings.get_settings()
        # Remote textures and HDRIs load asynchronously: captures come out black or white otherwise.
        settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
        settings.set("/omni/replicator/asyncRendering", False)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        props = Props(stage, app)
        chosen = json.loads((ctx.job.stage_dir("fetch_assets") / "scene.json").read_text())
        scene = scenes.load(spec.scene.template)(
            stage, props, rng, skies=[str(ctx.cache.root / p) for p in chosen["skies"]],
            distractors=spec.scene.distractors, ground=[str(ctx.cache.root / p) for p in chosen["ground"]])
        scene.build()
        paint = Paint(stage)

        w, h = cam_spec.resolution
        cam = UsdGeom.Camera.Define(stage, "/World/Cam")
        aperture = 20.955
        cam.CreateHorizontalApertureAttr(aperture)
        cam.CreateVerticalApertureAttr(aperture * h / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000))
        cam.CreateFocalLengthAttr(aperture / 2 / math.tan(math.radians(cam_spec.hfov_deg) / 2))

        assets = json.loads((ctx.job.stage_dir("qa_assets") / "assets.json").read_text())
        scatter = PropScatter(props, cam_spec.hfov_deg)
        for p in spec.props:
            models = []
            for a in (a for a in assets if a.get("role") == "prop" and a["class"] == p.id):
                dim = a["dims"][MEASURES.index(p.size_m.measure)]
                models.append((str(ctx.cache.root / a["usd"]), (math.hypot(*a["dims"][:2]) / 2, a["dims"][2]),
                               (p.size_m.range[0] / dim, p.size_m.range[1] / dim)))
            if models:
                scatter.add_kind(p.max_per_frame, models, f"/World/Props/{p.id}")
        if spec.props:
            ctx.log(f"{len(scatter.kinds)} kinds of props, {len(scatter.prims)} instances")
        assets = [a for a in assets if a.get("role") != "prop"]
        targets: dict[str, list] = {}
        for i, a in enumerate(assets):
            c = spec.cls(a["class"])
            prim = props.add(f"/World/Targets/t{i:03d}", str(ctx.cache.root / a["usd"]), label=a["class"])
            # Real-world scale from the spec, drawn once per asset (stable across runs and jobs).
            dim = a["dims"][MEASURES.index(c.size_m.measure)]
            scale = random.Random(f"{gen.seed}-{a['uid']}").uniform(*c.size_m.range) / dim
            size = max(a["dims"]) * scale
            targets.setdefault(a["class"], []).append((a, prim, scale, size))
        for _ in range(10):
            app.update()
        props.recenter([t[1] for ts in targets.values() for t in ts] + scatter.prims)
        scene.after_load()
        for _ in range(30):
            app.update()

        rep.orchestrator.set_capture_on_play(False)
        rp = rep.create.render_product(str(cam.GetPath()), (w, h))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(out), rgb=True, bounding_box_2d_tight=True)
        writer.attach([rp])

        classes = [c.id for c in spec.classes if c.id in targets]
        visible, labels = [], []
        for f in range(gen.frames):
            scene.randomize(change_sky=f % 20 == 0)
            cls = spec.background_class if rng.random() < gen.empty_ratio else rng.choice(classes)
            shown, painted, a, radius, height = [], False, None, 0.0, 0.0
            if cls != spec.background_class:
                a, prim, scale, size = rng.choice(targets[cls])
                target = scene.place_target(prim, scale)
                painted = paint.apply(prim, rng, force=a.get("no_color", False))
                shown.append(prim)
                radius, height = math.hypot(*a["dims"][:2]) / 2 * scale, a["dims"][2] * scale
                dist = camera_distance(size, rng.uniform(*cam_spec.target_fill), cam_spec.hfov_deg,
                                       cam_spec.distance_m)
            else:
                target = (0.0, 0.0, 0.0)
                lo, hi = cam_spec.distance_m
                dist = rng.uniform(max(lo, 5.0), max(min(hi, 35.0), lo))  # background views at typical range
            az = math.radians(rng.uniform(0, 360))
            ch = rng.uniform(*cam_spec.height_m)
            eye = (target[0] + dist * math.cos(az), target[1] + dist * math.sin(az), ch)
            j = dist * 0.12
            look_at(cam.GetPrim(), eye, (target[0] + rng.uniform(-j, j), target[1] + rng.uniform(-j, j),
                                         rng.uniform(0.3, 1.2)))
            shown += scene.place_distractors(eye)
            placed, hidden = scatter.place(eye, target, radius, height, rng)
            shown += placed
            for pr in visible:
                set_visible(pr, False)
            for pr in shown:
                set_visible(pr, True)
            visible = shown
            rep.orchestrator.step(rt_subframes=4)
            labels.append({"frame": f, "class": cls, "uid": a["uid"] if a else None, "cam_dist": round(dist, 2),
                           "cam_h": round(ch, 2), "painted": painted,
                           "props": len(placed), "hidden": round(hidden, 2)})
            if f % 200 == 0:
                ctx.log(f"[{f}/{gen.frames}] {cls}")
        rep.orchestrator.wait_until_complete()
        (out / "labels.json").write_text(json.dumps(labels, indent=1))
        counts = {}
        for lab in labels:
            counts[lab["class"]] = counts.get(lab["class"], 0) + 1
        ctx.log(f"{len(labels)} frames {counts}")
        return {"frames": counts}
