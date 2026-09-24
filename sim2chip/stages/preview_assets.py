"""Four views of every converted asset on a neutral floor, for automatic quality control.

Views are framed by the asset's real size; the last one looks down steeply, which reveals plinths and
bases. Only runs when qa.mode is "auto".
Output: preview_assets/<uid>_<view>.png, preview_assets/views.json
"""
from __future__ import annotations

import json
import math
import random

from ..spec import MEASURES
from .base import Context, Stage

VIEWS = [(35, 0.35), (145, 0.35), (270, 0.12), (215, 1.1)]  # (azimuth deg, relative elevation)
SIZE = (512, 384)


class PreviewAssets(Stage):
    name = "preview_assets"
    runtime = "isaac"
    uses_cache = True
    uses_db = True
    gpu = "rtx"
    uses = ("classes",)  # qa only decides whether the stage runs, not what it renders
    after = ("convert_assets",)

    def enabled(self, spec) -> bool:
        return spec.qa.mode == "auto" and any(c.assets for c in spec.classes)

    def run(self, ctx: Context) -> dict:
        import shutil

        import carb
        import omni.replicator.core as rep
        import omni.usd
        from omni.kit.app import get_app
        from pxr import Gf, UsdGeom, UsdLux

        from ..scenes.common import Props, look_at, make_quad, set_visible
        from ..scenes.skies import SKIES

        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        raw = out / "_raw"
        raw.mkdir(parents=True)
        app = get_app()
        settings = carb.settings.get_settings()
        settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
        settings.set("/omni/replicator/asyncRendering", False)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        floor, _ = make_quad(stage, "/World/Ground", 400, 400, 0.0)
        floor.Set(Gf.Vec3f(0.32, 0.32, 0.30))
        dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
        sky = next((s for s in SKIES if s.weather == "clear" and (ctx.cache.hdri / s.file).exists()), None)
        if sky:
            dome.CreateTextureFileAttr().Set(str(ctx.cache.hdri / sky.file))
        dome.CreateIntensityAttr().Set(1000)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr().Set(1500)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 30, 0))
        cam = UsdGeom.Camera.Define(stage, "/World/Cam")
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * SIZE[1] / SIZE[0])
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000))
        cam.CreateFocalLengthAttr(18.0)

        from ..asset_db import AssetDB
        from .qa_assets import concept_of

        assets = json.loads((ctx.job.stage_dir("convert_assets") / "assets.json").read_text())
        db = AssetDB(ctx.cache.db)
        known = [a for a in assets if db.lookup(a["uid"], concept_of(a))]
        assets = [a for a in assets if a not in known]  # already reviewed: no preview needed
        ctx.log(f"{len(known)} assets already reviewed, previewing {len(assets)}")
        if not assets:
            shutil.rmtree(raw, ignore_errors=True)
            (out / "views.json").write_text("{}")
            return {"assets": 0}
        props = Props(stage, app)
        items = []
        for i, a in enumerate(assets):
            c = ctx.spec.cls(a["class"])
            dim = a["dims"][MEASURES.index(c.size_m.measure)]
            scale = random.Random(f"{ctx.spec.generation.seed}-{a['uid']}").uniform(*c.size_m.range) / dim
            items.append((a, props.add(f"/World/Targets/t{i:03d}", str(ctx.cache.root / a["usd"])), scale))
        for _ in range(10):
            app.update()
        props.recenter([p for _, p, _ in items])
        for _ in range(30):
            app.update()

        rep.orchestrator.set_capture_on_play(False)
        rp = rep.create.render_product(str(cam.GetPath()), SIZE)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(raw), rgb=True)
        writer.attach([rp])
        frames = []
        shown = None
        for a, prim, scale in items:
            if shown is not None:
                set_visible(shown, False)
            set_visible(prim, True)
            shown = prim
            props.set_pose(prim, (0, 0, 0), 0, scale)
            size = [d * scale for d in a["dims"]]
            L = max(size) * 1.9
            for v, (az, el) in enumerate(VIEWS):
                t = math.radians(az)
                look_at(cam.GetPrim(), (L * math.cos(t), L * math.sin(t), L * el + 0.5), (0, 0, min(size) * 0.4))
                rep.orchestrator.step(rt_subframes=8)
                frames.append((a["uid"], v))
        rep.orchestrator.wait_until_complete()
        views = {}
        for k, (uid, v) in enumerate(frames):
            src = raw / f"rgb_{k:04d}.png"
            dst = out / f"{uid}_{v}.png"
            src.rename(dst)
            views.setdefault(uid, []).append(dst.name)
        shutil.rmtree(raw, ignore_errors=True)
        (out / "views.json").write_text(json.dumps(views, indent=1))
        ctx.log(f"{len(views)} assets previewed, {len(frames)} views")
        return {"assets": len(views)}
