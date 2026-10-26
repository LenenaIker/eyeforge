"""Synthetic frames in the walk scenes the user chose (generation.scenes: their own or public library scenes of
route "walk", made apart from training and given to the job in input/scenes.json by library.export_scenes). The class's 3D models
(qa_assets) are placed on each scene's free ground and rendered over its splat (sim2chip/walkscene): the plan in
this Python, the backgrounds in gsplat's, the objects in Isaac Sim's. A scene is only the place, any class goes in
it (user, 2026-10-26).

Output: scene_render/frames.json  [{"file", "scene", "boxes": [[class, x0, y0, x1, y1], ...]}] (pixels),
                                  classifying also "class": the view's one class (render.scene_frames reads it)
        scene_render/plan.json, background.json, render.json, render_NNNN.png, obj_NNNN.png
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from .base import Context, Stage

VIEWS = os.environ.get("SIM2CHIP_SCENE_VIEWS")  # frames per job instead of the spec's generation.frames (tests)
MIN_WIDTH = 640    # frames are rendered at least this wide (the spec's resolution times a whole factor)
ROUTE = "walk"
NO_SCENE = "none of the chosen scenes is ready"


def scene_folders(ctx: Context) -> dict[str, Path]:
    """The job's walk scenes, those of generation.scenes the library gave it (input/scenes.json): {id: bundle
    folder}. SIM2CHIP_WALK_SCENES (folders, comma separated) takes their
    place, to measure on scenes the library does not hold."""
    if os.environ.get("SIM2CHIP_WALK_SCENES"):
        return {Path(f).parent.name if Path(f).name == "bundle" else Path(f).name: Path(f)
                for f in os.environ["SIM2CHIP_WALK_SCENES"].split(",") if f}
    from ..library import SCENES
    f = ctx.job.input / SCENES
    ready = json.loads(f.read_text()).get("ready", []) if f.exists() else []
    chosen = set(ctx.spec.generation.scenes)
    return {s["id"]: ctx.job.input / s["dir"] for s in ready if s.get("route") == ROUTE and s["id"] in chosen}


def resolve_asset(usd: str, out_dir: Path) -> str:
    """The asset as Isaac can open it: the converted wrapper itself when its reference resolves, else a copy in
    out_dir whose reference points at the same file name in the wrapper's own raw/ folder (wrappers converted before
    the folder rename of 2026-10-01 carry absolute paths to a folder that no longer exists)."""
    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(usd)
    spec = layer.GetPrimAtPath("/Asset/Geo") if layer else None
    refs = (list(spec.referenceList.prependedItems) or list(spec.referenceList.explicitItems)) if spec else []
    if not refs:
        return usd
    target = Path(layer.ComputeAbsolutePath(refs[0].assetPath))
    local = Path(usd).parent / "raw" / target.name
    if target.exists() or not local.exists():
        return usd
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / Path(usd).name
    if not out.exists():
        copy = Sdf.Layer.CreateNew(str(out))
        copy.TransferContent(layer)
        geo = copy.GetPrimAtPath("/Asset/Geo")
        geo.referenceList.ClearEditsAndMakeExplicit()
        geo.referenceList.explicitItems = [Sdf.Reference(str(local.resolve()))]
        copy.Save()
    return str(out)


def frames_of(views: list[dict], report: list[dict], spec) -> list[dict]:
    """frames.json from the plan's views and what Isaac saw: an object hidden behind the scene (or under min_px)
    has no box; a classifying view whose one object is hidden is left out (its class would be a lie)."""
    seen = {r["name"]: r["objects"] for r in report}
    out = []
    for v in views:
        if v["name"] not in seen:
            continue
        boxes = [[o["class"], *b["box"]] for o, b in zip(v["objects"], seen[v["name"]]) if b]
        if v["class"] is not None and v["objects"] and not boxes:
            continue
        fr = {"file": f"render_{v['name']}.png", "scene": v["scene"], "boxes": boxes}
        if v["class"] is not None:
            fr["class"] = v["class"]
        out.append(fr)
    return out


class SceneRender(Stage):
    name = "scene_render"
    version = "1"
    runtime = "cpu"   # it runs the backgrounds in gsplat's Python and the objects in Isaac's
    gpu = "rtx"
    max_hours = 2
    reads_input = True
    uses_cache = True
    uses = ("classes", "camera", "generation", "task", "background_class")
    after = ("qa_assets", "ingest")
    skippable = "with photos"
    results = ("*.json", "render_*.png")

    def enabled(self, spec) -> bool:
        return spec.synthetic

    def powered_by(self, spec, backend: str, model: str | None = None) -> str:
        return f"NVIDIA Isaac Sim | {self.where(backend)}"

    def run(self, ctx: Context) -> dict:
        from .. import config, subproc
        from ..walkscene import Scene
        from ..walkscene.plan import plan

        scenes = scene_folders(ctx)
        if not scenes:
            raise RuntimeError(NO_SCENE)
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        assets = json.loads((ctx.job.stage_dir("qa_assets") / "assets.json").read_text())
        w, h = ctx.spec.camera.resolution
        k = max(1, math.ceil(MIN_WIDTH / w))
        n = int(VIEWS) if VIEWS else ctx.spec.generation.frames
        p = plan({s: Scene(f) for s, f in scenes.items()}, assets, ctx.spec, ctx.cache.root, n, (w * k, h * k))
        if not p["views"]:
            raise RuntimeError(p.get("why") or "no frame found room for the objects in the scenes")
        for v in p["views"]:
            for o in v["objects"]:
                o["usd"] = resolve_asset(o["usd"], out / "assets")
        (out / "plan.json").write_text(json.dumps(p, indent=1))
        (out / "scenes.json").write_text(json.dumps({s: str(f) for s, f in scenes.items()}, indent=1))
        ctx.log(f"{len(p['views'])} frames planned in {len(scenes)} scenes ({p.get('missed', 0)} found no room)")

        from ..backends.local import ISAAC_PYTHON
        ctx.job.logs.mkdir(parents=True, exist_ok=True)

        def step(python_var: str, default: str | None, module: str, log: str):
            exe = config.env(python_var, default)
            if not exe or not Path(exe).exists():
                raise RuntimeError(f"set {python_var}: the walk scenes render with it")
            code, tail = subproc.run_module([exe], module, ["--dir", str(out)], log_file=ctx.job.logs / log,
                                            on_line=lambda line: line.startswith("[sim2chip]") and ctx.log(
                                                line.rstrip().removeprefix("[sim2chip] ")),
                                            env=subproc.python_env(inherit_path=False))
            if code:
                raise RuntimeError(f"{module} failed: " + " | ".join(tail[-3:]))

        step("SIM2CHIP_GSPLAT_PYTHON", None, "sim2chip.walkscene.background", "scene_render_background.log")
        step("SIM2CHIP_ISAAC_PYTHON", ISAAC_PYTHON, "sim2chip.walkscene.isaac", "scene_render_isaac.log")

        report = json.loads((out / "render.json").read_text())["views"]
        frames = frames_of(p["views"], report, ctx.spec)
        (out / "frames.json").write_text(json.dumps(frames, indent=1))
        ctx.log(f"{len(frames)} frames in the walk scenes")
        return {"frames": len(frames), "planned": len(p["views"]), "scenes": sorted(scenes)}
