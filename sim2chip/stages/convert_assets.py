"""Convert downloaded models to USD, stand them upright, center them and measure them.

Conversion is class independent: the real-world scale is applied at render time from the spec, so a
reclassified asset needs no reconversion and the cache is shared by every job.

The converter is glb2usd (usd-core and trimesh, CPU only, one process per core): 536 models in 21 s where
Isaac Sim's Asset Converter needed about 2 s per model plus Kit's start, and the same renders and trained
model (experiments/NOTES.md, 2026-09-27). Models it cannot read (Draco or meshopt compression) go to the Asset
Converter in an Isaac Sim process, when there is one.

Cache:  usd/glb2usd/<uid>.usd (+ <uid>_textures/)   glb2usd output, in meters
        usd/raw/<uid>.usd                         Asset Converter output (fallback), in centimeters
        usd/norm/<uid>_v<N>.usd      wrapper that references the converted file with an upright transform
        usd/norm/<uid>_v<N>.json     {"dims": [length, width, height]} in the converted file's own units
Output: convert_assets/assets.json   manifest entries + usd (cache relative) + dims
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..spec import MEASURES
from . import base
from .base import Context, Stage

NORM_VERSION = 3  # bump when normalize() or the converter changes


def converted(cache, uid: str) -> Path | None:
    """The converted file of an asset: glb2usd's, else the Asset Converter's, else None."""
    for p in (cache.usd_glb2usd / f"{uid}.usd", cache.usd_raw / f"{uid}.usd"):
        if p.exists():
            return p
    return None


class ConvertAssets(Stage):
    name = "convert_assets"
    runtime = "cpu"
    uses_cache = True
    after = ("fetch_assets",)
    skippable = "with photos"
    version = str(NORM_VERSION)

    def enabled(self, spec) -> bool:
        return spec.synthetic and base.TEMPLATE  # its 3D models fed the scene template only

    def run(self, ctx: Context) -> dict:
        manifest = json.loads((ctx.job.stage_dir("fetch_assets") / "manifest.json").read_text())
        for d in (ctx.cache.usd_glb2usd, ctx.cache.usd_raw, ctx.cache.usd_norm):
            d.mkdir(parents=True, exist_ok=True)
        skipped = []
        todo = []
        for m in manifest:
            if (ctx.cache.usd_glb2usd / f"{m['uid']}.usd").exists():
                continue
            glb = ctx.cache.root / m["glb"]
            if glb.exists() and glb.stat().st_size > 0:
                todo.append((glb, m["uid"]))
            elif not converted(ctx.cache, m["uid"]):
                ctx.log(f"missing or empty download {m['glb']}")
                skipped.append(m["uid"])
        # Asset Converter only for what glb2usd cannot read and it has not converted before.
        failed = [(g, u) for g, u in _glb2usd(ctx, todo) if not converted(ctx.cache, u)]
        if failed:
            _asset_converter(ctx, failed)
        ready = []  # (manifest entry, converted file, wrapper)
        for m in manifest:
            raw = converted(ctx.cache, m["uid"])
            if raw is None:
                if m["uid"] not in skipped:
                    skipped.append(m["uid"])
                continue
            ready.append((m, raw, ctx.cache.usd_norm / f"{m['uid']}_v{NORM_VERSION}.usd"))
        _normalize_all(ctx, [(raw, norm) for _, raw, norm in ready if not norm.with_suffix(".json").exists()])
        done = []
        for m, _raw, norm in ready:
            meta = norm.with_suffix(".json")
            if not meta.exists():
                skipped.append(m["uid"])
                continue
            dims = json.loads(meta.read_text())["dims"]
            measure = ctx.spec.size_of(m["class"]).measure
            if dims[MEASURES.index(measure)] <= 1e-6 * max(max(dims), 1e-9):
                # A flat model (a road or wall as a plane) has no such size to scale by (2026-10-09: the
                # preview divided by 0 and the job failed).
                ctx.log(f"{m['uid']} ({m['class']}) is flat in its {measure}: skipped")
                skipped.append(m["uid"])
                continue
            done.append({**m, "usd": str(norm.relative_to(ctx.cache.root)), "dims": dims})
        if not done:
            raise RuntimeError(f"no asset could be converted ({len(skipped)} failed), see the log above")
        (ctx.out(self.name) / "assets.json").write_text(json.dumps(done, indent=1))
        ctx.log(f"{len(done)} assets ready ({len(todo)} converted now), {len(skipped)} failed")
        return {"assets": len(done), "converted": len(todo), "failed": skipped}


def _glb2usd(ctx: Context, todo: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Converts in parallel; returns the models glb2usd could not read."""
    if not todo:
        return []
    import time
    from concurrent.futures import ProcessPoolExecutor

    from .. import glb2usd

    opts = glb2usd.default_options()
    jobs = [(glb, ctx.cache.usd_glb2usd / f"{uid}.usd", opts) for glb, uid in todo]
    t0 = time.time()
    with ProcessPoolExecutor(max(1, min(os.cpu_count() or 1, len(jobs)))) as pool:
        results = list(pool.map(glb2usd._job, jobs))
    failed = []
    for (glb, uid), (_res, err) in zip(todo, results):
        if err:
            ctx.log(f"glb2usd could not convert {glb.name}: {err[1].strip().splitlines()[-1]}")
            failed.append((glb, uid))
    ctx.log(f"glb2usd: {len(todo) - len(failed)} of {len(todo)} 3D models converted in {time.time() - t0:.1f} s")
    return failed


def _normalize_one(raw_norm: tuple[Path, Path]) -> str | None:
    """None when done, else the error: one broken mesh must not stop the others."""
    raw, norm = raw_norm
    try:
        dims = normalize(raw, norm)
    except Exception as e:  # corrupt USD, degenerate hull...: whatever it is, only this model is lost
        norm.unlink(missing_ok=True)
        return f"{type(e).__name__}: {e}"
    if dims is not None:
        norm.with_suffix(".json").write_text(json.dumps({"dims": dims}))
    return None


def _normalize_all(ctx: Context, todo: list[tuple[Path, Path]]) -> None:
    """Upright wrappers and sizes, one process per core: big meshes take seconds each."""
    if not todo:
        return
    import time
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    with ProcessPoolExecutor(max(1, min(os.cpu_count() or 1, len(todo)))) as pool:
        errors = list(pool.map(_normalize_one, todo))
    for (raw, _), err in zip(todo, errors):
        if err:
            ctx.log(f"could not normalize {raw.name}, left out: {err}")
    ctx.log(f"normalized {len(todo)} 3D models in {time.time() - t0:.1f} s")


def _asset_converter(ctx: Context, todo: list[tuple[Path, str]]) -> None:
    """Isaac Sim's Asset Converter for what glb2usd cannot read, in its own Isaac Sim process."""
    import subprocess

    from ..backends.local import LocalBackend
    from ..workspace import REPO

    try:
        exe = LocalBackend().interpreter("isaac")
    except FileNotFoundError as e:
        ctx.log(f"{len(todo)} 3D models left out: {e}")
        return
    pairs = [[str(glb), str(ctx.cache.usd_raw / f"{uid}.usd")] for glb, uid in todo]
    ctx.log(f"{len(pairs)} 3D models to Isaac Sim's Asset Converter")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    # The pairs go in a file: one argv string is capped at 128 KiB on Linux.
    pairs_file = ctx.out("convert_assets") / "_isaac_pairs.json"
    pairs_file.write_text(json.dumps(pairs))
    r = subprocess.run(exe + ["-m", "sim2chip.isaac_convert", str(pairs_file)], env=env, cwd=REPO)
    pairs_file.unlink(missing_ok=True)
    if r.returncode:
        ctx.log(f"Isaac Sim's Asset Converter exited with {r.returncode}: 3D models it did not write are left out")


def world_points(stage, max_pts: int = 60000):
    """Vertices of every visible mesh in world coordinates (subsampled)."""
    import numpy as np
    from pxr import Usd, UsdGeom

    chunks = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        p = np.hstack([np.array(pts, dtype=np.float64), np.ones((len(pts), 1))]) @ m  # USD row-vector convention
        chunks.append(p[:, :3])
    if not chunks:
        return np.zeros((0, 3))
    # Drop large flat planes (invisible ground or shadow catchers): they inflate the measure and the
    # model ends up shrunk. License plates or windows are flat but small and stay.
    total = np.ptp(np.vstack(chunks), axis=0).max()
    solid = [c for c in chunks if not (len(c) <= 64 and np.sort(np.ptp(c, axis=0))[0] < 0.01 * np.ptp(c, axis=0).max()
                                        and np.ptp(c, axis=0).max() > 0.5 * total)]
    allp = np.vstack(solid or chunks)
    if len(allp) > max_pts:
        allp = allp[np.random.default_rng(0).choice(len(allp), max_pts, replace=False)]
    return allp


def min_area_rect(xy):
    """Minimum-area rectangle (rotating calipers on the convex hull): (long side angle, length, width, center)."""
    import numpy as np

    pts = np.unique(xy.round(6), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):  # z of the 2D cross product (np.cross on 2D vectors is deprecated)
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h
    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        a = np.arctan2(e[1], e[0])
        c, s_ = np.cos(a), np.sin(a)
        r = hull @ np.array([[c, -s_], [s_, c]])
        lo, hi = r.min(0), r.max(0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            ctr = ((lo + hi) / 2) @ np.array([[c, s_], [-s_, c]])
            best = (area, a, hi - lo, ctr)
    _, a, (d0, d1), ctr = best
    if d1 > d0:
        a, d0, d1 = a + np.pi / 2, d1, d0
    return a, d0, d1, ctr


def normalize(raw_usd, out_usd) -> list[float] | None:
    """Write a wrapper USD that stands the model upright (long side on X), centered, on z=0, unscaled.

    Returns [length, width, height] measured on the exact composed vertices. The length comes from the
    minimum-area rectangle seen from above: an axis-aligned box inflates models rotated inside their file.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    src = Usd.Stage.Open(str(raw_usd))
    up_i = 1 if UsdGeom.GetStageUpAxis(src) == UsdGeom.Tokens.y else 2
    # The converter puts its own rotation R on the root prim. The upright frame is p_local * to_z * R
    # (to_z = +90 on X for Y-up stages); to_z and R do not commute. The wrapper clears the root's
    # xformOps and applies a single matrix to p_local.
    R = UsdGeom.Xformable(src.GetDefaultPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    to_z = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)) if up_i == 1 else Gf.Matrix4d(1)
    U = to_z * R
    pw = world_points(src)
    if len(pw) < 4:
        return None
    homog = lambda a: np.hstack([a, np.ones((len(a), 1))])  # noqa: E731
    p_local = (homog(pw) @ np.array(R.GetInverse()))[:, :3]
    pz = (homog(p_local) @ np.array(U))[:, :3]
    ang, length, width, ctr = min_area_rect(pz[:, :2])
    if length <= 0:
        return None
    m_up = (Gf.Matrix4d().SetTranslate(Gf.Vec3d(-ctr[0], -ctr[1], -pz[:, 2].min()))
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -np.degrees(ang))))
    m_local = U * m_up
    fin = (homog(p_local) @ np.array(m_local))[:, :3]
    size = fin.max(0) - fin.min(0)

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    geo = UsdGeom.Xform.Define(stage, "/Asset/Geo")
    # Relative reference: the cache can be moved or mounted elsewhere (Nebius) without rewriting files.
    geo.GetPrim().GetReferences().AddReference(os.path.relpath(raw_usd, Path(out_usd).parent))
    xf = UsdGeom.Xformable(geo)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m_local)
    stage.GetRootLayer().Save()
    return [float(v) for v in size]
