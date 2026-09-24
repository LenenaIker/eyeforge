"""Convert downloaded models to USD, stand them upright, center them and measure them.

Conversion is class independent: the real-world scale is applied at render time from the spec, so a
reclassified asset needs no reconversion and the cache is shared by every job.

Cache:  usd/raw/<uid>.usd            converter output
        usd/norm/<uid>_v<N>.usd      wrapper that references ../raw/<uid>.usd with an upright transform
        usd/norm/<uid>_v<N>.json     {"dims": [length, width, height]} in the asset's own units
Output: convert_assets/assets.json   manifest entries + usd (cache relative) + dims
"""
from __future__ import annotations

import json

from .base import Context, Stage

NORM_VERSION = 2  # bump when normalize() changes


class ConvertAssets(Stage):
    name = "convert_assets"
    runtime = "isaac"
    uses_cache = True
    after = ("fetch_assets",)
    version = str(NORM_VERSION)

    def enabled(self, spec) -> bool:
        return any(c.assets for c in spec.classes)

    def run(self, ctx: Context) -> dict:
        import asyncio

        manifest = json.loads((ctx.job.stage_dir("fetch_assets") / "manifest.json").read_text())
        ctx.cache.usd_raw.mkdir(parents=True, exist_ok=True)
        ctx.cache.usd_norm.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        done, skipped = [], []
        for i, m in enumerate(manifest):
            raw = ctx.cache.usd_raw / f"{m['uid']}.usd"
            norm = ctx.cache.usd_norm / f"{m['uid']}_v{NORM_VERSION}.usd"
            meta = norm.with_suffix(".json")
            glb = ctx.cache.root / m["glb"]
            if not raw.exists() and (not glb.exists() or glb.stat().st_size == 0):
                ctx.log(f"missing or empty download {m['glb']}")  # the converter would crash on it
                skipped.append(m["uid"])
                continue
            if not raw.exists() and not loop.run_until_complete(_convert(str(glb), str(raw))):
                skipped.append(m["uid"])
                continue
            if not meta.exists():
                dims = normalize(raw, norm)
                if dims is None:
                    skipped.append(m["uid"])
                    continue
                meta.write_text(json.dumps({"dims": dims}))
            dims = json.loads(meta.read_text())["dims"]
            done.append({**m, "usd": str(norm.relative_to(ctx.cache.root)), "dims": dims})
            if (i + 1) % 25 == 0:
                ctx.log(f"[{i + 1}/{len(manifest)}]")
        if not done:
            raise RuntimeError(f"no asset could be converted ({len(skipped)} failed), see the log above")
        (ctx.out(self.name) / "assets.json").write_text(json.dumps(done, indent=1))
        ctx.log(f"{len(done)} assets ready, {len(skipped)} failed")
        return {"assets": len(done), "failed": skipped}


async def _convert(src: str, dst: str) -> bool:
    import omni.kit.asset_converter as converter

    c = converter.AssetConverterContext()
    c.ignore_materials = False
    c.ignore_animations = True
    c.ignore_camera = True
    c.ignore_light = True
    c.embed_textures = True
    task = converter.get_instance().create_converter_task(src, dst, None, c)
    ok = await task.wait_until_finished()
    if not ok:
        print("[sim2chip] conversion error", src, task.get_error_message(), flush=True)
    return ok


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

    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
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
    geo.GetPrim().GetReferences().AddReference(f"../raw/{raw_usd.name}")
    xf = UsdGeom.Xformable(geo)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m_local)
    stage.GetRootLayer().Save()
    return [float(v) for v in size]
