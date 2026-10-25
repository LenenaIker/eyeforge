"""Far ground as one shared image (test L0b): the scene's ground seen from above, painted once beyond the walks'
reach + MARGIN, then turned back into a ground mesh that replaces the mesh's torn far ground. One image for the whole
ground, so nothing can contradict itself the way views painted one by one did (ws_farfill.py, FF/FX).

1. ortho: the M1 mesh (TSDF of the splat, mesh_M1/mesh.ply) seen from below through its up-facing triangles only, so
   each CELL metres cell gets the lowest up-facing surface: height and colour. A ground plane is fitted to the cells
   within the reach (trimmed); a cell is ground when it is smooth (within SMOOTH_TOL of its neighbours' median, so
   ramps count) and not far above the plane (MAX_ABOVE: wall tops, canopies). The fill mask is every cell between the
   reach + MARGIN and FAR metres from the centre that is not ground.
2. paint: LaMa (SIM2CHIP_SAM3_PYTHON env) or Qwen-Image-Edit (SIM2CHIP_ERASE_PYTHON env, empty cells in magenta)
   paints every cell that is not ground (left black, they read as black ground to both), and only the fill mask
   takes its pixels.
3. mesh: a height-field mesh from the reach + MARGIN to FAR (heights: the measured ground where there is one, and
   away from it the nearest measured height fading into the plane over FADE metres), coloured by the painted
   orthophoto; the
   M1 mesh loses its far ground triangles (beyond the seam, within GROUND_TOL of the surface) and gains this one.
   The splat loses its far ground Gaussians the same way (splat variant, for a USD with the splat and this ground).
4. usd: the meshes to USD (edge-sdg env, usd-core) and a stage with the cropped splat and the ground mesh.

    <panogen venv> experiments/worldstereo/ws_ground.py ortho --case photo21_np
    <SIM2CHIP_SAM3_PYTHON> experiments/worldstereo/ws_ground.py paint --model lama
    <SIM2CHIP_ERASE_PYTHON> experiments/worldstereo/ws_ground.py paint --model qwen
    <panogen venv> experiments/worldstereo/ws_ground.py mesh --model lama|qwen
    <gsplat env> experiments/worldstereo/ws_ground.py crop          # G2 without its far ground Gaussians
    <edge-sdg> experiments/worldstereo/ws_ground.py usd --model lama|qwen
    <panogen venv> experiments/worldstereo/ws_ground.py sheet

Writes data/worldstereo_test/<case>/ground/."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

CELL = 0.05        # metres per orthophoto pixel
FAR = 25.0         # metres from the centre the ground reaches
MARGIN = 1.0       # as ws_farfill.py: the far zone starts this far beyond the walks' reach
GROUND_TOL = 0.3   # metres above the new ground where the mesh's or splat's old far ground is dropped
SMOOTH_TOL = 0.15  # metres from the median of its MEDIAN_M neighbourhood a ground cell may be
MEDIAN_M = 1.0
MAX_ABOVE = 1.5    # metres above the plane where the ground stops (wall tops, canopies)
UP = 0.85          # |normal z| of an up-facing triangle
FADE = 3.0         # metres over which the measured height, carried past the data, fades into the plane
MAGENTA = (255, 0, 255)
QWEN_PROMPT = ('This is a top-down orthophoto of a narrow country road between hedges. Remove the magenta areas from '
               'the image, filling them with the road that continues in its own direction and width, and grass, soil '
               'and leaves beside it, while preserving the remaining elements, maintaining realism and original '
               'details.')


def out_dir(case: str) -> Path:
    return ws.DATA / case / "ground"


def reach_of(case: str) -> float:
    w2c = np.load(ws.DATA / case / "splat" / "views.npz")["w2c"]
    return float(np.max(np.linalg.norm(np.linalg.inv(w2c)[:, :2, 3], axis=1)))


def grid():
    n = int(round(2 * FAR / CELL))
    c = (np.arange(n) + 0.5) * CELL - FAR
    x, y = np.meshgrid(c, -c)  # image rows go from +y (top) to -y, columns from -x to +x
    return x, y


def plane(x, y):
    return np.stack([np.ones_like(x), x, y], -1)


def fit_plane(x, y, z, iters=4, tol=0.3):
    """Least squares plane z(x, y), trimmed: each round drops the points beyond tol metres."""
    keep = np.ones(len(z), bool)
    for _ in range(iters):
        coef = np.linalg.lstsq(plane(x[keep], y[keep]), z[keep], rcond=None)[0]
        keep = np.abs(plane(x, y) @ coef - z) < tol
    return coef


def ortho(case: str):
    import open3d as o3d
    m = o3d.t.io.read_triangle_mesh(str(ws.DATA / case / "mesh_M1" / "mesh.ply"))
    m.compute_triangle_normals()
    up = np.abs(m.triangle.normals.numpy()[:, 2]) > UP
    v = m.vertex.positions.numpy()
    t = m.triangle.indices.numpy()[up]
    col = m.vertex.colors.numpy().astype(np.float32)
    if col.max() > 1.5:
        col /= 255
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(t.astype(np.uint32)))
    x, y = grid()
    rays = np.stack([x, y, np.full_like(x, -20.0), 0 * x, 0 * x, np.ones_like(x)], -1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    tid, uv, th = hit["primitive_ids"].numpy(), hit["primitive_uvs"].numpy(), hit["t_hit"].numpy()
    ok = tid != scene.INVALID_ID
    z = np.where(ok, th - 20.0, np.nan)
    rgb = np.zeros(x.shape + (3,), np.float32)
    tt = t[tid[ok]]
    b = np.stack([1 - uv[ok, 0] - uv[ok, 1], uv[ok, 0], uv[ok, 1]], -1)
    rgb[ok] = (col[tt] * b[..., None]).sum(1)
    reach = reach_of(case)
    r = np.hypot(x, y)
    near = ok & (r <= reach + MARGIN)
    coef = fit_plane(x[near], y[near], z[near])
    zp = plane(x, y) @ coef
    from scipy import ndimage
    med = ndimage.median_filter(np.where(ok, z, zp), size=int(round(MEDIAN_M / CELL)) | 1)
    ground = ok & (z < zp + MAX_ABOVE) & (np.abs(z - med) < SMOOTH_TOL)
    fill = (r > reach + MARGIN) & (r < FAR) & ~ground
    out = out_dir(case)
    out.mkdir(exist_ok=True)
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img[~ground] = 0
    Image.fromarray(img).save(out / "ortho.png")
    Image.fromarray((fill * 255).astype(np.uint8)).save(out / "fill.png")
    np.savez_compressed(out / "ortho.npz", z=np.where(ground, z, np.nan).astype(np.float32), coef=coef,
                        ground=ground, fill=fill)
    meta = {"case": case, "reach_m": round(reach, 2), "seam_m": round(reach + MARGIN, 2), "far_m": FAR, "cell_m": CELL,
            "ground_share_near": round(float(ground[r <= reach + MARGIN].mean()), 3),
            "ground_share_far": round(float(ground[(r > reach + MARGIN) & (r < FAR)].mean()), 3),
            "surface": [round(float(c), 4) for c in coef]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


def paint(case: str, model: str):
    out = out_dir(case)
    img = Image.open(out / "ortho.png").convert("RGB")
    fill = np.asarray(Image.open(out / "fill.png")) > 127
    unknown = ~np.load(out / "ortho.npz")["ground"]
    sys.path.insert(0, str(ws.ROOT / "sim2chip" / "photo2scene" / "steps"))
    if model == "lama":
        from bg_prepare import inpaint
        painted = inpaint(img, unknown)
    else:
        import torch
        from erase_qwen import load
        pipe = load(4, lightning=True)
        src = np.asarray(img).copy()
        src[unknown] = MAGENTA
        painted = pipe(image=[Image.fromarray(src)], prompt=QWEN_PROMPT, negative_prompt=" ", true_cfg_scale=1.0,
                       num_inference_steps=4, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        painted = painted.resize(img.size, Image.Resampling.LANCZOS)
    a = np.asarray(Image.fromarray((fill * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
                   np.float32)[..., None] / 255
    a = np.maximum(a, fill[..., None])
    res = np.asarray(painted, np.float32) * a + np.asarray(img, np.float32) * (1 - a)
    Image.fromarray(np.clip(res + 0.5, 0, 255).astype(np.uint8)).save(out / f"ortho_{model}.png")
    print(out / f"ortho_{model}.png")


def heights(case: str):
    """The ground's height per cell: the measured ground (smoothed) where there is one; elsewhere the nearest
    measured height above or below the plane, fading into the plane over FADE metres."""
    from scipy import ndimage
    z = np.load(out_dir(case) / "ortho.npz")
    meta = json.loads((out_dir(case) / "meta.json").read_text())
    x, y = grid()
    zp = plane(x, y) @ z["coef"]
    meas = np.nan_to_num(z["z"] - zp)
    w = z["ground"].astype(np.float32)
    s = 0.4 / CELL
    num, den = ndimage.gaussian_filter(meas * w, s), ndimage.gaussian_filter(w, s)
    known = den > 0.2
    resid = np.where(known, num / np.maximum(den, 1e-6), 0)
    dist, (ii, jj) = ndimage.distance_transform_edt(~known, return_indices=True)
    resid = np.where(known, resid, resid[ii, jj] * np.exp(-dist * CELL / FADE))
    return x, y, zp + resid, meta, zp


def mesh(case: str, model: str):
    import open3d as o3d
    out = out_dir(case)
    x, y, h, meta, _ = heights(case)
    r = np.hypot(x, y)
    col = np.asarray(Image.open(out / f"ortho_{model}.png").convert("RGB"), np.float32) / 255
    inside = (r >= meta["seam_m"] - CELL) & (r <= FAR)
    idx = -np.ones(x.shape, np.int64)
    idx[inside] = np.arange(inside.sum())
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    q = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    tris = np.concatenate([np.stack([a[q], c[q], b[q]], -1), np.stack([b[q], c[q], d[q]], -1)])
    g = o3d.t.geometry.TriangleMesh()
    g.vertex.positions = o3d.core.Tensor(np.stack([x[inside], y[inside], h[inside]], -1).astype(np.float32))
    g.vertex.colors = o3d.core.Tensor(col[inside].astype(np.float32))
    g.triangle.indices = o3d.core.Tensor(tris.astype(np.int32))
    o3d.t.io.write_triangle_mesh(str(out / f"ground_{model}.ply"), g)
    # M1 without its far ground: triangles past the seam whose centre lies within GROUND_TOL of the new ground
    m = o3d.t.io.read_triangle_mesh(str(ws.DATA / case / "mesh_M1" / "mesh.ply"))
    v, t = m.vertex.positions.numpy(), m.triangle.indices.numpy()
    cen = v[t].mean(1)
    gi = np.clip(((FAR - cen[:, 1]) / CELL).astype(int), 0, x.shape[0] - 1)
    gj = np.clip(((cen[:, 0] + FAR) / CELL).astype(int), 0, x.shape[1] - 1)
    drop = (np.hypot(cen[:, 0], cen[:, 1]) > meta["seam_m"]) & (cen[:, 2] < h[gi, gj] + GROUND_TOL)
    cols = m.vertex.colors.numpy().astype(np.float32)
    if cols.max() > 1.5:
        cols /= 255
    n0 = len(v)
    merged = o3d.t.geometry.TriangleMesh()
    merged.vertex.positions = o3d.core.Tensor(np.concatenate([v, g.vertex.positions.numpy()]).astype(np.float32))
    merged.vertex.colors = o3d.core.Tensor(np.concatenate([cols, col[inside]]).astype(np.float32))
    merged.triangle.indices = o3d.core.Tensor(np.concatenate([t[~drop], tris + n0]).astype(np.int32))
    o3d.t.io.write_triangle_mesh(str(out / f"M1_ground_{model}.ply"), merged)
    print(f"ground {len(tris)} tris; M1 lost {int(drop.sum())} far ground tris -> {out / f'M1_ground_{model}.ply'}")


def crop(case: str, arm: str = "G2"):
    """The arm's splat without the Gaussians past the seam that lie on the ground (within GROUND_TOL above it)."""
    from plyfile import PlyData, PlyElement
    x, y, h, meta, _ = heights(case)
    p = PlyData.read(ws.DATA / case / "splat" / "arms" / arm / "splat.ply")
    v = p["vertex"].data
    px, py, pz = v["x"], v["y"], v["z"]
    gi = np.clip(((FAR - py) / CELL).astype(int), 0, x.shape[0] - 1)
    gj = np.clip(((px + FAR) / CELL).astype(int), 0, x.shape[1] - 1)
    drop = (np.hypot(px, py) > meta["seam_m"]) & (pz < h[gi, gj] + GROUND_TOL)
    dst = out_dir(case) / f"{arm}_crop.ply"
    PlyData([PlyElement.describe(v[~drop], "vertex")]).write(dst)
    print(f"{arm}: dropped {int(drop.sum())} of {len(v)} Gaussians -> {dst}")


def ply_to_usd(src: Path, dst: Path):
    from plyfile import PlyData
    from pxr import Sdf, Usd, UsdGeom, Vt
    p = PlyData.read(src)
    v, f = p["vertex"], p["face"]
    pts = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    names = v.data.dtype.names
    col = np.stack([v[k] for k in (("red", "green", "blue") if "red" in names else ("r", "g", "b"))], -1)
    col = col.astype(np.float32) / (255 if col.dtype == np.uint8 else 1)
    idx = np.vstack(f["vertex_indices"]).astype(np.int32)
    st = Usd.Stage.CreateNew(str(dst))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(st, "/World/mesh")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(idx), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx.ravel()))
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                         UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(col))
    st.GetRootLayer().Save()


def usd(case: str, model: str):
    """M1 + ground and the ground alone as USD; a stage that references the cropped splat's USDZ and the ground."""
    from pxr import Usd, UsdGeom
    out = out_dir(case)
    for name in (f"ground_{model}", f"M1_ground_{model}"):
        ply_to_usd(out / f"{name}.ply", out / f"{name}.usd")
    st = Usd.Stage.CreateNew(str(out / f"G2crop_ground_{model}.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(root.GetPrim())
    st.DefinePrim("/World/splat").GetReferences().AddReference("./G2_crop.usdz")
    st.DefinePrim("/World/ground").GetReferences().AddReference(f"./ground_{model}.usd")
    st.GetRootLayer().Save()
    print(out / f"G2crop_ground_{model}.usda")


def sheet(case: str, models: list[str]):
    """Orthophotos before and after, and the meshes from ws_farfill.py's views (ray cast, as mesh_L0a's sheet)."""
    import open3d as o3d
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ws_farfill import VIEWS, view_w2c
    out = out_dir(case)
    W, H, F = 512, 288, 144.0
    j, i = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    cam = np.stack([(j - W / 2) / F, (i - H / 2) / F, np.ones_like(j)], -1)

    def render(path):
        m = o3d.t.io.read_triangle_mesh(str(path))
        cols = m.vertex.colors.numpy().astype(np.float32)
        if cols.max() > 1.5:
            cols /= 255
        tris = m.triangle.indices.numpy()
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(m)
        ims = []
        for _, yaw, d, back in VIEWS:
            c2w = np.linalg.inv(view_w2c(yaw, d, back))
            dirs = cam @ c2w[:3, :3].T
            dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
            rays = np.concatenate([np.broadcast_to(c2w[:3, 3], dirs.shape), dirs], -1).astype(np.float32)
            hit = sc.cast_rays(o3d.core.Tensor(rays))
            tid, uv = hit["primitive_ids"].numpy(), hit["primitive_uvs"].numpy()
            ok = tid != sc.INVALID_ID
            img = np.zeros((H, W, 3), np.float32)
            b = np.stack([1 - uv[ok, 0] - uv[ok, 1], uv[ok, 0], uv[ok, 1]], -1)
            img[ok] = (cols[tris[tid[ok]]] * b[..., None]).sum(1)
            ims.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        return ims

    rows = [("M1", ws.DATA / case / "mesh_M1" / "mesh.ply")] + \
           [(f"M1 + ground ({m})", out / f"M1_ground_{m}.ply") for m in models]
    renders = {lab: render(p) for lab, p in rows}
    meta = json.loads((out / "meta.json").read_text())
    for k, (label, *_) in enumerate(VIEWS):
        sh = Image.new("RGB", (W * len(rows), H + 24), (20, 20, 20))
        d = ImageDraw.Draw(sh)
        d.text((6, 6), f"{label}  (seam {meta['seam_m']} m)", fill=(255, 255, 255))
        for c, (lab, _) in enumerate(rows):
            sh.paste(Image.fromarray(renders[lab][k]), (c * W, 24))
            d.text((c * W + 6, 30), lab, fill=(255, 255, 0))
        sh.save(out / f"view_{k:02d}.jpg", quality=92)
    tiles = [("orthophoto (ground only)", out / "ortho.png")] + [(f"painted: {m}", out / f"ortho_{m}.png")
                                                                 for m in models]
    size = 700
    sh = Image.new("RGB", (size * len(tiles), size + 24), (20, 20, 20))
    d = ImageDraw.Draw(sh)
    for c, (lab, p) in enumerate(tiles):
        im = Image.open(p).convert("RGB").resize((size, size))
        dr = ImageDraw.Draw(im)
        for rad in (meta["seam_m"], FAR):
            e = rad / (2 * FAR) * size
            dr.ellipse([size / 2 - e, size / 2 - e, size / 2 + e, size / 2 + e], outline=(255, 0, 0))
        sh.paste(im, (c * size, 24))
        d.text((c * size + 6, 6), lab + "  (red: seam and far edge)", fill=(255, 255, 0))
    sh.save(out / "sheet_ortho.jpg", quality=90)
    views = [Image.open(out / f"view_{k:02d}.jpg") for k in range(len(VIEWS))]
    sv = Image.new("RGB", (views[0].width, sum(v.height for v in views)))
    for k, v in enumerate(views):
        sv.paste(v, (0, k * v.height))
    sv.save(out / "sheet_views.jpg", quality=88)
    print(out / "sheet_ortho.jpg", out / "sheet_views.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ortho", "paint", "mesh", "crop", "usd", "sheet"])
    ap.add_argument("--case", default="photo21_np")
    ap.add_argument("--model", choices=["lama", "qwen"], default="lama")
    ap.add_argument("--models", nargs="+", default=["lama", "qwen"])
    a = ap.parse_args()
    {"ortho": lambda: ortho(a.case), "paint": lambda: paint(a.case, a.model), "mesh": lambda: mesh(a.case, a.model),
     "crop": lambda: crop(a.case), "usd": lambda: usd(a.case, a.model),
     "sheet": lambda: sheet(a.case, a.models)}[a.cmd]()
