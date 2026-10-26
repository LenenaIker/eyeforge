"""A mesh straight from WorldStereo's frames, no splat (test M2): every view of ws_splat.py's views.npz (the panorama's
27 cuts and every generated frame, with their metric depth: the panorama's points where the guide saw something,
MoGe-2 scaled to them in the holes) fused into one TSDF at the cameras the walks asked for. M1 fused the G2 splat's
depth instead, so it carried the splat's floaters and blur; here each pixel comes from a frame.

    <panogen venv> experiments/worldstereo/ws_mesh.py tsdf --case e2e_photo21    # Open3D, CUDA
    <edge-sdg> experiments/worldstereo/ws_mesh.py usd --case e2e_photo21
    <panogen venv> experiments/worldstereo/ws_mesh.py sheet --case e2e_photo21   # mesh next to the G2 splat
    <panogen venv> experiments/worldstereo/ws_mesh.py tsdf --case e2e_photo21 --arm D1   # a 2DGS arm's renders

With --arm, the views are that splat arm's own renders (ws_splat.py, renders.npz: its colour and median depth at every
view) and the mesh goes to mesh_<arm>/.

Writes data/worldstereo_test/<case>/mesh_M2/: mesh.ply, mesh.usd, info.json, sheet.jpg."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

VOXEL = 0.06       # metres (M1: 4 cm, but to 40 m the grid outgrows the 3090)
TRUNC = 4.0        # truncation, in voxels
DEPTH_MAX = 40.0   # metres; ws_splat.MAX_DEPTH, beyond it the guide squashed the far zone (ws_test.prep)
BLOCKS = 120000    # voxel blocks the grid may hold (~10 GB on the GPU)
MIN_WEIGHT = 2.0   # a surface needs at least this many observations (one view alone leaves floaters)
MIN_TRIS = 2000    # connected pieces smaller than this are dropped
SKY = 0            # ws_splat.py's pixel code for sky


def out_dir(case: str, arm: str | None = None) -> Path:
    return ws.DATA / case / (f"mesh_{arm}" if arm else "mesh_M2")


def tsdf(case: str, arm: str | None = None):
    import open3d as o3d
    import open3d.core as o3c
    t0 = time.time()
    dev = o3c.Device("CUDA:0")
    z = np.load(ws.DATA / case / "splat" / ("arms/" + arm + "/renders.npz" if arm else "views.npz"))
    img, code, depth, w2c = z["img"], z["code"], z["depth"], z["w2c"]
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
                                        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                        attr_channels=((1), (1), (3)), voxel_size=VOXEL, block_resolution=16,
                                        block_count=BLOCKS, device=dev)
    K = o3c.Tensor(np.array(ws.intrinsic()), o3c.float64)
    for k in range(len(img)):
        d = np.where(code[k] == SKY, 0, depth[k].astype(np.float32))
        dt = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d))).to(dev)
        ct = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(img[k].astype(np.float32) / 255))).to(dev)
        ext = o3c.Tensor(w2c[k].astype(np.float64))
        blocks = vbg.compute_unique_block_coordinates(dt, K, ext, depth_scale=1.0, depth_max=DEPTH_MAX,
                                                      trunc_voxel_multiplier=TRUNC)
        vbg.integrate(blocks, dt, ct, K, K, ext, depth_scale=1.0, depth_max=DEPTH_MAX, trunc_voxel_multiplier=TRUNC)
    mesh = vbg.extract_triangle_mesh(weight_threshold=MIN_WEIGHT).to_legacy()
    n0 = len(mesh.triangles)
    tri_cluster, count, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(count)[np.asarray(tri_cluster)] < MIN_TRIS)
    mesh.remove_unreferenced_vertices()
    out = out_dir(case, arm)
    out.mkdir(exist_ok=True)
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    info = {"views": len(img), "voxel": VOXEL, "triangles_raw": n0, "triangles": len(mesh.triangles),
            "seconds": round(time.time() - t0)}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(info, out / "mesh.ply")


def usd(case: str, arm: str | None = None):
    from ws_ground import ply_to_usd
    ply_to_usd(out_dir(case, arm) / "mesh.ply", out_dir(case, arm) / "mesh.usd")
    print(out_dir(case, arm) / "mesh.usd")


def cameras():
    """ws_splat.py's novel views (yaw 0/90/180/270: raised 1.4 m looking down, then 2.5 m to the side)."""
    cams = []
    for yaw in (0, 90, 180, 270):
        cams.append((f"yaw {yaw} up 1.4 m", ws.w2c([0, 0, 1.4], yaw, -25.0)))
    for yaw in (0, 90, 180, 270):
        r = math.radians(yaw + 20)
        cams.append((f"yaw {yaw} 2.5 m off, beside", ws.w2c([2.5 * math.cos(r), 2.5 * math.sin(r), 0], yaw, -5.0)))
    return cams


def render(path: Path, cams) -> list[np.ndarray]:
    """Ray-cast the coloured mesh at ws_test's frame size."""
    import open3d as o3d
    m = o3d.t.io.read_triangle_mesh(str(path))
    cols = m.vertex.colors.numpy().astype(np.float32)
    cols = cols / 255 if cols.max() > 1.5 else cols
    tris = m.triangle.indices.numpy()
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(m)
    j, i = np.meshgrid(np.arange(ws.W) + 0.5, np.arange(ws.H) + 0.5)
    cam = np.stack([(j - ws.W / 2) / ws.F, (i - ws.H / 2) / ws.F, np.ones_like(j)], -1)
    ims = []
    for _, w2c in cams:
        c2w = np.linalg.inv(w2c)
        dirs = cam @ c2w[:3, :3].T
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        rays = np.concatenate([np.broadcast_to(c2w[:3, 3], dirs.shape), dirs], -1).astype(np.float32)
        hit = sc.cast_rays(o3d.core.Tensor(rays))
        tid, uv = hit["primitive_ids"].numpy(), hit["primitive_uvs"].numpy()
        ok = tid != sc.INVALID_ID
        im = np.zeros((ws.H, ws.W, 3), np.float32)
        b = np.stack([1 - uv[ok, 0] - uv[ok, 1], uv[ok, 0], uv[ok, 1]], -1)
        im[ok] = (cols[tris[tid[ok]]] * b[..., None]).sum(1)
        ims.append((np.clip(im, 0, 1) * 255).astype(np.uint8))
    return ims


def sheet(case: str, arm: str | None = None):
    """The G2 splat's novel views (its novel.npz) over the same views of the mesh; with an arm, rows G2 splat, the
    arm's splat, M2 and the arm's mesh."""
    if arm:
        return sheet_arm(case, arm)
    cams = cameras()
    nov = np.load(ws.DATA / case / "splat" / "arms" / "G2" / "novel.npz")
    by_label = {str(lab): im for lab, im in zip(nov["label"], nov["img"])}
    mesh = render(out_dir(case) / "mesh.ply", cams)
    info = json.loads((out_dir(case) / "info.json").read_text())
    cw, ch, pad = 400, 231, 30
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    rows = [("G2 splat", [by_label[lab] for lab, _ in cams[:4]]), ("M2 mesh", mesh[:4]),
            ("G2 splat", [by_label[lab] for lab, _ in cams[4:]]), ("M2 mesh", mesh[4:])]
    sh = Image.new("RGB", (cw * 4, pad + len(rows) * ch), (0, 0, 0))
    ImageDraw.Draw(sh).text((8, 4), f"{case}: G2 splat vs M2 mesh from the frames ({info['triangles'] / 1e6:.1f}M "
                                     f"triangles, {VOXEL * 100:.0f} cm voxels)", fill=(255, 255, 255), font=font)
    for r, (name, ims) in enumerate(rows):
        for k, im in enumerate(ims):
            t = Image.fromarray(im).resize((cw, ch))
            ImageDraw.Draw(t).text((6, 4), f"{name}: {cams[(r // 2) * 4 + k][0]}", fill=(255, 255, 0), font=small)
            sh.paste(t, (k * cw, pad + r * ch))
    sh.save(out_dir(case) / "sheet.jpg", quality=88)
    print(out_dir(case) / "sheet.jpg")


def sheet_arm(case: str, arm: str):
    cams = cameras()
    nov = {a: dict(zip(map(str, z["label"]), z["img"]))
           for a in ("G2", arm) for z in [np.load(ws.DATA / case / "splat" / "arms" / a / "novel.npz")]}
    rows = [("G2 splat", [nov["G2"][lab] for lab, _ in cams]), (f"{arm} 2DGS splat", [nov[arm][lab] for lab, _ in cams]),
            ("M2 mesh (MoGe depth)", render(out_dir(case) / "mesh.ply", cams)),
            (f"{arm} mesh (2DGS depth)", render(out_dir(case, arm) / "mesh.ply", cams))]
    info = json.loads((out_dir(case, arm) / "info.json").read_text())
    cw, ch, pad = 400, 231, 34
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    sh = Image.new("RGB", (cw * len(cams), pad + len(rows) * ch), (0, 0, 0))
    ImageDraw.Draw(sh).text((8, 4), f"{case}: G2 splat, {arm} 2DGS splat, M2 mesh, {arm} mesh "
                                     f"({info['triangles'] / 1e6:.1f}M triangles, {VOXEL * 100:.0f} cm voxels)",
                            fill=(255, 255, 255), font=font)
    for r, (label, ims) in enumerate(rows):
        for k, im in enumerate(ims):
            t = Image.fromarray(im).resize((cw, ch))
            ImageDraw.Draw(t).text((6, 4), f"{label}: {cams[k][0]}", fill=(255, 255, 0), font=small)
            sh.paste(t, (k * cw, pad + r * ch))
    sh.save(out_dir(case, arm) / "sheet.jpg", quality=85)
    print(out_dir(case, arm) / "sheet.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tsdf", "usd", "sheet"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--arm", help="a 2DGS splat arm of ws_splat.py (renders.npz) instead of the frames")
    a = ap.parse_args()
    {"tsdf": tsdf, "usd": usd, "sheet": sheet}[a.cmd](a.case, a.arm)
