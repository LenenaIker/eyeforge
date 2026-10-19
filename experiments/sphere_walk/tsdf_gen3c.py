"""GEN3C's geometry, not only its colour, into the site's scene: every frame of the batch (gen3c_batch.py) with
MoGe's depth, scaled to our proxy, fused with the panorama's own cuts in one TSDF volume (Open3D). Behind the columns
GEN3C shows what is there at its own depth, where our back layer (B_layers) guessed one flat distance and stepped.
The panorama's cuts go in with our own depth and more weight, so what the sphere saw keeps its real
colour and place. B_layers' scene stays behind
it as a backstop (BACKSTOP), seen only where the fusion has nothing. Run in the SIM2CHIP_SCENE_PYTHON env
(torch, MoGe, Open3D) after gen3c_batch.py fetch:

    python experiments/sphere_walk/tsdf_gen3c.py --dir data/sphere_test/<place>

Writes <dir>/walk/TS_scene/ (scene.npz with a vertex-coloured "fused" layer and the far proxy, layers.json,
info.json) for render_walk.py --scene."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("MOGE_DEPS", "/media/lenena-iker/Crucial/tools/moge_deps"))
sys.path.insert(0, os.environ.get("MOGE_ROOT", "/media/lenena-iker/Crucial/tools/MoGe"))
from fuse_gen3c import MOGE, frames  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

BACKSTOP = 1.08   # our proxy behind the fused mesh, this much further from the capture point
VOXEL_M = 0.04
TRUNC_M = 0.2
MAX_DEPTH_M = 25.0
REAL_WEIGHT = 4   # the panorama's cuts are integrated this many times
SCALE = 2         # frames and cuts integrated at half resolution
HEADINGS = tuple(range(0, 360, 30))
SEEN_TOL = 0.04   # a fused corner this near (share of the distance) to where the panorama measured it: seen
PITCHES = (0.0, 40.0, 75.0)  # the panorama's cuts: the horizon and down to the feet (GEN3C's frames are level)
# The fused ground ripples by millimetres at the voxel's size; seen from 2 m higher than the capture point its tilted
# faces showed their backs as dark arcs (2026-10-19). Corners this near the ground plane go onto it, and the
# faces that end up facing down (the ripple's folds) are dropped. A flat floor is also what objects stand on.
GROUND_SNAP_M = 0.03


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--add", nargs="*", default=[],
                    help="more clips, as <folder under walk>/<clip name> (e.g. a gen3c_probe.py turn)")
    ap.add_argument("--out", default="TS_scene")
    args = ap.parse_args()
    d = Path(args.dir)
    w = d / "walk"
    src, out = w / "G_gen3c", w / args.out
    out.mkdir(exist_ok=True)
    t0 = time.time()
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    # A site without the straight batch (gen3c_batch.py) fuses only the added clips (nbv.py's rounds).
    paths = json.loads((src / "paths.json").read_text())["paths"] if (src / "paths.json").exists() else []
    for a in args.add:
        folder, name = a.split("/")
        p = next(q for q in json.loads((w / folder / "paths.json").read_text())["paths"] if q["name"] == name)
        paths.append({**p, "dir": w / folder})
    v0 = paths[0]["view"] if paths else {"fov_deg": 80.0, "size": [1280, 704]}  # GEN3C's frames (gen3c_walk)
    W, H = v0["size"]
    fx = capture.focal(W, v0["fov_deg"])
    intr = o3d.camera.PinholeCameraIntrinsic(W // SCALE, H // SCALE, fx / SCALE, fx / SCALE,
                                             W / SCALE / 2 - 0.5, H / SCALE / 2 - 0.5)
    moge = MoGeModel.from_pretrained(MOGE).cuda().eval()
    # Our proxy's points, for each image's depth scale (MoGe's depth is relative).
    proxy = np.load(w / "walk.npz")
    ppts = torch.as_tensor(proxy["proxy_pts"], dtype=torch.float32, device="cuda")

    def depth_of(img, pose):
        x = torch.as_tensor(img, device="cuda").permute(2, 0, 1).float() / 255
        with torch.no_grad():
            m = moge.infer(x, fov_x=v0["fov_deg"])
        dep = torch.nan_to_num(m["depth"], nan=0, posinf=0) * m["mask"]
        pose = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device="cuda")
        c = (ppts - pose[:3, 3]) @ pose[:3, :3]
        ok = c[:, 2] > 0.5
        u = (fx * c[ok, 0] / c[ok, 2] + W / 2).long()
        v = (fx * c[ok, 1] / c[ok, 2] + H / 2).long()
        z = c[ok, 2]
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zb = torch.full((H * W,), float("inf"), device="cuda")
        zb.scatter_reduce_(0, v[ins] * W + u[ins], z[ins], "amin")  # the proxy's nearest point per pixel
        zb = zb.reshape(H, W)
        have = torch.isfinite(zb) & (dep > 0) & (zb < MAX_DEPTH_M)
        k = torch.median(zb[have] / dep[have]) if int(have.sum()) > 500 else torch.tensor(1.0)
        dep = dep * k
        dep = F.interpolate(dep[None, None], scale_factor=1 / SCALE, mode="nearest")[0, 0]
        return dep.cpu().numpy().astype(np.float32), float(k)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=VOXEL_M, sdf_trunc=TRUNC_M,
                                                       color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pd = np.load(d / "proxy_distance.npz")
    pdist = np.where(pd["valid"], pd["distance"], 0).astype(np.float32)

    def proxy_depth(v):
        """A view from the capture point: our own z-depth, no estimate (what the sphere saw, where it saw it)."""
        rays_l = capture.view_rays(v)
        x, y = capture.to_pixel(render_plan.pano_rays(rays_l, R), pdist.shape[1], pdist.shape[0])
        dd = pdist[np.clip(np.round(y).astype(int), 0, pdist.shape[0] - 1), np.round(x).astype(int) % pdist.shape[1]]
        z = dd * (rays_l @ capture.rotation(v["yaw_deg"], v["pitch_deg"])[:, 2])
        return np.ascontiguousarray(z[SCALE // 2::SCALE, SCALE // 2::SCALE]).astype(np.float32), 1.0

    def add(img, pose, times=1, depth=None):
        dep, k = depth if depth is not None else depth_of(img, pose)
        small = np.ascontiguousarray(np.asarray(Image.fromarray(img).resize((W // SCALE, H // SCALE), Image.Resampling.BILINEAR)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(small), o3d.geometry.Image(dep), depth_scale=1.0, depth_trunc=MAX_DEPTH_M,
            convert_rgb_to_intensity=False)
        extr = np.linalg.inv(np.asarray(pose, dtype=np.float64))
        for _ in range(times):
            vol.integrate(rgbd, intr, extr)
        return k

    # The panorama's own cuts at the capture point (the never-seen ground painted, B_layers' front texture).
    tex = np.asarray(Image.open(w / "BL_scene" / "tex_front.png").convert("RGB"))
    scales = []
    for hd, pitch in [(a, b) for a in HEADINGS for b in PITCHES]:
        v = {**v0, "yaw_deg": float(hd), "pitch_deg": pitch}
        img = np.clip(capture.sample(tex, render_plan.pano_rays(capture.view_rays(v), R)) + 0.5, 0, 255).astype(np.uint8)
        scales.append(add(img, render_plan.camera_pose(v, h), REAL_WEIGHT, proxy_depth(v)))
    real = len(scales)
    n = 0
    for p in paths:
        vid = p.get("dir", src) / f"{p['name']}.mp4"
        if not vid.exists():
            continue
        for k, img in frames(vid).items():
            scales.append(add(img, p["poses"][k]))
            n += 1
        print(p["name"], f"{time.time() - t0:.0f} s", flush=True)
    mesh = vol.extract_triangle_mesh()
    mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices).astype(np.float32)
    Fc = np.asarray(mesh.triangles).astype(np.int32)
    C = srgb_to_linear(np.asarray(mesh.vertex_colors)).astype(np.float32)  # emitted as is: linear light
    V[np.abs(V[:, 2]) < GROUND_SNAP_M, 2] = 0
    flat = (V[Fc, 2] == 0).all(1)
    nz = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])[:, 2]
    Fc = Fc[~flat | (nz > 0)]
    # What the sphere saw keeps the panorama's full resolution: a fused face whose corners lie where the panorama
    # measured them (SEEN_TOL of our distance, from the capture point) takes the texture by UVs, the rest keeps
    # its vertex colours (GEN3C's paint).
    eye = np.array([0.0, 0.0, h], np.float32)
    lvl = np.stack([V[:, 0], h - V[:, 2], V[:, 1]], -1)  # Isaac's world back to the levelled frame
    rng = np.linalg.norm(lvl, axis=1)
    dirs_p = render_plan.pano_rays(lvl / np.maximum(rng[:, None], 1e-6), R)
    px, py = capture.to_pixel(dirs_p, pdist.shape[1], pdist.shape[0])
    measured = pdist[np.clip(np.round(py).astype(int), 0, pdist.shape[0] - 1), np.round(px).astype(int) % pdist.shape[1]]
    vseen = (measured > 0) & (np.abs(rng - measured) < SEEN_TOL * measured)
    tw, th = Image.open(w / "BL_scene" / "tex_front.png").size
    tx, ty = capture.to_pixel(dirs_p, tw, th)
    uv = np.stack([(tx + 0.5) / tw, 1 - (ty + 0.5) / th], -1).astype(np.float32)
    fseen = vseen[Fc].all(1)
    from prep import face_uvs
    seen_f, paint_f = Fc[fseen], Fc[~fseen]

    # Behind it, our whole front proxy pushed BACKSTOP further from the capture point: it shows only where the
    # fused mesh has nothing (far away, sky edges), never in front of it.
    # Both of B_layers' layers: the proxy is torn at depth steps and its back layer fills the tears.
    bl = np.load(w / "BL_scene" / "scene.npz")
    push = lambda p: eye + (p - eye) * BACKSTOP  # noqa: E731
    np.savez(out / "scene.npz", fused_pts=V, fused_faces=paint_f, fused_col=C,
             seen_pts=V, seen_faces=seen_f, seen_uv=face_uvs(uv, seen_f),
             far_pts=push(bl["front_pts"]), far_faces=bl["front_faces"], far_uv=bl["front_uv"],
             farback_pts=push(bl["back_pts"]), farback_faces=bl["back_faces"], farback_uv=bl["back_uv"])
    (out / "layers.json").write_text(json.dumps([{"name": "seen", "tex": "../BL_scene/tex_front.png", "winding": "own"},
                                                 {"name": "fused", "color": "vertex"},
                                                 {"name": "far", "tex": "../BL_scene/tex_front.png"},
                                                 {"name": "farback", "tex": "../BL_scene/tex_back.png"}]))
    # The sky for a look around in Isaac's GUI (render_walk.py --save): the erased panorama, levelled.
    from matrix3d_walk import levelled
    sky = levelled(np.asarray(Image.open(d / "background.jpg").convert("RGB")), R, (2048, 1024))
    Image.fromarray(np.clip(sky + 0.5, 0, 255).astype(np.uint8)).save(out / "sky.png")
    info = {"images": n + real, "gen3c_frames": n, "fused_faces": int(len(Fc)), "seen_share": round(float(fseen.mean()), 3),
            "depth_scale": [round(float(q), 2) for q in np.percentile(scales[real:], [10, 50, 90])],
            "seconds": round(time.time() - t0, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
