"""The splat behind every frame of plan.json (plan.py), in the gsplat Python (SIM2CHIP_GSPLAT_PYTHON):

    <gsplat python> -m sim2chip.walkscene.background --dir <stage folder>

Per view: bg_NAME.png (the splat's colour, the panorama's sky where it shows nothing) and depth_NAME.npy (its
z-depth, float16, 0 where it shows nothing). Per object, the height of the splat's ground under it (a camera looking
straight down from just under eye height, below the trees' crowns); per view, the ground's median colour (Isaac's
ground plane takes it, so its exposure gain matches the background's); per scene, the sky's mean colour (the dome's
light). All in background.json; the scenes' folders come from plan.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import intrinsic, pano_lookup, w2c

ALPHA_MIN = 0.5          # a splat pixel with less opacity shows nothing
DOWN_FROM_M = 0.3        # the look-down camera's height above eye height (z = 0 is the walks' eye height)
DOWN_SIZE, DOWN_FOV = 16, 20.0
GROUND_SLACK_M = 0.6     # a ground under an object this far from the scene's is a roof or a crown: not taken
GROUND_BAND_M = 0.15     # a pixel within this of the plane counts as ground for its colour
GROUND_FAR_M = 30.0


def load_splat(path: Path, dev):
    """A 3DGS .ply as gsplat's rasterization arguments (wsroute.student.load_splat)."""
    import torch
    import torch.nn.functional as tF
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]

    def g(*k):
        return torch.as_tensor(np.stack([np.asarray(v[x], np.float32) for x in k], -1), device=dev)
    return dict(means=g("x", "y", "z"), quats=tF.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1),
                scales=torch.exp(g("scale_0", "scale_1", "scale_2")), opacities=torch.sigmoid(g("opacity")[:, 0]),
                colors=g("f_dc_0", "f_dc_1", "f_dc_2")[:, None])


def render(sp, w2cs: np.ndarray, K: np.ndarray, w: int, h: int):
    """RGB (uint8), z-depth and alpha of the splat for each world-to-camera, as numpy."""
    import torch
    from gsplat import rasterization
    dev = sp["means"].device
    m = torch.as_tensor(np.asarray(w2cs), dtype=torch.float32, device=dev)
    k = torch.as_tensor(K, dtype=torch.float32, device=dev)[None].repeat(len(m), 1, 1)
    with torch.no_grad():
        out, alpha, _ = rasterization(**sp, viewmats=m, Ks=k, width=w, height=h, sh_degree=0,
                                      render_mode="RGB+ED", near_plane=0.05, far_plane=500, packed=False)
    return ((out[..., :3].clamp(0, 1) * 255).byte().cpu().numpy(), out[..., 3].cpu().numpy(),
            alpha[..., 0].cpu().numpy())


def ground_under(sp, xys: list, camera_height: float) -> list[float]:
    """The splat's ground height (scene z) under each point."""
    if not xys:
        return []
    K = intrinsic(DOWN_SIZE, DOWN_SIZE, DOWN_FOV)
    mats = np.stack([w2c([x, y, DOWN_FROM_M], 0.0, -90.0) for x, y in xys]).astype(np.float32)
    _, dep, alpha = render(sp, mats, K, DOWN_SIZE, DOWN_SIZE)
    out = []
    for d, a in zip(dep, alpha):
        ok = a > ALPHA_MIN
        z = DOWN_FROM_M - float(np.median(d[ok])) if ok.sum() > DOWN_SIZE else -camera_height
        out.append(z if abs(z + camera_height) <= GROUND_SLACK_M else -camera_height)
    return out


def ground_colour(rgb, dep, m, K, plane_z: float) -> list[float] | None:
    """Median sRGB of the pixels whose splat point lies on the ground plane, None with too few."""
    h, w = dep.shape
    j, i = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam = np.stack([(j - K[0, 2]) / K[0, 0] * dep, (i - K[1, 2]) / K[1, 1] * dep, dep], -1)
    z = cam @ np.linalg.inv(m)[2, :3] + np.linalg.inv(m)[2, 3]
    on = (dep > 0) & (dep < GROUND_FAR_M) & (np.abs(z - plane_z) < GROUND_BAND_M)
    return np.median(rgb[on], 0).round(1).tolist() if on.sum() > 200 else None


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    plan = json.loads((d / "plan.json").read_text())
    scenes = json.loads((d / "scenes.json").read_text())   # {scene id: bundle folder}
    w, h = plan["size"]
    K = intrinsic(w, h, plan["fov_deg"])
    dev = torch.device("cuda")
    out = {"views": {}, "sky_rgb": {}}
    for sid, folder in scenes.items():
        views = [v for v in plan["views"] if v["scene"] == sid]
        if not views:
            continue
        meta = json.loads((Path(folder) / "scene.json").read_text())
        hgt = float(meta["camera_height"])
        sky = np.asarray(Image.open(Path(folder) / "sky.jpg").convert("RGB"))
        sp = load_splat(Path(folder) / "splat.ply", dev)
        sky_px = []
        for k in range(0, len(views), 8):
            batch = views[k:k + 8]
            mats = np.asarray([v["w2c"] for v in batch], np.float32)
            rgbs, deps, alphas = render(sp, mats, K, w, h)
            for v, m, rgb, dep, alpha in zip(batch, mats, rgbs, deps, alphas):
                empty = alpha < ALPHA_MIN
                back = pano_lookup(sky, m, K, w, h)
                a3 = alpha[..., None]
                col = np.where(empty[..., None], back, rgb * a3 + back * (1 - a3))
                Image.fromarray(col.clip(0, 255).astype(np.uint8)).save(d / f"bg_{v['name']}.png")
                np.save(d / f"depth_{v['name']}.npy", np.where(empty, 0, dep).astype(np.float16))
                sky_px.append(back[empty & (np.arange(h)[:, None] < h / 2)].reshape(-1, 3)[::7])
                zs = ground_under(sp, [o["xy"] for o in v["objects"]], hgt)
                plane = float(np.median(zs)) if zs else -hgt
                out["views"][v["name"]] = {"ground_z": [round(z, 3) for z in zs], "plane_z": round(plane, 3),
                                           "ground_rgb": ground_colour(rgb, dep * (~empty), m, K, plane),
                                           "sky_share": round(float(empty.mean()), 3)}
        px = np.concatenate(sky_px) if sky_px else np.zeros((0, 3))
        if len(px) < 100:  # the splat covers the sky: the panorama's upper half
            px = sky[: sky.shape[0] // 2].reshape(-1, 3)[::97]
        out["sky_rgb"][sid] = np.mean(px, 0).round(1).tolist()
        print(f"[sim2chip] {sid}: {len(views)} backgrounds", flush=True)
        del sp
        torch.cuda.empty_cache()
    (d / "background.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
