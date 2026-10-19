"""Walk test arm B_layers: WorldGen's two-layer idea on our own proxy, so the fill is painted once and every view
agrees on it (arm B paints each view on its own). Front: arm A's proxy mesh, its never-seen ground painted. Back:
behind everything that stands nearer than its surroundings (MoGe's measured distance under FG_RATIO of the farthest
within FAR_WIN, ground left out), the panorama painted by LaMa on the surroundings' distance. Run in the
SIM2CHIP_SAM3_PYTHON env (torch) after prep.py:

    python experiments/sphere_walk/layers_ours.py --dir data/sphere_test/<place>

Writes <dir>/walk/BL_scene/scene.npz, layers.json, tex_front.png, tex_back.png, fg.png and info.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshes import LAMA, grid_mesh, lama_inpaint  # noqa: E402

FAR_WIN = 81      # px of the 1024-wide distance map (~28 deg): the surroundings an occluder stands in front of
FG_RATIO = 0.75   # nearer than this share of the surroundings' farthest: something stands in front
GROW = 6          # px the occluders grow on the distance map, so their torn edges are covered too
SKY_M = 1000.0


def dilate(a, k):
    """Grey dilation (max over a k x k window) of a 2D array, on the GPU."""
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32)).cuda()[None, None]
    return F.max_pool2d(t, k, 1, k // 2)[0, 0].cpu().numpy()


def resize(a, size, method=Image.Resampling.NEAREST):
    """A uint8 mask or image to (width, height)."""
    return np.asarray(Image.fromarray(a).resize(size, method))


def occluders(dist, valid, ground):
    """What stands in front of its surroundings and the distance behind it: the farthest non-ground distance
    around (a grey dilation, on a horizontally wrapped map so the panorama's seam is no edge)."""
    pad = FAR_WIN
    walls = np.where(valid & ~ground, dist, 0).astype(np.float32)
    wrap = np.concatenate([walls[:, -pad:], walls, walls[:, :pad]], 1)
    far = dilate(wrap, FAR_WIN)[:, pad:-pad]
    fg = valid & ~ground & (dist < FG_RATIO * far)
    fg = dilate(fg, 2 * GROW + 1) > 0
    return fg, np.where(fg, far, dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    w = d / "walk"
    out = w / "BL_scene"
    out.mkdir(exist_ok=True)
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    pd = np.load(d / "proxy_distance.npz")
    dist, valid, ground = pd["distance"].astype(np.float32), pd["valid"], pd["ground"]
    H, W = dist.shape

    t = time.time()
    fg, behind = occluders(dist, valid, ground)

    # Textures: the panorama at the map's resolution x2, the never-seen ground painted on the front, everything
    # under the occluders painted on the back.
    full = np.asarray(Image.open(w / "tex_proxy.png").convert("RGB"))
    TW, TH = 2 * W, 2 * H
    tex = np.asarray(Image.fromarray(full).resize((TW, TH), Image.Resampling.LANCZOS))
    unseen = (tex[..., 0] > 200) & (tex[..., 1] < 90) & (tex[..., 2] > 200)
    unseen = (dilate(unseen, 5) > 0).astype(np.uint8)
    lama = torch.jit.load(LAMA, map_location="cuda").eval()
    front = np.asarray(lama_inpaint(lama, tex, unseen))
    # The front keeps the panorama's full resolution: LaMa's paint only where the sphere never saw.
    big = resize(unseen * 255, full.shape[1::-1]) > 0
    front_full = full.copy()
    front_full[big] = resize(front, full.shape[1::-1], Image.Resampling.BICUBIC)[big]
    fg_t = resize(fg.astype(np.uint8) * 255, (TW, TH)) > 0
    back = lama_inpaint(lama, front, fg_t)
    Image.fromarray(front_full).save(out / "tex_front.png")
    back.save(out / "tex_back.png")
    Image.fromarray((fg * 255).astype(np.uint8)).save(out / "fg.png")

    # Front: arm A's proxy (its own faces, UVs into the same panorama); back: a grid over the occluders.
    a = np.load(w / "walk.npz")
    bpts, bfaces, buv = grid_mesh(np.where(valid | fg, behind, SKY_M), fg, R, h)
    np.savez(out / "scene.npz", front_pts=a["proxy_pts"], front_faces=a["proxy_faces"], front_uv=a["proxy_uv"],
             back_pts=bpts, back_faces=bfaces, back_uv=buv)
    (out / "layers.json").write_text(json.dumps([{"name": "front", "tex": "tex_front.png"},
                                                 {"name": "back", "tex": "tex_back.png"}]))
    info = {"fg_share": float(fg.mean()), "back_faces": int(len(bfaces)), "seconds": round(time.time() - t, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
