"""Walk test arm C, WorldGen's way (github.com/ZiYang-xie/WorldGen, Apache 2.0) on our erased panorama, without
its FLUX panorama generator: DA-2 360 depth (Apache 2.0), OneFormer ADE20K foreground (MIT) and LaMa behind the
foreground, re-predicted depth matched to the first, as WorldGen.inpaint_bg_splat does. Its splats become two
textured mesh layers for render_walk.py (front: the panorama on DA-2's depth, torn at depth jumps as splats are;
back: the inpainted panorama on its own depth, only under the grown foreground). DA-2 is scale free: it is scaled
to our proxy's measured ground (proxy_distance.npz), and the sky (ours) is left out of the meshes. Run in the worldgen env
(/media/lenena-iker/Crucial/envs/worldgen):

    python experiments/sphere_walk/worldgen_layers.py --dir data/sphere_test/<place> [--width 2048]

Writes <dir>/walk/C_scene/scene.npz, layers.json, tex_front.png, tex_back.png, fg.png and info.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# WorldGen's package imports pytorch3d for its splat writer, unused here: a stand-in keeps it out of the env.
_p3d = types.ModuleType("pytorch3d")
_p3d.transforms = types.ModuleType("pytorch3d.transforms")
_p3d.transforms.matrix_to_quaternion = None
sys.modules.update({"pytorch3d": _p3d, "pytorch3d.transforms": _p3d.transforms})
# Its __init__ imports the FLUX generator (nunchaku), unused too: the package without it.
_wg = types.ModuleType("worldgen")
_wg.__path__ = ["/media/lenena-iker/Crucial/tools/WorldGen/src/worldgen"]
sys.modules["worldgen"] = _wg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshes import grid_mesh, lama_inpaint  # noqa: E402
from sim2chip.photo2scene.sphere import render_plan  # noqa: E402,F401
from worldgen.pano_depth import build_depth_model, pred_pano_depth  # noqa: E402
from worldgen.pano_seg import build_segment_model, seg_pano_fg  # noqa: E402
from worldgen.utils.general_utils import depth_match  # noqa: E402

from meshes import LAMA  # noqa: E402  (the same big-lama WorldGen pulls through iopaint)
SKY_M = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--width", type=int, default=2048, help="the meshes' panorama width (vertices per row)")
    args = ap.parse_args()
    d = Path(args.dir)
    out = d / "walk" / "C_scene"
    out.mkdir(parents=True, exist_ok=True)
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    W, H = args.width, args.width // 2
    bg = Image.open(d / "background.jpg").convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    seen = np.asarray(Image.open(d / "seen.png").resize((W, H))) > 127
    stuff = np.load(d / "stuff_bg.npz")
    pd = np.load(d / "proxy_distance.npz")
    proxy_valid = cv2.resize(pd["valid"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    sky = seen & ~proxy_valid
    ground = cv2.resize(pd["ground"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    proxy_d = cv2.resize(pd["distance"], (W, H), interpolation=cv2.INTER_NEAREST)

    # What the sphere never saw: above the capture, the sky's mean colour (and sky); below, the median ground.
    pano = np.asarray(bg).copy()
    rows = np.arange(H)[:, None].repeat(W, 1)
    up = ~seen & (rows < H // 2)
    pano[up] = np.median(pano[sky], 0) if sky.any() else (180, 190, 200)
    pano[~seen & ~up] = np.median(pano[seen & ground], 0)
    sky |= up
    pano_img = Image.fromarray(pano)

    t = time.time()
    dev = torch.device("cuda")
    depth_model = build_depth_model(dev)
    seg_processor, seg_model = build_segment_model(dev)
    lama = torch.jit.load(LAMA, map_location=dev).eval()
    t_load = time.time() - t

    t = time.time()
    with torch.inference_mode():
        init = pred_pano_depth(depth_model, pano_img)
        fg = seg_pano_fg(seg_processor, seg_model, pano_img, init["distance"])
        grown = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=10)  # as WorldGen
        back_img = lama_inpaint(lama, pano_img, grown)
        back = pred_pano_depth(depth_model, back_img)
        back = depth_match(init, back, 1 - grown)
    to_np = lambda x: cv2.resize(x.float().cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    d_front, d_back = to_np(init["distance"]), to_np(back["distance"])
    fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    grown = cv2.resize(grown, (W, H), interpolation=cv2.INTER_NEAREST) > 0

    # Metric scale from our measured ground (DA-2's max is set to 20 m, not measured).
    ok = ground & proxy_valid & seen & (proxy_d > 0)
    scale = float(np.median(proxy_d[ok] / np.maximum(d_front[ok], 1e-6)))
    d_front, d_back = d_front * scale, d_back * scale
    d_front[sky], d_back[sky] = SKY_M, SKY_M
    t_run = time.time() - t

    # Front: torn at depth jumps and without the foreground's edge band, as WorldGen masks its splats.
    edge = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8)) - cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8))
    # No sky faces: far geometry renders black in Isaac (2026-10-18); post.py fills the sky from the panorama.
    front = grid_mesh(d_front, (edge == 0) & ~sky, R, h)
    # The back layer only where it is behind the front: with houses near the camera as foreground (42 % of the
    # panorama on PXL_20261016_172729712) depth_match put it in front and it covered every view.
    backm = grid_mesh(d_back, grown & ~sky & (d_back > 1.02 * d_front), R, h)
    np.savez(out / "scene.npz", front_pts=front[0], front_faces=front[1], front_uv=front[2],
             back_pts=backm[0], back_faces=backm[1], back_uv=backm[2])
    pano_img.save(out / "tex_front.png")
    back_img.resize((W, H)).save(out / "tex_back.png")
    Image.fromarray((fg * 255).astype(np.uint8)).save(out / "fg.png")
    (out / "layers.json").write_text(json.dumps([{"name": "front", "tex": "tex_front.png"},
                                                 {"name": "back", "tex": "tex_back.png"}]))
    info = {"scale": scale, "fg_share": float(fg.mean()), "load_s": round(t_load, 1), "run_s": round(t_run, 1),
            "front_faces": int(len(front[1])), "back_faces": int(len(backm[1])),
            "ground_err_m": float(np.median(np.abs(d_front[ok] - proxy_d[ok])))}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
