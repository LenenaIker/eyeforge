"""Walk test W.1 (docs/plan_moverse_photosphere_2026-10-18.md), prep, in edge-sdg: the sphere's proxy mesh with
face-varying UVs into the erased panorama, and the common grid of views: 8 headings, the camera at the capture point
and moved 3, 5, 8 and 13 m (--distances) forward, back, left and right of each heading.

    python experiments/sphere_walk/prep.py --dir data/sphere_test/<place>

Writes <dir>/walk/walk.npz, tex_proxy.png, views.json and real_hNNN.png (the panorama's cut at the capture point)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

HEADINGS = range(0, 360, 45)
MOVES = {"f": (1, 0), "b": (-1, 0), "r": (0, 1), "l": (0, -1)}  # forward, back, right, left of the heading
DISTANCES = (3, 5, 8, 13)  # metres from the capture point (1, 2, 3, 5 on the first two places, 2026-10-18)
VIEW = {"pitch_deg": 3.0, "fov_deg": 68.0, "size": [1024, 768]}
CLEARANCE_M = 1.0  # a moved camera keeps this far from anything standing (user, 2026-10-18)
BODY_M = (0.2, 0.3)  # what stands: proxy points from this high over the ground to this much over the camera


def reachable(off, obstacles, clearance=CLEARANCE_M):
    """Whether a person carrying the camera walks from the capture point to `off` (world xy) without passing
    within `clearance` of anything standing (obstacles: their xy), the end point included."""
    a, b = np.zeros(2), np.asarray(off[:2], float)
    ab = b - a
    t = np.clip(((obstacles - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
    return bool(np.linalg.norm(obstacles - (a + t[:, None] * ab), axis=1).min() >= clearance)


def face_uvs(uv, faces):
    """Face-varying UVs; a face across the panorama's seam gets its small u's moved by 1 (the texture repeats)."""
    fu = uv[faces].copy()
    u = fu[..., 0]
    wrap = (u.max(1) - u.min(1)) > 0.5
    u[wrap] = np.where(u[wrap] < 0.5, u[wrap] + 1, u[wrap])
    return fu.reshape(-1, 2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--distances", default=",".join(map(str, DISTANCES)))
    args = ap.parse_args()
    dists = [int(x) for x in args.distances.split(",")]
    d = Path(args.dir)
    out = d / "walk"
    out.mkdir(exist_ok=True)
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    bg = np.asarray(Image.open(d / "background.jpg").convert("RGB"))
    H, W = bg.shape[:2]
    seen = np.asarray(Image.open(d / "seen.png")) > 127
    tex = bg.copy()
    tex[~seen] = (255, 0, 255)  # never seen (the ground under the camera): a hole for post.py, not a guess
    Image.fromarray(tex).save(out / "tex_proxy.png")

    proxy = np.load(d / "proxy.npz")
    V, F = proxy["vertices"].astype(np.float64), proxy["faces"]
    dirs = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    x, y = capture.to_pixel(render_plan.pano_rays(dirs, R), W, H)
    uv = np.stack([(x + 0.5) / W, 1 - (y + 0.5) / H], -1)
    np.savez(out / "walk.npz", proxy_pts=render_plan.to_world(V, h).astype(np.float32),
             proxy_faces=F.astype(np.int32), proxy_uv=face_uvs(uv, F))

    # What stands: the proxy's non-ground points between knee and over head height, in world xy.
    pw = render_plan.to_world(V, h)
    stand = ~proxy["ground"] & (pw[:, 2] > BODY_M[0]) & (pw[:, 2] < h + BODY_M[1])
    obstacles = pw[stand, :2]
    views = []
    for hd in HEADINGS:
        v = {**VIEW, "yaw_deg": float(hd)}
        Image.fromarray(render_plan.background_cut(bg, v, R)).save(out / f"real_h{hd:03d}.png")
        fwd = render_plan.to_world(capture.rotation(hd, 0) @ np.array([0, 0, 1.0]), 0)
        right = render_plan.to_world(capture.rotation(hd, 0) @ np.array([1.0, 0, 0]), 0)
        moves = [("0", 0, 0)] + [(f"{k}{m}", f * m, r * m) for k, (f, r) in MOVES.items() for m in dists]
        for label, f, r in moves:
            off = f * fwd + r * right
            off[2] = 0
            views.append({**v, "name": f"h{hd:03d}_{label}", "heading": hd, "move": label,
                          "offset": off.tolist(), "pose": render_plan.camera_pose(v, h, off),
                          "reachable": label == "0" or reachable(off, obstacles)})
    (out / "views.json").write_text(json.dumps({"camera_height_m": h, "views": views}, indent=1))
    print(len(V), "proxy vertices,", len(views), "views,", sum(v["reachable"] for v in views), "reachable")


if __name__ == "__main__":
    main()
