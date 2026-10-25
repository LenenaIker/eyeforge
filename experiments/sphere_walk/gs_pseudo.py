"""Random walk views within RADIUS_M of the capture point (reachable as prep.py says), for gs_train.py --pseudo: B_layers'
scene rendered there by Isaac teaches the splat what the panorama saw, from where no GEN3C clip passed (2026-10-20:
GS1 smeared the near ground 3-5 m away). Not the walk test's own grid (views.json), so the test stays a test.

    python experiments/sphere_walk/gs_pseudo.py --dir data/sphere_test/<place>
    python.sh experiments/sphere_walk/render_walk.py --dir <dir> --scene <dir>/walk/BL_scene \\
        --views <dir>/walk/pseudo.json --raw <dir>/walk/BL_pseudo_raw

Writes <dir>/walk/pseudo.json."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep import BODY_M, VIEW, reachable  # noqa: E402
from sim2chip.photo2scene.sphere import render_plan  # noqa: E402

N_VIEWS = 240
RADIUS_M = 5.5
PITCH_DEG = (-5.0, 25.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    rng = np.random.default_rng(0)
    h = json.loads((d / "geometry.json").read_text())["camera_height_m"]
    proxy = np.load(d / "proxy.npz")
    pw = render_plan.to_world(proxy["vertices"].astype(np.float64), h)
    stand = ~proxy["ground"] & (pw[:, 2] > BODY_M[0]) & (pw[:, 2] < h + BODY_M[1])
    obstacles = pw[stand, :2]
    views = []
    while len(views) < N_VIEWS:
        r, a = RADIUS_M * math.sqrt(rng.random()), rng.random() * 2 * math.pi
        off = np.array([r * math.cos(a), r * math.sin(a), 0.0])
        if not reachable(off, obstacles):
            continue
        v = {**VIEW, "yaw_deg": float(rng.random() * 360), "pitch_deg": float(rng.uniform(*PITCH_DEG))}
        views.append({**v, "name": f"p{len(views):03d}", "move": "P", "heading": 0, "offset": off.tolist(),
                      "pose": render_plan.camera_pose(v, h, off)})
    (d / "walk" / "pseudo.json").write_text(json.dumps({"camera_height_m": h, "views": views}))
    print(len(views), "views")


if __name__ == "__main__":
    main()
