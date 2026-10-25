"""Step 1.4 of the scene flow, the gate after depth: the ground plane on MoGe-2's points of the erased photo
(bg_depth.py) gives the camera's height, pitch and roll and where the horizon crosses the image. They are checked
against the route (a pedestrian camera stands 0.5 to 4 m up, level within a few degrees of roll) and against the
VLM's horizon guess, until 2026-10-15: Cosmos Reason 2 put it 0.4 to 0.5 image heights off on most streets. The
horizon the photo's own straight verticals give (lean.vertical, when they are clear) is written beside, not
checked: church_corner's and terrace's streets slope 5 degrees, their ground's horizon 0.2 image heights under the
verticals', and piece_build.py now stands the scene on those verticals with the floor keeping its slope.
Env with numpy, scipy and PIL (the mesh one):

    python sim2chip/photo2scene/steps/flow_camera.py --dir data/street_test/flow/basque_street

Writes <dir>/camera.json: {"ok", "height_m", "pitch_deg", "roll_deg", "horizon_y", "horizon_y_vlm",
"horizon_vlm_spread", "horizon_y_edges", "ground_inliers", "checks", "why"}. Exit code 3 when a check fails.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from bg_build import fit_plane
import lean

VLM_MAX_SPREAD = 0.1  # the VLM's horizon samples must agree this closely to be written down
HEIGHT = {'ground': (0.05, 1.0), 'pedestrian': (0.5, 4.0), 'raised': (2.0, 15.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    observed = json.load(open(os.path.join(args.dir, 'observe.json')))
    obs = observed['final']
    dp = np.load(os.path.join(args.dir, 'depth_background.npz'))
    g = np.load(os.path.join(args.dir, 'ground.npz'))
    erased = np.asarray(__import__('PIL.Image', fromlist=['Image']).open(
        os.path.join(args.dir, 'erased.png'))) > 0
    ground = np.any(g['masks'], 0) & (dp['mask'] > 0) & ~erased
    P = dp['points'][ground]
    P = P[np.isfinite(P).all(1)]
    rng = np.random.default_rng(0)
    if len(P) > 60000:
        P = P[rng.choice(len(P), 60000, replace=False)]
    tol = 0.01 * float(np.median(np.linalg.norm(P, axis=1)))  # MoGe's noise grows with distance
    n, dd, inl = fit_plane(P, iters=800, tol=tol)
    h, w = dp['depth'].shape
    k = dp['intrinsics']
    fy, cy = k[1, 1] * h, k[1, 2] * h
    # OpenCV camera: +Y down. n points from the ground to the camera, so "up" is n.
    pitch = math.degrees(math.asin(np.clip(n[2], -1, 1)))  # >0: the camera looks down
    roll = math.degrees(math.atan2(n[0], -n[1]))
    # Horizon at the image centre column: rays r = (0, (y - cy)/fy, 1) with r.n = 0.
    horizon = (cy - fy * n[2] / n[1]) / h if abs(n[1]) > 1e-6 else float('nan')
    vlm = obs.get('horizon_y')
    # The VLM's samples must agree for its horizon to judge ours. Looking down into a room (user_desk,
    # 2026-10-08) they gave 0.5, 0.42 and -0.1, taking the wall-ceiling edge for the horizon, while the
    # streets never spread more than 0.1: a VLM that disagrees with itself is not asked.
    guesses = [s['horizon_y'] for s in observed.get('samples', []) if isinstance(s.get('horizon_y'), (int, float))]
    spread = max(guesses) - min(guesses) if guesses else 0.0
    # Spread past 0.1 is already too much: on a street going uphill (the pots' photo 27, 2026-10-13) its samples
    # spread 0.15 around 0.55 while the parked cars' roofs, at the camera's height, sat on our 0.70.
    if spread > VLM_MAX_SPREAD:
        vlm = None
    # The photo's verticals: the true horizon where enough straight edges agree on one vertical (lean.USE_*).
    fx, cx = k[0, 0] * w, k[0, 2] * w
    edges_up = lean.photo_vertical(args.dir, n, (fx, fy, cx, cy), (w, h))
    u = edges_up['up']
    horizon_edges = (cy - fy * u[2] / u[1]) / h if edges_up['used'] and abs(u[1]) > 1e-6 else None
    lo, hi = HEIGHT.get(obs['camera_height'], (0.05, 50))
    checks = {
        'ground found': inl > 0.25,
        'height fits the route': lo <= dd <= hi,
        'level camera (roll under 8 deg)': abs(roll) < 8,
    }
    checks = {k2: bool(v) for k2, v in checks.items()}
    why = [k2 for k2, v in checks.items() if not v]
    out = {'ok': not why, 'height_m': round(float(dd), 3), 'pitch_deg': round(pitch, 2), 'roll_deg': round(roll, 2),
           'horizon_y': round(float(horizon), 3), 'horizon_y_vlm': vlm,
           'horizon_vlm_spread': round(spread, 3),
           'horizon_y_edges': None if horizon_edges is None else round(float(horizon_edges), 3),
           'edges': {k2: v for k2, v in edges_up.items() if k2 != 'up'}, 'ground_inliers': round(inl, 3),
           'tolerance_m': round(tol, 3), 'checks': checks, 'why': why}
    with open(os.path.join(args.dir, 'camera.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))
    sys.exit(0 if out['ok'] else 3)


if __name__ == '__main__':
    main()
