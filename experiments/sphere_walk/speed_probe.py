"""How fast GEN3C may walk (2026-10-20, docs/plan_gs_gen3c_2026-10-20.md): its 121 frames over 5 m are a frame
every 4 cm, ~30 times denser than the splat needs (cover_report.py), so a clip could walk much further. Three clips
of the same kind and growing length: out of the capture point, then along a circle round it, the camera looking
to the right of the walk (sideways, the user's pick: lateral moves show the most), every corner a turn. Planned on
the site's scene like nbv.py's clips (its GEN3C cache), launched with nbv.py launch --round ROUND.

    python experiments/sphere_walk/speed_probe.py --place <place> [--lengths 8,15,25]

Writes <dir>/walk/N<ROUND>/ (paths.json, the caches, probe.png: the paths from above)."""
import argparse
import json
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

import nbv
from nbv import Scene, cache_npz, clip, draw, g, heading, obstacles, place_dir, round_dir, smooth

ROUND = 9             # a round number the NBV rounds never reach: the probe is not one of them
KEY_EVERY = 20        # cache keys every this many frames (nbv.py's KEY_FRAMES has 5 over 121)
TURN_FRAMES = 15      # a corner's turn spread over this many frames, not one


def polyline(points: np.ndarray, n: int) -> np.ndarray:
    """n points at even steps along the polyline, eased at both ends (as nbv.py's smooth)."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = smooth(np.linspace(0, 1, n)) * s[-1]
    return np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], -1)


def path_of(length: float, radius: float, start_az: float, way: float) -> np.ndarray:
    """Out along start_az to radius, then along the circle (way +1: headings growing), then, if the circle alone
    falls short, back in to the capture point along its last azimuth and out again halfway round the arc."""
    u = lambda az: np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])  # noqa: E731
    pts = [np.zeros(2), radius * u(start_az)]
    arc = max(0.0, min(length - radius, math.pi * radius))  # half a circle at most: walls rarely leave more
    span = math.degrees(arc / radius)
    for a in np.linspace(0, span, max(2, int(span / 3))):
        pts.append(radius * u(start_az + way * a))
    rest = length - radius - arc
    if rest > 0:
        pts.append(max(0.0, radius - rest) * u(start_az + way * span))
    if rest > radius:
        pts.append(min(rest - radius, radius) * u(start_az + way * span / 2))
    return polyline(np.array(pts), g.FRAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", required=True)
    ap.add_argument("--lengths", default="8,15,25")
    ap.add_argument("--scene", help="default: nbv.py's for the round")
    args = ap.parse_args()
    d = place_dir(args.place)
    out = round_dir(args.place, ROUND)
    out.mkdir(parents=True, exist_ok=True)
    h = json.loads((d / "geometry.json").read_text())["camera_height_m"]
    scene = Scene(d / "walk" / (args.scene or nbv.default_scene(args.place, ROUND)))
    obs = obstacles(d)
    tree = cKDTree(obs) if len(obs) else None
    nbv.KEY_FRAMES = tuple(range(0, g.FRAMES, KEY_EVERY)) + (g.FRAMES - 1,)
    clips, picked = [], []
    for i, length in enumerate(float(x) for x in args.lengths.split(",")):
        best = None
        # the clearest path of this length: the radius as large as the site lets, every start and way round
        for radius in (nbv.STUDENT_RADIUS_M - 1, nbv.STUDENT_RADIUS_M - 2, nbv.STUDENT_RADIUS_M - 3):
            for az in range(0, 360, 15):
                for way in (1.0, -1.0):
                    xy = path_of(length, radius, az, way)
                    clear = tree.query(xy)[0].min() if tree is not None else math.inf
                    if clear >= nbv.CLEARANCE_M and (best is None or radius > best[0]):
                        best = (radius, az, way, xy)
            if best:
                break
        if best is None:
            raise SystemExit(f"no clear path of {length} m on this site")
        radius, az, way, xy = best
        step = np.diff(xy, axis=0)
        walk = np.array([heading(*s) if np.linalg.norm(s) > 1e-6 else np.nan for s in step])
        walk = np.r_[walk, walk[-1]]
        walk = np.degrees(np.unwrap(np.radians(np.where(np.isnan(walk), np.nanmean(walk), walk))))
        walk = uniform_filter1d(walk, TURN_FRAMES, mode="nearest")
        c = clip(f"speed_{int(length):02d}m_r{radius:g}_az{az:03d}", "speed", xy, walk + 90, slot=None)
        c["name"] = f"{args.place}_N{ROUND:02d}_{i}_{c['name']}"
        frames = list(range(0, g.FRAMES, 10))
        hole = scene.cast(nbv.poses_of(c, h, frames), nbv.SCORE_SIZE, g.HFOV)["hole"].mean((1, 2))
        c.update(gain=0.0, start_hole=float(hole[0]), max_hole=float(hole.max()))
        clips.append(cache_npz(c, scene, h, out))
        picked.append(c)
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        cm = 100 * length / (g.FRAMES - 1)
        turn = np.abs(np.diff(walk)).max()
        print(f"{c['name']}: {length:.1f} m, {cm:.1f} cm/frame, sharpest turn {turn:.1f} deg/frame, holes "
              f"{100 * hole[0]:.1f} % at start, {100 * hole.max():.1f} % max", flush=True)
    (out / "paths.json").write_text(json.dumps({"place": args.place, "paths": clips}))
    a = nbv.audit(d, scene)
    draw(a, [], picked, obs, out / "probe.png")
    print("map", out / "probe.png")


if __name__ == "__main__":
    main()
