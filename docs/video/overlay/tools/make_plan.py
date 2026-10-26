"""plan.json for the Isaac Replicator animation: an orbit around the f3c_photo21 scene looking down at its centre,
bins spawning and despawning on the free ground, and a student camera (drawn later, in post) taking shots.
Writes plan.json + scenes.json for sim2chip/walkscene (background.py, isaac.py) and anim.json for the overlay."""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lenena-iker/work/eyeforge")
sys.path.insert(0, str(ROOT))
from sim2chip.walkscene import Scene  # noqa: E402

OUT = Path(sys.argv[1])
BUNDLE = ROOT / "data/worldstereo_test/f3c_photo21/bundle"
DATA = ROOT / "data"
W, H, FPS, SECONDS = 1280, 720, 24, 14.0
ORBIT_R, ORBIT_UP, FOV = 5.0, 13.0, 66.0     # metres from the centre, metres over the ground, horizontal degrees
LIVE = 6                                   # bins on the ground at a time
LIFE = (2.5, 4.0)                          # seconds a bin stays
SHOT_EVERY = 1.3                           # seconds between the student camera's shots
rng = random.Random(7)

sc = Scene(BUNDLE)
centre = np.asarray(sc.meta["centre"], float)
ground = float(sc.meta["ground_z"])
free = sc.free_points(within=4.0)
qa = json.load(open(DATA / "jobs/sphere_bins/qa_assets/assets.json"))
verdicts = json.load(open(DATA / "jobs/sphere_bins/qa_assets/verdicts.json"))
models = [a for a in qa if verdicts.get(a["uid"], {}).get("decision") == "keep"
          and (DATA / "cache" / a["usd"]).exists()]
print(len(models), "bin models,", len(free), "free points")


def look_at(eye, target):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0, 0, 1.0))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, :3] = np.stack([r, d, f])
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def free_spot(taken, radius):
    for _ in range(200):
        xy = free[rng.randrange(len(free))]
        if all(np.hypot(*(xy - p)) > radius + r + 0.6 for p, r in taken):
            return xy
    return None


n = int(SECONDS * FPS)
# bin lifetimes: LIVE slots, each one bin after another with a short gap
lives = []
for s in range(LIVE):
    t = rng.uniform(0, LIFE[1])
    while t < SECONDS:
        dur = rng.uniform(*LIFE)
        a = rng.choice(models)
        size = rng.uniform(0.9, 1.4)
        lives.append({"start": t, "end": t + dur, "model": a, "size": size, "radius": 0.5 * size, "yaw": rng.uniform(0, 360)})
        t += dur + rng.uniform(0.4, 1.0)
# positions: no two bins alive at once closer than their sizes
for k, lv in enumerate(lives):
    taken = [(o["xy"], o["radius"]) for o in lives[:k] if "xy" in o and o["start"] < lv["end"] and lv["start"] < o["end"]]
    lv["xy"] = free_spot(taken, lv["radius"])
lives = [lv for lv in lives if lv["xy"] is not None]

def aim_at_bin(t):
    """A student camera 2.5-4.5 m from a bin alive at t, on free ground clear of the other bins, looking at it."""
    alive = [lv for lv in lives if lv["start"] <= t and t + 0.3 < lv["end"]]
    rng.shuffle(alive)
    for lv in alive:
        for _ in range(60):
            d, b = rng.uniform(2.5, 4.5), rng.uniform(0, 2 * math.pi)
            xy = np.asarray(lv["xy"]) + d * np.array([math.cos(b), math.sin(b)])
            if not sc.is_free(xy[None])[0]:
                continue
            if any(np.hypot(*(xy - np.asarray(o["xy"]))) < o["radius"] + 0.5 for o in alive):
                continue
            yaw = math.degrees(math.atan2(lv["xy"][1] - xy[1], lv["xy"][0] - xy[0]))
            pitch = math.degrees(math.atan2(lv["size"] / 2 - 1.6, d))
            return {"t": t, "pos": [float(xy[0]), float(xy[1]), ground + 1.6], "yaw": yaw, "pitch": pitch}
    return None


shots = []
t = 0.4
while t < SECONDS:
    s = aim_at_bin(t)
    if s:
        shots.append(s)
    t += SHOT_EVERY

views, frames = [], []
for f in range(n):
    t = f / FPS
    ang = math.radians(200 + 360 * t / SECONDS)  # a full turn, starting behind the photo's camera
    eye = np.array([centre[0] + ORBIT_R * math.cos(ang), centre[1] + ORBIT_R * math.sin(ang), ground + ORBIT_UP])
    m = look_at(eye, (centre[0], centre[1], ground))
    objs = []
    for lv in lives:
        if lv["start"] <= t < lv["end"]:
            a = lv["model"]
            objs.append({"key": a["uid"], "usd": str(DATA / "cache" / a["usd"]), "class": "bin", "measure": 2,
                         "size_m": round(lv["size"], 3), "xy": [round(float(v), 3) for v in lv["xy"]],
                         "yaw_deg": round(lv["yaw"], 1)})
    views.append({"name": f"{f:04d}", "scene": "f3c_photo21", "class": "bin", "objects": objs, "w2c": m.round(6).tolist()})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plan.json").write_text(json.dumps({"size": [W, H], "fov_deg": FOV, "views": views, "missed": 0}, indent=1))
(OUT / "scenes.json").write_text(json.dumps({"f3c_photo21": str(BUNDLE)}, indent=1))
(OUT / "anim.json").write_text(json.dumps({"fps": FPS, "fov_deg": FOV, "size": [W, H], "shots": shots,
                                           "bins": [{k: (v if k != "model" else v["uid"]) for k, v in lv.items()}
                                                    for lv in lives]}, indent=1, default=lambda v: v.tolist()))
print(n, "frames,", len(lives), "bin lives,", len(shots), "shots, max alive",
      max(len(v["objects"]) for v in views))
