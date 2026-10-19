"""What a free camera sees in a walk scene (render_walk.py --save), from every reachable point of a 1 m grid within
RADIUS_M of the capture point and every HEADING: per pixel, which layer it hits (the panorama's own texture, GEN3C's
paint, the backstop) or nothing. A view of 72 fixed moves (views.json) left most points and directions unchecked
(user, 2026-10-19). Run with Isaac's python.sh:

    python.sh experiments/sphere_walk/coverage.py --dir data/sphere_test/<place> --scene walk/TS_scene

Writes <scene>/coverage.json (shares per point and heading, and their totals) and coverage.png (a map from above:
per point, the worst heading's share of backstop and holes below the horizon)."""
import argparse
import json
import math
import os
import sys

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--scene", default="walk/TS_scene")
args = ap.parse_args()
app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.semantics import add_update_semantics  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

RADIUS_M = 5.0
GRID_M = 1.0
HEADINGS = tuple(range(0, 360, 45))
PITCH_DEG = 15.0  # a little down: a camera on a pole or a person's hand
FOV_DEG = 68.0
SIZE = (512, 384)
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
SENSOR_WIDTH_MM = 20.955
GUESS = ("far", "farback")  # the backstop: our proxy pushed back, right only from the capture point


def main():
    d = args.dir
    scene = os.path.join(d, args.scene)
    layers = [l["name"] for l in json.load(open(os.path.join(scene, "layers.json")))]
    h = json.load(open(os.path.join(d, "geometry.json")))["camera_height_m"]
    from prep import BODY_M, reachable
    proxy = np.load(os.path.join(d, "proxy.npz"))
    pw = render_plan.to_world(proxy["vertices"].astype(np.float64), h)
    obs = pw[~proxy["ground"] & (pw[:, 2] > BODY_M[0]) & (pw[:, 2] < h + BODY_M[1]), :2]  # as gen3c_batch.obstacles
    r = np.arange(-RADIUS_M, RADIUS_M + 1e-6, GRID_M)
    points = [(float(x), float(y)) for x in r for y in r
              if math.hypot(x, y) <= RADIUS_M + 1e-6 and ((x, y) == (0, 0) or reachable(np.array([x, y]), obs))]

    omni.usd.get_context().open_stage(os.path.join(scene, "scene.usd"))
    stage = omni.usd.get_context().get_stage()
    for n in layers:
        add_update_semantics(stage.GetPrimAtPath(f"/World/{n}"), n)
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
    w, hh = SIZE
    cam.CreateHorizontalApertureAttr(SENSOR_WIDTH_MM)
    cam.CreateVerticalApertureAttr(SENSOR_WIDTH_MM * hh / w)
    cam.CreateFocalLengthAttr(SENSOR_WIDTH_MM / 2 / math.tan(math.radians(FOV_DEG) / 2))
    op = UsdGeom.Xformable(cam).GetOrderedXformOps()[0]
    rp = rep.create.render_product("/World/Cam", SIZE)
    seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    seg.attach([rp])
    for _ in range(100):
        app.update()
    # Below the horizon a ray must hit something; above it nothing is the sky (the dome).
    below = None
    rows = []
    for x, y in points:
        for hd in HEADINGS:
            v = {"yaw_deg": float(hd), "pitch_deg": PITCH_DEG, "fov_deg": FOV_DEG, "size": list(SIZE)}
            if below is None:
                below = capture.view_rays(v).reshape(hh, w, 3)[..., 1] > 0  # levelled frame: y down
            op.Set(Gf.Matrix4d((np.asarray(render_plan.camera_pose(v, h, (x, y, 0))) @ CV2GL).T.tolist()))
            rep.orchestrator.step(rt_subframes=2)
            out = seg.get_data()
            ids = np.asarray(out["data"]).reshape(hh, w)
            names = {int(k): v.get("class", "") for k, v in out["info"]["idToLabels"].items()}
            share = {n: float(np.mean(np.isin(ids, [k for k, c in names.items() if c == n]))) for n in layers}
            hit = np.isin(ids, [k for k, c in names.items() if c in layers])
            share["hole"] = float(np.mean(~hit & below))
            rows.append({"x": x, "y": y, "heading": hd, **{k: round(s, 4) for k, s in share.items()}})
    total = {k: round(float(np.mean([q[k] for q in rows])), 4) for k in [*layers, "hole"]}
    bad = lambda q: sum(q[k] for k in GUESS if k in q) + q["hole"]  # noqa: E731
    worst = sorted(rows, key=bad)[-5:]
    json.dump({"points": len(points), "views": len(rows), "total": total, "worst": worst, "rows": rows},
              open(os.path.join(scene, "coverage.json"), "w"), indent=1)
    print("coverage", len(points), "points", len(rows), "views", total, flush=True)

    from PIL import Image, ImageDraw
    s, m = 60, 40
    img = Image.new("RGB", (int(2 * RADIUS_M * s + 2 * m),) * 2, (255, 255, 255))
    dr = ImageDraw.Draw(img)
    c = lambda x, y: (m + (x + RADIUS_M) * s, m + (RADIUS_M - y) * s)  # noqa: E731
    dr.ellipse([*c(-RADIUS_M, RADIUS_M), *c(RADIUS_M, -RADIUS_M)], outline=(150, 150, 150))
    for x, y in points:
        q = max(bad(r) for r in rows if (r["x"], r["y"]) == (x, y))
        col = (int(255 * min(1, q / 0.3)), int(255 * (1 - min(1, q / 0.3))), 60)
        px, py = c(x, y)
        dr.ellipse([px - 14, py - 14, px + 14, py + 14], fill=col)
        dr.text((px - 10, py - 5), f"{100 * q:.0f}", fill=(0, 0, 0))
    img.save(os.path.join(scene, "coverage.png"))


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    app.close()
    os._exit(1)
app.close()
