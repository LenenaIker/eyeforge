"""Facade feasibility test: planters on the balconies of a real building, seen from the street.

Loads one NVIDIA building (Rivermark, Isaac Sim 5.1 assets), finds support surfaces automatically
(upward faces above the ground floor, near the outer walls, away from the roof), drops NVIDIA ArchVis
planters on them and renders a few frames from street level looking up. No building-specific tuning:
the same search would work on any building model.

Run with Isaac Sim's python:
    /opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh spike/facade/facade_test.py --out data/facade_test
"""
import argparse
import math
import random
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--building", default="kasa_bldg_01")
ap.add_argument("--frames", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
args, _ = ap.parse_known_args()
app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

from sim2chip.scenes.common import NVIDIA_ASSETS, Props, look_at, make_quad, set_visible  # noqa: E402

RIVERMARK = f"{NVIDIA_ASSETS}/Isaac/5.1/Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content"
BUILDINGS = {
    "kasa_bldg_01": f"{RIVERMARK}/usa/country_assets/props_structure/kasa_bldg_01/kasa_bldg_01.usd",
    "kasa_bldg_02": f"{RIVERMARK}/usa/country_assets/props_structure/kasa_bldg_02/kasa_bldg_02.usd",
    "kasa_house_01": f"{RIVERMARK}/usa/country_assets/props_structures/kasa_house_01/kasa_house_01.usd",
}
PLANTERS = [f"{NVIDIA_ASSETS}/ArchVis/Residential/Outdoors/Planters/{n}.usd" for n in
            ["Anza_Medium", "Anza_Small", "ChippedPlanterSmall", "Fluted_Medium", "Flutted_Low",
             "GardenPlanter_Small", "GardenPlanter_Medium", "SquareGardenPlanterMedium", "SquareGardenPlanter_Long"]] + \
           [f"{NVIDIA_ASSETS}/ArchVis/Residential/Plants/{n}.usd" for n in ["PlantPot", "Plant_01", "YuccaCane"]]
SKY = "data/cache/hdri/polyhaven/modern_buildings_2k.hdr"


def wait_bounds(stage, prim, bc_purposes, tries=300):
    for _ in range(tries):
        bc = UsdGeom.BBoxCache(0, bc_purposes)
        r = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        if not r.IsEmpty() and (r.GetMax() - r.GetMin())[2] > 0:
            return r
        app.update()
    return None


def support_faces(stage, root, ground_z, top_z, box, min_h=2.2, roof_gap=1.5, edge=3.0, min_area=0.15):
    """Upward-facing triangles where a small object can stand: above the ground floor, below the roof and
    within `edge` meters of the building's outer footprint (balconies, sills, ledges)."""
    tris, areas = [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        p = np.hstack([np.array(pts, dtype=np.float64), np.ones((len(pts), 1))]) @ m
        p = p[:, :3]
        idx = np.array(idx)
        k = 0
        for c in counts:
            f = idx[k:k + c]
            k += c
            for j in range(1, c - 1):
                a, b, d = p[f[0]], p[f[j]], p[f[j + 1]]
                n = np.cross(b - a, d - a)
                area = np.linalg.norm(n) / 2
                if area < 1e-4:
                    continue
                nz = n[2] / (2 * area)
                cz = (a[2] + b[2] + d[2]) / 3
                if abs(nz) < 0.95 or cz < ground_z + min_h or cz > top_z - roof_gap:
                    continue
                cx, cy = (a[0] + b[0] + d[0]) / 3, (a[1] + b[1] + d[1]) / 3
                to_edge = min(cx - box[0], box[2] - cx, cy - box[1], box[3] - cy)
                if to_edge > edge:
                    continue
                tris.append((a, b, d))
                areas.append(area)
    # Winding may be flipped in some meshes, hence |nz|: the renders show whether that matters.
    return tris, np.array(areas)


def sample_point(tris, areas, rng):
    i = rng.choices(range(len(tris)), weights=areas)[0]
    a, b, c = tris[i]
    u, v = rng.random(), rng.random()
    if u + v > 1:
        u, v = 1 - u, 1 - v
    return a + u * (b - a) + v * (c - a)


def main():
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateTextureFileAttr(str(Path(SKY).resolve()))
    dome.CreateIntensityAttr(1000)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 30))

    bldg = stage.DefinePrim("/World/Bldg", "Xform")
    stage.DefinePrim("/World/Bldg/Ref", "Xform").GetReferences().AddReference(BUILDINGS[args.building])
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    r = wait_bounds(stage, bldg, purposes)
    if r is None:
        raise SystemExit("building did not load")
    size = r.GetMax() - r.GetMin()
    print(f"[facade] raw bounds {r.GetMin()} {r.GetMax()} size {size}", flush=True)
    # Buildings are a few to ~40 m tall: anything bigger is in centimeters.
    scale = 0.01 if size[2] > 300 else 1.0
    xf = UsdGeom.Xformable(bldg)
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    for _ in range(5):
        app.update()
    r = wait_bounds(stage, bldg, purposes)
    lo, hi = r.GetMin(), r.GetMax()
    print(f"[facade] building {args.building}: {hi - lo} m", flush=True)
    floor, _ = make_quad(stage, "/World/Ground", 400, 400, float(lo[2]) - 0.01)
    floor.Set(Gf.Vec3f(0.35, 0.35, 0.33))

    box = (lo[0], lo[1], hi[0], hi[1])
    tris, areas = support_faces(stage, bldg, lo[2], hi[2], box)
    print(f"[facade] {len(tris)} support triangles, {areas.sum():.1f} m2", flush=True)
    if not tris:
        raise SystemExit("no support surfaces found")
    heights = sorted(round(float(np.mean([t[0][2], t[1][2], t[2][2]])), 1) for t in tris)
    levels = sorted({h for h in heights})
    print(f"[facade] support heights (m): {levels[:40]}", flush=True)

    props = Props(stage, app)
    pots = [props.add(f"/World/Pots/p{i:02d}", u, label="maceta") for i, u in enumerate(PLANTERS)]
    for _ in range(10):
        app.update()
    props.recenter(pots)
    pots = props.fit_height(pots, (0.25, 0.6), rng)
    print(f"[facade] {len(pots)} planters loaded", flush=True)

    w, h = 640, 480
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    aperture = 20.955
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000))
    cam.CreateFocalLengthAttr(aperture / 2 / math.tan(math.radians(65) / 2))
    for _ in range(30):
        app.update()

    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product(str(cam.GetPath()), (w, h))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(out), rgb=True, bounding_box_2d_tight=True)
    writer.attach([rp])
    ctr = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
    for f in range(args.frames):
        for p in pots:
            set_visible(p, False)
        shown = rng.sample(pots, rng.randint(2, 6))
        spots = [sample_point(tris, areas, rng) for _ in shown]
        for p, s in zip(shown, spots):
            props.set_pose(p, (s[0], s[1], s[2]), rng.uniform(0, 360))
            set_visible(p, True)
        # Camera on the street, outside the footprint, on the side of the first pot, looking up at it.
        tgt = spots[0]
        d = tgt[:2] - ctr
        side = np.array([np.sign(d[0]), 0.0]) if abs(d[0]) / (hi[0] - lo[0]) > abs(d[1]) / (hi[1] - lo[1]) \
            else np.array([0.0, np.sign(d[1])])
        dist = rng.uniform(6, 22)
        lateral = np.array([-side[1], side[0]]) * rng.uniform(-8, 8)
        eye2 = tgt[:2] + side * dist + lateral
        eye = (eye2[0], eye2[1], lo[2] + rng.uniform(1.4, 1.8))
        look_at(cam.GetPrim(), eye, (tgt[0], tgt[1], tgt[2] + rng.uniform(-1.5, 1.5)))
        rep.orchestrator.step(rt_subframes=4)
        print(f"[facade] frame {f}: pots at {[round(float(s[2]), 1) for s in spots]} m, camera {dist:.0f} m away",
              flush=True)
    rep.orchestrator.wait_until_complete()
    print(f"[facade] wrote {args.frames} frames to {out}", flush=True)


try:
    main()
finally:
    app.close()
