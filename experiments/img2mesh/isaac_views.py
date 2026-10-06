"""Two Isaac Sim views of every USD in a folder, on a grey floor, to check that a converted asset (mesh,
texture, scale, up axis) renders as it should. Run with Isaac's python.sh:

    python.sh isaac_views.py --usd <dir with *.usd> --out <dir>   # one <name>.png per asset, both views
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=float, default=0.3, help="longest side in metres")
args = ap.parse_args()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2chip.scenes.common import Props, look_at, make_quad, set_visible  # noqa: E402

RES = (512, 384)
VIEWS = [(35, 0.45), (215, 0.45)]

settings = carb.settings.get_settings()
settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
settings.set("/omni/replicator/asyncRendering", False)
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
floor, _ = make_quad(stage, "/World/Ground", 50, 50, 0.0)
floor.Set(Gf.Vec3f(0.32, 0.32, 0.30))
UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr().Set(1000)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr().Set(1500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 30, 0))
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955 * RES[1] / RES[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
cam.CreateFocalLengthAttr(18.0)

files = sorted(Path(args.usd).glob("*.usd"))
props = Props(stage, app)
items = [(f.stem, props.add(f"/World/T/t{i:03d}", str(f.resolve()))) for i, f in enumerate(files)]
for _ in range(10):
    app.update()
props.recenter([p for _, p in items])
for _ in range(30):
    app.update()

bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
out = Path(args.out)
raw = out / "_raw"
shutil.rmtree(raw, ignore_errors=True)
raw.mkdir(parents=True)
rep.orchestrator.set_capture_on_play(False)
rp = rep.create.render_product(str(cam.GetPath()), RES)
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=str(raw), rgb=True)
writer.attach([rp])
shown = None
for name, prim in items:
    if shown is not None:
        set_visible(shown, False)
    set_visible(prim, True)
    shown = prim
    r = bc.ComputeWorldBound(prim).ComputeAlignedRange()
    scale = args.size / max(r.GetSize())
    props.set_pose(prim, (0, 0, 0), 0, scale)
    L = args.size * 1.6
    for az, el in VIEWS:
        t = math.radians(az)
        look_at(cam.GetPrim(), (L * math.cos(t), L * math.sin(t), L * el), (0, 0, args.size * 0.25))
        rep.orchestrator.step(rt_subframes=8)
rep.orchestrator.wait_until_complete()
for k, (name, _) in enumerate(items):
    views = [np.asarray(Image.open(raw / f"rgb_{2 * k + v:04d}.png").convert("RGB")) for v in range(len(VIEWS))]
    Image.fromarray(np.concatenate(views, axis=1)).save(out / f"{name}.png")
shutil.rmtree(raw, ignore_errors=True)
app.close()
