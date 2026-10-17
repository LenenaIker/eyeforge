"""S.4 of the sphere route, render: Isaac Sim renders each view of render/plan.json (render_plan.py) twice, without
the objects (B) and with them (A), over the proxy mesh, and composites them on the view's background by differential
rendering, as photo2scene's bg_isaac.py does on one photo (Debevec 1998): the background keeps the real panorama
times the shadow ratio A/B, the objects take A at the exposure that maps B to the background. The proxy also hides
the objects behind it: where A's depth is the proxy's, the pixel is background. Run with Isaac's python.sh
(SIM2CHIP_ISAAC_PYTHON):

    python.sh render.py --dir data/sphere_test/<place>

Light: a dome of the sky's mean colour (Alhaija et al. 2018: the matching environment map barely matters), or the
plan's sky texture (render_plan.overcast_sky), and, when
plan.json has one, a sun.

Writes <dir>/render/ (or --sub) render_NAME.png (composite), raw_NAME.png (pass A), obj_NAME.png (the objects'
visible pixels), inst_NAME.png (1 + the index of the object each pixel shows) and render.json (exposure gains, each
object's visible pixels and tight box, seconds).
"""
import argparse
import json
import math
import os
import time

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--sub", default="render", help="the plan's folder under --dir (render_plan.write)")
ap.add_argument("--subframes", type=int, default=16)
ap.add_argument("--path-tracing", action="store_true", help="path tracing (soft sky shadows) instead of real time")
ap.add_argument("--spp", type=int, default=64)
ap.add_argument("--warmup", type=int, default=120, help="app updates after loading, for the textures")
args = ap.parse_args()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
LEVEL_TO_WORLD = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # as render_plan.py
SENSOR_WIDTH_MM = 20.955
DOME_INTENSITY = 1000.0  # absolute level is free: the exposure is fitted against the background
SUN_INTENSITY = 3000.0   # per unit of the plan's sun share
SHADOW_BLUR_PX = 2.0     # the passes are smoothed this much before their ratio
DARK = 6.0               # B under this (0-255) is too dark for a ratio

def smooth(x):
    """Separable Gaussian blur of an (H, W, C) float image, edges repeated."""
    r = int(3 * SHADOW_BLUR_PX)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / SHADOW_BLUR_PX) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (r, r), (0, 0)), mode="edge")
    rows = sum(k[i] * pad[i:i + x.shape[0]] for i in range(2 * r + 1))
    return sum(k[i] * rows[:, i:i + x.shape[1]] for i in range(2 * r + 1))


def main():
    src = os.path.join(args.dir, args.sub)
    plan = json.load(open(os.path.join(src, "plan.json")))
    height = plan["camera_height_m"]

    settings = carb.settings.get_settings()
    settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
    settings.set("/omni/replicator/asyncRendering", False)
    if args.path_tracing:
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", args.spp)
        settings.set("/rtx/pathtracing/totalSpp", args.spp)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


    def surface(path, roughness):
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat, sh


    # Proxy: the levelled depth mesh in world coordinates, coloured per vertex by the erased panorama.
    proxy = np.load(os.path.join(args.dir, "proxy.npz"))
    pts = (proxy["vertices"] @ LEVEL_TO_WORLD.T + np.array([0.0, 0.0, height])).astype(np.float32)
    faces = proxy["faces"]
    m = UsdGeom.Mesh.Define(stage, "/World/Proxy")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(np.load(os.path.join(src, "proxy_colors.npy")).astype(np.float32)))
    mat, sh = surface("/World/Looks/Proxy", 0.9)
    rd = UsdShade.Shader.Define(stage, "/World/Looks/Proxy/Color")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

    # Light.
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr().Set(DOME_INTENSITY)
    if plan.get("sky_texture"):  # its pole is the dome's local +Z on this Z-up stage, as piece_isaac.py found
        dome.CreateTextureFileAttr().Set(plan["sky_texture"])
        dome.CreateTextureFormatAttr().Set("latlong")
    else:
        dome.CreateColorAttr().Set(Gf.Vec3f(*plan["sky_rgb"]))
    if plan.get("sun"):
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateAngleAttr().Set(float(plan["sun"].get("angle_deg", 1.0)))
        sun.CreateIntensityAttr().Set(SUN_INTENSITY * float(plan["sun"]["share"]))
        z = -np.asarray(plan["sun"]["travel_world"], float)  # a distant light shines along its -Z
        z /= np.linalg.norm(z)
        x = np.cross([0.0, 0.0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        rot = np.eye(4)
        rot[:3, 0], rot[:3, 1], rot[:3, 2] = x, np.cross(z, x), z
        sun.AddTransformOp().Set(Gf.Matrix4d(rot.T.tolist()))

    # Objects: one prim per object key (an object's "key", else its USD, so two objects of one asset in a view are
    # two prims), moved and shown per view. The reference sits on a child prim (some assets carry their own double3
    # ops) and is recentred on its measured box: base on z = 0, centred in x, y.
    def key(o):
        return o.get("key", o["usd"])

    assets = {}
    for v in plan["views"]:
        for o in v["objects"]:
            if key(o) in assets:
                continue
            k = len(assets)
            prim = stage.DefinePrim(f"/World/Objects/a{k}", "Xform")
            ref = stage.DefinePrim(f"/World/Objects/a{k}/Ref", "Xform")
            ref.GetReferences().AddReference(o["usd"])
            xf = UsdGeom.Xformable(prim)
            assets[key(o)] = (prim, xf.AddTranslateOp(), xf.AddRotateZOp(), xf.AddScaleOp(), ref)
    for _ in range(20):
        app.update()
    lengths = {}  # each asset's measured long side, in its own units: the scale comes from the real length
    for usd, (prim, _, _, _, ref) in assets.items():  # usd: the object's key
        r = None
        for _ in range(60):  # a heavy asset may still be loading: wait for its geometry
            bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            r = bc.ComputeLocalBound(ref).ComputeAlignedRange()
            if not r.IsEmpty():
                break
            app.update()
        if r.IsEmpty():
            print(f"no geometry in {usd}: left out", flush=True)
            continue
        c = (r.GetMin() + r.GetMax()) / 2
        UsdGeom.Xformable(ref).AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -r.GetMin()[2]))
        # Objaverse meshes often come single-sided with some normals flipped: the renderer culls those faces and
        # the object shows the background through itself (court 2026-10-16). Every mesh is drawn on both sides.
        for q in Usd.PrimRange(ref):
            if q.IsA(UsdGeom.Mesh):
                UsdGeom.Mesh(q).CreateDoubleSidedAttr().Set(True)
        size = r.GetMax() - r.GetMin()
        lengths[usd] = max(size[0], size[1])
        print(f"asset {os.path.basename(usd)}: size {[round(v, 3) for v in size]}", flush=True)

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000))
    cam_op = cam.AddTransformOp()
    rep.orchestrator.set_capture_on_play(False)
    sizes = {tuple(v["size"]) for v in plan["views"]}
    if len(sizes) != 1:
        raise SystemExit(f"the views of a plan share one size, got {sorted(sizes)}")
    rp = rep.create.render_product(str(cam.GetPath()), sizes.pop())
    prod = (rp, rep.AnnotatorRegistry.get_annotator("rgb"), rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"))
    prod[1].attach([rp])
    prod[2].attach([rp])
    # Which object each pixel shows, for the labels: Replicator's instance ids, mapped back to the object prims.
    inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
    inst.attach([rp])
    for _ in range(args.warmup):
        app.update()


    def capture(prod):
        rep.orchestrator.step(rt_subframes=1 if args.path_tracing else args.subframes)
        return prod[1].get_data()[..., :3].astype(np.float32), np.nan_to_num(prod[2].get_data(), posinf=0.0)


    def show(view):
        used = {key(o) for o in view["objects"]}
        for k, (prim, *_r) in assets.items():
            UsdGeom.Imageable(prim).MakeVisible() if k in used else UsdGeom.Imageable(prim).MakeInvisible()
        for o in view["objects"]:
            _, t_op, r_op, s_op, _ = assets[key(o)]
            t_op.Set(Gf.Vec3d(*o["pos"]))
            r_op.Set(float(o.get("yaw_deg", 0.0)))
            s = float(o["length_m"]) / lengths[key(o)] if lengths.get(key(o)) else 0.0
            s_op.Set(Gf.Vec3f(s, s, s))


    def hide_all():
        for prim, *_r in assets.values():
            UsdGeom.Imageable(prim).MakeInvisible()


    report = []
    for view in plan["views"]:
        t = time.time()
        w, h = view["size"]
        cam.CreateHorizontalApertureAttr(SENSOR_WIDTH_MM)
        cam.CreateVerticalApertureAttr(SENSOR_WIDTH_MM * h / w)
        cam.CreateFocalLengthAttr(SENSOR_WIDTH_MM / 2 / math.tan(math.radians(view["fov_deg"]) / 2))
        cam_op.Set(Gf.Matrix4d((np.asarray(view["pose"]) @ CV2GL).T.tolist()))
        hide_all()
        b, db = capture(prod)
        show(view)
        a, da = capture(prod)
        unlit = np.asarray(Image.open(os.path.join(src, f"bg_{view['name']}.png")).convert("RGB")).astype(np.float32)
        obj = (da > 0) & ((db == 0) | (da < db - 0.01))
        lit = ~obj & (db > 0) & (b.mean(-1) > 10)
        gain = np.median(unlit[lit] / np.maximum(b[lit], 1), axis=0) if lit.sum() > 100 else np.ones(3)
        # The ratio of two noisy renders is noisy where B is dark: taken on both passes smoothed, and left at 1
        # where B is nearly black or sees nothing (sky).
        sa, sb = smooth(a), smooth(b)
        shadow = np.clip(sa / np.maximum(sb, 1), 0, 1.2)
        shadow[(db == 0) | (sb.mean(-1) < DARK)] = 1.0
        out = np.where(obj[..., None], a * gain, unlit * shadow)
        name = view["name"]
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(os.path.join(src, f"render_{name}.png"))
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(src, f"raw_{name}.png"))
        Image.fromarray((obj * 255).astype(np.uint8)).save(os.path.join(src, f"obj_{name}.png"))
        # Instance map of pass A: 1 + the object's index in the view, 0 elsewhere (proxy, sky).
        seg = inst.get_data()
        ids, labels = np.asarray(seg["data"]), seg["info"]["idToLabels"]
        index = np.zeros(ids.shape, np.uint8)
        prims = {str(assets[key(o)][0].GetPath()): k + 1 for k, o in enumerate(view["objects"]) if key(o) in assets}
        for i, path in labels.items():
            path = path if isinstance(path, str) else str(path)
            for root, k in prims.items():
                if path == root or path.startswith(root + "/"):
                    index[ids == int(i)] = k
        index[~obj] = 0
        Image.fromarray(index).save(os.path.join(src, f"inst_{name}.png"))
        boxes = []
        for k, o in enumerate(view["objects"]):
            ys, xs = np.nonzero(index == k + 1)
            boxes.append(None if not len(xs) else
                         {"pixels": int(len(xs)), "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                                          int(ys.max()) + 1]})
        report.append({"name": name, "gain": np.round(gain, 3).tolist(), "object_pixels": int(obj.sum()),
                       "objects": boxes,
                       "seconds": round(time.time() - t, 1)})
        print(json.dumps(report[-1]), flush=True)
    json.dump({"views": report}, open(os.path.join(src, "render.json"), "w"), indent=1)


# Isaac's app swallows a script's exceptions and exits 0: a failed render must fail the step.
try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    app.close()
    os._exit(1)
app.close()
