"""The objects of plan.json (plan.py) rendered in Isaac Sim and composited on their backgrounds (background.py), in
Isaac's Python (SIM2CHIP_ISAAC_PYTHON):

    python.sh -m sim2chip.walkscene.isaac --dir <stage folder>

Each view is rendered twice over a ground plane at the splat's ground height, without the objects (B) and with them
(A), and composited by differential rendering (Debevec 1998; the sphere route's render step did the same over its
proxy mesh): the background keeps the splat times the shadow ratio A/B where the splat's surface is the ground, the
objects take A at the exposure gain that maps B to the background on the ground. An object pixel shows only where
the object is nearer than the splat (a hedge or a parked car of the scene hides it). The plane takes the ground's
colour with the sky's divided out, the dome the sky's colour as a CIE overcast sky: the gain then carries the
scene's light onto the objects.

Writes render_NAME.png (composite), obj_NAME.png (the objects' visible pixels) and render.json (per view: the gain,
each object's visible pixels and tight box; per object None when hidden)."""
import argparse
import json
import math
import os
import time

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--subframes", type=int, default=16)
ap.add_argument("--warmup", type=int, default=60, help="app updates after loading, for the textures")
args = ap.parse_args()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

from sim2chip.walkscene import isaac_world  # noqa: E402

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
SENSOR_WIDTH_MM = 20.955  # Isaac's default horizontal aperture; only its ratio to the focal length matters
DOME_INTENSITY = 1000.0   # the absolute level is free: the gain is fitted against the background
PLANE_M = 400.0
ALBEDO_MAX = 0.9
SHADOW_BLUR_PX = 2.0      # the passes are smoothed this much before their ratio
DARK = 6.0                # B under this (0-255) is too dark for a ratio
ON_GROUND = 0.08          # the splat's surface within this share of the plane's depth is the ground
OCCLUDE = 0.03            # an object pixel this share nearer than the splat shows
SKY_SIZE = (512, 256)
GROUND_BOUNCE = 0.2       # the dome under the horizon, as a share of the horizon's light


def srgb_to_linear(c):
    c = np.asarray(c, np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def write_hdr(path, rgb):
    """A float (H, W, 3) image as Radiance RGBE (.hdr), flat scanlines."""
    m = rgb.max(-1)
    mant, exp = np.frexp(m)
    scale = np.where(m > 1e-32, mant * 256 / np.maximum(m, 1e-32), 0)
    rgbe = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(m > 1e-32, exp + 128, 0).astype(np.uint8)
    h, w = m.shape
    with open(path, "wb") as f:
        f.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode())
        f.write(rgbe.tobytes())


def overcast_sky(path, rgb):
    """A latlong dome of the CIE standard overcast sky (luminance (1 + 2 sin e) / 3 at elevation e) in the sky's
    colour, a dim ground under the horizon. Row 0 is the zenith, as Isaac's latlong domes on a Z-up stage."""
    w, h = SKY_SIZE
    e = np.radians(90 - (np.arange(h) + 0.5) / h * 180)
    lum = np.where(e >= 0, (1 + 2 * np.sin(e)) / 3, GROUND_BOUNCE / 3)
    write_hdr(path, np.repeat(lum[:, None, None], w, 1) * np.asarray(rgb, np.float32))


def smooth(x):
    """Separable Gaussian blur of an (H, W, C) float image, edges repeated."""
    r = int(3 * SHADOW_BLUR_PX)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / SHADOW_BLUR_PX) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (r, r), (0, 0)), mode="edge")
    rows = sum(k[i] * pad[i:i + x.shape[0]] for i in range(2 * r + 1))
    return sum(k[i] * rows[:, i:i + x.shape[1]] for i in range(2 * r + 1))


def main():
    d = args.dir
    plan = json.load(open(os.path.join(d, "plan.json")))
    bgs = json.load(open(os.path.join(d, "background.json")))
    scenes = json.load(open(os.path.join(d, "scenes.json")))
    heights = {sid: json.load(open(os.path.join(f, "scene.json")))["camera_height"] for sid, f in scenes.items()}
    w, h = plan["size"]
    min_px = max(40, int(0.0005 * w * h))
    views = sorted(plan["views"], key=lambda v: (v["scene"], v["name"]))  # one dome texture per scene

    settings = carb.settings.get_settings()
    settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
    settings.set("/omni/replicator/asyncRendering", False)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Ground plane, moved under each camera at the view's ground height, in the view's ground colour.
    s = PLANE_M / 2
    plane = UsdGeom.Mesh.Define(stage, "/World/Ground")
    plane.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0), Gf.Vec3f(s, s, 0),
                                          Gf.Vec3f(-s, s, 0)]))
    plane.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    plane.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    plane.CreateDoubleSidedAttr(True)
    plane_op = UsdGeom.Xformable(plane).AddTranslateOp()
    mat = UsdShade.Material.Define(stage, "/World/Looks/Ground")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/Ground/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    albedo = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat)

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr().Set(DOME_INTENSITY)
    dome.CreateTextureFormatAttr().Set("latlong")
    skies = {}
    for sid, rgb in bgs["sky_rgb"].items():
        lin = srgb_to_linear(rgb)
        skies[sid] = lin / max(float(lin.max()), 1e-6)
        overcast_sky(os.path.join(d, f"sky_{sid}.hdr"), skies[sid])

    # Objects: one prim per model, moved and shown per view; the reference on a child prim, recentred on its
    # measured box (base on z = 0, centred in x and y) and drawn on both sides (Objaverse meshes come with flipped
    # normals, court 2026-10-16).
    assets = {}
    for v in views:
        for o in v["objects"]:
            if o["key"] in assets:
                continue
            k = len(assets)
            prim = stage.DefinePrim(f"/World/Objects/a{k}", "Xform")
            ref = stage.DefinePrim(f"/World/Objects/a{k}/Ref", "Xform")
            ref.GetReferences().AddReference(o["usd"])
            xf = UsdGeom.Xformable(prim)
            assets[o["key"]] = (prim, xf.AddTranslateOp(), xf.AddRotateZOp(), xf.AddScaleOp(), ref)
    for _ in range(20):
        app.update()
    dims = {}  # measured length, width, height in the model's own units
    for key, (prim, _, _, _, ref) in assets.items():
        r = None
        for _ in range(60):
            bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            r = bc.ComputeLocalBound(ref).ComputeAlignedRange()
            if not r.IsEmpty():
                break
            app.update()
        if r.IsEmpty():
            print(f"[sim2chip] no geometry in {key}: left out", flush=True)
            continue
        c = (r.GetMin() + r.GetMax()) / 2
        UsdGeom.Xformable(ref).AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -r.GetMin()[2]))
        for q in Usd.PrimRange(ref):
            if q.IsA(UsdGeom.Mesh):
                UsdGeom.Mesh(q).CreateDoubleSidedAttr().Set(True)
        size = r.GetMax() - r.GetMin()
        dims[key] = (max(size[0], size[1]), min(size[0], size[1]), size[2])

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000))
    cam.CreateHorizontalApertureAttr(SENSOR_WIDTH_MM)
    cam.CreateVerticalApertureAttr(SENSOR_WIDTH_MM * h / w)
    cam.CreateFocalLengthAttr(SENSOR_WIDTH_MM / 2 / math.tan(math.radians(plan["fov_deg"]) / 2))
    cam_op = cam.AddTransformOp()
    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product(str(cam.GetPath()), (w, h))
    rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
    dep_a = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
    for an in (rgb_a, dep_a, inst):
        an.attach([rp])
    for _ in range(args.warmup):
        app.update()

    def capture():
        rep.orchestrator.step(rt_subframes=args.subframes)
        return rgb_a.get_data()[..., :3].astype(np.float32), np.nan_to_num(dep_a.get_data(), posinf=0.0)

    def show(view, ground_z, height):
        used = {o["key"] for o in view["objects"]}
        for key, (prim, *_r) in assets.items():
            UsdGeom.Imageable(prim).MakeVisible() if key in used else UsdGeom.Imageable(prim).MakeInvisible()
        for o, z in zip(view["objects"], ground_z):
            if o["key"] not in dims:
                continue
            _, t_op, r_op, s_op, _ = assets[o["key"]]
            t_op.Set(Gf.Vec3d(o["xy"][0], o["xy"][1], z + height))
            r_op.Set(float(o["yaw_deg"]))
            sc = float(o["size_m"]) / max(dims[o["key"]][o["measure"]], 1e-9)
            s_op.Set(Gf.Vec3f(sc, sc, sc))

    def hide_all():
        for prim, *_r in assets.values():
            UsdGeom.Imageable(prim).MakeInvisible()

    report, sky_now, last_gain = [], None, np.ones(3)
    for view in views:
        t = time.time()
        name, sid = view["name"], view["scene"]
        bgv, height = bgs["views"][name], float(heights[sid])
        if sky_now != sid:
            dome.GetTextureFileAttr().Set(os.path.join(d, f"sky_{sid}.hdr"))
            for _ in range(10):
                app.update()
            sky_now = sid
        m = np.asarray(view["w2c"])
        c2w = isaac_world(m, height)
        cam_op.Set(Gf.Matrix4d((c2w @ CV2GL).T.tolist()))
        plane_op.Set(Gf.Vec3d(c2w[0, 3], c2w[1, 3], bgv["plane_z"] + height))
        g = srgb_to_linear(bgv["ground_rgb"] or [128, 128, 128]) / np.maximum(skies[sid], 1e-3)
        albedo.Set(Gf.Vec3f(*(g * ALBEDO_MAX / max(float(g.max()), 1e-6)).tolist()))
        hide_all()
        b, db = capture()
        show(view, bgv["ground_z"], height)
        a, da = capture()
        seg = inst.get_data()

        bg = np.asarray(Image.open(os.path.join(d, f"bg_{name}.png")).convert("RGB")).astype(np.float32)
        sd = np.load(os.path.join(d, f"depth_{name}.npy")).astype(np.float32)
        ids, labels = np.asarray(seg["data"]), seg["info"]["idToLabels"]
        index = np.zeros(ids.shape, np.uint8)
        prims = {str(assets[o["key"]][0].GetPath()): k + 1 for k, o in enumerate(view["objects"])
                 if o["key"] in dims}
        for i, path in labels.items():
            path = path if isinstance(path, str) else str(path)
            for root, k in prims.items():
                if path == root or path.startswith(root + "/"):
                    index[ids == int(i)] = k
        obj = (index > 0) & (da > 0) & ((sd == 0) | (da < sd * (1 - OCCLUDE)))
        index[~obj] = 0
        ground = (db > 0) & ((sd == 0) | (sd > db * (1 - ON_GROUND)))   # the splat shows the ground (or nothing)
        lit = ground & ~obj & (sd > 0) & (np.abs(sd - db) < db * ON_GROUND) & (b.mean(-1) > 10)
        if lit.sum() > 100:
            last_gain = np.median(bg[lit] / np.maximum(b[lit], 1), axis=0)
        gain = last_gain
        sa, sb = smooth(a), smooth(b)
        shadow = np.clip(sa / np.maximum(sb, 1), 0, 1.2)
        shadow[~ground | (sb.mean(-1) < DARK)] = 1.0
        out = np.where(obj[..., None], a * gain, bg * shadow)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(os.path.join(d, f"render_{name}.png"))
        Image.fromarray((obj * 255).astype(np.uint8)).save(os.path.join(d, f"obj_{name}.png"))
        Image.fromarray(index.astype(np.uint8)).save(os.path.join(d, f"index_{name}.png"))  # overlay: per-bin ids
        boxes = []
        for k, o in enumerate(view["objects"]):
            ys, xs = np.nonzero(index == k + 1)
            boxes.append(None if len(xs) < min_px else
                         {"pixels": int(len(xs)), "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                                          int(ys.max()) + 1]})
        report.append({"name": name, "gain": np.round(gain, 3).tolist(), "lit_px": int(lit.sum()),
                       "objects": boxes, "seconds": round(time.time() - t, 1)})
        print(f"[sim2chip] view {name}: {json.dumps(report[-1])}", flush=True)
    json.dump({"views": report, "min_px": min_px}, open(os.path.join(d, "render.json"), "w"), indent=1)


# Isaac's app swallows a script's exceptions and exits 0: a failed render must fail the step.
try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    app.close()
    os._exit(1)
app.close()
