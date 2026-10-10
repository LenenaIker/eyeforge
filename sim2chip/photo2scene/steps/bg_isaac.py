"""Step 4: Isaac Sim render of the scene that bg_build.py exported (<dir>/isaac), composited on the photo by
differential rendering: each view is rendered with the objects (A) and without them (B); the background keeps the
photo seen from that camera (bg_unlit_NN.png) times the shadow ratio A/B, and the objects take A with the exposure
that maps B to the photo. Before that, the light (sun direction and strength) and each object's albedo are fitted
so the replica (view 0, objects where the real ones stood) matches the photo. Run with Isaac's python.sh:

    python.sh bg_isaac.py --dir data/scene3d_test/ours/bottles [--path-tracing]

Writes <dir>/isaac/render_NN.png (composite), raw_NN.png (pass A), fit.json and <dir>/sheet_isaac.jpg.
"""
import argparse
import json
import math
import os

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--path-tracing", action="store_true")
ap.add_argument("--light", type=float, nargs=3, default=None, metavar=("ELEV", "AZIM", "SHARE"),
                help="fixed sun: elevation and azimuth (degrees) and strength relative to the sky (default: fitted)")
ap.add_argument("--tonemap", type=int, default=None,
                help="RTX tone mapping operator (0 clamp, 1 linear/off, 6 ACES...); default: Isaac's")
ap.add_argument("--tag", default="", help="suffix for the sheet and fit files, to compare runs")
args = ap.parse_args()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
src = os.path.join(args.dir, "isaac")
scene = json.load(open(os.path.join(src, "scene.json")))
arr = np.load(os.path.join(src, "meshes.npz"))
W, H = scene["size"]

settings = carb.settings.get_settings()
settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
settings.set("/omni/replicator/asyncRendering", False)
print("tone mapping operator (Isaac default):", settings.get("/rtx/post/tonemap/op"), flush=True)
if args.tonemap is not None:
    settings.set("/rtx/post/tonemap/op", args.tonemap)
if args.path_tracing:
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", 64)
    settings.set("/rtx/pathtracing/totalSpp", 64)
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def mesh(path, points, faces):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr("none")
    return m


def surface(path, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, sh


# Background: the erased photo as diffuse texture (only its lighting matters, its colours come from bg_unlit).
bg = mesh("/World/Background", arr["bg_points"], arr["bg_faces"])
UsdGeom.PrimvarsAPI(bg).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
    Vt.Vec2fArray.FromNumpy(arr["bg_uv"]))
mat, sh = surface("/World/Looks/Background", 0.9)
tex = UsdShade.Shader.Define(stage, "/World/Looks/Background/Tex")
tex.CreateIdAttr("UsdUVTexture")
tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(os.path.join(src, "background.png")))
st = UsdShade.Shader.Define(stage, "/World/Looks/Background/St")
st.CreateIdAttr("UsdPrimvarReader_float2")
st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
UsdShade.MaterialBindingAPI.Apply(bg.GetPrim()).Bind(mat)

# Objects: SAM 3D meshes with vertex colours, one prim each, posed and shown per view.
objs = []
for i in range(scene["objects"]):
    xf = UsdGeom.Xform.Define(stage, f"/World/Objects/o{i}")
    m = mesh(f"/World/Objects/o{i}/mesh", arr[f"obj{i}_points"], arr[f"obj{i}_faces"])
    srgb = arr[f"obj{i}_colors"]  # glTF vertex colours are sRGB; displayColor is linear
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
    colour = m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    colour.Set(Vt.Vec3fArray.FromNumpy(linear))
    mat, sh = surface(f"/World/Looks/Obj{i}", 0.45)
    rd = UsdShade.Shader.Define(stage, f"/World/Looks/Obj{i}/Color")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    objs.append((xf, xf.AddTransformOp(), colour, linear))

# Light: soft sky plus a soft-edged sun. Its elevation (above the support), azimuth (around the support normal,
# 0 = straight ahead of the camera) and strength are fitted on view 0 unless --sun gives the travel direction.
up = np.array(scene["up"])
ahead = np.array([0, 0, 1.0]) - up[2] * up
ahead /= np.linalg.norm(ahead)
right = np.cross(ahead, up)  # image right (OpenCV: x = y cross z, y = -up)
UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr().Set(1000)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateAngleAttr().Set(6.0)
sun_op = sun.AddTransformOp()


def set_sun(travel, share):
    sun.GetIntensityAttr().Set(1000 * share * 3)
    z = -travel / np.linalg.norm(travel)  # a distant light shines along its -Z
    x = np.cross(up, z) if abs(up @ z) < 0.99 else ahead
    x /= np.linalg.norm(x)
    rot = np.eye(4)
    rot[:3, 0], rot[:3, 1], rot[:3, 2] = x, np.cross(z, x), z
    sun_op.Set(Gf.Matrix4d(rot.T.tolist()))


def sun_travel(elev, azim):
    e, a = math.radians(elev), math.radians(azim)
    return -(math.cos(e) * (math.cos(a) * ahead + math.sin(a) * right) + math.sin(e) * up)


# Camera with the photo's intrinsics (20.955 mm sensor width, square pixels).
px = 20.955 / W
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(W * px)
cam.CreateVerticalApertureAttr(H * px)
cam.CreateFocalLengthAttr(scene["fx"] * px)
cam.CreateHorizontalApertureOffsetAttr((scene["cx"] - W / 2) * px)
cam.CreateVerticalApertureOffsetAttr(-(scene["cy"] - H / 2) * px)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 2000))
cam_op = cam.AddTransformOp()

rep.orchestrator.set_capture_on_play(False)
rp = rep.create.render_product(str(cam.GetPath()), (W, H))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
dist = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
rgb.attach([rp])
dist.attach([rp])
for _ in range(20):
    app.update()


def capture():
    rep.orchestrator.step(rt_subframes=16 if not args.path_tracing else 1)
    return rgb.get_data()[..., :3].astype(np.float32), np.nan_to_num(dist.get_data(), posinf=0.0)


def to_linear(c):
    c = c / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def render_view(k):
    """Composite of view k: the photo's background times the objects' shadow, objects at the photo's exposure."""
    view = scene["views"][k]
    cam_op.Set(Gf.Matrix4d((np.asarray(view["pose"]) @ CV2GL).T.tolist()))
    for o in objs:
        o[0].MakeInvisible()
    b, db = capture()
    for i, t in view["objects"]:
        objs[i][0].MakeVisible()
        objs[i][1].Set(Gf.Matrix4d(np.asarray(t).T.tolist()))
    a, da = capture()
    unlit = np.asarray(Image.open(os.path.join(src, f"bg_unlit_{k:02d}.png")).convert("RGB")).astype(np.float32)
    obj = (da > 0) & ((db == 0) | (da < db - 0.002))
    lit = ~obj & (b.mean(-1) > 10)
    gain = np.median(unlit[lit] / np.maximum(b[lit], 1), axis=0)  # Isaac exposure -> photo exposure, per channel
    shadow = np.clip(a / np.maximum(b, 1), 0, 1.2)
    out = np.where(obj[..., None], a * gain, unlit * shadow)
    return np.clip(out, 0, 255).astype(np.uint8), a, obj, b


photo = np.asarray(Image.open(os.path.join(args.dir, "photo.png")).convert("RGB"))
photo_lin = to_linear(photo.astype(np.float32))
fit_masks = [np.asarray(Image.open(os.path.join(src, f"photo_mask_{i:02d}.png"))) > 0 for i in range(len(objs))]


# The photo's own shadows: photo / erased background on the support, where neither the photo nor the replica
# has an object. The light is fitted so the replica casts the same shadows (direction, softness, depth).
erased_bg = to_linear(np.asarray(Image.open(os.path.join(args.dir, "background.png")).convert("RGB")).astype(np.float32))
support = np.asarray(Image.open(os.path.join(args.dir, "support.png"))) > 0
photo_objects = np.asarray(Image.open(os.path.join(args.dir, "objects.png"))) > 0
real_shadow = np.log(np.maximum(photo_lin.mean(-1), 1e-4) / np.maximum(erased_bg.mean(-1), 1e-4))


def shadow_error(a, b, obj, block=None):
    """Real vs synthetic shadows on the support, compared as block averages (about 1/20 of the image) so a shadow
    in the right direction but a little off is not punished twice (dark where it is not, light where it is) and
    so does not lose against no shadow at all."""
    sel = support & ~photo_objects & ~obj & (b.mean(-1) > 10)
    synth = np.log(np.maximum(to_linear(a).mean(-1), 1e-4) / np.maximum(to_linear(b).mean(-1), 1e-4))
    real = np.clip(real_shadow, -2, 1)
    k = block or max(8, max(W, H) // 20)
    hh, ww = H // k * k, W // k * k

    def blocks(x):
        return x[:hh, :ww].reshape(hh // k, k, ww // k, k).sum((1, 3))

    cnt = blocks(sel.astype(np.float32))
    ok = cnt > 0.3 * k * k
    diff = (blocks(np.where(sel, real, 0)) - blocks(np.where(sel, synth, 0)))[ok] / cnt[ok]
    return float(np.mean(np.abs(diff)))


def fit_colours(iters=3):
    """One brightness gain per object so the replica matches its photo pixels (fixes SAM 3D's darkening); hue is
    left alone, a per-channel fit tinted the objects with the photo's white balance. Starts from SAM 3D's colours.
    Returns the gains and the brightness left unmatched (mean |log photo / replica|, e.g. albedo stuck at 1)."""
    gains = {}
    for i in range(len(objs)):
        objs[i] = objs[i][:3] + (base_albedo[i],)
        objs[i][2].Set(Vt.Vec3fArray.FromNumpy(base_albedo[i]))
    for it in range(iters + 1):
        out, a, obj, b = render_view(0)
        lum = to_linear(out.astype(np.float32)).mean(-1)
        resid = []
        for i, m in enumerate(fit_masks):
            sel = m & obj
            if sel.sum() < 200:
                continue
            ratio = np.median(photo_lin.mean(-1)[sel]) / max(np.median(lum[sel]), 1e-4)
            if it == iters:
                resid.append(abs(math.log(ratio)))
                continue
            g = float(np.clip(ratio, 0.33, 3))
            new = np.clip(objs[i][3] * g, 0, 1).astype(np.float32)
            objs[i] = objs[i][:3] + (new,)
            objs[i][2].Set(Vt.Vec3fArray.FromNumpy(new))
            gains.setdefault(i, []).append(round(g, 2))
    return gains, float(np.mean(resid)), shadow_error(a, b, obj)


# Light: the sun's elevation, azimuth and strength that best reproduce the photo on the replica, judged by the
# shadows on the support AND by whether the objects can reach the photo's brightness once their colour is fitted
# (a sun lights the support more than the objects' sides: white plastic stayed grey).
base_albedo = [o[3] for o in objs]
fit = {}
def shade_ratio(a, b):
    return to_linear(a).mean(-1) / np.maximum(to_linear(b).mean(-1), 1e-4)


def synthetic_darkness(a, b, sky_ratio):
    """How much the sun's shadows darken the support: A/B with the sun over A/B with sky only, median over the
    pixels the sun shadows (the umbrella's blocked sky darkens with or without sun and is left out)."""
    rel = shade_ratio(a, b) / np.maximum(sky_ratio, 1e-4)
    sel = support & (b.mean(-1) > 10) & (rel < 0.9)
    return float(np.median(rel[sel])) if sel.sum() > 200 else 1.0


sun_est = scene.get("sun")
if args.light:
    set_sun(sun_travel(args.light[0], args.light[1]), args.light[2])
elif sun_est and sun_est.get("shadow_darkness"):
    # Direction from the photo's shadows (bg_build); strength so the synthetic shadows are as dark as the real.
    travel = np.array(sun_est["travel"])
    set_sun(travel, 0.0)
    _, a, obj, b = render_view(0)
    sky_ratio = shade_ratio(a, b)
    best = None
    for share in (0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.0):
        set_sun(travel, share)
        _, a, obj, b = render_view(0)
        dark = synthetic_darkness(a, b, sky_ratio)
        print(f"sun from shadows, share {share}: synthetic darkness {dark:.3f} (photo {sun_est['shadow_darkness']:.3f})",
              flush=True)
        if best is None or abs(dark - sun_est["shadow_darkness"]) < best[0]:
            best = (abs(dark - sun_est["shadow_darkness"]), share)
    fit["light"] = {"from_shadows": sun_est, "sun_share": best[1]}
    set_sun(travel, best[1])
else:
    best = None
    for share in (0.0, 0.25, 0.5, 1.0):
        for elev in ((12, 25, 50, 75) if share else (90,)):
            for azim in ((0, 90, 180, 270) if share else (0,)):
                set_sun(sun_travel(elev, azim), share)
                _, resid, shadow = fit_colours(iters=2)
                score = shadow + resid
                print(f"light share {share} elev {elev} azim {azim}: shadow error {shadow:.4f}, "
                      f"brightness left {resid:.4f}, score {score:.4f}", flush=True)
                if best is None or score < best[0]:
                    best = (score, share, elev, azim)
    fit["light"] = dict(zip(("score", "sun_share", "elevation", "azimuth"), best))
    set_sun(sun_travel(best[2], best[3]), best[1])
fit["albedo_gain"], fit["brightness_left"], fit["shadow_error"] = fit_colours(iters=5)
print(json.dumps(fit), flush=True)
with open(os.path.join(src, f"fit{args.tag}.json"), "w") as f:
    json.dump(fit, f, indent=1)

tiles = [("photo", photo)]
for k in range(len(scene["views"])):
    out, a, obj, _ = render_view(k)
    Image.fromarray(out).save(os.path.join(src, f"render{args.tag}_{k:02d}.png"))
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(src, f"raw{args.tag}_{k:02d}.png"))
    tiles.append(("replica, photo camera" if k == 0 else f"view {k}", out))
    if k == 0 and args.tag:  # shadow maps for inspection: photo / erased background, and A / B
        _, _, _, b0 = render_view(0)
        sel = support & ~photo_objects & ~obj
        vis = lambda x: Image.fromarray((np.clip(np.where(sel, x, 0) + 1.5, 0, 1.5) / 1.5 * 255).astype(np.uint8))  # noqa: E731
        vis(np.clip(real_shadow, -2, 1)).save(os.path.join(src, "shadow_real.png"))
        vis(np.log(np.maximum(to_linear(a).mean(-1), 1e-4) / np.maximum(to_linear(b0).mean(-1), 1e-4))).save(
            os.path.join(src, f"shadow_synth{args.tag}.png"))
    if k == 0:
        print(f"pass A pixels at 255 (burnt): objects {(a[obj] >= 254).any(-1).mean():.2%}, "
              f"background {(a[~obj] >= 254).any(-1).mean():.2%}", flush=True)
        lum, ph = to_linear(out.astype(np.float32)).mean(-1), photo_lin.mean(-1)
        print("replica vs photo, median linear brightness per object:",
              [(round(float(np.median(ph[m & obj])), 3), round(float(np.median(lum[m & obj])), 3)) for m in fit_masks],
              flush=True)

tw, th = 300, round(300 * H / W)
cols = 4
sheet = Image.new("RGB", (tw * cols, th * math.ceil(len(tiles) / cols)), "white")
for i, (name, im) in enumerate(tiles):
    x, y = (i % cols) * tw, (i // cols) * th
    sheet.paste(Image.fromarray(im).resize((tw, th)), (x, y))
    ImageDraw.Draw(sheet).text((x + 5, y + 5), name, fill=(255, 40, 40))
sheet.save(os.path.join(args.dir, f"sheet_isaac{args.tag}.jpg"), quality=88)
app.close()
