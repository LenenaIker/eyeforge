"""Isaac Sim render of the scene that piece_build.py built in pieces (<dir>/isaac): textured planes for the floor,
the support and the walls, the SAM 3D objects, an HDRI sky, a sun and, indoors, a room lamp. Everything is
synthetic, so the camera can go anywhere (bg_isaac.py composites on the photo instead). Run with Isaac's
python.sh:

    python.sh piece_isaac.py --dir data/scene3d_test/ours/office_pieces [--seed 0]

Writes <dir>/isaac/render_NN.png and <dir>/sheet_isaac.jpg.
"""
import argparse
import glob
import json
import math
import os

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sweep", action="store_true", help="view 0 under a fixed set of lights, to isaac/sweep/")
ap.add_argument("--hdri-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/hdri"))
args = ap.parse_args()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
EXPOSE_TOP, EXPOSE_MEDIAN_RANGE, EXPOSE_MAX_GAIN = 0.85, (0.28, 0.52), 4.0
# Colours cut from a photo already hold its daylight; lit again here they came out too pale (user, 2026-10-07,
# mostly the ground): albedo is the photo colour times these.
ALBEDO, ALBEDO_GROUND = 0.8, 0.7
src = os.path.join(args.dir, "isaac")
scene = json.load(open(os.path.join(src, "scene.json")))
arr = np.load(os.path.join(src, "meshes.npz"))
W, H = scene["size"]
rng = np.random.default_rng(args.seed)

settings = carb.settings.get_settings()
settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
settings.set("/omni/replicator/asyncRendering", False)
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def mesh(path, points, faces):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).reshape(-1)))
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


# Pieces: one quad each, the photo's material (made seamless in piece_build) tiled by its size in metres.
up0 = np.asarray(scene["up"], float) / np.linalg.norm(scene["up"])


def albedo(points):
    """Ground (a horizontal piece) or the rest."""
    c = np.asarray(points, float)
    n = np.cross(c[1] - c[0], c[3] - c[0])
    return ALBEDO_GROUND if abs(n @ up0) > 0.9 * np.linalg.norm(n) else ALBEDO


for i, pc in enumerate(scene["pieces"]):
    m = mesh(f"/World/Pieces/p{i}", pc["points"], [[0, 1, 2], [0, 2, 3]])
    mat, sh = surface(f"/World/Looks/Piece{i}", 0.8)
    if pc["texture"]:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(pc["uv"], np.float32)))
        tex = UsdShade.Shader.Define(stage, f"/World/Looks/Piece{i}/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(pc["texture"])
        for wrap in ("wrapS", "wrapT"):
            tex.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set("repeat")
        st = UsdShade.Shader.Define(stage, f"/World/Looks/Piece{i}/St")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        g = np.asarray(pc.get("gain") or [1, 1, 1], float) * albedo(pc["points"])  # flow_photometry.py gains
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*g.tolist(), 1))
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(linear(pc["colour"]) * albedo(pc["points"]) * np.asarray(pc.get("gain") or [1, 1, 1])).tolist()))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

# Objects: SAM 3D meshes with vertex colours, one prim each, posed and shown per view. Faces the photo saw
# (env_add.py --photo-texture) take the photo itself as texture, through a face subset with its own material.
objs, refs = [], []
for i in range(scene["objects"]):
    xf = UsdGeom.Xform.Define(stage, f"/World/Objects/o{i}")
    if f"obj{i}_usd" in arr.files:  # an NVIDIA tree or shrub (env_add.py vegetation): the asset itself, posed
        asset = UsdGeom.Xform.Define(stage, f"/World/Objects/o{i}/asset")
        asset.AddTransformOp().Set(Gf.Matrix4d(np.asarray(arr[f"obj{i}_usd_pose"]).T.tolist()))
        ref = stage.DefinePrim(f"/World/Objects/o{i}/asset/Ref")
        ref.GetReferences().AddReference(str(arr[f"obj{i}_usd"]))
        refs.append(ref)
        objs.append((xf, xf.AddTransformOp()))
        continue
    m = mesh(f"/World/Objects/o{i}/mesh", arr[f"obj{i}_points"], arr[f"obj{i}_faces"])
    m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(linear(arr[f"obj{i}_colors"]) * ALBEDO))
    mat, sh = surface(f"/World/Looks/Obj{i}", 0.45)
    rd = UsdShade.Shader.Define(stage, f"/World/Looks/Obj{i}/Color")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    photo = np.nonzero(arr[f"obj{i}_photo_faces"])[0] if f"obj{i}_photo_faces" in arr.files else []
    if len(photo):
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(arr[f"obj{i}_uv"], np.float32)))
        pmat, psh = surface(f"/World/Looks/ObjPhoto{i}", 0.8)
        tex = UsdShade.Shader.Define(stage, f"/World/Looks/ObjPhoto{i}/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(arr[f"obj{i}_texture"]))
        for wrap in ("wrapS", "wrapT"):
            tex.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set("clamp")
        st = UsdShade.Shader.Define(stage, f"/World/Looks/ObjPhoto{i}/St")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        g = np.asarray(arr[f"obj{i}_gain"] if f"obj{i}_gain" in arr.files else [1, 1, 1], float) * ALBEDO
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*g.tolist(), 1))
        psh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        sub = UsdGeom.Subset.CreateGeomSubset(m, "photo", UsdGeom.Tokens.face, Vt.IntArray(photo.tolist()),
                                              "materialBind")
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(pmat)
        rest = np.setdiff1d(np.arange(len(arr[f"obj{i}_faces"])), photo)
        if len(rest):
            sub = UsdGeom.Subset.CreateGeomSubset(m, "colour", UsdGeom.Tokens.face, Vt.IntArray(rest.tolist()),
                                                  "materialBind")
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    objs.append((xf, xf.AddTransformOp()))

# Remote assets load in the background: render only once each has a size (an unloaded tree measured 0 and came
# out missing; Isaac caches them in ~/.cache/ov for the next run).
if refs:
    cache = UsdGeom.BBoxCache(0, ["default", "render"])
    for _ in range(600):
        cache.Clear()
        if all(not cache.ComputeWorldBound(r).ComputeAlignedRange().IsEmpty() for r in refs):
            break
        app.update()
    else:
        print("some vegetation assets did not load", flush=True)

# Light: an HDRI sky (one per scene, turned per view) and a sun outdoors; indoors (closed room) a ceiling panel
# and a lamp moved per view.
up = np.array(scene["up"])
hdris = sorted(glob.glob(os.path.join(args.hdri_dir, "*.hdr")))
dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateIntensityAttr().Set(1000)
if hdris:
    dome.CreateTextureFileAttr().Set(os.path.abspath(hdris[rng.integers(len(hdris))]))
    dome.CreateTextureFormatAttr().Set("latlong")
dome_op = dome.AddTransformOp()
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateAngleAttr().Set(2.0)
sun.CreateIntensityAttr().Set(0 if scene["indoor"] else 2500)
sun_op = sun.AddTransformOp()
lamp_op = None
panel_size, window_lights, ceiling = (1.2, 1.2), [], None
PANEL, WINDOW, LAMP = 2000.0, 12000.0, 5000.0  # base intensities, scaled by the light set_indoor() picks
if scene["indoor"]:
    # A panel over most of the ceiling: a room's light is mostly bounced and even. A 1.2 m panel and a hot lamp
    # left home_office 4 times darker than the photo and warm; the photometry's per-channel gains then tinted the
    # walls green and the sideboard blue (2026-10-08).
    flat = [np.asarray(p["points"]) for p in scene["pieces"] if np.ptp(np.asarray(p["points"]) @ up) < 0.05]
    ceiling = max(flat, key=lambda c: float(np.mean(c @ up))) if len(flat) > 1 else None
    panel = UsdLux.RectLight.Define(stage, "/World/Panel")
    if ceiling is not None:
        panel_size = (0.8 * float(np.linalg.norm(ceiling[1] - ceiling[0])),
                      0.8 * float(np.linalg.norm(ceiling[3] - ceiling[0])))
    panel.CreateWidthAttr().Set(panel_size[0])
    panel.CreateHeightAttr().Set(panel_size[1])
    panel.CreateIntensityAttr().Set(PANEL)
    panel_op = panel.AddTransformOp()
    lamp = UsdLux.SphereLight.Define(stage, "/World/Lamp")
    lamp.CreateRadiusAttr().Set(0.15)
    lamp.CreateIntensityAttr().Set(LAMP)
    lamp_op = lamp.AddTranslateOp()
    # Daylight through the photo's windows (piece_build.py), a panel on each facing into the room.
    for i, wd in enumerate(scene.get("windows", [])):
        wl = UsdLux.RectLight.Define(stage, f"/World/Window{i}")
        wl.CreateWidthAttr().Set(wd["size"][0])
        wl.CreateHeightAttr().Set(wd["size"][1])
        wl.CreateIntensityAttr().Set(WINDOW)
        wl.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.96, 1.0))
        rot = np.eye(4)  # local X along the wall, Y up, -Z (where a rect light shines) into the room
        rot[:3, 0], rot[:3, 1] = -np.asarray(wd["along"]), up  # -along keeps the frame right-handed
        rot[:3, 2], rot[:3, 3] = -np.asarray(wd["normal"]), wd["centre"]
        wl.AddTransformOp().Set(Gf.Matrix4d(rot.T.tolist()))
        window_lights.append(wl)


def frame(z):
    """4x4 (row-major for Gf) whose local -Z points along z: lights shine along their -Z."""
    z = -np.asarray(z, float) / np.linalg.norm(z)
    x = np.cross(up, z) if abs(up @ z) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    rot = np.eye(4)
    rot[:3, 0], rot[:3, 1], rot[:3, 2] = x, np.cross(z, x), z
    return Gf.Matrix4d(rot.T.tolist())


SUN = 2500.0
SUN_LEVELS = (1.0, 0.3)  # the sweep's sun strengths: full, and soft (a lane under trees, light through leaves)
SKY = 1000.0
# The sweep's sky strengths: the dome as before, and three times as bright. Under a tree canopy the light is mostly
# the sky's; with the dome at 1000 the crowns came out black over a white sunlit road (roads, 2026-10-13).
SKY_LEVELS = (1.0, 3.0)


def set_light(e, a, sun_on=True, sky=1.0):
    """Sun at elevation e and azimuth a (radians, a about the scene's up from a fixed side), the dome turned by a.
    sun_on: True or a strength (a share of SUN); False or 0: the sky alone (overcast). sky: the dome's strength,
    a share of SKY."""
    dome.CreateIntensityAttr().Set(SKY * float(sky))
    side = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    other = np.cross(up, side)
    sun_op.Set(frame(-(math.cos(e) * (math.cos(a) * side + math.sin(a) * other) + math.sin(e) * up)))
    sun.CreateIntensityAttr().Set(0 if scene["indoor"] or not sun_on else SUN * float(sun_on))
    # The dome's latlong pole is its local +Z in RTX (the product renders with a Z-up stage): align it with the
    # scene's up, then turn it about that axis. With +Y the sky lay on its side (street HDRIs, 2026-10-06).
    z = up / np.linalg.norm(up)
    x = np.cross(z, side)
    x = math.cos(a) * x + math.sin(a) * np.cross(z, x)
    rot = np.eye(4)
    rot[:3, 0], rot[:3, 1], rot[:3, 2] = x, np.cross(z, x), z
    dome_op.Set(Gf.Matrix4d(rot.T.tolist()))


def set_indoor(panel_k, window_k, lamp_k=1.0):
    """Indoors: the ceiling panel, the windows' daylight and the lamp as multiples of their base intensities."""
    panel.CreateIntensityAttr().Set(PANEL * panel_k)
    lamp.CreateIntensityAttr().Set(LAMP * lamp_k)
    for wl in window_lights:
        wl.CreateIntensityAttr().Set(WINDOW * window_k)


def randomize_light():
    # The other views keep the photo's kind of light (flow_light.py's match): a soft sun stays soft, the sky alone
    # stays overcast; only where it comes from changes.
    light = scene.get("light") or {}
    set_light(math.radians(rng.uniform(30, 70)), rng.uniform(0, 2 * math.pi), light.get("sun", True),
              light.get("sky", 1.0))
    if scene["indoor"]:
        set_indoor(rng.uniform(0.5, 2.0), rng.uniform(0.3, 2.0), rng.uniform(0.0, 1.5))



def place_lights(view):
    """Indoors: the panel on the ceiling facing down (a rect light shines along its -Z), the view's lamp."""
    if lamp_op is None:
        return
    if ceiling is not None:  # its sides along the ceiling's, just under it
        rot = np.eye(4)
        x = (ceiling[1] - ceiling[0]) / np.linalg.norm(ceiling[1] - ceiling[0])
        rot[:3, 0], rot[:3, 1], rot[:3, 2] = x, np.cross(up, x), up
        rot[:3, 3] = ceiling.mean(0) - 0.03 * up
        m = Gf.Matrix4d(rot.T.tolist())
    else:
        m = frame(-up)
        m.SetTranslateOnly(Gf.Vec3d(*scene["lamp"]))
    panel_op.Set(m)
    lamp_op.Set(Gf.Vec3d(*view["lamp"]))


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
rgb.attach([rp])
for _ in range(20):
    app.update()


def expose(img, fixed=None):
    """Auto exposure in linear light, as a camera would: the brightest parts (97th percentile) just below white,
    with the median kept within EXPOSE_MEDIAN_RANGE. Taking every view to one median (sim2chip/stages/render.py)
    turned front-lit street views flat grey, nothing in shadow to balance them, and burnt the sunlit ground of
    side-lit ones (2026-10-06)."""
    lin = (img / 255) ** 2.2
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    med, top = float(np.median(lum)), float(np.percentile(lum, 97))
    gain = EXPOSE_TOP ** 2.2 / max(top, 1e-6)
    lo, hi = (m ** 2.2 / max(med, 1e-6) for m in EXPOSE_MEDIAN_RANGE)
    gain = min(EXPOSE_MAX_GAIN, max(1 / EXPOSE_MAX_GAIN, min(hi, max(lo, gain))))
    # The photometry's exposure ratio (flow_photometry.py, photo over render): an overcast photo's place is
    # rendered as dim as the photo in every view, not only in the photo's own.
    gain *= scene.get("exposure_scale", 1.0)
    if fixed is not None:  # manual exposure: the photometry rounds must not be undone by the auto exposure
        gain = fixed
    expose.last = gain
    return (np.clip(lin * gain, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


photo = np.asarray(Image.open(os.path.join(args.dir, "photo.png")).convert("RGB"))
tiles = [("photo", photo)]
if args.sweep:
    # Step 2.3b of the scene flow: view 0 (the photo's camera) under a fixed set of lights, for flow_light.py to
    # pick the one whose light falls like the photo's.
    lights = [(el, az, k if k != 1.0 else True, sky) for sky in SKY_LEVELS for k in SUN_LEVELS
              for el in (20, 45, 70) for az in range(0, 360, 45)] + [(60, 0, False, sky) for sky in SKY_LEVELS]
    if scene["indoor"]:  # no sun indoors: how strong the ceiling, the windows and the lamp are against each other
        lights = [dict(panel=p, window=wk, lamp=lk) for p in (0.3, 1.0, 3.0)
                  for wk in ((0.0, 1.0, 3.0) if window_lights else (0.0,)) for lk in (0.0, 1.0)]
    os.makedirs(os.path.join(src, "sweep"), exist_ok=True)
    view = scene["views"][0]
    cam_op.Set(Gf.Matrix4d((np.asarray(view["pose"]) @ CV2GL).T.tolist()))
    for i, (xf, _) in enumerate(objs):
        (xf.MakeVisible if i in {j for j, _ in view["objects"]} else xf.MakeInvisible)()
    for i, t in view["objects"]:
        objs[i][1].Set(Gf.Matrix4d(np.asarray(t).T.tolist()))
    place_lights(view)
    for k, L in enumerate(lights):
        if scene["indoor"]:
            set_indoor(L["panel"], L["window"], L["lamp"])
        else:
            set_light(math.radians(L[0]), math.radians(L[1]), L[2], L[3])
        rep.orchestrator.step(rt_subframes=16)
        Image.fromarray(expose(rgb.get_data()[..., :3].astype(np.float32))).save(
            os.path.join(src, "sweep", f"sweep_{k:02d}.png"))
    json.dump(lights if scene["indoor"] else
              [{"elevation": el, "azimuth": az, "sun": on, "sky": sky} for el, az, on, sky in lights],
              open(os.path.join(src, "sweep", "lights.json"), "w"))
    app.close()
    raise SystemExit(0)

for k, view in enumerate(scene["views"]):
    randomize_light()
    if k == 0 and scene.get("light"):  # the light flow_light.py matched to the photo
        L = scene["light"]
        if scene["indoor"]:
            set_indoor(L["panel"], L["window"], L["lamp"])
        else:
            set_light(math.radians(L["elevation"]), math.radians(L["azimuth"]), L["sun"], L.get("sky", 1.0))
    place_lights(view)
    cam_op.Set(Gf.Matrix4d((np.asarray(view["pose"]) @ CV2GL).T.tolist()))
    shown = {i for i, _ in view["objects"]}
    for i, (xf, _) in enumerate(objs):
        (xf.MakeVisible if i in shown else xf.MakeInvisible)()
    for i, t in view["objects"]:
        objs[i][1].Set(Gf.Matrix4d(np.asarray(t).T.tolist()))
    rep.orchestrator.step(rt_subframes=16)
    out = expose(rgb.get_data()[..., :3].astype(np.float32), scene.get("exposure_0") if k == 0 else None)
    if k == 0:  # flow_photometry.py fixes it for the next rounds
        json.dump({"gain": expose.last}, open(os.path.join(src, "exposure_0.json"), "w"))
    Image.fromarray(out).save(os.path.join(src, f"render_{k:02d}.png"))
    tiles.append((f"view {k}", out))

tw, th = 300, round(300 * H / W)
cols = 4
sheet = Image.new("RGB", (tw * cols, th * math.ceil(len(tiles) / cols)), "white")
for i, (name, im) in enumerate(tiles):
    x, y = (i % cols) * tw, (i // cols) * th
    sheet.paste(Image.fromarray(im).resize((tw, th)), (x, y))
    ImageDraw.Draw(sheet).text((x + 5, y + 5), name, fill=(255, 40, 40))
sheet.save(os.path.join(args.dir, "sheet_isaac.jpg"), quality=88)
app.close()
