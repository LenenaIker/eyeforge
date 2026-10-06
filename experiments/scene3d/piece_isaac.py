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
ap.add_argument("--hdri-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/hdri"))
args = ap.parse_args()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
EXPOSE_TOP, EXPOSE_MEDIAN_RANGE, EXPOSE_MAX_GAIN = 0.92, (0.32, 0.58), 4.0
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
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear(pc["colour"]).tolist()))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

# Objects: SAM 3D meshes with vertex colours, one prim each, posed and shown per view.
objs = []
for i in range(scene["objects"]):
    xf = UsdGeom.Xform.Define(stage, f"/World/Objects/o{i}")
    m = mesh(f"/World/Objects/o{i}/mesh", arr[f"obj{i}_points"], arr[f"obj{i}_faces"])
    m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(linear(arr[f"obj{i}_colors"])))
    mat, sh = surface(f"/World/Looks/Obj{i}", 0.45)
    rd = UsdShade.Shader.Define(stage, f"/World/Looks/Obj{i}/Color")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    objs.append((xf, xf.AddTransformOp()))

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
if scene["indoor"]:
    panel = UsdLux.RectLight.Define(stage, "/World/Panel")
    panel.CreateWidthAttr().Set(1.2)
    panel.CreateHeightAttr().Set(1.2)
    panel.CreateIntensityAttr().Set(8000)
    panel_op = panel.AddTransformOp()
    lamp = UsdLux.SphereLight.Define(stage, "/World/Lamp")
    lamp.CreateRadiusAttr().Set(0.15)
    lamp.CreateIntensityAttr().Set(2e4)
    lamp_op = lamp.AddTranslateOp()


def frame(z):
    """4x4 (row-major for Gf) whose local -Z points along z: lights shine along their -Z."""
    z = -np.asarray(z, float) / np.linalg.norm(z)
    x = np.cross(up, z) if abs(up @ z) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    rot = np.eye(4)
    rot[:3, 0], rot[:3, 1], rot[:3, 2] = x, np.cross(z, x), z
    return Gf.Matrix4d(rot.T.tolist())


def randomize_light():
    e, a = math.radians(rng.uniform(30, 70)), rng.uniform(0, 2 * math.pi)
    side = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    other = np.cross(up, side)
    sun_op.Set(frame(-(math.cos(e) * (math.cos(a) * side + math.sin(a) * other) + math.sin(e) * up)))
    # The dome's latlong pole is its local +Z in RTX (the product renders with a Z-up stage): align it with the
    # scene's up, then turn it about that axis. With +Y the sky lay on its side (street HDRIs, 2026-10-06).
    z = up / np.linalg.norm(up)
    x = np.cross(z, side)
    x = math.cos(a) * x + math.sin(a) * np.cross(z, x)
    rot = np.eye(4)
    rot[:3, 0], rot[:3, 1], rot[:3, 2] = x, np.cross(z, x), z
    dome_op.Set(Gf.Matrix4d(rot.T.tolist()))



def place_lights(view):
    """Indoors: the panel on the ceiling facing down (a rect light shines along its -Z), the view's lamp."""
    if lamp_op is None:
        return
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


def expose(img):
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
    return (np.clip(lin * gain, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


photo = np.asarray(Image.open(os.path.join(args.dir, "photo.png")).convert("RGB"))
tiles = [("photo", photo)]
for k, view in enumerate(scene["views"]):
    randomize_light()
    place_lights(view)
    cam_op.Set(Gf.Matrix4d((np.asarray(view["pose"]) @ CV2GL).T.tolist()))
    shown = {i for i, _ in view["objects"]}
    for i, (xf, _) in enumerate(objs):
        (xf.MakeVisible if i in shown else xf.MakeInvisible)()
    for i, t in view["objects"]:
        objs[i][1].Set(Gf.Matrix4d(np.asarray(t).T.tolist()))
    rep.orchestrator.step(rt_subframes=16)
    out = expose(rgb.get_data()[..., :3].astype(np.float32))
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
