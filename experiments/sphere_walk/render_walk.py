"""Walk test W.1, render, with Isaac's python.sh: the proxy mesh emissive with the erased panorama as its texture,
no lights, from every view of <dir>/walk/views.json. HdrColor is linear light without tone mapping, so the texture
comes back as it went in (gain 0.997 against the panorama's cut, 2026-10-18).

    python.sh experiments/sphere_walk/render_walk.py --dir data/sphere_test/<place>

Writes <dir>/walk/raw/lin_NAME.npy (linear RGB, float16) and depth_NAME.npy (distance to the image plane, 0 where
no mesh was hit, float16)."""
import argparse
import json
import math
import os
import time

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--scene", help="a folder with scene.npz (<layer>_pts, _faces, _uv) and layers.json "
                "([{name, tex}]); default arm A's proxy in <dir>/walk")
ap.add_argument("--raw", help="output folder (default <dir>/walk/raw)")
ap.add_argument("--only", default="", help="comma-separated view names (default all)")
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--save", help="write the stage to this .usd for a look around in Isaac's GUI (the scene's sky.png as "
                "a dome, a camera at the capture point) and render no views")
args = ap.parse_args()
app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
SENSOR_WIDTH_MM = 20.955
DOME_INTENSITY = 1140.0  # a white dome of intensity 1 comes back as 0.00088 in HdrColor (2026-10-19): this matches the emissive scene


CHUNK = 50_000  # faces per USD mesh: one mesh of WorldGen's ~2M faces rendered near black (2026-10-18)


def material(stage, name, tex):
    """An emissive material showing `tex` through the "st" UVs, nothing else (no lights in the stage)."""
    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    rd = UsdShade.Shader.Define(stage, path + "/St")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, path + "/Tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(tex))
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def vertex_material(stage, name):
    """An emissive material showing the mesh's displayColor (linear RGB per vertex)."""
    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    rd = UsdShade.Shader.Define(stage, path + "/Col")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def vertex_mesh(stage, data, name):
    """A vertex-coloured layer (a fused mesh, its own winding kept: it was seen from many points)."""
    pts, faces, col = data[f"{name}_pts"], data[f"{name}_faces"], data[f"{name}_col"]
    if os.environ.get("WALK_FLIP"):
        faces = faces[:, [0, 2, 1]]
    mat = vertex_material(stage, name)
    for k in range(0, len(faces), CHUNK):
        f = faces[k:k + CHUNK]
        used, local = np.unique(f, return_inverse=True)
        m = UsdGeom.Mesh.Define(stage, f"/World/{name}/c{k // CHUNK}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts[used])))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(local.reshape(-1).astype(np.int32)))
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(col[used])))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def mesh(stage, data, name, tex, height, own_winding=False):
    """One layer: its mesh in chunks of CHUNK faces, emissive with its texture through face-varying UVs.
    own_winding keeps the faces as given (a fused mesh, already facing free space) instead of turning them to the
    capture point: a fused face turned that way showed its dark back to a camera 2 m higher (2026-10-19)."""
    # One winding, facing the capture point (the camera height above the origin): RTX emits only from front faces,
    # and two windings z-fought into stripes (2026-10-18). A moved camera that sees a face from behind sees it dark.
    pts = data[f"{name}_pts"]
    faces = data[f"{name}_faces"]
    uv = data[f"{name}_uv"].reshape(-1, 3, 2)
    p = pts.astype(np.float64)
    n = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
    away = ((p[faces].mean(1) - np.array([0.0, 0.0, height])) * n).sum(1) > 0  # normal pointing away from it
    away &= not own_winding
    faces = np.where(away[:, None], faces[:, [0, 2, 1]], faces)
    uv = np.where(away[:, None, None], uv[:, [0, 2, 1]], uv)
    mat = material(stage, name, tex)
    for k in range(0, len(faces), CHUNK):
        f, u = faces[k:k + CHUNK], uv[k:k + CHUNK].reshape(-1, 2)
        used, local = np.unique(f, return_inverse=True)
        m = UsdGeom.Mesh.Define(stage, f"/World/{name}/c{k // CHUNK}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts[used])))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(local.reshape(-1).astype(np.int32)))
        m.CreateDoubleSidedAttr(True)
        m.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                             UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(u)))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def main():
    src = os.path.join(args.dir, "walk")
    out = args.raw or os.path.join(src, "raw")
    os.makedirs(out, exist_ok=True)
    plan = json.load(open(os.path.join(src, "views.json")))
    data = np.load(os.path.join(src, "walk.npz"))
    s = carb.settings.get_settings()
    s.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
    s.set("/omni/replicator/asyncRendering", False)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    if args.scene:
        data = np.load(os.path.join(args.scene, "scene.npz"))
        layers = [(l["name"], os.path.join(args.scene, l["tex"]) if "tex" in l else None, l.get("winding") == "own")
                  for l in json.load(open(os.path.join(args.scene, "layers.json")))]
    else:
        layers = [("proxy", os.path.join(src, "tex_proxy.png"), False)]
    for name, tex, own in layers:
        if tex is None:
            vertex_mesh(stage, data, name)
        else:
            mesh(stage, data, name, tex, plan["camera_height_m"], own)

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000))
    op = cam.AddTransformOp()
    if args.save:
        sky = os.path.join(args.scene or src, "sky.png")
        if os.path.exists(sky):  # far geometry does not render: the sky comes from a dome
            # Isaac's latlong dome (Z up) shows the texture's longitude L towards our heading L - 90 (measured with
            # a coloured texture, 2026-10-19): the panorama goes in rolled a quarter turn.
            from PIL import Image
            pano = np.asarray(Image.open(sky).convert("RGB"))
            dome_tex = os.path.join(os.path.dirname(os.path.abspath(args.save)), "sky_dome.png")
            Image.fromarray(np.roll(pano, pano.shape[1] // 4, axis=1)).save(dome_tex)
            dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
            dome.CreateTextureFileAttr(dome_tex)
            dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
            dome.CreateIntensityAttr(DOME_INTENSITY)
        v = next(v for v in plan["views"] if v["move"] == "0" and v["heading"] == 0)
        op.Set(Gf.Matrix4d((np.asarray(v["pose"]) @ CV2GL).T.tolist()))
        stage.GetRootLayer().Export(args.save)
        print("saved", args.save, flush=True)
        return
    rep.orchestrator.set_capture_on_play(False)
    w, h = plan["views"][0]["size"]
    rp = rep.create.render_product(str(cam.GetPath()), (w, h))
    hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
    dist = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    hdr.attach([rp])
    dist.attach([rp])
    for _ in range(args.warmup):
        app.update()
    t0 = time.time()
    views = [v for v in plan["views"] if not args.only or v["name"] in args.only.split(",")]
    for v in views:
        cam.CreateHorizontalApertureAttr(SENSOR_WIDTH_MM)
        cam.CreateVerticalApertureAttr(SENSOR_WIDTH_MM * h / w)
        cam.CreateFocalLengthAttr(SENSOR_WIDTH_MM / 2 / math.tan(math.radians(v["fov_deg"]) / 2))
        op.Set(Gf.Matrix4d((np.asarray(v["pose"]) @ CV2GL).T.tolist()))
        rep.orchestrator.step(rt_subframes=8)
        lin = np.asarray(hdr.get_data())[..., :3].astype(np.float16)
        dep = np.nan_to_num(np.asarray(dist.get_data()), posinf=0.0).astype(np.float16)
        np.save(os.path.join(out, f"lin_{v['name']}.npy"), lin)
        np.save(os.path.join(out, f"depth_{v['name']}.npy"), dep)
    print(f"{len(views)} views in {time.time() - t0:.0f} s", flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    app.close()
    os._exit(1)
app.close()
