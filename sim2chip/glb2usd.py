#!/usr/bin/env python3
"""GLB/glTF to USD without Isaac Sim (usd-core + trimesh only), for synthetic data in Isaac Sim.

sim2chip's convert_assets calls convert() on every downloaded model; Isaac's Asset Converter is only the
fallback for what this cannot read. Also a standalone bulk converter:

    python -m sim2chip.glb2usd models/ -o usd/ -r -j 8

What it does:
  - PBR materials to UsdPreviewSurface (base color, metallic/roughness, normal, emissive, occlusion, alpha);
    textures in <model>_textures/ next to the .usd, each image written once. --material omnipbr (or both)
    also writes an MDL OmniPBR material, not validated in Isaac yet.
  - Z-up stage in meters and kilograms with a clean defaultPrim, ready to reference. The Y-up to Z-up
    rotation and the scale sit on the child /<Model>/geometry, so a pose set on the root does not undo them.
  - Optional UsdPhysics collisions (--collision), rigid body and mass, and a semantic 'class' label (--label)
    in the new format (UsdSemantics.LabelsAPI, Isaac 5.x) and/or the legacy one (SemanticsAPI, Isaac 4.x).
  - Whole folders in parallel, one process per core.

Why not Isaac or Blender: Isaac Sim uses OpenUSD inside, so the file is the same format, but this starts no
Kit, renderer or GPU: a model takes milliseconds (41 models in 5 s on 8 cores, against minutes through
Isaac, experiments/NOTES.md). What it loses against Isaac's converter: Draco/meshopt compressed GLBs and
animation or skinning (static meshes only).

Formats: only .glb and .gltf. trimesh also reads OBJ, STL, PLY and DAE, but each would need its own axis and
unit rules (STL often comes Z-up in mm); FBX needs Blender first.
"""
import argparse
import hashlib
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

try:
    from pxr import UsdSemantics  # OpenUSD >= 25.x (Isaac Sim 5.x uses LabelsAPI)
except ImportError:
    UsdSemantics = None


# --------------------------------------------------------------------------- utils
def _v3(a):
    return Gf.Vec3f(*(float(x) for x in a))


def _v4(a):
    return Gf.Vec4f(*(float(x) for x in a))


_valid = re.compile(r"[^A-Za-z0-9_]")


def safe_name(name, fallback="prim"):
    """A valid USD prim name."""
    n = _valid.sub("_", str(name or "")) or fallback
    if n[0].isdigit():
        n = "_" + n
    return n


def unique(name, used):
    base, i = name, 1
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def as_float_color(c, n=4):
    """trimesh stores factors sometimes as uint8 (0-255), sometimes as float."""
    if c is None:
        return None
    c = np.asarray(c, dtype=np.float64).flatten()[:n]
    if c.max() > 1.0 + 1e-6:
        c = c / 255.0
    return c


# ----------------------------------------------------------------------- textures
class TextureWriter:
    """Writes each image once (deduplicated by content)."""

    def __init__(self, tex_dir: Path, rel_prefix: str):
        self.tex_dir = tex_dir
        self.rel_prefix = rel_prefix
        self.cache = {}

    def write(self, img: Image.Image, hint: str) -> str:
        key = id(img)
        if key in self.cache:
            return self.cache[key]
        digest = hashlib.md5(img.tobytes()).hexdigest()[:10]
        is_jpeg = (getattr(img, "format", None) == "JPEG") and img.mode in ("RGB", "L")
        ext = ".jpg" if is_jpeg else ".png"
        fname = f"{safe_name(hint)}_{digest}{ext}"
        path = self.tex_dir / fname
        if not path.exists():
            self.tex_dir.mkdir(parents=True, exist_ok=True)
            if is_jpeg:
                img.save(path, quality=95)
            else:
                img.save(path, compress_level=1)  # fast; the size barely changes
        rel = f"{self.rel_prefix}/{fname}"
        self.cache[key] = rel
        return rel


# ---------------------------------------------------------------------- materials
def make_material(stage, looks_path, mat, name, texw, used, mode="preview", emissive_intensity=None):
    """glTF material (trimesh) -> UsdShade.Material.
    mode: 'preview' = UsdPreviewSurface (outputs:surface)
          'omnipbr' = MDL OmniPBR (outputs:mdl:surface), not validated in Isaac yet
          'both'    = both in the same material; Isaac RTX uses the MDL and any
                      other USD viewer the PreviewSurface."""
    mname = unique(safe_name(getattr(mat, "name", None) or name, "Material"), used)
    mpath = looks_path.AppendChild(mname)
    material = UsdShade.Material.Define(stage, mpath)
    if mode in ("preview", "both"):
        _preview_surface(stage, material, mpath, mat, mname, texw)
    if mode in ("omnipbr", "both"):
        _omnipbr(stage, material, mpath, mat, mname, texw, emissive_intensity)
    return material


def _omnipbr(stage, material, mpath, mat, mname, texw, emissive_intensity):
    """glTF PBR -> OmniPBR.mdl (Isaac resolves it from its MDL library when loading).

    Experimental: written without Isaac, so it still has to be checked in Isaac. Choices:
      - baseColorFactor -> diffuse_tint (multiplies the texture).
      - metallicRoughness (+ occlusion) -> a new ORM image (R=AO, G=rough, B=metal) with the glTF
        factors baked in.
      - alpha (BLEND/MASK) -> opacity_texture = baseColor (alpha channel).
    """
    sh = UsdShade.Shader.Define(stage, mpath.AppendChild("OmniPBR"))
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")

    def const(n, t, v):
        sh.CreateInput(n, t).Set(v)

    def tex(n, img, hint, colorspace):
        a = sh.CreateInput(n, Sdf.ValueTypeNames.Asset)
        a.Set(Sdf.AssetPath(texw.write(img, f"{mname}_{hint}")))
        a.GetAttr().SetColorSpace(colorspace)

    F, C3, B = Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Bool

    # --- non-PBR materials (SimpleMaterial: OBJ/MTL...): diffuse only
    if not isinstance(mat, trimesh.visual.material.PBRMaterial):
        img = getattr(mat, "image", None)
        diffuse = as_float_color(getattr(mat, "diffuse", None))
        if img is not None:
            tex("diffuse_texture", img, "diffuseTex", "sRGB")
        elif diffuse is not None:
            const("diffuse_color_constant", C3, _v3(diffuse[:3]))
        const("reflection_roughness_constant", F, 0.5)
        const("metallic_constant", F, 0.0)
        return

    bcf = as_float_color(mat.baseColorFactor)
    if bcf is None:
        bcf = np.ones(4)
    alpha_mode = (mat.alphaMode or "OPAQUE").upper()

    if mat.baseColorTexture is not None:
        tex("diffuse_texture", mat.baseColorTexture, "baseColorTex", "sRGB")
        if not np.allclose(bcf[:3], 1.0):
            const("diffuse_tint", C3, _v3(bcf[:3]))
    else:
        const("diffuse_color_constant", C3, _v3(bcf[:3]))

    metal = 1.0 if mat.metallicFactor is None else float(mat.metallicFactor)
    rough = 1.0 if mat.roughnessFactor is None else float(mat.roughnessFactor)
    if mat.metallicRoughnessTexture is not None:
        mr = np.asarray(mat.metallicRoughnessTexture.convert("RGB"), dtype=np.float32)
        orm = np.empty_like(mr)
        orm[..., 0] = 255.0
        occ = mat.occlusionTexture
        if occ is not None:
            o = occ.convert("RGB")
            if o.size != mat.metallicRoughnessTexture.size:
                o = o.resize(mat.metallicRoughnessTexture.size)
            orm[..., 0] = np.asarray(o, dtype=np.float32)[..., 0]
        orm[..., 1] = mr[..., 1] * rough
        orm[..., 2] = mr[..., 2] * metal
        orm_img = Image.fromarray(np.clip(orm, 0, 255).astype(np.uint8), "RGB")
        const("enable_ORM_texture", B, True)
        tex("ORM_texture", orm_img, "ORMTex", "raw")
    else:
        const("reflection_roughness_constant", F, rough)
        const("metallic_constant", F, metal)

    if mat.normalTexture is not None:
        tex("normalmap_texture", mat.normalTexture, "normalTex", "raw")

    ef = as_float_color(mat.emissiveFactor, 3)
    has_emissive_tex = mat.emissiveTexture is not None
    if has_emissive_tex or (ef is not None and ef.max() > 0):
        const("enable_emission", B, True)
        if has_emissive_tex:
            tex("emissive_color_texture", mat.emissiveTexture, "emissiveTex", "sRGB")
        if ef is not None:
            const("emissive_color", C3, _v3(ef[:3]))
        if emissive_intensity is not None:
            const("emissive_intensity", F, float(emissive_intensity))

    if alpha_mode != "OPAQUE":
        const("enable_opacity", B, True)
        const("opacity_constant", F, float(bcf[3]))
        if mat.baseColorTexture is not None and mat.baseColorTexture.mode in ("RGBA", "LA", "P"):
            const("enable_opacity_texture", B, True)
            tex("opacity_texture", mat.baseColorTexture, "baseColorTex", "sRGB")
            const("opacity_mode", Sdf.ValueTypeNames.Int, 0)  # 0 = mono_alpha
        if alpha_mode == "MASK":
            cutoff = mat.alphaCutoff if mat.alphaCutoff is not None else 0.5
            const("opacity_threshold", F, float(cutoff))


def _preview_surface(stage, material, mpath, mat, mname, texw):
    """glTF PBRMaterial (trimesh) -> UsdPreviewSurface."""
    shader = UsdShade.Shader.Define(stage, mpath.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    reader = None

    def uv_reader():
        nonlocal reader
        if reader is None:
            reader = UsdShade.Shader.Define(stage, mpath.AppendChild("uvReader"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        return reader

    def texture(img, tname, colorspace, fallback=None):
        t = UsdShade.Shader.Define(stage, mpath.AppendChild(tname))
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texw.write(img, f"{mname}_{tname}"))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_reader().ConnectableAPI(), "result")
        if fallback is not None:
            t.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(_v4(fallback))
        return t

    inp = shader.CreateInput
    F, C3 = Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Color3f

    # --- non-PBR materials (SimpleMaterial): diffuse color only
    if not isinstance(mat, trimesh.visual.material.PBRMaterial):
        img = getattr(mat, "image", None)
        diffuse = as_float_color(getattr(mat, "diffuse", None))
        if img is not None:
            t = texture(img, "diffuseTex", "sRGB")
            t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            inp("diffuseColor", C3).ConnectToSource(t.ConnectableAPI(), "rgb")
        elif diffuse is not None:
            inp("diffuseColor", C3).Set(_v3(diffuse[:3]))
        inp("roughness", F).Set(0.5)
        inp("metallic", F).Set(0.0)
        return material

    # --- base color (+ alpha)
    bcf = as_float_color(mat.baseColorFactor)
    if bcf is None:
        bcf = np.ones(4)
    alpha_mode = (mat.alphaMode or "OPAQUE").upper()

    if mat.baseColorTexture is not None:
        t = texture(mat.baseColorTexture, "baseColorTex", "sRGB")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(_v4(bcf))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        inp("diffuseColor", C3).ConnectToSource(t.ConnectableAPI(), "rgb")
        if alpha_mode != "OPAQUE" and mat.baseColorTexture.mode in ("RGBA", "LA", "P"):
            t.CreateOutput("a", F)
            inp("opacity", F).ConnectToSource(t.ConnectableAPI(), "a")
        elif alpha_mode != "OPAQUE":
            inp("opacity", F).Set(float(bcf[3]))
    else:
        inp("diffuseColor", C3).Set(_v3(bcf[:3]))
        if alpha_mode != "OPAQUE":
            inp("opacity", F).Set(float(bcf[3]))

    if alpha_mode == "MASK":
        cutoff = mat.alphaCutoff if mat.alphaCutoff is not None else 0.5
        inp("opacityThreshold", F).Set(float(cutoff))

    # --- metallic / roughness (glTF: G = roughness, B = metallic)
    metal = 1.0 if mat.metallicFactor is None else float(mat.metallicFactor)
    rough = 1.0 if mat.roughnessFactor is None else float(mat.roughnessFactor)
    if mat.metallicRoughnessTexture is not None:
        t = texture(mat.metallicRoughnessTexture, "metalRoughTex", "raw")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, rough, metal, 1))
        t.CreateOutput("g", F)
        t.CreateOutput("b", F)
        inp("roughness", F).ConnectToSource(t.ConnectableAPI(), "g")
        inp("metallic", F).ConnectToSource(t.ConnectableAPI(), "b")
    else:
        inp("roughness", F).Set(rough)
        inp("metallic", F).Set(metal)

    # --- normal map ([0,1] -> [-1,1])
    if mat.normalTexture is not None:
        t = texture(mat.normalTexture, "normalTex", "raw")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        inp("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")

    # --- emissive
    ef = as_float_color(mat.emissiveFactor, 3)
    if mat.emissiveTexture is not None:
        t = texture(mat.emissiveTexture, "emissiveTex", "sRGB")
        s = ef if ef is not None else np.ones(3)
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(_v4(list(s[:3]) + [1]))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        inp("emissiveColor", C3).ConnectToSource(t.ConnectableAPI(), "rgb")
    elif ef is not None and ef.max() > 0:
        inp("emissiveColor", C3).Set(_v3(ef[:3]))

    # --- occlusion (glTF: R channel)
    if mat.occlusionTexture is not None:
        t = texture(mat.occlusionTexture, "occlusionTex", "raw")
        t.CreateOutput("r", F)
        inp("occlusion", F).ConnectToSource(t.ConnectableAPI(), "r")

    return material


# --------------------------------------------------------------------------- mesh
def write_mesh(stage, path, geom: trimesh.Trimesh, xform, double_sided):
    mesh = UsdGeom.Mesh.Define(stage, path)
    v = np.asarray(geom.vertices, dtype=np.float32)
    f = np.asarray(geom.faces, dtype=np.int32)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)  # polygon mesh, no subdivision
    mesh.CreateDoubleSidedAttr(bool(double_sided))
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)

    # normals (the glTF's if present, else computed by trimesh)
    n = np.asarray(geom.vertex_normals, dtype=np.float32)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # extent (bounding box): Isaac and Replicator use it for bounding boxes
    mesh.CreateExtentAttr(Vt.Vec3fArray([_v3(v.min(0)), _v3(v.max(0))]))

    pv = UsdGeom.PrimvarsAPI(mesh)
    vis = geom.visual
    # UVs: trimesh already flips them to the OpenGL convention (origin bottom left), as USD
    uv = getattr(vis, "uv", None)
    if uv is not None and len(uv) == len(v):
        st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))

    # vertex colors
    if getattr(vis, "kind", None) == "vertex":
        vc = np.asarray(vis.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        dc = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                              UsdGeom.Tokens.vertex)
        dc.Set(Vt.Vec3fArray.FromNumpy(vc))

    # the glTF node's transform (world, relative to the file's root)
    m = np.asarray(xform, dtype=np.float64)
    if not np.allclose(m, np.eye(4)):
        UsdGeom.Xformable(mesh).AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))  # USD uses rows
    return mesh


# ------------------------------------------------------------------------ convert
def convert(src: Path, dst: Path, opts) -> dict:
    t0 = time.perf_counter()
    scene = trimesh.load(str(src), force="scene", process=False, maintain_order=True)

    if dst.exists():
        dst.unlink()
    stage = Usd.Stage.CreateNew(str(dst))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root_name = safe_name(opts.root_name or src.stem, "Root")
    root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{root_name}"))
    stage.SetDefaultPrim(root.GetPrim())
    UsdModelAPI = Usd.ModelAPI(root.GetPrim())
    UsdModelAPI.SetKind("component")

    # Intermediate node with the axis fix and the scale, so the root stays identity and a pose set on the
    # reference (Replicator, the caller) does not undo the rotation.
    geo = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("geometry"))
    gx = UsdGeom.Xformable(geo)
    if not opts.keep_y_up:
        gx.AddRotateXOp().Set(90.0)  # glTF Y-up -> USD Z-up
    if opts.scale != 1.0:
        gx.AddScaleOp().Set(Gf.Vec3f(opts.scale, opts.scale, opts.scale))

    looks = UsdGeom.Scope.Define(stage, root.GetPath().AppendChild("Looks")).GetPath()
    rel_tex = "./" + (dst.stem + "_textures")
    texw = TextureWriter(dst.parent / (dst.stem + "_textures"), rel_tex)

    mat_cache, used_mats, used_meshes = {}, set(), set()
    n_meshes = n_tris = 0

    for node in scene.graph.nodes_geometry:
        xform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue  # points and lines are skipped

        mname = unique(safe_name(node, "mesh"), used_meshes)
        mat = getattr(geom.visual, "material", None)
        double_sided = bool(getattr(mat, "doubleSided", False)) if mat is not None else False
        mesh = write_mesh(stage, geo.GetPath().AppendChild(mname), geom, xform, double_sided)
        n_meshes += 1
        n_tris += len(geom.faces)

        if mat is not None and not opts.no_materials:
            key = id(mat)
            if key not in mat_cache:
                mat_cache[key] = make_material(stage, looks, mat, geom_name, texw, used_mats,
                                              opts.material, opts.emissive_intensity)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat_cache[key])

        if opts.collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mc.CreateApproximationAttr().Set(opts.collision)

    if opts.rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        if opts.mass:
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(float(opts.mass))

    if opts.label:
        label = src.stem if opts.label == "auto" else opts.label
        add_semantic_label(root.GetPrim(), label, opts.semantics_format)

    stage.GetRootLayer().Save()
    return dict(file=src.name, meshes=n_meshes, tris=n_tris,
                materials=len(mat_cache), secs=time.perf_counter() - t0)


def add_semantic_label(prim, label, fmt):
    """'class' label for Replicator.
    new    -> UsdSemantics.LabelsAPI (Isaac Sim 5.x)
    legacy -> SemanticsAPI (Isaac Sim 4.x, attrs semantic:Semantics:params:*)
    both   -> both (safe when unsure which version will read it)"""
    if fmt in ("new", "both"):
        if UsdSemantics is None:
            print("  warning: this usd-core has no UsdSemantics, using the legacy format",
                  file=sys.stderr)
            fmt = "legacy"
        else:
            api = UsdSemantics.LabelsAPI.Apply(prim, "class")
            api.CreateLabelsAttr().Set([label])
    if fmt in ("legacy", "both"):
        prim.AddAppliedSchema("SemanticsAPI:Semantics")
        prim.CreateAttribute("semantic:Semantics:params:semanticType",
                             Sdf.ValueTypeNames.String).Set("class")
        prim.CreateAttribute("semantic:Semantics:params:semanticData",
                             Sdf.ValueTypeNames.String).Set(label)


# --------------------------------------------------------------------------- main
def _job(args):
    src, dst, opts = args
    try:
        return convert(src, dst, opts), None
    except Exception:
        return None, (src.name, traceback.format_exc())


def build_parser():
    ap = argparse.ArgumentParser(description="GLB/glTF -> USD sin Isaac Sim")
    ap.add_argument("input", help=".glb/.gltf file or folder")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("-e", "--ext", default="usd", choices=["usd", "usda", "usdc"],
                    help="usd/usdc = binary (fast), usda = text (debugging)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="parallel processes (default: every core)")
    ap.add_argument("-r", "--recursive", action="store_true", help="look in subfolders")
    ap.add_argument("--skip-existing", action="store_true", help="do not convert again when the output exists")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale factor (e.g. 0.01 when the GLB is in cm)")
    ap.add_argument("--keep-y-up", action="store_true", help="do not rotate to Z-up")
    ap.add_argument("--root-name", default=None, help="root prim name (default: file name)")
    ap.add_argument("--no-materials", action="store_true")
    ap.add_argument("--material", default="preview", choices=["preview", "omnipbr", "both"],
                    help="preview = UsdPreviewSurface (default, tested); omnipbr = MDL OmniPBR "
                         "(not validated in Isaac); both = both in the same material")
    ap.add_argument("--emissive-intensity", type=float, default=None,
                    help="OmniPBR only: emissive_intensity (default: OmniPBR's own)")
    ap.add_argument("--collision", default=None,
                    choices=["none", "convexHull", "convexDecomposition", "boundingCube",
                             "boundingSphere", "meshSimplification"],
                    help="adds UsdPhysics CollisionAPI with this approximation")
    ap.add_argument("--rigid-body", action="store_true", help="RigidBodyAPI on the root")
    ap.add_argument("--mass", type=float, default=None, help="mass in kg (with --rigid-body)")
    ap.add_argument("--label", default=None,
                    help="semantic 'class' label; 'auto' = the file name")
    ap.add_argument("--semantics-format", default="both", choices=["new", "legacy", "both"])
    return ap


def default_options(**overrides):
    """CLI defaults as an options object, for callers that import convert() (sim2chip's convert_assets)."""
    opts = build_parser().parse_args(["-", "-o", "-"])
    for k, v in overrides.items():
        setattr(opts, k, v)
    return opts


def main():
    opts = build_parser().parse_args()

    inp, out = Path(opts.input), Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    if inp.is_dir():
        pat = "**/*" if opts.recursive else "*"
        files = sorted(p for p in inp.glob(pat) if p.suffix.lower() in (".glb", ".gltf"))
    else:
        files = [inp]

    jobs = []
    for f in files:
        rel = f.relative_to(inp).parent if inp.is_dir() else Path()
        (out / rel).mkdir(parents=True, exist_ok=True)
        dst = out / rel / f"{f.stem}.{opts.ext}"
        if opts.skip_existing and dst.exists():
            continue
        jobs.append((f, dst, opts))

    if not jobs:
        print("Nothing to convert.")
        return

    t0, ok, errors = time.perf_counter(), 0, []
    workers = max(1, min(opts.jobs or 1, len(jobs)))
    print(f"Converting {len(jobs)} file(s) with {workers} process(es)...")

    if workers == 1:
        results = map(_job, jobs)
    else:
        pool = ProcessPoolExecutor(workers)
        results = (fut.result() for fut in as_completed([pool.submit(_job, j) for j in jobs]))

    for i, (res, err) in enumerate(results, 1):
        if err:
            errors.append(err)
            print(f"[{i}/{len(jobs)}] ERROR {err[0]}")
        else:
            ok += 1
            print(f"[{i}/{len(jobs)}] {res['file']}: {res['meshes']} meshes, "
                  f"{res['tris']:,} tris, {res['materials']} mats, {res['secs']:.2f}s")

    print(f"\nDone: {ok} ok, {len(errors)} errors, {time.perf_counter() - t0:.1f}s in total")
    if errors:
        log = out / "glb2usd_errors.log"
        log.write_text("\n\n".join(f"== {n}\n{tb}" for n, tb in errors))
        print(f"Error details in {log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
