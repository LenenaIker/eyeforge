#!/usr/bin/env python3
"""
glb2usd.py - Convierte GLB/glTF a USD sin Isaac Sim (solo usd-core + trimesh).
==============================================================================

Pensado para pipelines de synthetic data generation (SDG) en Isaac Sim.

QUÉ HACE
--------
  - Materiales PBR -> UsdPreviewSurface (baseColor, metallic/roughness, normal,
    emissive, occlusion, alpha). Texturas en <modelo>_textures/ junto al .usd.
  - Stage Z-up, metros, kilogramos, defaultPrim limpio (listo para referenciar).
    La rotación Y-up -> Z-up va en el hijo /<Modelo>/geometry, así que poner la
    pose en la raíz (Replicator, tu código) no la pisa.
  - Opcional: colisiones UsdPhysics (--collision), RigidBody + masa, y etiqueta
    semántica 'class' (--label) en formato nuevo (UsdSemantics.LabelsAPI,
    Isaac 5.x) y/o legacy (SemanticsAPI, Isaac 4.x).
  - Conversión en paralelo de carpetas enteras (bulk).

POR QUÉ usd-core (OpenUSD de Pixar) Y NO ISAAC / BLENDER
--------------------------------------------------------
  - Isaac Sim usa OpenUSD por dentro: el .usd resultante es el mismo formato.
  - No arranca Kit, ni renderer, ni GPU: cada conversión tarda milisegundos
    (vs. decenas de segundos solo en arrancar Isaac headless, o ~1-2 s por
    archivo con Blender headless).
  - Control total de ejes, unidades, nombres, colisiones y etiquetas.
  - Lo que se pierde frente al Asset Converter de Isaac: GLB con compresión
    Draco/meshopt y animaciones/skinning (solo se exporta malla estática).
    Los materiales MDL/OmniPBR de Isaac se pueden generar con la opción
    OPCIONAL --material omnipbr (ver sección MATERIALES).
  Medido: 5 modelos de ejemplo de Khronos (casco, pato, zorro...) en 0,6 s.

INSTALACIÓN
-----------
    pip install usd-core trimesh pillow numpy

  Si un .usd no abre en tu Isaac, instala una versión de usd-core cercana a la
  de tu Isaac Sim, o exporta en texto con `-e usda` (siempre compatible).

USO BÁSICO
----------
    python -m sim2chip.glb2usd modelo.glb -o salida/
    python -m sim2chip.glb2usd carpeta_glbs/ -o salida/ -j 8 --collision convexHull --label auto

BULK REQUEST (carpeta X -> carpeta Y, todo en paralelo)
-------------------------------------------------------
    python -m sim2chip.glb2usd /datos/modelos_raw -o /datos/modelos_usd -r -j 32 \\
        --skip-existing --label auto --collision convexHull

  - -r              recorre subcarpetas y replica la estructura en la salida.
  - -j N            N procesos en paralelo (multiproceso, no multithread: en
                    Python el GIL impide que varios hilos procesen mallas a la
                    vez; varios procesos sí, uno por núcleo, escala ~lineal).
  - --skip-existing solo convierte lo nuevo. Convierte una vez, reutiliza en
                    todas las ejecuciones de SDG.
  - Errores: se registran en <salida>/glb2usd_errors.log sin parar el lote.

  En paralelo al pipeline principal (no bloquea):

    import subprocess
    conv = subprocess.Popen(["python", "glb2usd.py", SRC, "-o", DST,
                             "-r", "-j", "32", "--skip-existing"])
    # ... el pipeline sigue: arrancar Isaac, preparar escena ...
    conv.wait()   # esperar solo cuando necesites todos los modelos

MATERIALES (--material)  -- OmniPBR ES OPCIONAL Y HAY QUE TESTEARLO
-------------------------------------------------------------------
  --material preview  (por defecto) UsdPreviewSurface, el estándar genérico de
                      USD. Isaac RTX lo renderiza bien. Es la opción probada.
  --material omnipbr  (OPCIONAL) MDL OmniPBR, el formato nativo de Omniverse,
                      el mismo tipo de material que genera el Asset Converter
                      de Isaac. Se genera SIN Isaac: el shader apunta a
                      OmniPBR.mdl y Isaac lo resuelve desde su librería al
                      cargar. NO está validado en Isaac: hay que testearlo.
  --material both     (OPCIONAL) los dos en el mismo material. Isaac RTX usa
                      el MDL; cualquier otro visor USD usa el PreviewSurface.

  Cuándo importa: solo si el pipeline MODIFICA parámetros del material original
  (p.ej. diffuse_tint, reflection_roughness_constant de OmniPBR). Si solo
  renderizas los modelos tal cual o los sustituyes por materiales aleatorios con
  Replicator, 'preview' es suficiente. Con 'preview', los parámetros equivalentes
  se llaman inputs:diffuseColor, inputs:roughness, inputs:metallic...

  Cómo mapea omnipbr desde glTF:
    - baseColorFactor -> diffuse_tint (multiplica la textura).
    - metallicRoughness + occlusion -> textura ORM nueva (R=AO, G=roughness,
      B=metallic) con los factores glTF ya horneados; enable_ORM_texture=1.
    - normal -> normalmap_texture; emissive -> enable_emission + textura/color
      (intensidad: --emissive-intensity, si no, la de OmniPBR por defecto).
    - alpha BLEND/MASK -> enable_opacity + opacity_texture (alpha del
      baseColor) + opacity_threshold para MASK.
  Coste: ~2x más lento que preview (se genera y guarda la textura ORM).

  TEST A HACER ANTES DE USARLO (decide si cambian mucho los resultados y si nos
  importa que cambien):
    1. Convierte los mismos 5-10 modelos representativos con
       --material preview, --material omnipbr y con el Asset Converter de Isaac.
    2. Renderiza los tres en Isaac con la misma escena, cámara y luces
       (idealmente un frame de tu propio pipeline de Replicator).
    3. Compara: color base, brillo/rugosidad, metales, relieve del normal map
       (si el relieve parece invertido, el canal verde va al revés), emisivos
       (intensidad) y transparencias (en RTX puede hacer falta activar la
       opacidad fraccional / cutout en los ajustes de render).
    4. Si omnipbr no aporta diferencias relevantes para el dataset, o las que
       hay no afectan al modelo que vais a entrenar, quedaos con 'preview'.
       Si necesitáis parámetros OmniPBR para randomizar, usad omnipbr/both.

DIMENSIONADO (-j, RAM, disco)
-----------------------------
  - Solo usa CPU; la GPU no aporta nada a la conversión.
  - -j = vCPUs que dejes libres para convertir. Si Isaac corre en la misma
    máquina, resérvale ~8-16 vCPUs (Isaac también usa mucha CPU).
  - RAM: ~4 GB por proceso va sobrado (más si hay muchas texturas 4K). Si ves
    swapping, baja -j.
  - Disco: entrada y salida en NVMe local, no en disco de red ni HDD.
  - Las texturas son lo más lento (se recodifican). Mejora posible: copiar las
    imágenes embebidas del GLB tal cual.
  - Estimación: ~0,01-0,5 s por modelo y núcleo -> 10.000 modelos en ~2-5 min
    con ~20 procesos. Mide antes con ~200 modelos reales y extrapola.

  Local (desarrollo): Ryzen 9 7900X (24 hilos) -> -j 20-24, o -j 12-16 si
  Isaac está renderizando a la vez.

DESPLIEGUE EN NEBIUS (hackathon)
--------------------------------
  IMPORTANTE: Isaac Sim necesita GPUs con RT cores. H100/H200/B200 NO sirven
  para renderizar. En Nebius: L40S (eu-north1) o RTX PRO 6000 (us-central1,
  uk-south2, eu-south1).

  Opción A (recomendada): UNA sola VM con GPU para todo.
    - L40S + AMD EPYC, 1 GPU, hasta 48 vCPU / 288 GiB RAM.
      Isaac ~8-16 vCPU, conversión en segundo plano con -j 28-32.
    - RTX PRO 6000 (1 GPU) trae solo 24 vCPU -> conversión con -j <=12.
    - Sin almacenamiento compartido ni transferencias: menos cosas que fallen
      en directo.

  Opción B: VM solo-CPU para convertir + VM con GPU para Isaac.
    - CPU: AMD EPYC Genoa (cpu-d3), presets 4 GB por vCPU; 32-64 vCPU basta.
    - Compartir archivos: shared filesystem de Nebius montado en ambas VMs
      (virtiofs, lectura/escritura desde varias VMs a la vez):
          sudo mount -t virtiofs <nombre-fs> /mnt/fs
      La VM CPU escribe en /mnt/fs/modelos_usd y la de Isaac lee de ahí.
      Alternativa: rsync entre VMs de la misma región al terminar.
    - Solo compensa con muchísimos modelos (decenas de miles con texturas 4K)
      o si quieres que la GPU no gaste ni un ciclo en convertir.
    - Recuerda copiar cada .usd JUNTO a su carpeta <modelo>_textures/
      (las rutas de textura son relativas).

FORMATOS
--------
  Actualmente solo recoge .glb / .gltf. trimesh también lee OBJ (+mtl, solo
  color/textura difusa), STL (sin materiales; suele venir Z-up y en mm), PLY
  (colores por vértice) y DAE (con `pip install pycollada`), pero habría que
  añadir la extensión y reglas de ejes/unidades por formato para no dejarlos
  tumbados o 1000x más grandes. FBX no lo lee trimesh: pasarlo por Blender
  headless (o convertir a GLB antes).

CHECKLIST ANTES DE LANZAR EL LOTE
---------------------------------
  - Abre 3-4 modelos convertidos en Isaac y compara materiales y escala con
    los que convierte Isaac.
  - Revisa glb2usd_errors.log (Draco/meshopt, archivos corruptos...).
  - Si un modelo viene en cm, usa --scale 0.01.
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
    from pxr import UsdSemantics  # OpenUSD >= 25.x (Isaac Sim 5.x usa LabelsAPI)
except ImportError:
    UsdSemantics = None


# --------------------------------------------------------------------------- utils
def _v3(a):
    return Gf.Vec3f(*(float(x) for x in a))


def _v4(a):
    return Gf.Vec4f(*(float(x) for x in a))


_valid = re.compile(r"[^A-Za-z0-9_]")


def safe_name(name, fallback="prim"):
    """Nombre de prim válido en USD."""
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
    """trimesh guarda los factores a veces en uint8 (0-255) y a veces en float."""
    if c is None:
        return None
    c = np.asarray(c, dtype=np.float64).flatten()[:n]
    if c.max() > 1.0 + 1e-6:
        c = c / 255.0
    return c


# ----------------------------------------------------------------------- textures
class TextureWriter:
    """Escribe cada imagen una sola vez (dedupe por contenido)."""

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
                img.save(path, compress_level=1)  # rápido; el tamaño apenas cambia
        rel = f"{self.rel_prefix}/{fname}"
        self.cache[key] = rel
        return rel


# ---------------------------------------------------------------------- materials
def make_material(stage, looks_path, mat, name, texw, used, mode="preview", emissive_intensity=None):
    """glTF material (trimesh) -> UsdShade.Material.
    mode: 'preview' = UsdPreviewSurface (outputs:surface)
          'omnipbr' = MDL OmniPBR (outputs:mdl:surface)   [OPCIONAL, testear]
          'both'    = los dos en el mismo material; Isaac RTX usa el MDL y
                      cualquier otro visor USD usa el PreviewSurface."""
    mname = unique(safe_name(getattr(mat, "name", None) or name, "Material"), used)
    mpath = looks_path.AppendChild(mname)
    material = UsdShade.Material.Define(stage, mpath)
    if mode in ("preview", "both"):
        _preview_surface(stage, material, mpath, mat, mname, texw)
    if mode in ("omnipbr", "both"):
        _omnipbr(stage, material, mpath, mat, mname, texw, emissive_intensity)
    return material


def _omnipbr(stage, material, mpath, mat, mname, texw, emissive_intensity):
    """glTF PBR -> OmniPBR.mdl (lo resuelve Isaac desde su librería MDL al cargar).

    OPCIONAL / EXPERIMENTAL: generado sin Isaac, así que hay que validarlo en
    Isaac (ver sección MATERIALES del README). Decisiones tomadas:
      - baseColorFactor -> diffuse_tint (multiplica la textura).
      - metallicRoughness (+ occlusion) -> un ORM nuevo (R=AO, G=rough, B=metal),
        con los factores glTF ya horneados en la imagen.
      - alpha (BLEND/MASK) -> opacity_texture = baseColor (canal alpha).
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

    # --- materiales no-PBR (SimpleMaterial: OBJ/MTL...): solo difuso
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

    # --- materiales no-PBR (SimpleMaterial): solo color difuso
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

    # --- occlusion (glTF: canal R)
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
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)  # malla poligonal, sin subdiv
    mesh.CreateDoubleSidedAttr(bool(double_sided))
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)

    # normales (las del glTF si existen; si no, las calcula trimesh)
    n = np.asarray(geom.vertex_normals, dtype=np.float32)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # extent (bounding box) - Isaac/Replicator lo usan para bboxes
    mesh.CreateExtentAttr(Vt.Vec3fArray([_v3(v.min(0)), _v3(v.max(0))]))

    pv = UsdGeom.PrimvarsAPI(mesh)
    vis = geom.visual
    # UVs: trimesh ya las invierte a convención OpenGL (origen abajo-izq), igual que USD
    uv = getattr(vis, "uv", None)
    if uv is not None and len(uv) == len(v):
        st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))

    # colores por vértice
    if getattr(vis, "kind", None) == "vertex":
        vc = np.asarray(vis.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        dc = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                              UsdGeom.Tokens.vertex)
        dc.Set(Vt.Vec3fArray.FromNumpy(vc))

    # transform del nodo glTF (world, relativo a la raíz del archivo)
    m = np.asarray(xform, dtype=np.float64)
    if not np.allclose(m, np.eye(4)):
        UsdGeom.Xformable(mesh).AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))  # USD usa filas
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

    # Nodo intermedio con la corrección de ejes + escala. Así la raíz queda en identidad
    # y Replicator / tu código pueden poner la pose de la referencia sin pisar el giro.
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
            continue  # puntos / líneas: se ignoran

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
    """Etiqueta 'class' para Replicator.
    new    -> UsdSemantics.LabelsAPI (Isaac Sim 5.x)
    legacy -> SemanticsAPI (Isaac Sim 4.x, attrs semantic:Semantics:params:*)
    both   -> las dos (seguro si no sabes qué versión lo va a leer)"""
    if fmt in ("new", "both"):
        if UsdSemantics is None:
            print("  aviso: esta versión de usd-core no trae UsdSemantics; uso legacy",
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
    ap.add_argument("input", help="archivo .glb/.gltf o carpeta")
    ap.add_argument("-o", "--out", required=True, help="carpeta de salida")
    ap.add_argument("-e", "--ext", default="usd", choices=["usd", "usda", "usdc"],
                    help="usd/usdc = binario (rápido), usda = texto (depurar)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="procesos en paralelo (por defecto: todos los cores)")
    ap.add_argument("-r", "--recursive", action="store_true", help="buscar en subcarpetas")
    ap.add_argument("--skip-existing", action="store_true", help="no reconvertir si ya existe")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="factor de escala (p.ej. 0.01 si el GLB viene en cm)")
    ap.add_argument("--keep-y-up", action="store_true", help="no rotar a Z-up")
    ap.add_argument("--root-name", default=None, help="nombre del prim raíz (def: nombre archivo)")
    ap.add_argument("--no-materials", action="store_true")
    ap.add_argument("--material", default="preview", choices=["preview", "omnipbr", "both"],
                    help="preview = UsdPreviewSurface (defecto, probado); omnipbr = MDL OmniPBR "
                         "(OPCIONAL, testear en Isaac); both = ambos en el mismo material")
    ap.add_argument("--emissive-intensity", type=float, default=None,
                    help="solo OmniPBR: emissive_intensity (si no, el valor por defecto de OmniPBR)")
    ap.add_argument("--collision", default=None,
                    choices=["none", "convexHull", "convexDecomposition", "boundingCube",
                             "boundingSphere", "meshSimplification"],
                    help="añade UsdPhysics CollisionAPI con esa aproximación")
    ap.add_argument("--rigid-body", action="store_true", help="RigidBodyAPI en la raíz")
    ap.add_argument("--mass", type=float, default=None, help="masa en kg (con --rigid-body)")
    ap.add_argument("--label", default=None,
                    help="etiqueta semántica 'class'; 'auto' = nombre del archivo")
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
        print("Nada que convertir.")
        return

    t0, ok, errors = time.perf_counter(), 0, []
    workers = max(1, min(opts.jobs or 1, len(jobs)))
    print(f"Convirtiendo {len(jobs)} archivo(s) con {workers} proceso(s)...")

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
            print(f"[{i}/{len(jobs)}] {res['file']}: {res['meshes']} mallas, "
                  f"{res['tris']:,} tris, {res['materials']} mats, {res['secs']:.2f}s")

    print(f"\nListo: {ok} ok, {len(errors)} errores, {time.perf_counter() - t0:.1f}s en total")
    if errors:
        log = out / "glb2usd_errors.log"
        log.write_text("\n\n".join(f"== {n}\n{tb}" for n, tb in errors))
        print(f"Detalle de errores en {log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
