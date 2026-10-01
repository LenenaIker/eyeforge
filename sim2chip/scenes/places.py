"""Generic scene plus procedural places (scene.contexts): a building front with balconies and sills seen from
the street, a balcony seen from inside, and a room. Layouts come from places_layout.py and change every few
frames; each material is one mesh rewritten in place, so a new building costs a few attribute writes.
Frames of the "ground" context are the plain generic scene. Import only inside a running Isaac Sim app.
"""
from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

from ..spec import CONTEXTS
from . import places_layout as pl
from .common import set_visible
from .generic import GenericScene

MATERIALS = ("wall", "glass", "slab", "railing", "sill", "fabric", "floor", "furniture")
# Plain colors (linear RGB) when there is no texture, and for the parts that never have one.
WALL_COLORS = [(0.75, 0.68, 0.55), (0.85, 0.83, 0.78), (0.55, 0.28, 0.18), (0.6, 0.6, 0.58), (0.8, 0.6, 0.45)]
RAILING_COLORS = [(0.05, 0.05, 0.05), (0.8, 0.8, 0.8), (0.25, 0.3, 0.35), (0.4, 0.25, 0.15)]
FABRIC_COLORS = [(0.7, 0.55, 0.3), (0.3, 0.45, 0.25), (0.6, 0.2, 0.15), (0.75, 0.72, 0.6), (0.2, 0.3, 0.5)]
FURNITURE_COLORS = [(0.35, 0.2, 0.1), (0.8, 0.78, 0.72), (0.15, 0.15, 0.15), (0.55, 0.4, 0.25)]


class PlacesScene(GenericScene):
    def __init__(self, stage, props, rng, skies: list[str], distractors: list[str], ground: list[str] = (),
                 walls: list[str] = (), contexts=(), railings: list[str] = (), ground_tile: float | None = None):
        super().__init__(stage, props, rng, skies, distractors, ground, ground_tile)
        self.walls, self.railings = list(walls), list(railings)
        self.contexts = {c.kind: c for c in contexts}
        self.parts: dict[str, dict[str, tuple]] = {}
        self.building = self.room = None

    def build(self) -> None:
        super().build()
        for group in ("building", "room"):
            self.parts[group] = {}
            for m in MATERIALS:
                path = f"/World/Places/{group}/{m}"
                mesh = UsdGeom.Mesh.Define(self.stage, path)
                mesh.CreateDoubleSidedAttr(True)
                textured = (m == "wall" and self.walls) or (m == "floor" and self.ground)
                inputs = _material(self.stage, f"{path}Mat", mesh, textured)
                self.parts[group][m] = (mesh, inputs)
        # Balconies and rooms get little sky light (the floor above, the walls): a soft light near the camera
        # stands in for a room lamp or daylight bouncing around.
        self.fill = UsdLux.SphereLight.Define(self.stage, "/World/Places/Fill")
        self.fill.CreateRadiusAttr(0.5)
        UsdGeom.Xformable(self.fill).AddTranslateOp()
        self.new_layout()
        self.show(None)

    def new_layout(self) -> None:
        """A new building and a new room, with new colors and textures."""
        rng = self.rng
        self.building, self.room = pl.building(rng, self.railings or None), pl.room(rng)
        for group, lay in (("building", self.building), ("room", self.room)):
            for m, (mesh, inputs) in self.parts[group].items():
                _write(mesh, lay.boxes.get(m, []))
                if "file" in inputs:
                    inputs["file"].Set(rng.choice(self.walls if m == "wall" else self.ground))
                    k = rng.uniform(0.7, 1.15)
                    inputs["scale"].Set(Gf.Vec4f(k, k, k, 1))
                else:
                    inputs["color"].Set(Gf.Vec3f(*_color(m, rng)))
                    inputs["roughness"].Set(0.08 if m == "glass" else 0.4 if m == "railing" else 0.9)
                    inputs["metallic"].Set(0.5 if m in ("glass", "railing") else 0.0)

    def show(self, kind: str | None) -> None:
        if kind is None or kind == "ground":
            self.fill.CreateIntensityAttr().Set(0.0)
        for group, on in (("building", kind in ("facade", "balcony")), ("room", kind == "indoor")):
            for mesh, _ in self.parts[group].values():
                set_visible(mesh.GetPrim(), on)

    def pick_context(self) -> str:
        kinds = list(self.contexts)
        return self.rng.choices(kinds, weights=[self.contexts[k].share for k in kinds])[0]

    def shot(self, kind: str, items: list[tuple[float, float]], hfov: float, aspect: float):
        """Camera and object spots for one frame of a place context. items: (footprint radius, height).
        Returns (positions or None per item, eye, look)."""
        c, rng = self.contexts[kind], self.rng
        supports = list(c.supports) or list(CONTEXTS[kind])
        shot = None
        if kind == "balcony":
            shot = pl.balcony_shot(self.building, supports, rng, c.eye_height)
        elif kind == "indoor":
            shot = pl.room_shot(self.room, supports, rng, c.eye_height)
        if shot is None:  # facade, or a building without balconies
            shot = pl.facade_shot(self.building, supports, c.distance_m or (6.0, 25.0), hfov, aspect, rng,
                                  c.eye_height)
        eye, look, surfaces = shot
        lit = kind in ("balcony", "indoor") or (kind == "facade" and rng.random() < 0.2)
        UsdGeom.Xformable(self.fill).GetOrderedXformOps()[0].Set(Gf.Vec3d(eye[0], eye[1], eye[2] + 0.8))
        self.fill.CreateIntensityAttr().Set(rng.uniform(3e3, 2.5e4) if lit else 0.0)
        return pl.place(surfaces, items, rng), eye, look


def _color(m: str, rng):
    palette = {"wall": WALL_COLORS, "railing": RAILING_COLORS, "fabric": FABRIC_COLORS,
               "furniture": FURNITURE_COLORS}.get(m)
    if palette:
        return [min(1.0, c * rng.uniform(0.8, 1.2)) for c in rng.choice(palette)]
    if m == "glass":
        return (0.05, 0.07, 0.09)
    k = rng.uniform(0.55, 0.85)  # slab, sill, floor: stone and concrete greys
    return (k, k * 0.98, k * 0.94)


def _material(stage, path: str, mesh, textured: bool) -> dict:
    mat = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    out = {"roughness": surface.CreateInput("roughness", Sdf.ValueTypeNames.Float),
           "metallic": surface.CreateInput("metallic", Sdf.ValueTypeNames.Float)}
    out["roughness"].Set(0.9)
    if textured:
        reader = UsdShade.Shader.Define(stage, f"{path}/St")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        out["file"] = tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        out["scale"] = tex.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    else:
        out["color"] = surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return out


def _write(mesh, boxes: list) -> None:
    pts, counts, idx, uv = pl.mesh(boxes)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(idx))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in uv]))
    if pts:  # a stale extent makes the renderer cull the new geometry
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        mesh.CreateExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
