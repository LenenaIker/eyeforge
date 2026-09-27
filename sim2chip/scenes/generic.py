"""Generic scene built from data: a textured ground plane, a sky map and optional distractors.

Nothing in it is specific to one kind of place: the spec's ground_tags and sky_tags choose Poly Haven
textures and HDRIs (a forest floor and a forest sky, a warehouse floor and an indoor HDRI...), and the
sky map also gives the background and most of the light. Targets stand on the ground facing any
direction. Import only inside a running Isaac Sim app.
"""
from __future__ import annotations

import math

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .common import PEOPLE, TREES, VERGES, make_quad

HALF = 300.0   # ground half size (m)
TILE = 2.0     # texture repeat (m) at scale 1


def make_textured_ground(stage, path: str, half: float, tile: float):
    """Ground plane with a repeating diffuse texture. Returns (file input, brightness input, prim)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    reps = 2 * half / tile
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 UsdGeom.Tokens.varying)
    st.Set([(0, 0), (reps, 0), (reps, reps), (0, reps)])

    mat = UsdShade.Material.Define(stage, f"{path}Mat")
    surface = UsdShade.Shader.Define(stage, f"{path}Mat/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    reader = UsdShade.Shader.Define(stage, f"{path}Mat/St")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}Mat/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    file_in = tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    scale_in = tex.CreateInput("scale", Sdf.ValueTypeNames.Float4)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    xf = UsdGeom.Xformable(mesh)
    xf.AddRotateZOp()
    xf.AddScaleOp()
    return file_in, scale_in, mesh.GetPrim()


class GenericScene:
    def __init__(self, stage, props, rng, skies: list[str], distractors: list[str], ground: list[str] = ()):
        self.stage, self.props, self.rng = stage, props, rng
        self.skies, self.ground = list(skies), list(ground)
        self.want = set(distractors)
        self.trees, self.people = [], []

    def build(self) -> None:
        if self.ground:
            self.ground_file, self.ground_scale, self.ground_prim = make_textured_ground(
                self.stage, "/World/Ground", HALF, TILE)
        else:
            self.ground_color, self.ground_prim = make_quad(self.stage, "/World/Ground", HALF, HALF, 0.0)
        self.dome = UsdLux.DomeLight.Define(self.stage, "/World/Sky")
        UsdGeom.Xformable(self.dome).AddRotateZOp()
        self.sun = UsdLux.DistantLight.Define(self.stage, "/World/Sun")
        UsdGeom.Xformable(self.sun).AddRotateXYZOp()
        if "tree" in self.want:
            self.trees = [self.props.add(f"/World/Distractors/tree{i}", u) for i, u in enumerate(TREES)]
        if "person" in self.want:
            self.people = [self.props.add(f"/World/Distractors/person{i}", u) for i, u in enumerate(PEOPLE)]

    def after_load(self) -> None:
        self.trees = self.props.fit_height(self.trees, (6, 16), self.rng)
        self.people = self.props.fit_height(self.people, (1.6, 1.9), self.rng)

    def randomize(self, change_sky: bool) -> None:
        rng = self.rng
        ops = UsdGeom.Xformable(self.ground_prim).GetOrderedXformOps()
        ops[0].Set(rng.uniform(0, 360))
        s = rng.uniform(0.6, 1.5)  # texture scale: tiles of 1.2 to 3 m
        ops[1].Set(Gf.Vec3f(s, s, 1))
        if self.ground:
            k = rng.uniform(0.7, 1.2)
            self.ground_scale.Set(Gf.Vec4f(k, k, k, 1))
            if change_sky:  # swapping textures every frame would stall on texture loads
                self.ground_file.Set(rng.choice(self.ground))
        else:
            k = rng.uniform(0.6, 1.4)
            self.ground_color.Set(Gf.Vec3f(*(c * k for c in rng.choice(VERGES))))
        if change_sky:
            self.dome.CreateTextureFileAttr().Set(rng.choice(self.skies))
        self.dome.CreateIntensityAttr().Set(rng.uniform(400, 1500))
        UsdGeom.Xformable(self.dome).GetOrderedXformOps()[0].Set(rng.uniform(0, 360))
        # The sky map already lights the scene; an extra sun only sometimes (never strong indoors-looking).
        self.sun.CreateIntensityAttr().Set(rng.uniform(0, 3000) if rng.random() < 0.6 else 0.0)
        UsdGeom.Xformable(self.sun).GetOrderedXformOps()[0].Set(
            Gf.Vec3f(rng.uniform(-70, 70), rng.uniform(-70, 70), 0))

    def place_target(self, prim, scale: float):
        """On the ground at the origin, facing any direction. Returns its position."""
        pos = (0.0, 0.0, 0.0)
        self.props.set_pose(prim, pos, self.rng.uniform(0, 360), scale)
        return pos

    def place_distractors(self, eye) -> list:
        """Trees around and people nearby, never between the camera and the target."""
        shown, rng = [], self.rng

        def clear(pos, margin: float) -> bool:
            if math.dist(pos[:2], eye[:2]) < margin or math.hypot(*pos[:2]) < margin:
                return False
            # distance from pos to the camera-target segment (the target is at the origin)
            ex, ey = eye[0], eye[1]
            t = max(0.0, min(1.0, (pos[0] * ex + pos[1] * ey) / (ex * ex + ey * ey or 1)))
            return math.dist(pos[:2], (t * ex, t * ey)) > margin

        for group, n, radius, margin, scale in ((self.trees, 3, (8, 40), 6, (0.8, 1.2)),
                                                (self.people, 2, (2, 12), 1.5, (1, 1))):
            for pr in rng.sample(group, rng.randint(0, min(n, len(group)))):
                for _ in range(20):
                    r, a = rng.uniform(*radius), rng.uniform(0, 2 * math.pi)
                    pos = (r * math.cos(a), r * math.sin(a), 0.0)
                    if clear(pos, margin):
                        self.props.set_pose(pr, pos, rng.uniform(0, 360), rng.uniform(*scale))
                        shown.append(pr)
                        break
        return shown
