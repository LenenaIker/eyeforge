"""USD helpers shared by scene templates. Import only inside a running Isaac Sim app."""
from __future__ import annotations

import math

from pxr import Gf, Sdf, Semantics, UsdGeom, UsdLux, UsdShade

from ..workspace import say

NVIDIA_ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"

# Distractors and ground colors shared by the outdoor templates (road, generic).
TREES = [f"{NVIDIA_ASSETS}/Vegetation/Trees/{n}.usd"
         for n in ["American_Beech", "Black_Oak", "Norway_Spruce", "Red_Maple", "White_Pine"]]
PEOPLE = [f"{NVIDIA_ASSETS}/Isaac/5.1/Isaac/People/Characters/{n}/{n}.usd"
          for n in ["male_adult_construction_01_new", "female_adult_police_01_new", "F_Business_02", "M_Medical_01"]]
VERGES = [(0.18, 0.28, 0.10), (0.30, 0.25, 0.16), (0.35, 0.33, 0.30), (0.22, 0.30, 0.14)]  # grass, dirt, gravel


class Props:
    """Referenced props (targets and distractors) with a base scale each, shown and moved per frame."""

    def __init__(self, stage, app):
        self.stage, self.app = stage, app
        self.base_scale: dict[str, float] = {}

    def add(self, path: str, url: str, label: str | None = None):
        # The reference goes on a child prim: some assets carry their own (double3) xformOps, which
        # would clash with the float ops added here.
        prim = self.stage.DefinePrim(path, "Xform")
        self.stage.DefinePrim(f"{path}/Ref", "Xform").GetReferences().AddReference(url)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp()
        xf.AddRotateZOp()
        xf.AddScaleOp()
        set_visible(prim, False)
        if label:
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set(label)
        return prim

    def set_pose(self, prim, pos, yaw_deg: float = 0.0, scale: float = 1.0) -> None:
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(*pos))
        ops[1].Set(float(yaw_deg))
        s = scale * self.base_scale.get(str(prim.GetPath()), 1.0)
        ops[2].Set(Gf.Vec3f(s, s, s))

    def recenter(self, prims) -> None:
        """Center each prop on the origin and put it on z=0, measuring the composed bounding box.

        Offsets computed on the converted file do not always survive USD composition.
        """
        bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for prim in prims:
            ref = self.stage.GetPrimAtPath(f"{prim.GetPath()}/Ref")
            r = bc.ComputeLocalBound(ref).ComputeAlignedRange()
            if r.IsEmpty():
                continue
            c = (r.GetMin() + r.GetMax()) / 2
            UsdGeom.Xformable(ref).AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -r.GetMin()[2]))

    def fit_height(self, prims, h_range, rng) -> list:
        """Base scale so each prop is h_range meters tall (remote assets come in cm or m).

        Remote assets may still be loading: wait a bit and drop the ones without geometry (with scale 1
        a tree in cm would be kilometers tall).
        """
        ok = []
        for prim in prims:
            ref = self.stage.GetPrimAtPath(f"{prim.GetPath()}/Ref")
            h = 0.0
            for _ in range(60):
                bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                r = bc.ComputeLocalBound(ref).ComputeAlignedRange()
                h = r.GetMax()[2] - r.GetMin()[2] if not r.IsEmpty() else 0.0
                if h > 0:
                    break
                self.app.update()
            if h <= 0:
                say(f"prop dropped, no geometry: {prim.GetPath()}")
                continue
            self.base_scale[str(prim.GetPath())] = rng.uniform(*h_range) / h
            ok.append(prim)
        return ok


def set_visible(prim, visible: bool) -> None:
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


def look_at(cam_prim, eye, target) -> None:
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


def make_quad(stage, path: str, sx: float, sy: float, z: float):
    """Flat colored plane. Returns its color input and prim (with editable Z rotation and scale)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-sx, -sy, z), (sx, -sy, z), (sx, sy, z), (-sx, sy, z)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mat = UsdShade.Material.Define(stage, f"{path}Mat")
    shader = UsdShade.Shader.Define(stage, f"{path}Mat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    xf = UsdGeom.Xformable(mesh)
    xf.AddRotateZOp()
    xf.AddScaleOp()
    return color, mesh.GetPrim()


class Paint:
    """Random solid paint for targets: always for models without color, with the class's `repaint`
    chance for the rest (see ClassSpec).

    Bound with strongerThanDescendants to override the model's own materials. Every part takes the same
    color (a car's wheels and windows too): at the model's input resolution this matters little.
    """
    COLORS = [(0.85, 0.85, 0.85), (0.03, 0.03, 0.03), (0.55, 0.56, 0.58), (0.25, 0.26, 0.28), (0.55, 0.04, 0.03),
              (0.04, 0.12, 0.45), (0.05, 0.22, 0.10), (0.75, 0.55, 0.05), (0.55, 0.47, 0.35), (0.35, 0.08, 0.08)]

    def __init__(self, stage, path: str = "/World/Looks/Paint"):
        self.mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        self.color = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        self.rough = sh.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        self.metal = sh.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        self.mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    def apply(self, prim, rng, force: bool, p: float = 0.0, metallic: tuple[float, float] = (0.0, 0.15)) -> bool:
        api = UsdShade.MaterialBindingAPI.Apply(prim)
        if not force and rng.random() > p:
            api.UnbindDirectBinding()
            return False
        base = rng.choice(self.COLORS)
        self.color.Set(Gf.Vec3f(*(min(1.0, max(0.0, c * rng.uniform(0.8, 1.2))) for c in base)))
        self.rough.Set(rng.uniform(0.2, 0.55))
        self.metal.Set(rng.uniform(*metallic))
        api.Bind(self.mat, UsdShade.Tokens.strongerThanDescendants)
        return True


class OutdoorScene:
    """What the outdoor templates share: a sky map with a sun, trees and bystanders as distractors.
    Template method: render drives the steps below, each template fills its own (patterns: docs/architecture.md).

    A template adds its ground in build() and its layout in randomize(); the interface render uses is
    build, after_load, randomize(change_sky), place_target(prim, scale) and place_distractors(eye)."""
    SUN_SHARE = 1.0  # frames with a sun besides the sky map's own light
    BACKLIT_SHARE = 0.25  # frames shot against the sun (aim_sun); the rest have it behind or beside the camera

    def __init__(self, stage, props, rng, skies: list[str], distractors: list[str], ground: list[str] = (),
                 ground_tile: float | None = None):
        self.stage, self.props, self.rng = stage, props, rng
        self.skies, self.ground = list(skies), list(ground)
        self.ground_tile = ground_tile  # meters one ground texture covers (None: the template's default)
        self.want = set(distractors)
        self.trees, self.people = [], []

    def build_sky_and_distractors(self) -> None:
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

    def randomize_light(self, change_sky: bool) -> None:
        rng = self.rng
        if change_sky:
            self.dome.CreateTextureFileAttr().Set(rng.choice(self.skies))
        self.dome.CreateIntensityAttr().Set(rng.uniform(400, 1500))
        UsdGeom.Xformable(self.dome).GetOrderedXformOps()[0].Set(rng.uniform(0, 360))
        sun = rng.uniform(500, 3000) if self.SUN_SHARE >= 1 or rng.random() < self.SUN_SHARE else 0.0
        self.sun.CreateIntensityAttr().Set(sun)
        UsdGeom.Xformable(self.sun).GetOrderedXformOps()[0].Set(
            Gf.Vec3f(rng.uniform(-70, 70), rng.uniform(-70, 70), 0))

    def aim_sun(self, eye, look) -> None:
        """Turn the sun to the camera once it is placed: behind or beside it in most frames, as people take photos
        (a random sun left a facade against the light in about half the frames, nearly black), against it in
        BACKLIT_SHARE of them."""
        rng = self.rng
        # A distant light shines along its -Z: tilted by X, then turned by Z, it travels towards
        # (-sin az, cos az) on the ground, here the way the camera looks.
        az = math.degrees(math.atan2(-(look[0] - eye[0]), look[1] - eye[1])) + rng.uniform(-60, 60)
        if rng.random() < self.BACKLIT_SHARE:
            az += 180
        UsdGeom.Xformable(self.sun).GetOrderedXformOps()[0].Set(Gf.Vec3f(rng.uniform(25, 70), 0, az))
