"""Road scene: a straight road with verges, sky and sun, trees and bystanders as distractors.

Targets drive along their lane in either direction. Import only inside a running Isaac Sim app.
"""
from __future__ import annotations

import math

from pxr import Gf, UsdGeom, UsdLux

from .common import PEOPLE, TREES, VERGES, make_quad



class RoadScene:
    def __init__(self, stage, props, rng, skies: list[str], distractors: list[str], ground: list[str] = ()):
        # ground textures are not used: the road template paints its verges with plain colors
        self.stage, self.props, self.rng, self.skies = stage, props, rng, skies
        self.want = set(distractors)
        self.trees, self.people = [], []
        self.yaw, self.half_w = 0.0, 3.0

    def build(self) -> None:
        self.ground_color, _ = make_quad(self.stage, "/World/Ground", 400, 400, 0.0)
        self.road_color, self.road = make_quad(self.stage, "/World/Road", 400, 1, 0.02)
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
        k = rng.uniform(0.6, 1.4)
        self.ground_color.Set(Gf.Vec3f(*(c * k for c in rng.choice(VERGES))))
        g = rng.uniform(0.04, 0.18)
        self.road_color.Set(Gf.Vec3f(g, g, g * rng.uniform(0.95, 1.1)))
        self.yaw = rng.uniform(0, 180)
        self.half_w = rng.uniform(2.5, 4.5)
        ops = UsdGeom.Xformable(self.road).GetOrderedXformOps()
        ops[0].Set(self.yaw)
        ops[1].Set(Gf.Vec3f(1, self.half_w, 1))
        if change_sky:
            self.dome.CreateTextureFileAttr().Set(rng.choice(self.skies))
        self.dome.CreateIntensityAttr().Set(rng.uniform(400, 1500))
        UsdGeom.Xformable(self.dome).GetOrderedXformOps()[0].Set(rng.uniform(0, 360))
        self.sun.CreateIntensityAttr().Set(rng.uniform(0, 3000))
        UsdGeom.Xformable(self.sun).GetOrderedXformOps()[0].Set(
            Gf.Vec3f(rng.uniform(-70, 70), rng.uniform(-70, 70), 0))

    def _on_road(self, t: float, lat: float):
        """Point t meters along the road and lat meters from its axis."""
        ry = math.radians(self.yaw)
        return (t * math.cos(ry) - lat * math.sin(ry), t * math.sin(ry) + lat * math.cos(ry), 0.0)

    def _lateral(self, pos) -> float:
        ry = math.radians(self.yaw)
        return abs(-pos[0] * math.sin(ry) + pos[1] * math.cos(ry))

    def place_target(self, prim, scale: float):
        """Put the target in its lane, facing either direction. Returns its position."""
        lane = self.rng.choice([-1, 1]) * self.half_w * self.rng.uniform(0.2, 0.6)
        heading = self.yaw + (0 if lane < 0 else 180) + self.rng.uniform(-10, 10)
        pos = self._on_road(0, lane)
        self.props.set_pose(prim, pos, heading, scale)
        return pos

    def place_distractors(self, eye) -> list:
        """Trees on the verges and bystanders on the shoulder, away from the camera and the target."""
        shown = []

        def clear(pos, margin: float) -> bool:
            return (math.dist(pos[:2], eye[:2]) > margin and math.hypot(*pos[:2]) > margin
                    and self._lateral(pos) > self.half_w + 1)

        rng = self.rng
        for pr in rng.sample(self.trees, rng.randint(0, min(3, len(self.trees)))):
            for _ in range(20):
                pos = self._on_road(rng.uniform(-40, 40), rng.choice([-1, 1]) * rng.uniform(self.half_w + 3, 30))
                if clear(pos, 7):
                    self.props.set_pose(pr, pos, rng.uniform(0, 360), rng.uniform(0.8, 1.2))
                    shown.append(pr)
                    break
        for pr in rng.sample(self.people, rng.randint(0, min(3, len(self.people)))):
            for _ in range(20):
                pos = self._on_road(rng.uniform(-15, 15), rng.choice([-1, 1]) * (self.half_w + rng.uniform(0.5, 5)))
                if clear(pos, 2):
                    self.props.set_pose(pr, pos, rng.uniform(0, 360))
                    shown.append(pr)
                    break
        return shown
