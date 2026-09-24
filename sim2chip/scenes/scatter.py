"""Background props scattered around the target. Pure Python: it only calls props.add and props.set_pose
(scenes.common.Props inside Isaac Sim), so the placement rules can be tested anywhere."""
from __future__ import annotations

import math


class PropScatter:
    """Background props from the spec, several instances each, placed around the target every frame.

    Most instances land inside the camera's view (behind, beside and sometimes in front of the target, so
    it is partly hidden now and then); the rest anywhere around. Instances never overlap each other, the
    target or the camera. Density changes per frame, from an empty place to a cluttered one.

    How much of the target the props in front hide is estimated from their angular size seen from the camera
    (horizontal overlap times relative height, ignoring the camera's height, so it errs on the high side):
    spots that would hide more than MAX_HIDDEN in total are refused, so the label always stays fair.
    """
    FRONT = 0.25       # chance of accepting a spot between the camera and the target
    MAX_HIDDEN = 0.4

    def __init__(self, props, hfov_deg: float):
        self.props, self.hfov = props, hfov_deg
        self.kinds: list[dict] = []  # {max, slots: [(prim, (radius, height) at scale 1, scale range)]}

    def add_kind(self, max_per_frame: int, models: list[tuple[str, tuple[float, float], tuple[float, float]]],
                 path: str):
        """models: (usd url, (footprint radius, height) at scale 1, scale range). One prim per model and slot."""
        slots = [(self.props.add(f"{path}/m{m}_{k}", url), size, scales)
                 for m, (url, size, scales) in enumerate(models) for k in range(max_per_frame)]
        self.kinds.append({"max": max_per_frame, "slots": slots})

    @property
    def prims(self) -> list:
        return [s[0] for kind in self.kinds for s in kind["slots"]]

    @staticmethod
    def hidden(eye, target, radius: float, height: float, pos, r: float, h: float) -> float:
        """Estimated fraction of the target hidden by a prop at pos (footprint radius r, height h)."""
        D = math.dist(eye[:2], target[:2])
        d = math.dist(eye[:2], pos[:2])
        if radius <= 0 or height <= 0 or d >= D - radius:
            return 0.0
        a_t = math.atan2(target[1] - eye[1], target[0] - eye[0])
        delta = (math.atan2(pos[1] - eye[1], pos[0] - eye[0]) - a_t + math.pi) % (2 * math.pi) - math.pi
        alpha, beta = math.atan2(radius, D), math.atan2(r, d)
        overlap = max(0.0, min(delta + beta, alpha) - max(delta - beta, -alpha)) / (2 * alpha)
        return overlap * min(1.0, (h / d) / (height / D))

    def place(self, eye, target, target_radius: float, target_height: float, rng) -> tuple[list, float]:
        """Shows some props for this frame. Returns them and the estimated hidden fraction of the target."""
        tx, ty = target[0], target[1]
        dist = math.hypot(eye[0] - tx, eye[1] - ty) or 1.0
        view = math.atan2(ty - eye[1], tx - eye[0])  # camera heading
        taken = [(tx, ty, target_radius), (eye[0], eye[1], 1.0)]
        density = rng.random()
        shown, hidden = [], 0.0
        for kind in self.kinds:
            n = min(len(kind["slots"]), round(density * kind["max"] + rng.uniform(-0.5, 0.5)))
            for prim, (radius, height), scales in rng.sample(kind["slots"], max(0, n)):
                scale = rng.uniform(*scales)
                r, h = radius * scale, height * scale
                for _ in range(15):
                    if rng.random() < 0.75:  # in view: an angle inside the field of view, a range around the target
                        a = view + math.radians(rng.uniform(-0.6, 0.6) * self.hfov)
                        d = rng.uniform(0.3, 2.5) * dist
                        x, y = eye[0] + d * math.cos(a), eye[1] + d * math.sin(a)
                        if d < 4 * r + 1 or (d < dist and rng.random() > self.FRONT):
                            continue  # too close to the lens, or in front of the target too often
                    else:
                        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(target_radius + r, 3 * dist)
                        x, y = tx + d * math.cos(a), ty + d * math.sin(a)
                    if any(math.hypot(x - ox, y - oy) <= r + orad + 0.2 for ox, oy, orad in taken):
                        continue
                    hid = self.hidden(eye, target, target_radius, target_height, (x, y), r, h)
                    if hidden + hid <= self.MAX_HIDDEN:
                        hidden += hid
                        self.props.set_pose(prim, (x, y, 0.0), rng.uniform(0, 360), scale)
                        taken.append((x, y, r))
                        shown.append(prim)
                        break
        return shown, hidden
