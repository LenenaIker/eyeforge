"""The settings of one scene, data/worldstereo_test/<case>/route.json; the defaults are recipe F3c's (user,
2026-10-25: whole-area targeted legs, then Fixer in three rounds over the whole area)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

from .paths import Case


@dataclass
class RouteConfig:
    case: str
    photo: str = ""                 # the input photo (data/panogen_test/in/<photo>.png); empty: the case starts at pano
    pano_seed: int = 0              # WorldGen's seed; the walks read pano_s<seed>_paste.jpg
    skip_legs: list[str] = field(default_factory=list)  # leg folders left out of the fit (another test's legs)
    radius: float = 10.0            # metres: the student's area around the photo's spot
    spots: list[list[float]] = field(default_factory=list)  # grid spots the sheets show (measure only, never targets)
    # Phase 2: legs planned over every student camera of the area, at what no generated frame backs.
    nbv_legs: int = 8
    nbv_max_unknown: float = 0.3    # share of a leg frame's guide that may be left to invent
    nbv_min_gain: float = 0.0003    # share of the student pixels a leg must add
    nbv_prefix: str = "nba_"        # their folders in the scene
    # Fixer (F3): student cameras from this band of distance to walked, rounds at 1/3, 2/3 and the whole way.
    fixer_rounds: int = 3
    fixer_views: int = 80
    fixer_walked: list[float] = field(default_factory=lambda: [1.0, 99.0])
    fixer_timestep: int = 200
    gpu_hours: float = 1.0          # the WorldStereo job's timeout on the VM

    @property
    def pano(self) -> str:
        return f"pano_s{self.pano_seed}_paste.jpg"

    def fixer_arms(self) -> list[str]:
        """F3a, F3b, ...: one splat per round, the last one is the scene."""
        return [f"F3{chr(ord('a') + k)}" for k in range(self.fixer_rounds)]

    def fixer_rounds_plan(self) -> list[tuple[str, str, int, str]]:
        """(splat it renders, tag, step, views file it adds to) per round: F3a renders G2 on views.npz, F3b renders
        F3a on views_f3a.npz, ..."""
        out, prev, base = [], "G2", "views.npz"
        for step, tag in enumerate(self.fixer_arms(), 1):
            out.append((prev, tag, step, base))
            prev, base = tag, arm_views(tag)
        return out

    @property
    def final_arm(self) -> str:
        return self.fixer_arms()[-1] if self.fixer_rounds else "G2"

    def save(self):
        path = Case(self.case).dir / "route.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))


def arm_views(arm: str) -> str:
    """The views file an arm is fitted to: G2_base the walks alone, G2 with the planned legs, F3<x> plus round x's
    pseudo views."""
    return {"G2_base": "views_base.npz", "G2": "views.npz"}.get(arm, f"views_{arm.lower()}.npz")


def load(case: str) -> RouteConfig:
    """route.json of the case, or the defaults when there is none."""
    path = Case(case).dir / "route.json"
    if not path.exists():
        return RouteConfig(case=case)
    known = {f.name for f in fields(RouteConfig)}
    return RouteConfig(**{k: v for k, v in json.loads(path.read_text()).items() if k in known})


def spots_arg(text: str) -> list[list[float]]:
    """"x,y;x,y" -> [[x, y], ...]."""
    return [[float(v) for v in t.split(",")] for t in text.split(";")] if text else []
