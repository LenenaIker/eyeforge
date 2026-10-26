"""Where the route reads and writes, and which interpreter runs each step.

Everything machine-specific comes from the environment, falling back to the repo's .env (sim2chip.config.env):

    SIM2CHIP_TOOLS             folder with the upstream repos and checkpoints (WorldStereo, MoGe, WorldGen, hf_cache)
    SIM2CHIP_SAM3_PYTHON       SAM 3 and the local VLM (transformers)
    SIM2CHIP_SCENE_PYTHON      MoGe-2 and pytorch3d (prep, views, the photo's distances)
    SIM2CHIP_ERASE_PYTHON      the Qwen eraser
    SIM2CHIP_PANOGEN_PYTHON    WorldGen and nunchaku
    SIM2CHIP_MAPANYTHING_PYTHON  MapAnything
    SIM2CHIP_GSPLAT_PYTHON     gsplat (splat fit, conflict, next-best-view, Fixer's pseudo views)
    SIM2CHIP_GRUT_DIR          3DGRUT's checkout with its .venv (the USDZ export)"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim2chip.config import env  # noqa: E402

DATA = ROOT / "data" / "worldstereo_test"
PANOGEN = ROOT / "data" / "panogen_test"
PHOTOS = PANOGEN / "in"            # <photo>.png and <photo>.json (field of view), the route's input
PANOS = PANOGEN / "out"            # <case>/pano_s<seed>_paste.jpg
ERASE = DATA / "scene_in"          # e2e_<photo>/: the erase step's files
FIXER = ROOT / "data" / "fixer"    # Fixer's code (src/) and weights (models/)
HERE = Path(__file__).resolve().parent
SPHERE_WALK = ROOT / "experiments" / "sphere_walk"
MODEL = "worldstereo-memory-dmd"   # WorldStereo's checkpoint: a generated leg is <leg>/<MODEL>_result.mp4

ENVS = {"sam3": "SIM2CHIP_SAM3_PYTHON", "scene": "SIM2CHIP_SCENE_PYTHON", "erase": "SIM2CHIP_ERASE_PYTHON",
        "pano": "SIM2CHIP_PANOGEN_PYTHON", "mapanything": "SIM2CHIP_MAPANYTHING_PYTHON",
        "gsplat": "SIM2CHIP_GSPLAT_PYTHON"}


def required(name: str) -> str:
    value = env(name)
    if not value:
        sys.exit(f"set {name} (environment or .env): see experiments/worldstereo/wsroute/paths.py")
    return value


def tools() -> Path:
    return Path(required("SIM2CHIP_TOOLS"))


def python_for(step_env: str) -> str:
    """The interpreter of one of ENVS, or this one ("here")."""
    return sys.executable if step_env == "here" else required(ENVS[step_env])


def moge_ckpt() -> str:
    return env("MOGE_CKPT") or str(tools() / "moge_ckpt/moge-2-vitl/model.pt")


def add_moge():
    """MoGe's code and its pinned dependencies on sys.path."""
    sys.path[:0] = [str(tools() / "MoGe"), str(tools() / "moge_deps")]


class Case:
    """The folders of one scene, data/worldstereo_test/<case>/."""

    def __init__(self, name: str):
        self.name = name
        self.dir = DATA / name

    @property
    def scene(self) -> Path:
        """WorldStereo's input: pano_bank/ and one folder per leg."""
        return self.dir / "scene"

    @property
    def results(self) -> Path:
        """WorldStereo's output, one <leg>/<MODEL>_result.mp4 per generated leg."""
        return self.dir / "out" / "out" / self.name

    @property
    def splat(self) -> Path:
        return self.dir / "splat"

    @property
    def depth(self) -> Path:
        return self.dir / "depth_ma"

    @property
    def prep_json(self) -> Path:
        return self.dir / "prep.json"

    @property
    def distance(self) -> Path:
        """MoGe-2's distance map of the panorama (cached by walks.prep)."""
        return self.dir / "distance.npz"

    def arm(self, arm: str) -> Path:
        return self.splat / "arms" / arm

    def splat_ply(self, arm: str) -> Path:
        return self.arm(arm) / "splat.ply"

    def conflict(self, arm: str) -> Path:
        return self.dir / "conflict" / arm

    def nbv(self, arm: str) -> Path:
        return self.dir / "nbv" / arm

    def leg_video(self, leg: str) -> Path:
        return self.results / leg / f"{MODEL}_result.mp4"

    def legs(self, skip: tuple = ()) -> list[str]:
        """Every generated leg (<folder>/traj<k>), but those whose folder starts with one of skip."""
        return sorted(f"{p.parts[-3]}/{p.parts[-2]}" for p in self.results.glob(f"*/traj*/{MODEL}_result.mp4")
                      if not p.parts[-3].startswith(tuple(skip)))
