"""Recipe F3c, step by step: each step names its interpreter, its command and the output that marks it done, so a
run after a failure picks up where it stopped. The two WorldStereo runs on the Nebius VM cost money (~0.3-0.5 $ each):
the driver never starts one unless --until names it or a later step (the user's budget approval)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import erase
from .config import RouteConfig, arm_views
from .paths import ERASE, PANOS, PHOTOS, ROOT, SPHERE_WALK, Case, python_for, required

ENTRY = Path(__file__).resolve().parents[1] / "ws_route.py"
PAID = ("walk_legs", "planned_legs")


@dataclass
class Step:
    name: str
    env: str                    # paths.ENVS key, "here" (this interpreter), "cloud" (here, with .env), "grut"
    args: list[str]             # ws_route.py arguments, or a script and its arguments (grut, erase)
    done: Callable[[], bool]
    script: str = ""            # a script other than ws_route.py


def exists(p: Path) -> Callable[[], bool]:
    return p.exists


def depth_done(case: Case) -> Callable[[], bool]:
    """Every generated leg has MapAnything's depth and a scale."""
    def done():
        f = case.depth / "scales.json"
        if not f.exists():
            return False
        return set(case.legs()) <= set(json.loads(f.read_text()))
    return done


def fixed_done(case: Case, tag: str, timestep: int) -> Callable[[], bool]:
    def done():
        meta, out = case.arm(tag) / "meta.npz", case.arm(tag) / f"t{timestep}"
        return meta.exists() and out.is_dir() and len(list(out.glob("*.png"))) == len(np.load(meta)["files"])
    return done


def recipe(cfg: RouteConfig) -> list[Step]:
    case = Case(cfg.case)
    c = ["--case", cfg.case]
    skip = ["--skip", *cfg.skip_legs] if cfg.skip_legs else []
    steps = []
    if cfg.photo:
        d, p = ERASE / cfg.case, ["--photo", cfg.photo, *c]
        steps += [
            Step("things", "sam3", ["things", *p], exists(d / "things.json")),
            Step("masks", "sam3", ["masks", *p], exists(d / "things.npz")),
            Step("split", "scene", ["split", *p], exists(d / "split.json")),
            Step("erase", "erase", ["--dir", str(d)], exists(d / "erase_qwen.json"),
                 script=str(ROOT / "sim2chip/photo2scene/steps/erase_qwen.py")),
            Step("publish", "here", ["publish", *p], exists(PHOTOS / f"{cfg.case}.png")),
            Step("caption", "sam3", ["caption", *p], exists(PHOTOS / f"{cfg.case}.txt")),
            Step("pano", "pano", ["pano", *c], exists(PANOS / cfg.case / cfg.pano)),
        ]
    steps += [
        Step("walks", "scene", ["walks", *c], exists(case.prep_json)),
        Step("walk_legs", "cloud", ["cloud", *c], exists(case.leg_video("reconstruct_0/traj0"))),
        Step("depth_walks", "mapanything", ["depth", *c], depth_done(case)),
        Step("views_walks", "scene", ["views", *c, "--arm", "G2_base", *skip], exists(case.splat / "views_base.npz")),
        Step("train_walks", "gsplat", ["train", *c, "--arm", "G2_base"], exists(case.splat_ply("G2_base"))),
        Step("audit", "gsplat", ["audit", *c, "--arm", "G2_base"], exists(case.nbv("G2_base") / "audit.json")),
        Step("plan", "gsplat", ["plan", *c, "--arm", "G2_base"], exists(case.nbv("G2_base") / "plan.json")),
        Step("stage", "gsplat", ["stage", *c, "--arm", "G2_base"], exists(case.scene / f"{cfg.nbv_prefix}00")),
        Step("planned_legs", "cloud", ["cloud", *c, "--resume"], exists(case.leg_video(f"{cfg.nbv_prefix}00/traj0"))),
        Step("depth_planned", "mapanything", ["depth", *c], depth_done(case)),
        Step("views", "scene", ["views", *c, "--arm", "G2", *skip], exists(case.splat / "views.npz")),
        Step("train", "gsplat", ["train", *c, "--arm", "G2"], exists(case.splat_ply("G2"))),
        Step("grid", "gsplat", ["grid", *c, "--arm", "G2"], exists(case.conflict("G2") / "grid.json")),
    ]
    for prev, tag, step, base in cfg.fixer_rounds_plan():
        steps += [
            Step(f"pseudo_{tag}", "gsplat", ["pseudo", *c, "--arm", prev, "--tag", tag, "--step", str(step)],
                 exists(case.arm(tag) / "meta.npz")),
            Step(f"fix_{tag}", "gsplat", ["fix", *c, "--tag", tag], fixed_done(case, tag, cfg.fixer_timestep)),
            Step(f"views_{tag}", "gsplat", ["fixer_views", *c, "--tag", tag, "--base", base],
                 exists(case.splat / arm_views(tag))),
            Step(f"train_{tag}", "gsplat", ["train", *c, "--arm", tag], exists(case.splat_ply(tag))),
        ]
    final = case.splat_ply(cfg.final_arm)
    steps += [
        Step("usdz", "grut", [str(final)], exists(final.with_suffix(".usdz")), script=str(SPHERE_WALK / "gs_usd.py")),
        Step("grid_final", "gsplat", ["grid", *c, "--arm", cfg.final_arm],
             exists(case.conflict(cfg.final_arm) / "grid.json")),
    ]
    if cfg.spots:
        arms = ["G2_base", "G2", cfg.final_arm]
        steps += [Step(f"sheet_{a}", "gsplat", ["sheet", *c, "--arm", a], exists(case.conflict(a) / "sheet_spots.jpg"))
                  for a in arms]
        steps.append(Step("compare", "gsplat", ["compare", *c, "--arms", ",".join(arms)],
                          exists(case.dir / "conflict" / f"sheet_compare_{'_'.join(arms)}.jpg")))
    return steps


def command(step: Step) -> list[str]:
    script = step.script or str(ENTRY)
    if step.env == "grut":
        grut = required("SIM2CHIP_GRUT_DIR")
        return ["bash", "-c", f"cd {grut} && source .venv/bin/activate && python {script} {' '.join(step.args)}"]
    if step.env == "cloud":
        # the VM runner reads the Nebius and Object Storage settings from .env (never traced: rule 5)
        run = f"exec {sys.executable} {script} {' '.join(step.args)}"
        return ["bash", "-c", f"set -a; . {ROOT / '.env'}; set +a; {run}"]
    return [python_for(step.env), script, *step.args]


def run(cfg: RouteConfig, until: str):
    """Every step for the case in order, the done ones skipped, through until (a step name)."""
    steps = recipe(cfg)
    names = [s.name for s in steps]
    if until not in names:
        sys.exit(f"--until must be one of: {', '.join(names)}")
    last = names.index(until)
    env = dict(os.environ, SIM2CHIP_REVIEW_MODEL=os.environ.get("SIM2CHIP_REVIEW_MODEL", erase.VLM))
    for k, step in enumerate(steps):
        if k > last:
            print(f"[{cfg.case}] stop before {step.name} (--until {until})")
            return
        if step.done():
            print(f"[{cfg.case}] {step.name}: done")
            continue
        if step.name == "erase" and erase.nothing_to_erase(cfg.photo, cfg.case):
            print(f"[{cfg.case}] erase: nothing to erase")
            continue
        if step.name == "walks":  # walks.prep caches the panorama's depth: stale once the panorama is made again
            made = PANOS / cfg.case / cfg.pano
            cache = Case(cfg.case).distance
            if cache.exists() and made.exists() and cache.stat().st_mtime < made.stat().st_mtime:
                cache.unlink()
        print(f"[{cfg.case}] {step.name} ...", flush=True)
        subprocess.run(command(step), check=True, cwd=ROOT, env=env)
    print(f"[{cfg.case}] done through {until}")


def status(cfg: RouteConfig):
    for s in recipe(cfg):
        print(f"{'done' if s.done() else '    '}  {s.name}{'  (paid)' if s.name in PAID else ''}")
