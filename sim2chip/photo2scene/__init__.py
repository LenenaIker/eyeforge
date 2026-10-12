"""photo2scene in the product (docs/plan_photo2scene.md): a place for the synthetic frames from one of the user's
photos, built once and reused by later jobs on the same dataset.

The support route here (one working surface seen from above: a table, a lawn, the ground close up): the photo
stays the background, its movable things are erased and meshed, and every job places them again at random on the
rebuilt support plane and renders new views composited on the photo. The street and room routes, a whole place
rebuilt from the photo, are in place.py; surfaces seen from above in surface.py.

Each step is a script in steps/, run in the Python that has its models (the step scripts' docstrings say which):
    sam3   SAM 3 and the VLM calls (the teachers' Python, SIM2CHIP_SAM3_PYTHON)
    mesh   SAM 3D Objects, MoGe-2 and the geometry (pyrender, trimesh), SIM2CHIP_SCENE_PYTHON
    erase  Qwen-Image-Edit 2511 with the Object-Remover LoRA, SIM2CHIP_ERASE_PYTHON
    isaac  Isaac Sim, SIM2CHIP_ISAAC_PYTHON
A step whose outputs exist is not run again, so a build that stopped resumes where it was.

build(): S.1 plan, S.2 segment and erase, S.3 depth, S.4 the erased things as 3D models. The folder it fills is
the scene the library keeps.
compose(): S.5 a new layout and views, S.6 Isaac render and labels, S.7 the VLM review. Every job runs it again
with its own seed on its own copy of the scene.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import config

STEPS = Path(__file__).resolve().parent / "steps"
PYTHONS = {"sam3": "SIM2CHIP_SAM3_PYTHON", "mesh": "SIM2CHIP_SCENE_PYTHON", "erase": "SIM2CHIP_ERASE_PYTHON",
           "isaac": "SIM2CHIP_ISAAC_PYTHON"}
# Where the steps find their weights and code, passed on when set (.env or the worker's environment).
PASS_ON = ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "MOGE_ROOT", "MOGE_DEPS", "MOGE_CKPT", "SIM2CHIP_SAM3D_DIR",
           "SIM2CHIP_LAMA", "SIM2CHIP_PLAN_MODEL", "SIM2CHIP_OBSERVE_MODEL", "SIM2CHIP_REVIEW_MODEL")
STEP_TIMEOUT_S = 30 * 60
MIN_SUPPORT = 0.03  # share of the photo the support must cover
LONG_SIDE = 1344    # the photo's long side in the scene (bg_prepare.py's default)
# What compose() writes into a scene folder: one job's layout, not part of the scene the library keeps.
COMPOSED = ("isaac", "report.json", "sheet.jpg", "scene.glb", "sheet_isaac.jpg", "review.jpg", "review.json",
            "groups.json")


class Stop(Exception):
    """A check failed: the photo makes no scene (the reason is the message)."""


def missing_pythons(names=("sam3", "mesh", "erase", "isaac")) -> list[str]:
    """The step Pythons this machine lacks (environment variable unset or file missing). In a cloud job the mesh
    and erase ones come from the cache bucket when a step needs them (cloud.py)."""
    from . import cloud
    names = [n for n in names if not (cloud.on() and n in cloud.KITS)]
    return [PYTHONS[n] for n in names if not (p := config.env(PYTHONS[n])) or not Path(p).exists()]


@dataclass
class Flow:
    d: Path                         # the scene folder
    usage_log: Path | None = None   # where the VLM calls are counted (the job's logs/llm_usage.jsonl)
    log: callable = print
    steps: dict = field(default_factory=dict)

    def __post_init__(self):
        self.d.mkdir(parents=True, exist_ok=True)
        (self.d / "logs").mkdir(exist_ok=True)

    def run(self, step: str, python: str, script: str, args: list, outputs: list[str] = (), ok=(0,),
            must: bool = True) -> int:
        """One step script, unless all its outputs exist. Returns its exit code; raises when it is not in ok, unless
        the step is a best effort (must=False: a retry that fills what a first run left out)."""
        if outputs and all((self.d / o).exists() for o in outputs):
            return 0
        from . import cloud
        env = {**os.environ, **{k: v for k in PASS_ON if (v := config.env(k))}, "PYOPENGL_PLATFORM": "egl",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1",
               **(cloud.ready(python, script, self.log) if cloud.on() else {})}
        if self.usage_log:
            env["SIM2CHIP_LLM_USAGE"] = str(self.usage_log)
        t0 = time.time()
        logf = self.d / "logs" / f"{step}.log"
        with open(logf, "w") as fh:
            try:
                code = subprocess.run([config.env(PYTHONS[python]), str(STEPS / script), *map(str, args)],
                                      stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=STEPS,
                                      timeout=STEP_TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                code = -1
        secs = round(time.time() - t0, 1)
        self.steps[step] = {"code": code, "seconds": secs}
        self.log(f"  {step}: exit {code} in {secs:.0f} s")
        if code not in ok and must:
            raise RuntimeError(f"{step} failed (exit {code}): {logf.read_text()[-800:]}")
        return code

    def json(self, name: str):
        return json.loads((self.d / name).read_text())


def build(flow: Flow, photo: Path, targets: list[str], sunny: bool = False) -> dict:
    """The scene of one photo into flow.d. targets: the phrases of the user's classes, segmented and erased first;
    each erased thing named like one of them is a target, the rest distractors. Returns its summary (route, the
    plan, the erased things); raises Stop when the photo does not fit the support route."""
    import shutil

    d = flow.d
    if not (d / "photo.png").exists():  # upright and at the size the steps work at (a phone's 12 MP PNG is
        from PIL import Image, ImageOps  # 10 MB, more than the VLM endpoint takes)
        with Image.open(photo) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
            im.save(d / "photo.png")
    image = d / "photo.png"
    if flow.run("S.1_plan", "sam3", "bg_plan.py", ["--image", image, "--out", d], ["plan.json"], ok=(0, 3)) == 3 \
            or not flow.json("plan.json")["one_flat_support"]:
        raise Stop("no single flat support holds the objects: " + flow.json("plan.json").get("reason", ""))
    plan = flow.json("plan.json")
    # The user's classes by the phrases the teacher uses, then whatever else the VLM saw on the support. (Closing
    # a class's meshes like the VLM's solids turned a bottle with a handle into a cone, 2026-10-11: left open.)
    objects = list(dict.fromkeys([t.lower() for t in targets] + plan["objects"]))
    flow.run("S.2_prepare", "sam3", "bg_prepare.py", ["--image", image, "--objects", ",".join(objects), "--support",
                                                     plan["support"], "--out", d] + (["--shadows"] if sunny else []),
             ["background.png", "instances.npz"])
    flow.run("S.2_erase", "erase", "erase_qwen.py", ["--dir", d, "--replace"], ["background_qwen.png"])
    prep = flow.json("prepare.json")
    if prep["support_frac"] <= MIN_SUPPORT or not prep["instances"]:
        raise Stop(f"the segmenter found no support ({prep['support_frac']:.0%} of the photo) or no objects")
    flow.run("S.3_depth", "mesh", "bg_depth.py", ["--dir", d], ["depth_photo.npz", "depth_background.npz"])
    flow.run("S.4_assets", "sam3", "flow_assets.py", ["--dir", d, "--targets", ",".join(targets)], ["assets.json"])
    mesh = ["--dir", d, "--masks", "assets.npz", "--image", "photo", "--out", "assets"]
    flow.run("S.4_meshes", "mesh", "flow_sam3d.py", mesh, ["assets/scene_graph.json"])
    assets = flow.json("assets.json")["assets"]
    if not all((d / a["mesh"]).exists() for a in assets):
        flow.run("S.4_meshes_retry", "mesh", "flow_sam3d.py", mesh, must=False)  # fills what a GPU memory failure
        # left out; when it fails again, the things without a mesh are left out (a sign twice out of GPU memory,
        # 2026-10-12)
    made = [a for a in assets if (d / a["mesh"]).exists()]
    if not any(a["role"] == "target" for a in made):
        raise Stop("none of the user's classes could be taken out of the photo as a 3D model")
    shutil.rmtree(d / "isaac", ignore_errors=True)
    return {"route": "support", "plan": {k: plan.get(k) for k in ("support", "objects", "solid", "fixed")},
            "objects": made}


def compose(flow: Flow, views: int, seed: int, groups: dict[str, str] | None = None, empty: float = 0.0) -> list[dict]:
    """New views of a built scene: its things placed again at random (seed), rendered by Isaac on the photo and
    labelled. groups ({thing's name: group}): each view shows the things of one group only (things in no group
    always), or of none in a share `empty` of the views. Returns [{"file": isaac/render_NN.png, "boxes": [{"box",
    "name", "role"}]}], the photo's own layout (view 0) left out; empty when the VLM review rejects the views."""
    d = flow.d
    plan = flow.json("plan.json")
    import shutil
    # Every job its own layout. Removed, never written over: the scene's files may be links to the library's.
    for out in COMPOSED:
        p = d / out
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    one = []
    if groups is not None:
        (d / "groups.json").write_text(json.dumps(groups))
        one = ["--groups", d / "groups.json", "--empty", empty]
    flow.run("S.5_build", "mesh", "bg_build.py", ["--dir", d, "--meshes", d / "assets", "--views", views, "--seed", seed,
                                                  "--solid", ",".join(plan.get("solid") or []),
                                                  "--fixed", ",".join(plan.get("fixed") or []), "--stack", "0.2", *one])
    flow.run("S.6_isaac", "isaac", "bg_isaac.py", ["--dir", d])
    flow.run("S.6_labels", "mesh", "view_labels.py", ["--dir", d])
    if flow.run("S.7_review", "sam3", "bg_review.py", ["--dir", d], ok=(0, 3)) == 3:
        review = flow.json("review.json")
        flow.log(f"  the review rejected the views (score {review.get('score')}): {'; '.join(review.get('worst', []))}")
        return []
    labels = flow.json("isaac/labels.json")
    return [{"file": f"isaac/{name}", "boxes": boxes} for name, boxes in sorted(labels.items())
            if name != "render_00.png" and (d / "isaac" / name).exists()]
