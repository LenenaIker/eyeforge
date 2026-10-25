"""WorldStereo's cheap knobs, one change per variant (K rows of docs/pruebas_reconstruccion_worldstereo.md).

The four quarter turns at the silver SUV (spg_0_*) are generated again with phase 1 in memory, as in S3, once per
variant, in one VM session and one model load per batch (each variant is a scene of its own under one input folder;
batch "knobs" ran k0 and failed on an align_nframe of 32, batch "knobs2" ran k1 and k2):

    k0_control   S3's settings; every leg gets a fresh generator seeded 1024 (upstream's run shares one generator
                 across legs and scenes, which would hand each variant different noise). Against S3's own legs it
                 measures what a new seed alone changes.
    k1_align20   the memory keeps 20 frames of each clip instead of 8 (align_nframe; a clip has 21 and upstream
                 never stores the first, so 20 is the most): retrieval picks, per target keyframe, the stored frame
                 that overlaps it most, and S3's picks overlapped only ~50 %.
    k2_caption   the turns get a caption without left or right: the walk's caption says "wall on the left, SUV on
                 the right", which stops being true once the camera turns.

max_reference (8) is left alone: no leg has ever retrieved more than 8 frames, so it never binds.

    python experiments/worldstereo/ws_knobs.py stage --batch knobs2   # scenes and resume under data/worldstereo_test/<batch>/
    python experiments/worldstereo/ws_knobs.py run --batch knobs2     # VM: start, generate, fetch, stop
    python experiments/worldstereo/ws_knobs.py link                   # every batch's results as legs of photo21

`set -a; . ./.env; set +a` first, like ws_vm.py."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

SRC = "photo21"
LEGS = ("spg_0_0", "spg_0_1", "spg_0_2", "spg_0_3")
PHASE1 = ("reconstruct_0", "reconstruct_1", "reconstruct_2", "wonder_0")
BATCHES = {"knobs": ("k0_control",), "knobs2": ("k1_align20", "k2_caption")}
ALIGN = {"k1_align20": 20}
NEUTRAL = ("A quiet residential street in daylight under a partly cloudy sky: a narrow, cracked asphalt road, a "
           "stone retaining wall covered in ivy and greenery, a silver SUV with a black roof box parked at the "
           "roadside, a few other parked cars, tall leafy trees, hedges and red-brick houses behind them. Soft, "
           "even light and calm weather.")


def stage(batch: str):
    """data/worldstereo_test/knobs/scene/<variant>/ (pano_bank, phase 1, the four turns) and, as already generated,
    phase 1's results under out/out/knobs/<variant>/."""
    src, root = ws.DATA / SRC, ws.DATA / batch
    if root.exists():
        shutil.rmtree(root)
    for v in BATCHES[batch]:
        scene = root / "scene" / v
        for d in ("pano_bank", *PHASE1, *LEGS):
            shutil.copytree(src / "scene" / d, scene / d)
        if v in ALIGN:
            (scene / "align_nframe.txt").write_text(str(ALIGN[v]))
        if v == "k2_caption":
            for leg in LEGS:
                (scene / leg / "traj0" / "traj_caption.json").write_text(json.dumps({"prompt": NEUTRAL}))
        done = root / "out" / "out" / batch / v
        for d in PHASE1:
            shutil.copytree(src / "out" / "out" / SRC / d, done / d)
    print(f"staged {len(BATCHES[batch])} variants under {root}")


# Into the job's run_ws.py, before upstream runs: a per-scene align_nframe, and a fresh generator for every leg.
PATCH = r'''
import src.retrieval_wm as rwm
_bank_init = rwm.SimpleMemoryBank.__init__
def _bank(self, cfg, root_path, *a, **k):
    f = os.path.join(root_path, "align_nframe.txt")
    if os.path.exists(f):
        k["align_nframe"] = int(open(f).read())
        print(f"{root_path}: align_nframe {k['align_nframe']}", flush=True)
    _bank_init(self, cfg, root_path, *a, **k)
rwm.SimpleMemoryBank.__init__ = _bank


class _FreshNoise:
    """The pipeline with its generator reseeded on every call: each leg gets the same noise in every variant."""
    def __init__(self, pipe):
        self.__dict__["_pipe"] = pipe
    def __call__(self, *a, **k):
        if k.get("generator") is not None:
            k["generator"] = torch.Generator(device=k["generator"].device).manual_seed(1024)
        return self._pipe(*a, **k)
    def __getattr__(self, name):
        return getattr(self._pipe, name)
    def __setattr__(self, name, value):
        setattr(self._pipe, name, value)
_fp_knobs = WorldStereo.from_pretrained.__func__
def _from_pretrained_knobs(cls, *a, **k):
    w = _fp_knobs(cls, *a, **k)
    w.pipeline = _FreshNoise(w.pipeline)
    return w
WorldStereo.from_pretrained = classmethod(_from_pretrained_knobs)
'''


def preflight(case: str, image: str):
    """ws_test.preflight for a folder of scenes."""
    import subprocess
    for scene in sorted((ws.DATA / case / "scene").iterdir()):
        legs = [t for t in scene.glob("*/traj*") if t.is_dir()]
        need = ("render.mp4", "render_mask.mp4", "camera.json", "traj_caption.json")
        missing = [f"{t.relative_to(scene)}/{n}" for t in legs for n in need if not (t / n).exists()]
        missing += [] if (scene / "pano_bank/cameras.json").exists() else ["pano_bank/cameras.json"]
        if not legs or missing:
            sys.exit(f"{scene.name} not ready ({len(legs)} legs), missing: {missing}")
    if subprocess.run(["docker", "manifest", "inspect", image], capture_output=True).returncode:
        sys.exit(f"{image} is not in the registry")
    print(f"preflight ok: {len(list((ws.DATA / case / 'scene').iterdir()))} scenes, image pushed")


def run(batch: str):
    import ws_vm
    ws.PREFIX = f"worldstereo_{batch}"  # its own outputs: fetch lists every key under the prefix
    marker = "WorldStereo.from_pretrained = classmethod(from_pretrained)\n"
    out_arg = "--input_path /tmp/scene/__CASE__ --output_path /tmp/out"
    assert marker in ws.JOB and out_arg in ws.JOB
    # each variant is a scene under /tmp/scene/knobs; outputs land in /tmp/out/knobs/<variant>, where resume put
    # phase 1
    ws.JOB = ws.JOB.replace(marker, marker + PATCH).replace(out_arg, out_arg + "/__CASE__")
    ws.preflight = preflight
    ws_vm.cmd_run(batch, "resume")


def link(batch: str = ""):
    """Each variant's four turns as legs of photo21 (spg_0_0 -> k1_align32_spg_0_0) in out/out/photo21 and
    legsets/knobs, so ws_depth.py's depth and consistency see them next to S3's."""
    res = ws.DATA / SRC / "out" / "out" / SRC
    sets = ws.DATA / SRC / "legsets" / "knobs"
    for b, v in [(b, v) for b, vs in BATCHES.items() for v in vs]:
        got = ws.DATA / b / "out" / "out" / b / v
        for leg in LEGS:
            name = f"{v}_{leg}"
            if not (got / leg / "traj0" / f"{ws.MODEL}_result.mp4").exists():
                print(f"missing {v}/{leg}")
                continue
            if (res / name).exists():
                shutil.rmtree(res / name)
            shutil.copytree(got / leg, res / name)
            (sets / name).mkdir(parents=True, exist_ok=True)
            shutil.copytree(ws.DATA / SRC / "scene" / leg / "traj0", sets / name / "traj0", dirs_exist_ok=True)
    print(f"linked into {res} and {sets}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "run", "link"])
    ap.add_argument("--batch", default="knobs2", choices=list(BATCHES))
    a = ap.parse_args()
    {"stage": stage, "run": run, "link": link}[a.cmd](a.batch)
