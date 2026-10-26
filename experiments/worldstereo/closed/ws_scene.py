"""End-to-end scene generation from one photo (the WorldStereo route), step 1: the photo emptied of every object that
would complicate the scene within the walks' area + MARGIN metres, before anything else sees it (user, 2026-10-24).
Like photo2scene's start (flow_things.py): the local VLM lists every thing with a box and calls it an object or part
of the place's structure, SAM 3's box prompt turns each box into a mask; MoGe-2 gives the photo metric points
(scaled so the ground lies ws_test.CAMERA_HEIGHT below the camera) and so each thing's distance on the ground. An
object is erased when its foot is within REACH + MARGIN metres; structure stays. The erased objects do not come back
(re-inserting them as 3D is future work).

    <SIM2CHIP_SAM3_PYTHON> experiments/worldstereo/ws_scene.py things --photo photo21     # VLM list (local server)
    <SIM2CHIP_SAM3_PYTHON> experiments/worldstereo/ws_scene.py masks --photo photo21      # SAM 3 (server stopped)
    <eyeforge-sam3d env>   experiments/worldstereo/ws_scene.py split --photo photo21      # MoGe-2 distances, rule
    <SIM2CHIP_ERASE_PYTHON> sim2chip/photo2scene/steps/erase_qwen.py --dir data/worldstereo_test/scene_in/e2e_photo21
    python experiments/worldstereo/ws_scene.py publish --photo photo21                    # erased photo -> panogen in/
    <SIM2CHIP_SAM3_PYTHON> experiments/worldstereo/ws_scene.py caption --photo photo21    # the one scene caption

The whole route for one photo, each step in its own env, a step skipped when its output is there (rerun after a
failure picks up where it stopped); it stops before the cloud unless --until says otherwise (the cloud run needs
the user's budget approval):

    python experiments/worldstereo/ws_scene.py run --photo photo21 [--until prep|cloud|all]

Reads data/panogen_test/in/<photo>.png and .json (EXIF field of view); writes data/worldstereo_test/scene_in/
e2e_<photo>/ (photo.png, things.json/npz, split.json, split.jpg, instances.npz, prepare.json, erased.png, the
eraser's background_qwen.png) and data/panogen_test/in/e2e_<photo>.png/.json, the case the rest of the route uses."""
import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
IN = ROOT / "data" / "panogen_test" / "in"
OUT = ROOT / "data" / "worldstereo_test" / "scene_in"
REACH = 10.0    # metres: the walks' longest reach (ws_test.MAX_FREE)
MARGIN = 5.0    # user, 2026-10-24: the area + 5 m
GROW = 15       # px the erase mask grows (contact shadows), as ws_nocar.py
FOOT = 0.1      # share of a mask's rows, from its bottom, taken as where it stands
CAMERA_HEIGHT = 1.6
PROMPT = """List every distinct physical thing in this outdoor photo, each as a whole object (a car is one car, a \
tree is one tree with its trunk and crown). Mark each one:
- "object": anything that stands on its own and could be taken away: vehicles, people, animals, bicycles, bins, \
signs, poles, street lights, bollards, benches, planters, a single tree or bush standing apart, parked things.
- "structure": what shapes the place itself: buildings, walls, fences, railings, hedges, rows of trees or bushes \
along an edge, slopes and banks.
Leave out the ground surfaces (road, pavement, grass) and the sky. List each thing once, at most 25 entries; \
things you cannot tell apart from far away (a line of parked cars, a hedge in pieces) are one entry whose box holds \
them all. Answer with ONE JSON object:
{"things": [{"name": "<1-3 words>", "kind": "object|structure", "box": [x0, y0, x1, y1]}]}
Box coordinates are normalised to 0-1000 (0,0 top left)."""


def case_dir(photo: str) -> Path:
    return OUT / f"e2e_{photo}"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def things(photo: str):
    """The VLM's list of things (object or structure, with boxes); a repeated box (IoU > 0.8) counts once. The
    local VLM server stays on the GPU afterwards: stop it before masks."""
    sys.path.insert(0, str(ROOT))
    from sim2chip.llm import LOCAL_VLM, Client, image_part, parse_json
    d = case_dir(photo)
    d.mkdir(parents=True, exist_ok=True)
    img = Image.open(IN / f"{photo}.png").convert("RGB")
    img.save(d / "photo.png")
    w, h = img.size
    client = Client(usage_log=ROOT / "runs" / "llm_usage.jsonl")
    msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, image_part(d / "photo.png")]}]

    def listed(v):
        return v if isinstance(v, list) else v.get("things") if isinstance(v, dict) else None

    def ok(v):
        lst = listed(v)
        return isinstance(lst, list) and all(isinstance(t, dict) and isinstance(t.get("box"), list)
                                             and len(t["box"]) == 4 for t in lst)
    model = os.environ.get("SIM2CHIP_REVIEW_MODEL", LOCAL_VLM)
    # Cosmos Reason 2 often answers with the bare list instead of {"things": [...]}: both are taken.
    a = None
    for _ in range(2):
        text = client.chat(msg, model, purpose="scene_things_outdoor", temperature=0.0, max_tokens=16384)
        try:
            v = parse_json(text)
        except (ValueError, TypeError):
            # It sometimes loops on one name and leaves the last entry broken: every complete entry is kept.
            v = []
            for frag in re.findall(r"\{[^{}]*\}", text):
                try:
                    v.append(json.loads(re.sub(r",\s*}", "}", frag)))
                except ValueError:
                    pass
            v = [t for t in v if isinstance(t.get("box"), list) and len(t["box"]) == 4]
            if not v:
                continue
        if ok(v):
            a = listed(v)
            break
    if a is None:
        sys.exit("the VLM gave no usable list of things")
    found = []
    for t in a:
        try:
            x0, y0, x1, y1 = (float(v) for v in t["box"])
        except (TypeError, ValueError):
            continue
        if x1 - x0 < 2 or y1 - y0 < 2 or any(iou([x0, y0, x1, y1], f["box"]) > 0.8 for f in found):
            continue
        found.append({"name": str(t.get("name", "thing")), "box": [x0, y0, x1, y1],
                      "kind": "structure" if t.get("kind") == "structure" else "object"})
    (d / "things.json").write_text(json.dumps({"things": found, "model": model}, indent=1))
    print(json.dumps([(t["name"], t["kind"]) for t in found]))


def masks(photo: str):
    """SAM 3's mask for each listed box."""
    import torch
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    d = case_dir(photo)
    img = Image.open(d / "photo.png").convert("RGB")
    w, h = img.size
    found = json.loads((d / "things.json").read_text())["things"]
    tracker = Sam3TrackerModel.from_pretrained("facebook/sam3").cuda().eval()
    tproc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    boxes = [[t["box"][0] * w / 1000, t["box"][1] * h / 1000, t["box"][2] * w / 1000, t["box"][3] * h / 1000]
             for t in found]
    masks = np.zeros((0, h, w), bool)
    if boxes:
        p = tproc(images=img, input_boxes=[boxes], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = tracker(**p, multimask_output=False)
        masks = tproc.post_process_masks(out.pred_masks.cpu(), p["original_sizes"].cpu())[0][:, 0].numpy()
    np.savez_compressed(d / "things.npz", masks=masks, names=np.array([t["name"] for t in found]))
    print(f"{len(found)} masks")


def split(photo: str):
    """Each thing's distance on the ground from MoGe-2's metric points; the objects within REACH + MARGIN go."""
    import torch
    sys.path[:0] = [str(Path(os.environ.get("TOOLS", "/media/lenena-iker/Crucial/tools")) / p)
                    for p in ("MoGe", "moge_deps")]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from moge.model.v2 import MoGeModel
    from ws_test import MOGE_CKPT
    d = case_dir(photo)
    img = np.asarray(Image.open(d / "photo.png").convert("RGB"))
    fov = json.loads((IN / f"{photo}.json").read_text())["hfov"]
    model = MoGeModel.from_pretrained(MOGE_CKPT).cuda().eval()
    x = torch.tensor(img / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None]
    with torch.no_grad():
        res = model.infer(x, fov_x=torch.tensor([fov], device="cuda"), apply_mask=True)
    pts = res["points"][0].cpu().numpy()          # OpenCV camera: x right, y down, z forward
    valid = res["mask"][0].cpu().numpy() & np.isfinite(pts).all(-1)
    z = np.load(d / "things.npz")
    masks, names = z["masks"], [str(n) for n in z["names"]]
    info = json.loads((d / "things.json").read_text())["things"]
    # Ground: the lowest third of the photo, outside every thing, gives the camera's height (y down): scale so it is
    # CAMERA_HEIGHT; a horizontal camera is assumed, as the panorama step does.
    h = img.shape[0]
    rows = np.arange(h)[:, None] >= 2 * h // 3
    free = valid & rows & ~(masks.any(0) if len(masks) else False)
    scale = CAMERA_HEIGHT / float(np.median(pts[free][:, 1])) if free.sum() > 500 else 1.0
    pts = pts * scale
    keep_masks, keep_names, report = [], [], []
    for m, n, t in zip(masks, names, info):
        ys = np.nonzero(m.any(1))[0]
        if not len(ys):
            continue
        foot = m & (np.arange(h)[:, None] >= ys.max() - max(2, int(FOOT * (ys.max() - ys.min() + 1)))) & valid
        dist = float(np.median(np.hypot(pts[foot][:, 0], pts[foot][:, 2]))) if foot.any() else float("inf")
        erase = t["kind"] == "object" and dist <= REACH + MARGIN
        report.append({"name": n, "kind": t["kind"], "distance_m": round(dist, 1), "erase": erase,
                       "share": round(float(m.mean()), 4)})
        if erase:
            keep_masks.append(m)
            keep_names.append(n)
    union = np.any(keep_masks, 0) if keep_masks else np.zeros(img.shape[:2], bool)
    Image.fromarray(union.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * GROW + 1)).save(d / "erased.png")
    np.savez_compressed(d / "instances.npz", masks=np.stack(keep_masks) if keep_masks else
                        np.zeros((0,) + img.shape[:2], bool), phrases=np.array(keep_names))
    (d / "prepare.json").write_text(json.dumps({"instances": keep_names}))
    (d / "split.json").write_text(json.dumps({"scale": scale, "limit_m": REACH + MARGIN, "things": report}, indent=1))
    # Sheet: red erased, blue objects left (too far), green structure.
    ov = img.astype(np.float32).copy()
    for m, r in zip(masks, report):
        c = (255, 40, 40) if r["erase"] else (60, 120, 255) if r["kind"] == "object" else (40, 200, 80)
        ov[m] = ov[m] * 0.45 + np.array(c) * 0.55
    sheet = Image.fromarray(ov.astype(np.uint8))
    dr = ImageDraw.Draw(sheet)
    for m, r in zip(masks, report):
        ys, xs = np.nonzero(m)
        if len(xs):
            dr.text((int(xs.mean()), int(ys.mean())), f"{r['name']} {r['distance_m']} m", fill=(255, 255, 255),
                    stroke_width=2, stroke_fill=(0, 0, 0))
    sheet.save(d / "split.jpg", quality=90)
    print(json.dumps(report))


def caption(photo: str):
    """One caption per scene, from the same VLM on the erased photo: WorldGen's prompt (in/e2e_<photo>.txt) and every
    WorldStereo walk's (ws_test.py caption --scene-caption)."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sim2chip.llm import LOCAL_VLM, Client, image_part
    from ws_test import CAPTION
    client = Client(usage_log=ROOT / "runs" / "llm_usage.jsonl")
    msg = [{"role": "user", "content": [{"type": "text", "text": CAPTION + PLACE_ONLY},
                                        image_part(IN / f"e2e_{photo}.png")]}]
    text = client.chat(msg, os.environ.get("SIM2CHIP_REVIEW_MODEL", LOCAL_VLM), purpose="scene_caption",
                       temperature=0.0, max_tokens=16384)
    text = " ".join(text.split("</think>")[-1].strip().strip('"').split())
    (IN / f"e2e_{photo}.txt").write_text(text)
    print(f"e2e_{photo}: {text}")


HOME = Path.home()
ENVS = {"sam3": os.environ.get("SIM2CHIP_SAM3_PYTHON", str(HOME / "miniforge3/envs/teacher-sam3/bin/python")),
        "sam3d": "/media/lenena-iker/Crucial/envs/eyeforge-sam3d/bin/python",
        "erase": os.environ.get("SIM2CHIP_ERASE_PYTHON", "/media/lenena-iker/Crucial/envs/eyeforge-qwenedit/bin/python"),
        "pano": "/media/lenena-iker/Crucial/tools/envs/panogen/bin/python",
        "gsplat": "/media/lenena-iker/Crucial/tools/envs/gsplat/bin/python",
        "edge": str(HOME / "miniforge3/envs/edge-sdg/bin/python"),
        "here": sys.executable}
GRUT = Path("/media/lenena-iker/Crucial/tools/3dgrut")
VLM = "nvidia/Nemotron-3.5-Super-VL"  # Cosmos Reason 2 looped and missed things in the listing (2026-10-24)
PANO_SEED = 0
# WorldGen draws whatever the caption names where it fits: "a red trailer and a white truck" on photo27 became a
# truck beside the camera (2026-10-24). The scene is the place alone; objects come later from the generator.
PLACE_ONLY = (" Describe only the place itself: never mention vehicles, people, animals, bicycles or any other "
              "object that could be moved away.")


def run(photo: str, until: str):
    """Every step for one photo, in order; see the module's docstring."""
    import subprocess
    case = f"e2e_{photo}"
    d, ws_dir = case_dir(photo), ROOT / "data" / "worldstereo_test" / case
    me = str(Path(__file__).resolve())
    wt, sp = str(Path(me).parent / "ws_test.py"), str(Path(me).parent / "ws_splat.py")
    pano = f"pano_s{PANO_SEED}_paste.jpg"
    splat = ws_dir / "splat" / "arms" / "G2" / "splat.ply"
    steps = [  # (name, env, args, output that marks it done, stage)
        ("things", "sam3", [me, "things", "--photo", photo], d / "things.json", "prep"),
        ("masks", "sam3", [me, "masks", "--photo", photo], d / "things.npz", "prep"),
        ("split", "sam3d", [me, "split", "--photo", photo], d / "split.json", "prep"),
        ("erase", "erase", [str(ROOT / "sim2chip/photo2scene/steps/erase_qwen.py"), "--dir", str(d)],
         d / "erase_qwen.json", "prep"),
        ("publish", "here", [me, "publish", "--photo", photo], IN / f"{case}.png", "prep"),
        ("caption", "sam3", [me, "caption", "--photo", photo], IN / f"{case}.txt", "prep"),
        ("pano", "pano", [str(ROOT / "experiments/panogen/panogen_test.py"), "pano", "--case", case],
         ROOT / "data/panogen_test/out" / case / pano, "prep"),
        ("prep", "sam3d", [wt, "prep", "--case", case, "--pano", pano], ws_dir / "prep.json", "prep"),
        ("cloud", "edge", [str(Path(me).parent / "ws_vm.py"), "run", case],
         ws_dir / "out" / "out" / case, "cloud"),
        ("views", "sam3d", [sp, "views", "--case", case], ws_dir / "splat" / "views.npz", "all"),
        ("train", "gsplat", [sp, "train", "--case", case, "--arm", "G2"], splat, "all"),
        ("usdz", "grut", [str(ROOT / "experiments/sphere_walk/gs_usd.py"), str(splat)],
         splat.with_suffix(".usdz"), "all"),
    ]
    order = ["prep", "cloud", "all"]
    env = dict(os.environ, SIM2CHIP_REVIEW_MODEL=os.environ.get("SIM2CHIP_REVIEW_MODEL", VLM))
    for name, e, args, done, stage in steps:
        if order.index(stage) > order.index(until):
            print(f"[{case}] stop before {name} (--until {until})")
            return
        if done.exists():
            print(f"[{case}] {name}: done")
            continue
        if name == "erase" and not json.loads((d / "prepare.json").read_text())["instances"]:
            print(f"[{case}] erase: nothing to erase")
            continue
        if name == "prep":  # ws_test.py caches the panorama's depth: stale once the panorama is made again
            cache, made = ws_dir / "distance.npz", ROOT / "data/panogen_test/out" / case / pano
            if cache.exists() and cache.stat().st_mtime < made.stat().st_mtime:
                cache.unlink()
        print(f"[{case}] {name} ...", flush=True)
        if e == "grut":
            cmd = ["bash", "-c", f"cd {GRUT} && source .venv/bin/activate && python {' '.join(args)}"]
        elif e == "edge":
            # the cloud runner reads the Nebius settings from .env (never traced: rule 5)
            cmd = ["bash", "-c", f"set -a; . {ROOT / '.env'}; set +a; exec {ENVS[e]} {' '.join(args)}"]
        else:
            cmd = [ENVS[e], *args]
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def publish(photo: str):
    """The eraser's photo as the case e2e_<photo> of the panorama step, with the original's field of view."""
    d = case_dir(photo)
    src = d / "background_qwen.png" if (d / "background_qwen.png").exists() else d / "photo.png"
    shutil.copy(src, IN / f"e2e_{photo}.png")
    shutil.copy(IN / f"{photo}.json", IN / f"e2e_{photo}.json")
    print(f"{src} -> {IN / f'e2e_{photo}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["things", "masks", "split", "publish", "caption", "run"])
    ap.add_argument("--photo", required=True)
    ap.add_argument("--until", choices=["prep", "cloud", "all"], default="prep")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.photo, a.until)
    else:
        {"things": things, "masks": masks, "split": split, "publish": publish, "caption": caption}[a.cmd](a.photo)
