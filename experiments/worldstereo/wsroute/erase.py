"""Step 1: the photo emptied of every object that would complicate the scene within the walks' area + MARGIN metres,
before anything else sees it (user, 2026-10-24). The VLM lists every thing with a box and calls it an object or part
of the place's structure, SAM 3's box prompt turns each box into a mask; MoGe-2 gives the photo metric points
(scaled so the ground lies CAMERA_HEIGHT below the camera) and so each thing's distance on the ground. An object is
erased when its foot is within REACH + MARGIN metres; structure stays. The erased objects do not come back
(re-inserting them as 3D is future work).

    things   SIM2CHIP_SAM3_PYTHON   the VLM's list (things.json)
    masks    SIM2CHIP_SAM3_PYTHON   SAM 3 per box (things.npz); stop the local VLM server first
    split    SIM2CHIP_SCENE_PYTHON  distances and the rule (split.json, split.jpg, instances.npz, erased.png)
    (erase)  SIM2CHIP_ERASE_PYTHON  wsroute/erase_qwen.py on that folder
    publish  any                    the erased photo as the case's input of the panorama step
    caption  SIM2CHIP_SAM3_PYTHON   the one scene caption (WorldGen's prompt and every walk's)

Reads data/panogen_test/in/<photo>.png and .json (EXIF field of view); writes data/worldstereo_test/scene_in/
<case>/ and data/panogen_test/in/<case>.png/.json/.txt (the case is e2e_<photo> by default)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .camera import CAMERA_HEIGHT
from .paths import ERASE, PHOTOS, ROOT, add_moge, env, moge_ckpt

REACH = 10.0    # metres: the walks' longest reach (walks.MAX_FREE)
MARGIN = 5.0    # user, 2026-10-24: the area + 5 m
GROW = 15       # px the erase mask grows (contact shadows)
FOOT = 0.1      # share of a mask's rows, from its bottom, taken as where it stands
VLM = "nvidia/Nemotron-3.5-Super-VL"  # Cosmos Reason 2 looped and missed things in the listing (2026-10-24)
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
CAPTION = ("Describe this scene in one paragraph of about 90 words for a video model that will walk forward into "
           "it: the ground, what lies ahead and on each side, the materials, the light and the weather. Plain "
           "description only, no camera words, no lists.")
# WorldGen draws whatever the caption names where it fits: "a red trailer and a white truck" on photo27 became a
# truck beside the camera (2026-10-24). The scene is the place alone; objects come later from the generator.
PLACE_ONLY = (" Describe only the place itself: never mention vehicles, people, animals, bicycles or any other "
              "object that could be moved away.")


def case_name(photo: str) -> str:
    """The default case of a photo."""
    return f"e2e_{photo}"


def case_dir(case: str) -> Path:
    return ERASE / case


def vlm() -> str:
    return env("SIM2CHIP_REVIEW_MODEL") or VLM


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def parse_things(text: str) -> list | None:
    """The VLM's list from its answer: {"things": [...]} or the bare list (Cosmos Reason 2 often answers that);
    an answer that looped and broke its last entry keeps every complete one. None when nothing usable is left."""
    from sim2chip.llm import parse_json

    def listed(v):
        return v if isinstance(v, list) else v.get("things") if isinstance(v, dict) else None
    try:
        v = parse_json(text)
    except (ValueError, TypeError):
        v = []
        for frag in re.findall(r"\{[^{}]*\}", text):
            try:
                v.append(json.loads(re.sub(r",\s*}", "}", frag)))
            except ValueError:
                pass
        v = [t for t in v if isinstance(t.get("box"), list) and len(t["box"]) == 4]
        if not v:
            return None
    lst = listed(v)
    ok = isinstance(lst, list) and all(isinstance(t, dict) and isinstance(t.get("box"), list) and len(t["box"]) == 4
                                       for t in lst)
    return lst if ok else None


def things(photo: str, case: str):
    """The VLM's list of things (object or structure, with boxes); a repeated box (IoU > 0.8) counts once."""
    from sim2chip.llm import Client, image_part
    d = case_dir(case)
    d.mkdir(parents=True, exist_ok=True)
    img = Image.open(PHOTOS / f"{photo}.png").convert("RGB")
    img.save(d / "photo.png")
    client = Client(usage_log=ROOT / "runs" / "llm_usage.jsonl")
    msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, image_part(d / "photo.png")]}]
    model = vlm()
    a = None
    for _ in range(2):
        a = parse_things(client.chat(msg, model, purpose="scene_things_outdoor", temperature=0.0, max_tokens=16384))
        if a is not None:
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


def masks(photo: str, case: str):
    """SAM 3's mask for each listed box."""
    import torch
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    d = case_dir(case)
    img = Image.open(d / "photo.png").convert("RGB")
    w, h = img.size
    found = json.loads((d / "things.json").read_text())["things"]
    tracker = Sam3TrackerModel.from_pretrained("facebook/sam3").cuda().eval()
    tproc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    boxes = [[t["box"][0] * w / 1000, t["box"][1] * h / 1000, t["box"][2] * w / 1000, t["box"][3] * h / 1000]
             for t in found]
    out_masks = np.zeros((0, h, w), bool)
    if boxes:
        p = tproc(images=img, input_boxes=[boxes], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = tracker(**p, multimask_output=False)
        out_masks = tproc.post_process_masks(out.pred_masks.cpu(), p["original_sizes"].cpu())[0][:, 0].numpy()
    np.savez_compressed(d / "things.npz", masks=out_masks, names=np.array([t["name"] for t in found]))
    print(f"{len(found)} masks")


def split(photo: str, case: str):
    """Each thing's distance on the ground from MoGe-2's metric points; the objects within REACH + MARGIN go."""
    import torch
    add_moge()
    from moge.model.v2 import MoGeModel
    d = case_dir(case)
    img = np.asarray(Image.open(d / "photo.png").convert("RGB"))
    fov = json.loads((PHOTOS / f"{photo}.json").read_text())["hfov"]
    model = MoGeModel.from_pretrained(moge_ckpt()).cuda().eval()
    x = torch.tensor(img / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None]
    with torch.no_grad():
        res = model.infer(x, fov_x=torch.tensor([fov], device="cuda"), apply_mask=True)
    pts = res["points"][0].cpu().numpy()          # OpenCV camera: x right, y down, z forward
    valid = res["mask"][0].cpu().numpy() & np.isfinite(pts).all(-1)
    z = np.load(d / "things.npz")
    all_masks, names = z["masks"], [str(n) for n in z["names"]]
    info = json.loads((d / "things.json").read_text())["things"]
    # Ground: the lowest third of the photo, outside every thing, gives the camera's height (y down): scale so it is
    # CAMERA_HEIGHT; a horizontal camera is assumed, as the panorama step does.
    h = img.shape[0]
    rows = np.arange(h)[:, None] >= 2 * h // 3
    free = valid & rows & ~(all_masks.any(0) if len(all_masks) else False)
    scale = CAMERA_HEIGHT / float(np.median(pts[free][:, 1])) if free.sum() > 500 else 1.0
    pts = pts * scale
    keep_masks, keep_names, report = [], [], []
    for m, n, t in zip(all_masks, names, info):
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
    split_sheet(d, img, all_masks, report)
    print(json.dumps(report))


def split_sheet(d: Path, img: np.ndarray, all_masks: np.ndarray, report: list):
    """Red erased, blue objects left (too far), green structure."""
    ov = img.astype(np.float32).copy()
    for m, r in zip(all_masks, report):
        c = (255, 40, 40) if r["erase"] else (60, 120, 255) if r["kind"] == "object" else (40, 200, 80)
        ov[m] = ov[m] * 0.45 + np.array(c) * 0.55
    sheet = Image.fromarray(ov.astype(np.uint8))
    dr = ImageDraw.Draw(sheet)
    for m, r in zip(all_masks, report):
        ys, xs = np.nonzero(m)
        if len(xs):
            dr.text((int(xs.mean()), int(ys.mean())), f"{r['name']} {r['distance_m']} m", fill=(255, 255, 255),
                    stroke_width=2, stroke_fill=(0, 0, 0))
    sheet.save(d / "split.jpg", quality=90)


def nothing_to_erase(photo: str, case: str) -> bool:
    return not json.loads((case_dir(case) / "prepare.json").read_text())["instances"]


def publish(photo: str, case: str):
    """The eraser's photo as the case's input of the panorama step, with the original's field of view."""
    d = case_dir(case)
    src = d / "background_qwen.png" if (d / "background_qwen.png").exists() else d / "photo.png"
    shutil.copy(src, PHOTOS / f"{case}.png")
    shutil.copy(PHOTOS / f"{photo}.json", PHOTOS / f"{case}.json")
    print(f"{src} -> {PHOTOS / f'{case}.png'}")


def caption(photo: str, case: str):
    """One caption per scene, from the same VLM on the erased photo: WorldGen's prompt and every walk's."""
    from sim2chip.llm import Client, image_part
    client = Client(usage_log=ROOT / "runs" / "llm_usage.jsonl")
    msg = [{"role": "user", "content": [{"type": "text", "text": CAPTION + PLACE_ONLY},
                                        image_part(PHOTOS / f"{case}.png")]}]
    text = client.chat(msg, vlm(), purpose="scene_caption", temperature=0.0, max_tokens=16384)
    text = " ".join(text.split("</think>")[-1].strip().strip('"').split())
    (PHOTOS / f"{case}.txt").write_text(text)
    print(f"{case}: {text}")
