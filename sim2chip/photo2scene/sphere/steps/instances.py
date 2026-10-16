"""S.3 of the sphere route, instances: SAM 3 on every cut for the targets (the spec's classes) and the movable
things (people, bikes, bins: the observation's phrases), each instance mapped onto the panorama, and the copies of
one object seen by overlapping cuts merged into one. Each object keeps the cut where it is most centred (its home)
and whether it reaches the edge of the capture (truncated: part of it was never photographed). Run in the sam3
Python (SIM2CHIP_SAM3_PYTHON):

    python instances.py --dir data/sphere_test/<place> --target car --movable person --movable bicycle

Writes <dir>/instances.npz (masks: packed bits of the panorama masks at OUT_SIZE, one row per object) and
<dir>/instances.json (per object: phrase, role, score, cuts that saw it, home cut, centre and angular extent in
degrees, area share of the sphere, truncated).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capture  # noqa: E402

OUT_SIZE = (2048, 1024)  # as depth.npz and stuff.npz
SAME_IOU = 0.3           # two cuts' instances overlapping this much on the panorama are one object
INSIDE = 0.6             # or one lies this much inside the other (a cut that sees only part of it)
MIN_PIXELS = 40          # smaller instances on the panorama are dropped (noise at a cut's edge)
EDGE_PX = 3              # a mask this close to a cut's side touches it


def segment(model, proc, image, phrase, threshold):
    inp = proc(images=image, text=phrase, return_tensors="pt").to("cuda")
    with torch.no_grad():
        res = model(**inp)
    r = proc.post_process_instance_segmentation(
        res, threshold=threshold, mask_threshold=0.5, target_sizes=inp.get("original_sizes").tolist())[0]
    return [(float(s), m.cpu().numpy().astype(bool)) for s, m in zip(r["scores"], r["masks"])]


def grow(m: np.ndarray, px: int) -> np.ndarray:
    """The mask grown by px pixels (square neighbourhood), without wrapping at the image's edges."""
    out = m.copy()
    for dy in range(-px, px + 1):
        for dx in range(-px, px + 1):
            out[max(dy, 0):m.shape[0] + min(dy, 0), max(dx, 0):m.shape[1] + min(dx, 0)] |= \
                m[max(-dy, 0):m.shape[0] + min(-dy, 0), max(-dx, 0):m.shape[1] + min(-dx, 0)]
    return out


def angular_box(idx: np.ndarray, W: int, H: int) -> dict:
    """Centre and extent (degrees) of panorama pixels, longitude measured around their circular mean."""
    y, x = np.divmod(idx, W)
    lon = (x + 0.5) / W * 360 - 180
    lat = 90 - (y + 0.5) / H * 180
    c = math.degrees(math.atan2(np.sin(np.radians(lon)).mean(), np.cos(np.radians(lon)).mean()))
    rel = (lon - c + 180) % 360 - 180
    return {"lon": round(c + (rel.max() + rel.min()) / 2, 2), "lat": round(float(lat.max() + lat.min()) / 2, 2),
            "width": round(float(rel.max() - rel.min()), 2), "height": round(float(lat.max() - lat.min()), 2)}


def merge(found: list[dict]) -> list[list[int]]:
    """Groups of the cuts' instances that are one object: same role, overlapping on the panorama."""
    parent = list(range(len(found)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            a, b = found[i], found[j]
            if a["role"] != b["role"] or a["cut"] == b["cut"]:
                continue
            inter = len(np.intersect1d(a["idx"], b["idx"], assume_unique=True))
            if not inter:
                continue
            if inter / (len(a["idx"]) + len(b["idx"]) - inter) > SAME_IOU or \
                    inter / min(len(a["idx"]), len(b["idx"])) > INSIDE:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(found)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--target", action="append", default=[], help="a target phrase (the spec's classes)")
    ap.add_argument("--movable", action="append", default=[], help="a movable thing's phrase (to erase too)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    info = json.load(open(os.path.join(args.dir, "sphere.json")))
    t0 = time.time()
    model = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")

    W, H = OUT_SIZE
    dirs = capture.pixel_rays(W, H)
    seen = np.asarray(Image.open(os.path.join(args.dir, "seen.png")).resize(OUT_SIZE, Image.Resampling.NEAREST)) > 127
    found = []
    for view in info["cuts"]:
        image = Image.open(os.path.join(args.dir, "cuts", view["file"])).convert("RGB")
        x, y, inside = capture.project(view, dirs)
        where = np.flatnonzero(inside)
        px, py = np.round(x.ravel()[where]).astype(int), np.round(y.ravel()[where]).astype(int)
        unseen = grow(capture.sample(seen.astype(np.uint8), capture.view_rays(view)) < 0.5, EDGE_PX)
        axis = capture.rotation(view["yaw_deg"], view["pitch_deg"])[:, 2]
        for role, phrases in (("target", args.target), ("movable", args.movable)):
            for phrase in phrases:
                for score, m in segment(model, proc, image, phrase, args.threshold):
                    idx = where[m[py, px]]
                    if len(idx) < MIN_PIXELS:
                        continue
                    ys, xs = np.nonzero(m)
                    side = xs.min() < EDGE_PX or xs.max() >= m.shape[1] - EDGE_PX or ys.min() < EDGE_PX or \
                        ys.max() >= m.shape[0] - EDGE_PX
                    centre = dirs.reshape(-1, 3)[idx].mean(0)
                    found.append({"role": role, "phrase": phrase, "score": score, "cut": view["file"],
                                  "idx": idx, "side": bool(side), "truncated": bool((m & unseen).any()),
                                  "centred": float(centre @ axis / max(np.linalg.norm(centre), 1e-9))})
        print(view["file"], "instances so far:", len(found), flush=True)

    objects, masks = [], []
    for group in merge(found):
        members = [found[i] for i in group]
        # A pixel belongs to the object when at least half of the cuts that see it say so.
        cover = np.zeros(W * H, np.int16)
        vote = np.zeros(W * H, np.int16)
        cuts = {m["cut"] for m in members}
        for view in info["cuts"]:
            if view["file"] in cuts:
                _, _, inside = capture.project(view, dirs)
                cover += inside.ravel()
        for m in members:
            vote[m["idx"]] += 1
        idx = np.flatnonzero((vote > 0) & (vote * 2 >= cover))
        if len(idx) < MIN_PIXELS:
            continue
        home = max(members, key=lambda m: (not m["side"], m["centred"]))
        phrase = max(members, key=lambda m: m["score"])["phrase"]
        objects.append({"phrase": phrase, "role": members[0]["role"], "score": round(max(m["score"] for m in members), 3),
                        "cuts": sorted(cuts), "home_cut": home["cut"], **angular_box(idx, W, H),
                        "area_share": round(len(idx) / (W * H), 6),
                        "truncated": any(m["truncated"] for m in members)})
        mask = np.zeros(W * H, bool)
        mask[idx] = True
        masks.append(np.packbits(mask))
    # A movable thing that is mostly a target (a van also called a car) is the target.
    keep = []
    for i, o in enumerate(objects):
        if o["role"] == "movable":
            mi = np.unpackbits(masks[i])[:W * H].astype(bool)
            if any(objects[j]["role"] == "target" and (mi & np.unpackbits(masks[j])[:W * H].astype(bool)).sum()
                   > INSIDE * mi.sum() for j in range(len(objects))):
                continue
        keep.append(i)
    objects = [{"id": k, **objects[i]} for k, i in enumerate(keep)]
    np.savez_compressed(os.path.join(args.dir, "instances.npz"), masks=np.stack([masks[i] for i in keep])
                        if keep else np.zeros((0, W * H // 8), np.uint8), size=np.array(OUT_SIZE))
    json.dump({"targets": args.target, "movables": args.movable, "objects": objects},
              open(os.path.join(args.dir, "instances.json"), "w"), indent=1)
    roles = {r: sum(o["role"] == r for o in objects) for r in ("target", "movable")}
    print(f"{len(found)} cut instances -> {len(objects)} objects {roles} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
