"""S.2 and S.3 of the sphere route, stuff: SAM 3 on every cut for each group of phrases (ground kinds, sky), the
union of a group's instances per cut, and back on the panorama as a vote of the cuts that see each pixel, each cut
weighted towards its centre (where it is least stretched). Run in the sam3 Python (SIM2CHIP_SAM3_PYTHON):

    python stuff.py --dir data/sphere_test/<place> --group road=road,street --group grass=grass \
        --group sky=sky --union ground=road,grass

Run again on the erased panorama (--pano background.jpg --out stuff_bg.npz) for the ground where the erased
objects stood, which the proxy and the placement need.

Writes <dir>/stuff.npz (or --out): one map per group and per union of groups (H, W, float16, the share of the cuts' weight
that called the pixel that group, OUT_SIZE) and seen_by (H, W, the cuts' total weight; 0 where no cut looks). A
union counts a cut's pixel once whichever of its groups called it, so road in one cut and sidewalk in the next
still make ground.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capture  # noqa: E402  (the sphere's geometry: numpy and PIL only)

OUT_SIZE = (2048, 1024)  # as depth.py's MERGE_SIZE: the maps line up pixel for pixel
CENTRE_POWER = 4         # weight of a cut's pixel: the cosine of its angle to the cut's axis, to this power


def segment(model, proc, image, phrase, threshold):
    inp = proc(images=image, text=phrase, return_tensors="pt").to("cuda")
    with torch.no_grad():
        res = model(**inp)
    r = proc.post_process_instance_segmentation(
        res, threshold=threshold, mask_threshold=0.5, target_sizes=inp.get("original_sizes").tolist())[0]
    return r["masks"].cpu().numpy().astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--group", action="append", required=True, help="name=phrase,phrase,...")
    ap.add_argument("--union", action="append", default=[], help="name=group,group,...")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--pano", help="cut this panorama in the place folder (the erased one, background.jpg) "
                    "instead of reading cuts/")
    ap.add_argument("--out", default="stuff.npz")
    args = ap.parse_args()
    split = lambda spec: (spec.split("=", 1)[0], [p.strip() for p in spec.split("=", 1)[1].split(",") if p.strip()])  # noqa: E731
    groups, unions = dict(map(split, args.group)), dict(map(split, args.union))
    info = json.load(open(os.path.join(args.dir, "sphere.json")))
    t0 = time.time()
    model = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")

    W, H = OUT_SIZE
    dirs = capture.pixel_rays(W, H)
    votes = {g: np.zeros((H, W), np.float32) for g in [*groups, *unions]}
    weight = np.zeros((H, W), np.float32)
    pano = np.asarray(Image.open(os.path.join(args.dir, args.pano)).convert("RGB")) if args.pano else None
    for view in info["cuts"]:
        if pano is None:
            image = Image.open(os.path.join(args.dir, "cuts", view["file"])).convert("RGB")
        else:
            image = Image.fromarray(np.clip(capture.sample(pano, capture.view_rays(view)) + 0.5, 0, 255)
                                    .astype(np.uint8))
        x, y, inside = capture.project(view, dirs)
        px, py = np.round(x[inside]).astype(int), np.round(y[inside]).astype(int)
        axis = capture.rotation(view["yaw_deg"], view["pitch_deg"])[:, 2]
        w = np.clip(dirs[inside] @ axis, 0, 1) ** CENTRE_POWER
        weight[inside] += w
        found, cut_masks = {}, {}
        for g, phrases in groups.items():
            m = np.zeros((image.height, image.width), bool)
            for phrase in phrases:
                for inst in segment(model, proc, image, phrase, args.threshold):
                    m |= inst
            votes[g][inside] += w * m[py, px]
            found[g], cut_masks[g] = round(float(m.mean()), 3), m
        for u, members in unions.items():
            m = np.any([cut_masks[g] for g in members], 0)
            votes[u][inside] += w * m[py, px]
        print(view["file"], "share of the cut:", found, flush=True)
    share = {g: (v / np.maximum(weight, 1e-6)).astype(np.float16) for g, v in votes.items()}
    np.savez_compressed(os.path.join(args.dir, args.out), seen_by=weight.astype(np.float16), **share)
    print(f"{len(info['cuts'])} cuts, {sum(map(len, groups.values()))} phrases in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
