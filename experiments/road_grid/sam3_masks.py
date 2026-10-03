"""SAM 3 masks of everything one can walk, ride or drive on, for the 3x3 road grid test. Env: SIM2CHIP_SAM3_PYTHON.

    python experiments/road_grid/sam3_masks.py PHOTOS OUT [--side 1008] [--phrases "trail,footpath"]

Every photo is shrunk to `side`, segmented once per phrase, and the union of each phrase's instances (score over
0.3) is saved as OUT/<phrase>/<photo>.png (0 or 255), so the cell rule can be changed without running SAM 3 again.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor

PHRASES = ["road", "sidewalk", "bike lane", "footpath", "dirt path", "crosswalk", "parking lot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--side", type=int, default=1008)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--phrases", default=",".join(PHRASES), help="comma list, English")
    args = ap.parse_args()
    model = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    photos = sorted(args.photos.glob("*.jpg"))
    for i, p in enumerate(photos):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((args.side, args.side))
        for phrase in args.phrases.split(","):
            dst = args.out / phrase.replace(" ", "_") / f"{p.stem}.png"
            if dst.exists():
                continue
            inp = proc(images=im, text=phrase, return_tensors="pt").to("cuda")
            with torch.no_grad():
                res = model(**inp)
            r = proc.post_process_instance_segmentation(res, threshold=args.threshold, mask_threshold=0.5,
                                                        target_sizes=inp.get("original_sizes").tolist())[0]
            mask = np.zeros((im.height, im.width), bool)
            for m in r["masks"]:
                mask |= m.cpu().numpy().astype(bool)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(dst)
        print(f"{i + 1}/{len(photos)} {p.name}", flush=True)


if __name__ == "__main__":
    main()
