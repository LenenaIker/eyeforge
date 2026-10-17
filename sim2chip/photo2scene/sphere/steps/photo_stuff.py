"""S.6 of the sphere route, context: SAM 3's ground kinds and built surfaces on the user's photos, as stuff.py does on
a sphere's cuts (same phrases, so context.py's shares map onto a sphere's votes). Run in the sam3 Python:

    python photo_stuff.py --photos <dir> --names a.jpg,b.jpg --out <dir> --group road=road,street ...

Writes <out>/<photo stem>.npz: one boolean map per group at SIZE on the long side (the size of photo_depth.py).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor

SIZE = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", action="append", required=True, help="name=phrase,phrase,...")
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()
    groups = {g.split("=", 1)[0]: [p.strip() for p in g.split("=", 1)[1].split(",") if p.strip()] for g in args.group}
    os.makedirs(args.out, exist_ok=True)
    model = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    for name in args.names.split(","):
        out = os.path.join(args.out, os.path.splitext(name)[0] + ".npz")
        if os.path.exists(out):
            continue
        img = ImageOps.exif_transpose(Image.open(os.path.join(args.photos, name))).convert("RGB")
        img.thumbnail((SIZE, SIZE))
        maps = {}
        for g, phrases in groups.items():
            m = np.zeros((img.height, img.width), bool)
            for phrase in phrases:
                inp = proc(images=img, text=phrase, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    res = model(**inp)
                r = proc.post_process_instance_segmentation(res, threshold=args.threshold, mask_threshold=0.5,
                                                            target_sizes=inp.get("original_sizes").tolist())[0]
                for inst in r["masks"].cpu().numpy().astype(bool):
                    m |= inst
            maps[g] = m
        np.savez_compressed(out, **maps)
        print(name, {g: round(float(m.mean()), 3) for g, m in maps.items()}, flush=True)


if __name__ == "__main__":
    main()
