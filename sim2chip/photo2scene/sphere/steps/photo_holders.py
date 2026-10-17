"""S.6 of the sphere route, support: SAM 3's instances of what holds the targets up in the user's photos (the
holders the vision model named, support.py: "metal rail", "wooden post"), one mask per instance so context.py can
measure each one with MoGe's points and hang the targets on it as in the photo. Run in the sam3 Python:

    python photo_holders.py --photos <dir> --out <dir> --phrases phrases.json

phrases.json: {photo name: [the holders named for its targets]}: each photo looks for what holds its own targets.

Writes <out>/<photo stem>.npz: masks (n, H, W) at SIZE on the long side (as photo_depth.py), scores (n,) and
phrases (n,); overlapping instances of several phrases are kept once, the higher score.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor

SIZE = 1024
SAME = 0.6  # intersection over union above which two instances are one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrases", required=True, help="JSON file {photo name: [phrase, ...]}")
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    wanted = json.load(open(args.phrases))
    for name, phrases in sorted(wanted.items()):
        out = os.path.join(args.out, os.path.splitext(name)[0] + ".npz")
        if os.path.exists(out):
            continue
        img = ImageOps.exif_transpose(Image.open(os.path.join(args.photos, name))).convert("RGB")
        img.thumbnail((SIZE, SIZE))
        found = []
        for phrase in phrases:
            inp = proc(images=img, text=phrase, return_tensors="pt").to("cuda")
            with torch.no_grad():
                res = model(**inp)
            r = proc.post_process_instance_segmentation(res, threshold=args.threshold, mask_threshold=0.5,
                                                        target_sizes=inp.get("original_sizes").tolist())[0]
            for m, s in zip(r["masks"].cpu().numpy().astype(bool), r["scores"].cpu().numpy()):
                found.append((float(s), phrase, m))
        kept = []
        for s, phrase, m in sorted(found, key=lambda f: -f[0]):
            if m.sum() and all((m & k[2]).sum() / (m | k[2]).sum() < SAME for k in kept):
                kept.append((s, phrase, m))
        shape = (img.height, img.width)
        np.savez_compressed(out, masks=np.array([k[2] for k in kept]).reshape(-1, *shape),
                            scores=np.array([k[0] for k in kept], np.float32), phrases=np.array([k[1] for k in kept]))
        print(name, len(kept), "holders", flush=True)


if __name__ == "__main__":
    main()
