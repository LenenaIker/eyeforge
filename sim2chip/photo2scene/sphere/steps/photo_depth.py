"""S.6 of the sphere route, context: MoGe-2 metric point maps of the user's photos (context.py measures where the
targets stand on them). Run in the mesh Python (SIM2CHIP_SCENE_PYTHON), as bg_depth.py:

    python photo_depth.py --photos <dir> --names a.jpg,b.jpg --out <dir>

Writes <out>/<photo stem>.npz: points (H, W, 3, OpenCV camera, metres) and mask (H, W), at SIZE on the long side.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.environ.get("MOGE_DEPS", "/media/lenena-iker/Crucial/tools/moge_deps"))
sys.path.insert(0, os.environ.get("MOGE_ROOT", "/media/lenena-iker/Crucial/tools/MoGe"))
from moge.model.v2 import MoGeModel  # noqa: E402

CKPT = os.environ.get("MOGE_CKPT", "/media/lenena-iker/Crucial/tools/moge_ckpt/moge-2-vitl/model.pt")
SIZE = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = MoGeModel.from_pretrained(CKPT).cuda().eval()
    for name in args.names.split(","):
        out = os.path.join(args.out, os.path.splitext(name)[0] + ".npz")
        if os.path.exists(out):
            continue
        img = ImageOps.exif_transpose(Image.open(os.path.join(args.photos, name))).convert("RGB")
        img.thumbnail((SIZE, SIZE))
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().cuda() / 255
        with torch.no_grad():
            res = model.infer(x)
        np.savez_compressed(out, points=res["points"].cpu().numpy().astype(np.float32),
                            mask=res["mask"].cpu().numpy() > 0)
        print(name, flush=True)


if __name__ == "__main__":
    main()
