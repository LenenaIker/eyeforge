"""S.5 of the sphere route, orient: which way each object faces, from Orient Anything V2 (Wang et al., NeurIPS 2025;
code CC-BY-4.0, checkpoint Viglong/OriAnyV2_ckpt) on its cutout. A box fitted to a silhouette (replica.py) cannot
tell a car seen at three quarters from one seen side-on, nor its front from its back; the model says both. Run in
the sam3 Python with the model's code and its two extra packages on the path:

    ORIANY_ROOT=<clone> ORIANY_DEPS=<scipy, einops> ORIANY_CKPT=<.pt> python orient.py --crops <dir> --out <json>

--crops holds RGBA cutouts (transparent outside the object), one PNG each. Writes {name: {"azimuth_deg",
"elevation_deg", "roll_deg", "symmetry", "confidence"}}: azimuth of the object's front as the model gives it
(calibrated against renders of known heading in replica.py), symmetry the number of equally likely fronts it fits
(0: no front, 1: one).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.environ["ORIANY_ROOT"]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.environ["ORIANY_DEPS"])
sys.modules.setdefault("rembg", type(sys)("rembg"))  # the background is already cut out: rembg is never called

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

os.chdir(ROOT)  # the model code reads assets relative to its folder
from utils.app_utils import preprocess_images, resize_foreground, val_fit_alpha  # noqa: E402
from vision_tower import VGGT_OriAny_Ref  # noqa: E402

FOREGROUND = 0.85  # the object's share of the square crop, as the model's own demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    model.load_state_dict(torch.load(os.environ["ORIANY_CKPT"], map_location="cpu"))
    model = model.eval().to("cuda")
    out = {}
    for p in sorted(Path(args.crops).glob("*.png")):
        img = resize_foreground(Image.open(p).convert("RGBA"), FOREGROUND)
        batch = preprocess_images([img], mode="pad").to(model.get_device()).unsqueeze(0)
        with torch.no_grad():
            enc = model(batch).view(1, -1).float()
        az = torch.sigmoid(enc[:, :360])
        out[p.stem] = {"azimuth_deg": int(torch.argmax(enc[0, :360])),
                       "elevation_deg": int(torch.argmax(enc[0, 360:540])) - 90,
                       "roll_deg": int(torch.argmax(enc[0, 540:900])) - 180,
                       "symmetry": int(val_fit_alpha(distribute=az.cpu().numpy())[0]),
                       "confidence": round(float(az.max()), 3)}
        print(p.stem, out[p.stem], flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
