"""MoGe's depth for every frame of the GEN3C clips of a site, for gs_train.py (its weak depth hint where the cache
had a hole, scaled there to our proxy). Run in the SIM2CHIP_SCENE_PYTHON env (torch, MoGe, imageio):

    python experiments/sphere_walk/gs_moge.py --dir data/sphere_test/<place> --clips N01

Writes <clip>_moge.npz next to each clip video: depth (frames, H/2, W/2) float16, relative (0 where MoGe has none)."""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("MOGE_DEPS", "/media/lenena-iker/Crucial/tools/moge_deps"))
sys.path.insert(0, os.environ.get("MOGE_ROOT", "/media/lenena-iker/Crucial/tools/MoGe"))
from moge.model.v2 import MoGeModel  # noqa: E402

MOGE = os.environ.get("MOGE_CKPT", "/media/lenena-iker/Crucial/tools/moge_ckpt/moge-2-vitl/model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--clips", nargs="+", default=["N01"], help="folders under walk with paths.json and the videos")
    args = ap.parse_args()
    w = Path(args.dir) / "walk"
    moge = MoGeModel.from_pretrained(MOGE).cuda().eval()
    for folder in args.clips:
        for p in json.loads((w / folder / "paths.json").read_text())["paths"]:
            vid = w / folder / f"{p['name']}.mp4"
            out = vid.with_name(p["name"] + "_moge.npz")
            if not vid.exists() or out.exists():
                continue
            deps = []
            for img in iio.imiter(vid):
                x = torch.as_tensor(img, device="cuda").permute(2, 0, 1).float() / 255
                with torch.no_grad():
                    m = moge.infer(x, fov_x=p["view"]["fov_deg"])
                dep = torch.nan_to_num(m["depth"], nan=0, posinf=0) * m["mask"]
                deps.append(F.interpolate(dep[None, None], scale_factor=0.5, mode="nearest")[0, 0].cpu().numpy())
            np.savez_compressed(out, depth=np.stack(deps).astype(np.float16))
            print(out.name, len(deps), flush=True)


if __name__ == "__main__":
    main()
