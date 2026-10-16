"""S.2 of the sphere route, depth: MoGe-2 on views of the working panorama, fused into one 360 distance map with
MoGe's own panorama merge (log-distance gradients solved by least squares, coarse to fine). The fusion keeps the
shape but not the scale, so the merged map is brought back to the median of the views' metric distances; the
ground (geometry.py) sets the final scale. What was not captured is masked out of every view, so the black canvas
never reaches the fusion. Run in the mesh Python (SIM2CHIP_SCENE_PYTHON) with MOGE_ROOT, MOGE_DEPS and MOGE_CKPT:

    python depth.py --dir data/sphere_test/<place> [--image background.jpg --out depth_bg.npz]

Run on the panorama (pano.jpg) for the ground and the scale, and again on the erased one (S.3's background.jpg)
for the proxy, so the erased objects do not stay behind as invisible geometry.

Writes <dir>/depth.npz (or --out): distance (H, W, metres along each pixel's ray, MERGE_SIZE) and valid (H, W: a surface was
seen there, not sky, and the sphere was captured).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

for var in ("MOGE_DEPS", "MOGE_ROOT"):  # MoGe's own utils3d first: the env's is the one SAM 3D needs
    if not os.environ.get(var):
        sys.exit(f"{var} is not set (see .env)")
    sys.path.insert(0, os.environ[var])
try:
    import utils3d_moge as utils3d  # the version MoGe pins
except ImportError:
    import utils3d
from moge.model.v2 import MoGeModel  # noqa: E402
from moge.utils.panorama import (get_panorama_cameras, merge_panorama_depth, spherical_uv_to_directions,  # noqa: E402
                                 split_panorama_image)

VIEW_SIZE = 512             # MoGe's panorama views: 90 degrees, one per icosahedron vertex (12)
VIEW_FOV_DEG = 90.0
MERGE_SIZE = (2048, 1024)   # half the working panorama: about 5.7 px per degree
BATCH = 4
MIN_VIEW_SEEN = 0.05        # views with less of the sphere captured are not inferred


def view_scale(pano_distance, pano_valid, distances, masks, extrinsics, intrinsics):
    """Median ratio of the views' metric distances to the merged map, over the pixels both saw."""
    H, W = pano_distance.shape
    dirs = spherical_uv_to_directions(utils3d.np.uv_map(H, W))
    logs = []
    for dist, mask, ext, intr in zip(distances, masks, extrinsics, intrinsics):
        uv, z = utils3d.np.project_cv(dirs, extrinsics=ext, intrinsics=intr)
        inside = (z > 0) & (uv > 0).all(-1) & (uv < 1).all(-1) & pano_valid
        px = np.clip((uv[inside] * dist.shape[::-1]).astype(int), 0, np.array(dist.shape[::-1]) - 1)
        ok = mask[px[:, 1], px[:, 0]]
        logs.append(np.log(dist[px[ok, 1], px[ok, 0]]) - np.log(pano_distance[inside][ok]))
    return float(np.exp(np.median(np.concatenate(logs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--image", default="pano.jpg")
    ap.add_argument("--out", default="depth.npz")
    args = ap.parse_args()
    t0 = time.time()
    pano = np.asarray(Image.open(os.path.join(args.dir, args.image)).convert("RGB"))
    seen = np.asarray(Image.open(os.path.join(args.dir, "seen.png")))
    extrinsics, intrinsics = get_panorama_cameras()
    views = split_panorama_image(pano, extrinsics, intrinsics, VIEW_SIZE)
    seen_views = [v > 250 for v in split_panorama_image(seen, extrinsics, intrinsics, VIEW_SIZE)]
    keep = [i for i, s in enumerate(seen_views) if s.mean() >= MIN_VIEW_SEEN]

    model = MoGeModel.from_pretrained(os.environ["MOGE_CKPT"]).cuda().eval()
    distances, masks = [], []
    for k in range(0, len(keep), BATCH):
        idx = keep[k:k + BATCH]
        x = torch.tensor(np.stack([views[i] for i in idx]) / 255, dtype=torch.float32, device="cuda").permute(0, 3, 1, 2)
        with torch.no_grad():
            out = model.infer(x, fov_x=torch.full((len(idx),), VIEW_FOV_DEG, device="cuda"), apply_mask=False)
        distances += list(out["points"].norm(dim=-1).float().cpu().numpy())
        masks += [m & seen_views[i] for m, i in zip(out["mask"].cpu().numpy() > 0, idx)]
    ext, intr = [extrinsics[i] for i in keep], [intrinsics[i] for i in keep]
    print(f"{len(keep)} of {len(views)} views inferred in {time.time() - t0:.0f} s", flush=True)

    distance, valid = merge_panorama_depth(*MERGE_SIZE, distances, masks, ext, intr)
    seen_small = np.asarray(Image.fromarray(seen).resize(MERGE_SIZE, Image.Resampling.NEAREST)) > 127
    valid &= seen_small
    scale = view_scale(distance, valid, distances, masks, ext, intr)
    distance = (distance * scale).astype(np.float32)
    np.savez_compressed(os.path.join(args.dir, args.out), distance=distance, valid=valid)
    d = distance[valid]
    print(f"merged in {time.time() - t0:.0f} s; scale to the views {scale:.3f}; distance m: median {np.median(d):.2f}, "
          f"p5 {np.percentile(d, 5):.2f}, p95 {np.percentile(d, 95):.2f}; valid {valid.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
