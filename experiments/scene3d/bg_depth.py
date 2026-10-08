"""Step 2: MoGe-2 metric point maps of the photo and of the erased background (run in the `eyeforge-sam3d` env; MoGe's
own utils3d versions come from MOGE_DEPS, ahead of the env's, which SAM 3D needs as they are):

    python bg_depth.py --dir data/scene3d_test/ours/bottles

Writes depth_<name>.npz per image: points (H, W, 3, OpenCV camera: +X right, +Y down, +Z forward, metres),
depth (H, W), mask (H, W) and intrinsics (3x3, normalised by width and height).
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ.get('MOGE_DEPS', '/media/lenena-iker/Crucial/tools/moge_deps'))
sys.path.insert(0, os.environ.get('MOGE_ROOT', '/media/lenena-iker/Crucial/tools/MoGe'))
from moge.model.v2 import MoGeModel  # noqa: E402

CKPT = os.environ.get('MOGE_CKPT', '/media/lenena-iker/Crucial/tools/moge_ckpt/moge-2-vitl/model.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--only', choices=['photo', 'background'], help='one image (a room measures its things on the '
                    'photo before deciding what to erase)')
    args = ap.parse_args()
    model = MoGeModel.from_pretrained(CKPT).cuda().eval()
    for name in [args.only] if args.only else ('photo', 'background'):
        img = np.asarray(Image.open(os.path.join(args.dir, f'{name}.png')).convert('RGB'))
        x = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255
        with torch.no_grad():
            out = model.infer(x)
        res = {k: out[k].cpu().numpy() for k in ('points', 'depth', 'mask', 'intrinsics')}
        np.savez_compressed(os.path.join(args.dir, f'depth_{name}.npz'), **res)
        d = res['depth'][res['mask'] > 0]
        print(name, 'depth m: median %.3f, range %.3f-%.3f' % (np.median(d), d.min(), d.max()),
              'fov_x %.1f deg' % np.degrees(2 * np.arctan(0.5 / res['intrinsics'][0, 0])), flush=True)


if __name__ == '__main__':
    main()
