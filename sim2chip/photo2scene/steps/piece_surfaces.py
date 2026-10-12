"""Surfaces of the place in one photo, for the scene built in pieces (piece_build.py): SAM 3 masks of the walls,
the floor and the windows, on the photo that bg_prepare.py saved. Run in the `teacher-sam3` env:

    python piece_surfaces.py --dir data/scene3d_test/ours/office_pieces

Writes <dir>/surfaces.npz: masks (one per instance, photo size) and kinds ("wall", "floor" or "window").
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from transformers import Sam3Model, Sam3Processor

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from bg_prepare import segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--min-area', type=float, default=0.01, help='smallest mask kept, share of the image')
    args = ap.parse_args()
    photo = Image.open(os.path.join(args.dir, 'photo.png')).convert('RGB')
    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    masks, kinds = [], []
    for kind in ('wall', 'floor', 'window'):
        for _, m in segment(model, proc, photo, kind, args.threshold):
            # A window lies in a wall (its light goes in the room, piece_build.py): only windows exclude windows.
            same = [k for k, kk in zip(masks, kinds) if kind != 'window' or kk == 'window']
            if m.mean() >= args.min_area and not any((m & k).sum() / max(m.sum(), 1) > 0.5 for k in same):
                masks.append(m)
                kinds.append(kind)
    np.savez_compressed(os.path.join(args.dir, 'surfaces.npz'), masks=np.array(masks, bool).reshape(-1, *photo.size[::-1]),
                        kinds=np.array(kinds))
    print(json.dumps({'surfaces': [f'{k} {m.mean():.2f}' for k, m in zip(kinds, masks)]}))


if __name__ == '__main__':
    main()
