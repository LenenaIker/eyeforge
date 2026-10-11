"""Surface route, step 2: each photo re-framed in 3D (roadmap B.5). MoGe-2 turns the photo into a depth mesh coloured
by itself; new cameras look at the same ground from other angles, heights and turns within the measured camera range,
and only views that stay inside the rebuilt part are kept. A second render of the mesh coloured by pixel coordinates
says which photo pixel each view pixel shows, so the photo's surface masks (surf_masks.py) carry over to the view.
Run in the scene env (MoGe and pyrender):

    PYOPENGL_PLATFORM=egl python surf_views.py --photos photos.txt --masks masks/ --out views/ --views 4 \
        [--pitch 50,89] [--height 0.4,1.4] [--seed 0]

Writes <out>/<stem>_vNN.png and <out>/views.json: {"photos": {stem: {"feats", "pitch_deg", "height_m"}},
"views": [{"file", "source", "feats", "pitch_deg", "height_m"}]}, where feats lists, per mask phrase, the share of the
image it covers, then the share of the middle half (the same numbers for the photo and its views).
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps

sys.path.insert(0, os.environ.get('MOGE_DEPS', '/media/lenena-iker/Crucial/tools/moge_deps'))
sys.path.insert(0, os.environ.get('MOGE_ROOT', '/media/lenena-iker/Crucial/tools/MoGe'))
from moge.model.v2 import MoGeModel  # noqa: E402

from bg_build import Renderer, fit_plane, grid_mesh, look_at_cv  # noqa: E402
from bg_depth import CKPT  # noqa: E402

SIDE = 768        # long side the photo is rebuilt and the views rendered at
MAX_EMPTY = 0.04  # largest share of a view outside the rebuilt mesh (filled by inpainting)
MAX_TURN = 25     # degrees a view turns about the vertical
MAX_TILT = 10     # degrees a view's pitch moves from the photo's


def feats(masks, where=None):
    """Per phrase, the share of the pixels it covers, then the share in the middle half; where: valid pixels."""
    m = masks if where is None else masks & where[None]
    n = where.sum() if where is not None else masks[0].size
    h, w = masks.shape[1:]
    mid = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    nm = where[mid].sum() if where is not None else masks[0][mid].size
    return [float(x) for x in m.sum((1, 2)) / max(n, 1)] + \
           [float(x) for x in m[:, mid[0], mid[1]].sum((1, 2)) / max(nm, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', type=int, default=4)
    ap.add_argument('--pitch', default=None, help='lo,hi: the measured pitch range, degrees (90 straight down)')
    ap.add_argument('--height', default=None, help='lo,hi: the measured camera height range, metres; MoGe\'s scale '
                    'differs (a lawn at 1 m came out at 2.5 m), so only its spread is used, about each photo\'s height')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    plo, phi = map(float, args.pitch.split(',')) if args.pitch else (0.0, 90.0)
    hlo, hhi = map(float, args.height.split(',')) if args.height else (0.0, 1e9)
    os.makedirs(args.out, exist_ok=True)
    model = MoGeModel.from_pretrained(CKPT).cuda().eval()
    out = {'photos': {}, 'views': []}
    for path in [line.strip() for line in open(args.photos) if line.strip()]:
        stem = os.path.splitext(os.path.basename(path))[0]
        masks = np.load(os.path.join(args.masks, stem + '.npz'))['masks']
        photo = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        photo.thumbnail((SIDE, SIDE))
        img = np.asarray(photo)
        h, w = img.shape[:2]
        with torch.no_grad():
            res = model.infer(torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255)
        pts, valid = res['points'].cpu().numpy(), res['mask'].cpu().numpy() > 0
        k = res['intrinsics'].cpu().numpy()
        fx, fy, cx, cy = k[0, 0] * w, k[1, 1] * h, k[0, 2] * w, k[1, 2] * h
        sel = pts[valid][::7]
        # Grass and gravel are rough: the plane takes points within 2 % of the distance (fit_plane's 4 mm is for a
        # table seen from 0.5 m).
        n, d, inl = fit_plane(sel, tol=0.02 * float(np.median(sel[:, 2])))
        pitch0 = math.degrees(math.asin(min(1.0, abs(float(n[2])))))
        out['photos'][stem] = {'feats': feats(masks), 'pitch_deg': round(pitch0, 1), 'height_m': round(float(d), 3)}
        if inl < 0.5:  # not one ground: rough terrain or a wall; no view is trusted
            print(stem, f'no single ground ({inl:.0%} of points on the plane)', flush=True)
            continue
        mesh = grid_mesh(pts, img, valid, step=2, max_jump=0.06)
        ys, xs = np.mgrid[0:h, 0:w]
        coord = np.stack([xs * 255 // max(w - 1, 1), ys * 255 // max(h - 1, 1), np.full_like(xs, 255)], -1).astype(np.uint8)
        cmesh = grid_mesh(pts, coord, valid, step=2, max_jump=0.06)
        rend = Renderer(w, h, fx, fy, cx, cy)
        # The ground point the photo looks at: its middle ray hits the plane there.
        t = -d / float(np.array([0, 0, 1.0]) @ n)
        pivot = np.array([0, 0, 1.0]) * (t if t > 0 else float(np.median(sel[:, 2])))
        rel = -pivot  # the camera from the pivot
        up_h = float(rel @ n)  # height above the plane
        horiz = rel - up_h * n
        made = tries = 0
        while made < args.views and tries < 30 * args.views:
            tries += 1
            turn = math.radians(rng.uniform(-MAX_TURN, MAX_TURN))
            pitch = float(np.clip(rng.uniform(plo, phi), pitch0 - MAX_TILT, min(89.0, pitch0 + MAX_TILT)))
            height = d * float(np.clip(rng.uniform(hlo, hhi) / math.sqrt(hlo * hhi), 0.75, 1.3)) if args.height \
                else d * rng.uniform(0.8, 1.2)
            rot = trimesh.transformations.rotation_matrix(turn, n)[:3, :3]
            hdir = rot @ (horiz / np.linalg.norm(horiz) if np.linalg.norm(horiz) > 1e-6 else np.array([0, 1.0, 0]))
            shift = (rng.uniform(-0.15, 0.15, 3) * height)
            target = pivot + shift - (shift @ n) * n
            eye = target + n * height + hdir * height / math.tan(math.radians(max(pitch, 1.0)))
            up = rot @ np.array([0, -1.0, 0])
            pose = look_at_cv(eye, target, up)
            roll = trimesh.transformations.rotation_matrix(math.radians(rng.normal(0, 3)), pose[:3, 2])
            pose[:3, :3] = roll[:3, :3] @ pose[:3, :3]
            col, dep, _ = rend.render(pose, mesh, [], n)
            empty = dep == 0
            if empty.mean() > MAX_EMPTY:
                continue
            col = cv2.inpaint(col, empty.astype(np.uint8), 5, cv2.INPAINT_TELEA)
            cmap = rend.render(pose, cmesh, [], n)[0]
            ok = (cmap[..., 2] > 128) & ~empty
            mh, mw = masks.shape[1:]
            src = masks[:, np.minimum(cmap[..., 1].astype(int) * mh // 256, mh - 1),
                        np.minimum(cmap[..., 0].astype(int) * mw // 256, mw - 1)]
            name = f'{stem}_v{made:02d}.png'
            Image.fromarray(col).save(os.path.join(args.out, name))
            out['views'].append({'file': name, 'source': stem, 'feats': feats(src, ok), 'pitch_deg': round(pitch, 1),
                                 'height_m': round(height, 3)})
            made += 1
        rend.r.delete()  # one GL context at a time: the next photo's opens its own
        print(stem, f'pitch {pitch0:.0f} deg, height {d:.2f} m, {made} views in {tries} tries', flush=True)
    with open(os.path.join(args.out, 'views.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
