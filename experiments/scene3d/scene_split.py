"""Step 1.3a of the scene flow for a room, after flow_things.py: what is scene and what is erased, by measured
size instead of by which words SAM 3 happens to find. MoGe-2's metric points of the photo (bg_depth.py --only
photo) and the floor plane give each thing's height above the floor and its size along it. The rule:

- one of the user's classes (flow_things.py --targets): erased, the generator places it again with a label;
- big (longest side >= --big metres): scene, built as a mesh (a desk, a bed, a shelf, a radiator);
- fixed to a wall and small (a switch, a socket): scene, left in the texture;
- the rest: erased, and placed again at random by the generator as distractors, so the model cannot learn
  where the photo's own small things stand (user, 2026-10-08).

Masks from segment_all.py that no thing covers, if present, are drawn as missed by the VLM. Run in the
`teacher-sam3` env:

    python experiments/scene3d/scene_split.py --dir data/scene3d_test/flow/user_desk

Writes <dir>/split.json (floor plane, per thing: name, kind, sizes), split.jpg (green scene mesh, blue scene
texture, red erased, yellow missed) and objects.npz (masks, phrases: what bg_prepare.py --object-masks erases).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

COLOURS = {'mesh': (40, 200, 80), 'texture': (60, 120, 255), 'erase': (255, 50, 50), 'missed': (255, 220, 0)}


def fit_floor(pts, iters=800, seed=0):
    """RANSAC plane n.p + d = 0 with n pointing to the camera (up), tolerance 1 % of the median distance."""
    rng = np.random.default_rng(seed)
    tol = 0.01 * float(np.median(np.linalg.norm(pts, axis=1)))
    best = None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(pts @ n - n @ a) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    p = pts[best]
    n = np.linalg.svd(p - p.mean(0), full_matrices=False)[2][2]
    d = -n @ p.mean(0)
    return (-n, -d) if d < 0 else (n, d)


def measure(points, mask, n, d):
    """Height above the floor of the bottom and the top, and the two sizes along the floor, in metres."""
    m = mask
    if mask.sum() > 400:  # off the edges, where depth bleeds into what stands behind
        t = torch.from_numpy(~mask)[None, None].float()
        m = F.max_pool2d(t, 5, 1, 2)[0, 0].numpy() < 0.5
    p = points[m]
    p = p[np.isfinite(p).all(1)]
    if len(p) < 20:
        return None
    h = p @ n + d
    flat = p - np.outer(h, n)
    flat = flat - flat.mean(0)
    axes = np.linalg.svd(flat, full_matrices=False)[2][:2]
    along = flat @ axes.T
    size = np.percentile(along, 97, 0) - np.percentile(along, 3, 0)
    return {'bottom': float(np.percentile(h, 3)), 'top': float(np.percentile(h, 97)),
            'width': float(size.max()), 'depth': float(size.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--big', type=float, default=0.6, help='longest side that makes a thing scene, metres')
    args = ap.parse_args()
    d = args.dir
    photo = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
    w, h = photo.size
    things = json.load(open(os.path.join(d, 'things.json')))['things']
    masks = np.load(os.path.join(d, 'things.npz'))['masks'].reshape(-1, h, w)

    dp = np.load(os.path.join(d, 'depth_photo.npz'))
    pts = dp['points']
    ground = np.load(os.path.join(d, 'ground.npz'))['masks'].any(0) & (dp['mask'] > 0)
    g = pts[ground]
    g = g[np.isfinite(g).all(1)]
    g = g[np.random.default_rng(0).choice(len(g), min(len(g), 60000), replace=False)]
    n, dd = fit_floor(g)

    rows = []
    for t, m in zip(things, masks):
        s = measure(pts, m, n, dd)
        if t.get('target') or s is None:
            kind = 'erase'
        elif max(s['width'], s['top'] - s['bottom']) >= args.big:
            kind = 'mesh'
        elif t.get('on') == 'wall':
            kind = 'texture'
        else:
            kind = 'erase'
        rows.append({'name': t['name'], 'on': t.get('on'), 'kind': kind, 'box': t['box'],
                     **({k: round(v, 2) for k, v in s.items()} if s else {})})

    covered = masks.any(0) | ground
    missed = []
    path = os.path.join(d, 'all_masks.npz')
    if os.path.exists(path):
        am = np.load(path)['masks']
        for i, m in enumerate(am):
            # Walls and ceiling are huge; things are what the VLM was asked for.
            if m.mean() < 0.08 and (m & ~covered).sum() > 0.7 * m.sum():
                missed.append(i)
    am_missed = np.load(path)['masks'][missed] if missed else np.zeros((0, h, w), bool)
    erase = [i for i, r in enumerate(rows) if r['kind'] == 'erase']
    np.savez_compressed(os.path.join(d, 'objects.npz'), masks=masks[erase].reshape(-1, h, w),
                        phrases=np.array([rows[i]['name'] for i in erase]))

    a = np.asarray(photo).astype(float) * 0.45
    ph = np.asarray(photo).astype(float)
    order = np.argsort([-m.sum() for m in masks])
    for i in order:
        a[masks[i]] = ph[masks[i]] * 0.45 + np.array(COLOURS[rows[i]['kind']]) * 0.55
    for m in am_missed:
        a[m] = ph[m] * 0.45 + np.array(COLOURS['missed']) * 0.55
    im = Image.fromarray(a.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for r, m in zip(rows, masks):
        ys, xs = np.nonzero(m)
        if len(xs):
            size = f" {max(r.get('width', 0), r.get('top', 0) - r.get('bottom', 0)):.2f}m" if 'top' in r else ''
            dr.text((xs.mean() - 20, ys.mean()), r['name'] + size, fill='white', stroke_width=2, stroke_fill='black')
    im.save(os.path.join(d, 'split.jpg'), quality=92)
    json.dump({'floor': {'normal': n.tolist(), 'offset': float(dd)}, 'things': rows, 'missed_masks': missed},
              open(os.path.join(d, 'split.json'), 'w'), indent=1)
    print(json.dumps({k: [r['name'] for r in rows if r['kind'] == k] for k in ('mesh', 'texture', 'erase')}))


if __name__ == '__main__':
    main()
