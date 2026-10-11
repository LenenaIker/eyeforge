"""Surface route, step 2: new borders between the user's surfaces (roadmap B.5, extrapolation). Two training photos of
different classes become one flat ground: each fills one side of a random wavy line, joined by a soft seam, mirrored
past its edges so the camera never runs off it; a camera within the measured range looks down at it. Run in the scene
env (pyrender):

    PYOPENGL_PLATFORM=egl python surf_mix.py --photos photos.txt --labels labels.json --out mix/ \
        --views 30 [--pitch 50,89] [--fov 60] [--seed 0]

labels.json maps each photo's stem to its class. Writes <out>/mix_NNN.png and <out>/mix.json: [{"file", "a", "b",
"pitch_deg"}].
"""
import argparse
import json
import math
import os

import numpy as np
import trimesh
from PIL import Image, ImageOps

from bg_build import Renderer, look_at_cv

SIDE = 768
SEAM_PX = 6  # the seam's soft width in the texture


def joined(a, b, rng):
    """Each pixel's weight for a (1) against b (0) across a random wavy line, with a soft seam. The line passes anywhere
    from a tenth to nine tenths of the way across: a lawn's border photos are mostly grass with a strip of pavement
    to one side, and lines through the middle made halves that looked like neither class."""
    h, w = a.shape[:2]
    ang = rng.uniform(0, math.pi)
    cx, cy = w * rng.uniform(0.1, 0.9), h * rng.uniform(0.1, 0.9)
    ys, xs = np.mgrid[0:h, 0:w]
    along = (xs - cx) * math.cos(ang) + (ys - cy) * math.sin(ang)
    across = -(xs - cx) * math.sin(ang) + (ys - cy) * math.cos(ang)
    wave = rng.uniform(0, 0.05) * w * np.sin(along / w * rng.uniform(2, 8) + rng.uniform(0, 6.3))
    dist = across - wave
    weight = np.clip(0.5 + dist / (2 * SEAM_PX), 0, 1)[..., None]
    return weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', type=int, default=30)
    ap.add_argument('--pitch', default=None)
    ap.add_argument('--fov', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    plo, phi = map(float, args.pitch.split(',')) if args.pitch else (70.0, 89.0)
    labels = json.load(open(args.labels))
    paths = {os.path.splitext(os.path.basename(p))[0]: p.strip() for p in open(args.photos) if p.strip()}
    stems = [s for s in paths if s in labels]
    if len(set(labels[s] for s in stems)) < 2:
        raise SystemExit('two classes are needed for a border')
    os.makedirs(args.out, exist_ok=True)
    first = ImageOps.exif_transpose(Image.open(paths[stems[0]])).convert('RGB')
    first.thumbnail((SIDE, SIDE))
    w, h = first.size
    fx = w / 2 / math.tan(math.radians(args.fov) / 2)
    rend = Renderer(w, h, fx, fx, w / 2, h / 2)
    out = []
    for k in range(args.views):
        a = stems[rng.integers(len(stems))]
        b = next(s for s in rng.permutation(stems) if labels[s] != labels[a])
        imgs = []
        for s in (a, b):
            im = ImageOps.exif_transpose(Image.open(paths[s])).convert('RGB').resize((w, h), Image.BILINEAR)
            if rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            imgs.append(np.asarray(im, np.float32))
        weight = joined(imgs[0], imgs[1], rng)
        tex = (imgs[0] * weight + imgs[1] * (1 - weight)).astype(np.uint8)
        # Mirrored past its edges (3 x 3), so a tilted or turned camera still sees ground; the middle is the joined pair.
        tiles = np.concatenate([np.concatenate([tex[::-1, ::-1], tex[::-1], tex[::-1, ::-1]], 1),
                                np.concatenate([tex[:, ::-1], tex, tex[:, ::-1]], 1),
                                np.concatenate([tex[::-1, ::-1], tex[::-1], tex[::-1, ::-1]], 1)], 0)
        # The ground: the joined pair spans what the camera would see from height 1 m straight down.
        span_x = 2 * math.tan(math.radians(args.fov) / 2)
        span_y = span_x * h / w
        X, Y = 3 * span_x / 2, 3 * span_y / 2
        def plane(image, step=3):
            """The ground as a grid coloured per vertex (a texture came out dark in the unlit render)."""
            img = image[::step, ::step]
            gh, gw = img.shape[:2]
            gy, gx = np.mgrid[0:gh, 0:gw]
            verts = np.stack([gx / (gw - 1) * 2 * X - X, Y - gy / (gh - 1) * 2 * Y, np.zeros_like(gx, float)], -1)
            idx = np.arange(gh * gw).reshape(gh, gw)
            a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
            faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
            m = trimesh.Trimesh(verts.reshape(-1, 3), faces, process=False)
            m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.concatenate(
                [img.reshape(-1, 3), np.full((gh * gw, 1), 255, np.uint8)], 1))
            return m
        n = np.array([0, 0, 1.0])
        pitch = float(rng.uniform(plo, phi))
        height = float(rng.uniform(0.85, 1.0))
        turn = rng.uniform(0, 2 * math.pi)
        hdir = np.array([math.cos(turn), math.sin(turn), 0])
        target = np.array([rng.uniform(-0.1, 0.1) * span_x, rng.uniform(-0.1, 0.1) * span_y, 0])
        eye = target + n * height + hdir * height / math.tan(math.radians(max(pitch, 1.0)))
        pose = look_at_cv(eye, target, -hdir if pitch < 89 else np.array([0, 1.0, 0]))
        col, dep, _ = rend.render(pose, plane(tiles), [], n)
        if (dep == 0).mean() > 0.01:
            continue
        name = f'mix_{k:03d}.png'
        Image.fromarray(col).save(os.path.join(args.out, name))
        out.append({'file': name, 'a': a, 'b': b, 'pitch_deg': round(pitch, 1)})
    with open(os.path.join(args.out, 'mix.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f'{len(out)} joined grounds of {args.views}', flush=True)


if __name__ == '__main__':
    main()
