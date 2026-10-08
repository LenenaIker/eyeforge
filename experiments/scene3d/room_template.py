"""Step 3.2 of the scene flow for a room (route indoor_floor): the place from phase 2 is already closed
(piece_build.py --indoor builds the walls the photo does not show and a ceiling), so creating the place is only
new cameras inside it. Each camera stands at a person's or a robot's height, clear of the walls and the
furniture, and looks at one of the photo's pieces of furniture or along a wall, slightly down. Run in the
`trellis2` env:

    python experiments/scene3d/room_template.py --scene data/scene3d_test/flow/home_office/place \
        --out data/scene3d_test/flow/home_office/extended --views 16

Writes <out>/isaac/scene.json and meshes.npz (view 0 is still the photo camera) and <out>/photo.png.
"""
import argparse
import json
import os
import shutil

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from crossroad import look


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='the env_add.py place of a closed room')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', type=int, default=12)
    ap.add_argument('--clearance', type=float, default=0.5, help='metres from a camera to a wall or a mesh')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src = os.path.join(args.scene, 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = dict(np.load(os.path.join(src, 'meshes.npz')))
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    floor = np.asarray(scene['pieces'][0]['points'])
    level = float(np.mean(floor @ up))
    ceiling = max(float(np.max(np.asarray(p['points']) @ up)) for p in scene['pieces'])
    # Points over every surface but the floor and the ceiling: walls and furniture a camera must keep away from.
    pts, centres = [], []
    for e in scene['env']:
        m = trimesh.Trimesh(arr[f"obj{e['index']}_points"], arr[f"obj{e['index']}_faces"], process=False)
        s = trimesh.sample.sample_surface(m, 4000, seed=0)[0]
        pts.append(s)
        centres.append(s.mean(0))
    for p in scene['pieces'][1:]:
        c = np.asarray(p['points'])
        if np.ptp(c @ up) < 0.1:  # the ceiling
            continue
        uv = rng.uniform(0, 1, (600, 2))
        pts.append(c[0] + uv[:, :1] * (c[1] - c[0]) + uv[:, 1:] * (c[3] - c[0]))
    tree = cKDTree(np.concatenate(pts))
    # The middle of each wall: a target for a few views along the room. Mostly the furniture, though: on
    # home_office (2026-10-08) half the views aimed at bare walls, a plain backdrop the VLM still passed.
    walls = [np.asarray(p['points']).mean(0) for p in scene['pieces'][1:]
             if np.ptp(np.asarray(p['points']) @ up) >= 0.1]

    v0 = scene['views'][0]
    views, tries = [v0], 0
    top = min(ceiling - level - 0.25, 1.8)
    while len(views) <= args.views and tries < 200 * args.views:
        tries += 1
        a, b = rng.uniform(0.05, 0.95, 2)  # the floor quad: corners 0, 1, 2, 3 in order
        base = (1 - a) * (1 - b) * floor[0] + a * (1 - b) * floor[1] + a * b * floor[2] + (1 - a) * b * floor[3]
        pos = base + rng.uniform(0.4, max(top, 0.5)) * up
        if tree.query(pos)[0] < args.clearance:
            continue
        pool = walls if (rng.random() < 0.2 or not centres) else centres
        aim = pool[rng.integers(len(pool))] + rng.normal(0, 0.2, 3) - pos
        if np.linalg.norm(aim - (aim @ up) * up) < 1.0:  # too close to frame anything
            continue
        fwd = aim / np.linalg.norm(aim)
        if fwd @ up > 0.3:  # mostly ceiling
            continue
        # The view must not start inside furniture: march along the ray for the first 0.8 m.
        if min(tree.query(pos + t * fwd)[0] for t in (0.2, 0.4, 0.6, 0.8)) < 0.08:
            continue
        lamp = v0.get('lamp')
        views.append({'pose': look(pos, fwd, up).tolist(), 'objects': list(v0['objects']), 'lamp': lamp})
    os.makedirs(os.path.join(args.out, 'isaac'), exist_ok=True)
    scene['views'] = views
    json.dump(scene, open(os.path.join(args.out, 'isaac', 'scene.json'), 'w'))
    np.savez_compressed(os.path.join(args.out, 'isaac', 'meshes.npz'), **arr)
    shutil.copy(os.path.join(args.scene, 'photo.png'), args.out)
    print(json.dumps({'views': len(views), 'tries': tries, 'room_height_m': round(ceiling - level, 2),
                      'meshes': len(scene['env'])}))


if __name__ == '__main__':
    main()
