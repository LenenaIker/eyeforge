"""A straight street from one photo, so the place can be seen from anywhere along it (template `straight_street`
of the scene flow; crossroad.py is the corner one).

The street's direction is the long axis of the road pixels on the ground plane, which holds whether the photo
looks along the street or straight at a facade across it. The surroundings meshes (env_add.py) are split by the
side of the street they stand on. A side the photo does not show is the shown side turned 180 degrees about the
street's centre line (turned, not mirrored: text keeps its handedness). Each side is repeated along the street,
every copy with another brick (crossroad.rebrick). Sidewalks take the photo's ground material, the road is
asphalt a curb lower. Views are cameras on the road or the sidewalks, looking along the street or at a side.
Run in the `trellis2` env:

    python street_template.py --scene data/street_test/flow/basque_street/place \
        --photo-scene data/street_test/flow/basque_street --out data/street_test/flow/basque_street/street

Writes <out>/isaac/{scene.json,meshes.npz}, <out>/photo.png and <out>/template.json.
"""
import argparse
import json
import math
import os
import re
import shutil

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from crossroad import ASPHALT, BRICKS, ROAD_WORDS, _axis_rotation, look, photo_keys, quad, rebrick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='an env_add.py scene (the place from the photo camera)')
    ap.add_argument('--photo-scene', required=True, help='its flow directory (depth, ground masks, pieces)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=2, help='copies of each side before and after the original')
    ap.add_argument('--curb', type=float, default=0.12)
    ap.add_argument('--reach', type=float, default=40.0, help='sidewalk depth beyond the curb, metres')
    ap.add_argument('--views', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src = os.path.join(args.scene, 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = np.load(os.path.join(src, 'meshes.npz'))
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    names = {e['index']: e['name'] for e in scene.get('env', [])}
    floor = scene['pieces'][0]
    level = float(np.mean(np.asarray(floor['points']) @ up))

    # Ground points of the photo, the road ones apart when a ground phrase names a road.
    dp = np.load(os.path.join(args.photo_scene, 'depth_background.npz'))
    g = np.load(os.path.join(args.photo_scene, 'ground.npz'))
    phrases = [str(p) for p in g['phrases']]
    road_sel = [i for i, p in enumerate(phrases) if any(w in p for w in ROAD_WORDS)]
    road = np.any(g['masks'][road_sel], 0) if road_sel else np.any(g['masks'], 0)
    pts = dp['points'][road & (dp['mask'] > 0)]
    pts = pts[np.isfinite(pts).all(1)]
    pts = pts[np.linalg.norm(pts, axis=1) < 40]  # far road pixels are few and noisy
    flat = pts - (pts @ up)[:, None] * up
    c = flat.mean(0)
    # The street's direction: the road's long axis on the ground (PCA of its points).
    ev, evec = np.linalg.eigh(np.cov((flat - c).T))
    a = evec[:, np.argmax(ev)]
    a -= (a @ up) * up
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    if b @ (-c) < 0:  # b points from the street's centre towards the camera's side
        b = -b
    across = (flat - c) @ b
    lo_b, hi_b = np.percentile(across, [5, 95])
    width = float(np.clip(hi_b - lo_b, 3.5, 14.0)) if road_sel else 7.0
    centre = c + (lo_b + hi_b) / 2 * b if road_sel else c
    centre = centre - (centre @ up) * up + level * up

    # Surroundings by side; their bases set on the sidewalk level (far ones come out floating or sunk).
    sides = {1: [], -1: []}
    for i, t in scene['views'][0]['objects']:
        if i not in names:
            continue
        p = arr[f'obj{i}_points'] @ np.asarray(t)[:3, :3].T + np.asarray(t)[:3, 3]
        p = p + (level - np.percentile(p @ up, 0.5)) * up
        s = 1 if float((p.mean(0) - centre) @ b) >= 0 else -1
        sides[s].append((p, arr[f'obj{i}_faces'], arr[f'obj{i}_colors'], names[i], photo_keys(arr, i)))
    # Facade planes env_add.py kept where no mesh stands: they go with their side, like the meshes.
    facades = {1: [], -1: []}
    for p in scene['pieces']:
        if re.search(r'tex_wall\d+\.png$', p['texture'] or ''):
            q = np.asarray(p['points'])
            facades[1 if float((q.mean(0) - centre) @ b) >= 0 else -1].append(p)
    shown = [s for s in (1, -1) if sides[s] or facades[s]]
    if not shown:
        raise SystemExit('no surroundings mesh to build a street from')
    if len(shown) == 1:  # the other side: the shown one turned 180 degrees about the centre line's vertical
        s0 = shown[0]
        # The road's own pixels rarely show its whole width (a parked car, the camera on one lane): the camera
        # stands in the street, so the centre line goes under it and the road reaches to the shown facades less
        # a sidewalk. A centre line from the road pixels put the turned copy on top of the camera (basque_street).
        pts_s = np.concatenate([p for p, *_ in sides[s0]] + [np.asarray(fp['points']) for fp in facades[s0]])
        # The facade line: the block's points nearest the street (signed b coordinate from the old centre).
        facade_b = s0 * float(np.percentile(s0 * ((pts_s - centre) @ b), 2))
        cam_b = float((np.zeros(3) - centre) @ b)
        gap = abs(cam_b - facade_b)
        centre = centre + cam_b * b
        half = float(np.clip(gap - 1.5, 2.0, 7.0))
        width = 2 * half
        rot = np.asarray(_axis_rotation(up, math.pi))
        # The turned side is a copy: rebricked, without the photo's texture.
        sides[-s0] = [((p - centre) @ rot.T + centre, f, col, n, {}) for p, f, col, n, _ in sides[s0]]
        facades[-s0] = [{**fp, 'points': ((np.asarray(fp['points']) - centre) @ rot.T + centre).tolist()}
                        for fp in facades[s0]]

    def along_street(fp):
        c = np.asarray(fp['points'])
        n = np.cross(c[1] - c[0], c[3] - c[0])
        return abs(n @ a) < 0.35 * np.linalg.norm(n)

    out_arr, objects = {}, 0
    lengths, facade_quads = {}, []
    for s, block in sides.items():
        allp = np.concatenate([p for p, *_ in block] + [np.asarray(fp['points']) for fp in facades[s]])
        along = (allp - centre) @ a
        a0, a1 = np.percentile(along, [1, 99])
        L = float(max(a1 - a0, 8.0))
        lengths[s] = L
        for k in range(-args.repeats, args.repeats + 1):
            brick = BRICKS[0] if k == 0 else BRICKS[1 + rng.integers(len(BRICKS) - 1)]
            shift = k * L * a
            for fp in facades[s]:
                # Copies along the street only for a facade that runs along it: an oblique one (a far house seen
                # at an angle), moved by a block's length, crossed the road in front of the camera (basque_street).
                if k and not along_street(fp):
                    continue
                facade_quads.append({**fp, 'points': (np.asarray(fp['points']) + shift).tolist()})
            for p, f, col, n, photo in block:
                if any(w in n for w in ('house', 'wall', 'building')):
                    col = rebrick(col, brick)
                out_arr[f'obj{objects}_points'] = (p + shift).astype(np.float32)
                out_arr[f'obj{objects}_faces'] = f
                out_arr[f'obj{objects}_colors'] = np.asarray(col, np.float32)
                if k == 0:  # the block where the photo has it keeps the photo's texture; copies are rebricked
                    out_arr.update({f'obj{objects}_{key}': v for key, v in photo.items()})
                objects += 1

    # Ground: the road along the street, a sidewalk with the photo's material on each side, curb faces.
    span = (args.repeats + 1) * max(lengths.values()) + 20
    half = width / 2
    road_pts = [centre + x * span * a + y * (half + args.reach) * b - args.curb * up
                for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    pieces = [quad(road_pts, colour=ASPHALT)] + facade_quads
    size = json.load(open(os.path.join(args.photo_scene, 'pieces.json')))['materials']['floor'].get('size_m', [1, 1])
    for s in (1, -1):
        c0 = centre + s * half * b
        side = [c0 - span * a, c0 + span * a, c0 + span * a + s * args.reach * b, c0 - span * a + s * args.reach * b]
        uv = [[0, 0], [2 * span / size[0], 0], [2 * span / size[0], args.reach / size[1]], [0, args.reach / size[1]]]
        pieces.append(quad(side, uv, floor['texture'], floor['colour']))
        pieces[-1]['gain'] = floor.get('gain')  # the photometry gain of the photo's ground
        pieces.append(quad([c0 - span * a, c0 + span * a, c0 + span * a - args.curb * up,
                            c0 - span * a - args.curb * up], colour=[0.55, 0.54, 0.52]))

    # Views: the photo's camera, then cameras over the road and sidewalks looking along the street or at a side.
    def quad_points(fp, n=300):
        c = np.asarray(fp['points'])
        uv = rng.uniform(0, 1, (n, 2))
        return c[0] + uv[:, :1] * (c[1] - c[0]) + uv[:, 1:] * (c[3] - c[0])
    env_pts = np.concatenate([quad_points(fp) for fp in facade_quads] + [trimesh.sample.sample_surface(
        trimesh.Trimesh(out_arr[f'obj{i}_points'], out_arr[f'obj{i}_faces'], process=False), 2000, seed=0)[0]
        for i in range(objects)])
    tree = cKDTree(env_pts)
    clearance = float(np.clip(half, 1.5, 5.0))  # narrow lanes leave little room
    views, tries = [{'pose': np.eye(4).tolist(), 'objects': [], 'lamp': None}], 0
    reach_a = (args.repeats + 0.5) * min(lengths.values())
    while len(views) <= args.views and tries < 5000:
        tries += 1
        pos = centre + rng.uniform(-reach_a, reach_a) * a + rng.uniform(-half - 2, half + 2) * b + \
            rng.uniform(1.3, 4.0) * up
        if tree.query(pos)[0] < clearance:
            continue
        if rng.random() < 0.6:  # along the street, either way
            yaw = (0 if rng.random() < 0.5 else math.pi) + math.radians(rng.uniform(-40, 40))
        else:  # at one side
            yaw = (math.pi / 2 if rng.random() < 0.5 else -math.pi / 2) + math.radians(rng.uniform(-50, 50))
        pitch = math.radians(rng.uniform(3, 18))
        fwd = math.cos(pitch) * (math.cos(yaw) * a + math.sin(yaw) * b) - math.sin(pitch) * up
        views.append({'pose': look(pos, fwd, up).tolist(), 'objects': [], 'lamp': None})
    for v in views:
        v['objects'] = [[i, np.eye(4).tolist()] for i in range(objects)]

    os.makedirs(os.path.join(args.out, 'isaac'), exist_ok=True)
    scene.update(objects=objects, views=views, pieces=pieces, env=[], indoor=False, lamp=None)
    scene.pop('floor_plane', None)  # the photo's floor; the template's ground is level
    json.dump(scene, open(os.path.join(args.out, 'isaac', 'scene.json'), 'w'))
    np.savez_compressed(os.path.join(args.out, 'isaac', 'meshes.npz'), **out_arr)
    shutil.copy(os.path.join(args.scene, 'photo.png'), args.out)
    rep = {'width_m': round(width, 2), 'road_phrases': [phrases[i] for i in road_sel],
           'sides_shown': shown, 'block_length_m': {str(k): round(v, 1) for k, v in lengths.items()},
           'meshes': objects, 'views': len(views), 'tries': tries, 'clearance_m': round(clearance, 2)}
    json.dump(rep, open(os.path.join(args.out, 'template.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
