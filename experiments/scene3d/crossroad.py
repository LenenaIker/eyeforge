"""One street corner turned into a crossroads by cheap tricks, so the scene can be seen from any direction.

The photo shows one corner block: its sidewalk, the house, hedge and garden wall (env_add.py) and the objects. The
block is laid on a grid (its axes from the facade, its corner at the curb, both measured on the photo's depth)
and copied to the other three corners by turning it 90, 180 and 270 degrees about the crossing's centre: turned,
not mirrored, so text and bikes keep their handedness. Each copy gets another brick (the house and wall colours
shifted in hue, saturation and value; window frames, roofs and plants stay), the roads are an asphalt plane
below the sidewalks, with curb faces. Views are cameras anywhere on the roads or by the curbs, looking any way.
Run in the `trellis2` env:

    python crossroad.py --scene data/scene3d_test/ours/street_env_lite --photo-scene data/scene3d_test/ours/street_orbit \
        --out data/scene3d_test/ours/street_cross [--drop 10,12]

Writes <out>/isaac/{scene.json,meshes.npz} and photo.png for piece_isaac.py.
"""
import argparse
import colorsys
import json
import math
import os
import re
import shutil

import numpy as np
import pyrender
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

# Bricks for the copies: (hue or None to keep it, saturation factor, value factor). The first keeps the photo's.
BRICKS = [(None, 1.0, 1.0), (0.02, 1.8, 0.75), (0.06, 1.3, 0.55), (None, 0.15, 1.25),
          (0.04, 1.5, 0.9), (0.09, 0.6, 0.8)]
ASPHALT = [0.24, 0.24, 0.25]
ROAD_WORDS = ('road', 'street', 'asphalt', 'lane', 'cobble')


PHOTO_KEYS = ('uv', 'photo_faces', 'texture', 'gain')


def photo_keys(arr, i):
    """An object's photo texture (env_add.py --photo-texture), to carry with it into a template."""
    return {key: arr[f'obj{i}_{key}'] for key in PHOTO_KEYS if f'obj{i}_{key}' in arr}


def rebrick(colours, brick):
    """Brick-coloured vertices (orange to red, not grey) moved to another brick; the rest is left alone."""
    hue, ks, kv = brick
    if hue is None and ks == 1 and kv == 1:
        return colours
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in colours])
    sel = (hsv[:, 1] > 0.12) & ((hsv[:, 0] < 0.17) | (hsv[:, 0] > 0.95))
    if hue is not None:
        hsv[sel, 0] = hue
    hsv[sel, 1] = np.clip(hsv[sel, 1] * ks, 0, 1)
    hsv[sel, 2] = np.clip(hsv[sel, 2] * kv, 0, 1)
    out = colours.copy()
    out[sel] = [colorsys.hsv_to_rgb(*c) for c in hsv[sel]]
    return out


def quad(corners, uv=None, texture=None, colour=None):
    return {'points': np.asarray(corners).tolist(), 'uv': uv if uv is not None else [[0, 0], [1, 0], [1, 1], [0, 1]],
            'texture': texture, 'colour': colour}


def look(pos, forward, up):
    """4x4 OpenCV camera to world: +Z along forward, +Y down."""
    z = forward / np.linalg.norm(forward)
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = np.cross(y, z), y, z, pos
    return pose


def impostor(points, faces, colours, axes, texdir, tag, px):
    """A far thing as a box with a tiny picture of it on each side: the mesh's flat colours seen square-on from
    outside each face (orthographic, unlit), px pixels along the longer edge; empty corners (above a gable)
    take the mean colour. Returns the box faces as quads (4 sides and the top) in the mesh's frame."""
    a, b, up = axes
    lo = np.array([np.min(points @ v) for v in axes])
    hi = np.array([np.max(points @ v) for v in axes])
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    m = trimesh.Trimesh(points, faces, vertex_colors=(np.clip(colours, 0, 1) * 255).astype(np.uint8), process=False)
    sc = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0] * 3)
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    mean = np.asarray(colours).mean(0)
    quads = []
    # (outward normal, its index, right as seen from outside, up as seen from outside)
    for nrm, ni, right, top in ((a, 0, b, up), (-a, 0, -b, up), (b, 1, -a, up), (-b, 1, a, up), (up, 2, a, b)):
        ri, ti = [int(np.argmax(np.abs([r @ v for v in axes]))) for r in (right, top)]
        w, h = 2 * half[ri], 2 * half[ti]
        c = sum(mid[i] * axes[i] for i in range(3)) + nrm * half[ni]
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, top, nrm, c + nrm * 1.0
        cam = sc.add(pyrender.OrthographicCamera(xmag=w / 2, ymag=h / 2, znear=0.01, zfar=2 * half[ni] + 2), pose=pose)
        tw, th = (px, max(2, round(px * h / w))) if w >= h else (max(2, round(px * w / h)), px)
        r = pyrender.OffscreenRenderer(tw * 4, th * 4)
        rgba = r.render(sc, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)[0]
        r.delete()
        sc.remove_node(cam)
        img = rgba[..., :3].astype(np.float32) / 255
        img[rgba[..., 3] == 0] = mean
        path = os.path.join(texdir, f'{tag}_{len(quads)}.png')
        Image.fromarray((img * 255).astype(np.uint8)).resize((tw, th), Image.Resampling.BOX).save(path)
        corners = [c - right * w / 2 - top * h / 2, c + right * w / 2 - top * h / 2,
                   c + right * w / 2 + top * h / 2, c - right * w / 2 + top * h / 2]
        quads.append((corners, os.path.abspath(path)))
    return quads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='an env_add.py scene (photo layout: objects as in view 0)')
    ap.add_argument('--photo-scene', required=True, help='its piece_build.py directory (depth, support, walls)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--road', type=float, default=7.0, help='road width, metres')
    ap.add_argument('--curb', type=float, default=0.12, help='sidewalk height over the road, metres')
    ap.add_argument('--reach', type=float, default=60.0, help='sidewalk size from the curb, metres')
    ap.add_argument('--views', type=int, default=12)
    ap.add_argument('--clearance', type=float, default=7.0, help='metres from a camera to anything at least')
    ap.add_argument('--drop', default='', help='env mesh indices to leave out (bad SAM 3D meshes)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-objects', action='store_true', help='the place only: objects come from Replicator')
    ap.add_argument('--far', type=float, default=25.0, help='surroundings farther than this from the curb corner '
                    '(metres) are not meshes')
    ap.add_argument('--far-mode', choices=['box', 'drop'], default='box',
                    help='box: a box with a tiny picture of it; drop: left to the HDRI')
    ap.add_argument('--far-px', type=int, default=24, help='box picture size, pixels along the longer edge')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src = os.path.join(args.scene, 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = np.load(os.path.join(src, 'meshes.npz'))
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    drop = {int(i) for i in args.drop.split(',') if i}
    names = {e['index']: e['name'] for e in scene.get('env', [])}

    # Grid axes: b into the block (against the facade's normal), a along the facade, to the photo's right.
    walls = json.load(open(os.path.join(args.photo_scene, 'pieces.json')))['walls']
    nw = np.asarray(walls[0]['normal'], float)
    b = -(nw - (nw @ up) * up)
    b /= np.linalg.norm(b)
    a = np.cross(up, b)
    a *= np.sign(a[0])
    # The curb: where the support (the sidewalk) ends, toward the camera (b) and toward the side street (a).
    pts = np.load(os.path.join(args.photo_scene, 'depth_photo.npz'))['points']
    sup = np.asarray(Image.open(os.path.join(args.photo_scene, 'support_raw.png'))) > 0
    gpath = os.path.join(args.photo_scene, 'ground.npz')
    if os.path.exists(gpath):  # the flow segments sidewalk and road apart: the curb is where the sidewalk ends
        g = np.load(gpath)
        side = [i for i, p in enumerate(g['phrases']) if not any(w in str(p) for w in ROAD_WORDS)]
        if side and np.any(g['masks'][side], 0).sum() > 2000:
            sup = np.any(g['masks'][side], 0)
    q = pts[sup]
    q = q[np.isfinite(q).all(1)]
    level = float(np.median(q @ up))
    a0, b0 = np.percentile(q @ a, 1), np.percentile(q @ b, 1)
    corner = a0 * a + b0 * b + level * up
    centre = corner - args.road / 2 * (a + b) - args.curb * up

    # The block: the objects where the photo has them, the surroundings; everything baked into the world frame.
    block = []
    for i, t in scene['views'][0]['objects']:
        if i in drop:
            continue
        t = np.asarray(t)
        p = arr[f'obj{i}_points'] @ t[:3, :3].T + t[:3, 3]
        name = names.get(i, 'object')
        if args.no_objects and i not in names:
            continue
        if i in names:
            # Far surroundings come out of the photo's depth floating or sunk (a house 4 m up): set them on the
            # sidewalk; their lowest points, not one stray vertex.
            p = p + (level - np.percentile(p @ up, 0.5)) * up
        flat = p - (p @ up)[:, None] * up
        far = float(np.linalg.norm(flat.mean(0) - (corner - (corner @ up) * up))) > args.far and i in names
        if far and args.far_mode == 'drop':
            continue
        block.append((p, arr[f'obj{i}_faces'], arr[f'obj{i}_colors'], name, far, photo_keys(arr, i)))
    # Facade planes env_add.py kept where no mesh stands go round with the block.
    facades = [p for p in scene['pieces'] if re.search(r'tex_wall\d+\.png$', p['texture'] or '')]
    out_arr, objects, boxes = {}, 0, []
    texdir = os.path.join(args.out, 'isaac', 'far')
    os.makedirs(texdir, exist_ok=True)
    for k in range(4):
        rot = np.eye(4)
        rot[:3, :3] = np.asarray(_axis_rotation(up, k * math.pi / 2))
        rot[:3, 3] = centre - rot[:3, :3] @ centre
        brick = BRICKS[0] if k == 0 else BRICKS[1 + rng.integers(len(BRICKS) - 1)]
        for fp in facades:
            boxes.append({**fp, 'points': (np.asarray(fp['points']) @ rot[:3, :3].T + rot[:3, 3]).tolist()})
        for j, (p, f, c, name, far, photo) in enumerate(block):
            if any(w in name for w in ('house', 'wall')):
                c = rebrick(c, brick)
            if far:
                for corners, tex in impostor(p, f, c, (a, b, up), texdir, f'c{k}_m{j}', args.far_px):
                    boxes.append(quad([rot[:3, :3] @ x + rot[:3, 3] for x in corners], texture=tex))
                continue
            out_arr[f'obj{objects}_points'] = (p @ rot[:3, :3].T + rot[:3, 3]).astype(np.float32)
            out_arr[f'obj{objects}_faces'] = f
            out_arr[f'obj{objects}_colors'] = np.asarray(c, np.float32)
            if k == 0:  # the corner as the photo shows it keeps the photo's texture; turned copies are rebricked
                out_arr.update({f'obj{objects}_{key}': v for key, v in photo.items()})
            objects += 1

    # Ground: the road under everything, per corner a sidewalk with the photo's material and its two curb faces.
    floor = next((p for p in scene['pieces'] if p['texture'] and p['texture'].endswith('tex_floor.png')), None)
    size = json.load(open(os.path.join(args.photo_scene, 'pieces.json')))['materials']['floor']['size_m']
    big = args.reach + args.road
    pieces = [quad([centre + s * big * a + t * big * b for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
                   colour=ASPHALT)] + boxes
    for k in range(4):
        r = np.asarray(_axis_rotation(up, k * math.pi / 2))
        ka, kb = r @ a, r @ b
        c0 = centre + args.road / 2 * (ka + kb) + args.curb * up
        side = [c0, c0 + args.reach * ka, c0 + args.reach * (ka + kb), c0 + args.reach * kb]
        uv = [[0, 0], [args.reach / size[0], 0], [args.reach / size[0], args.reach / size[1]], [0, args.reach / size[1]]]
        pieces.append(quad(side, uv, floor['texture'] if floor else None, floor['colour'] if floor else [0.5] * 3))
        pieces[-1]['gain'] = (floor or {}).get('gain')  # the photometry gain of the photo's ground
        for e in (ka, kb):
            pieces.append(quad([c0, c0 + args.reach * e, c0 + args.reach * e - args.curb * up, c0 - args.curb * up],
                               colour=[0.55, 0.54, 0.52]))

    # Views: the photo's camera, then cameras on the roads (or by the curbs) looking any way, and one from above.
    views = [{'pose': np.eye(4).tolist(), 'objects': [], 'lamp': None}]
    # Points over the surfaces, not the vertices: a decimated wall is a few big triangles.
    env_pts = np.concatenate([trimesh.sample.sample_surface(trimesh.Trimesh(out_arr[f'obj{i}_points'],
                                                                            out_arr[f'obj{i}_faces'], process=False),
                                                            3000, seed=0)[0] for i in range(objects)])
    for fp in (p for p in pieces if re.search(r'tex_wall\d+\.png$', p['texture'] or '')):  # facades too
        c = np.asarray(fp['points'])
        uv = rng.uniform(0, 1, (300, 2))
        env_pts = np.concatenate([env_pts, c[0] + uv[:, :1] * (c[1] - c[0]) + uv[:, 1:] * (c[3] - c[0])])
    tree = cKDTree(env_pts)
    while len(views) <= args.views:
        along = rng.uniform(-25, 25)
        across = rng.uniform(-args.road / 2 - 1.5, args.road / 2 + 1.5)
        u, v = (along, across) if rng.random() < 0.5 else (across, along)
        pos = centre + u * a + v * b + rng.uniform(1.3, 4.0) * up
        if tree.query(pos)[0] < args.clearance:
            continue  # against a wall or in a hedge: nothing but a texture in view
        # Toward one of the corners (its middle, about 8 m into the block), give or take 35 degrees.
        k = rng.integers(4)
        r = np.asarray(_axis_rotation(up, k * math.pi / 2))
        aim = centre + r @ ((args.road / 2 + 8) * (a + b)) - pos
        aim -= (aim @ up) * up
        yaw = math.atan2(aim @ b, aim @ a) + math.radians(rng.uniform(-35, 35))
        pitch = math.radians(rng.uniform(4, 18))
        fwd = math.cos(pitch) * (math.cos(yaw) * a + math.sin(yaw) * b) - math.sin(pitch) * up
        views.append({'pose': look(pos, fwd, up).tolist(), 'objects': [], 'lamp': None})
    views.append({'pose': look(centre + 45 * up - 20 * b, 20 * b - 45 * up, b).tolist(), 'objects': [],
                  'lamp': None})
    for v in views:
        v['objects'] = [[i, np.eye(4).tolist()] for i in range(objects)]

    os.makedirs(os.path.join(args.out, 'isaac'), exist_ok=True)
    scene.update(objects=objects, views=views, pieces=pieces, env=[], indoor=False, lamp=None)
    json.dump(scene, open(os.path.join(args.out, 'isaac', 'scene.json'), 'w'))
    np.savez_compressed(os.path.join(args.out, 'isaac', 'meshes.npz'), **out_arr)
    shutil.copy(os.path.join(args.scene, 'photo.png'), args.out)
    print(json.dumps({'corner_ab': [round(a0, 2), round(b0, 2)], 'meshes': objects, 'boxes_and_facades': len(boxes),
                      'views': len(views),
                      'pieces': len(pieces)}))


def _axis_rotation(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * k @ k


if __name__ == '__main__':
    main()
