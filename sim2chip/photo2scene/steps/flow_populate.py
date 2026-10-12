"""Step 3.5 of photo2scene: the erased things come back, at random. Every view of the extended place (a room or a
street, room_template.py, street_template.py, crossroad.py) gets a few of phase 1's assets (flow_assets.py: the
user's classes and the distractors) on whatever flat surface the camera sees: the floor, a desk top, a shelf
board, the top of another asset. A distractor that stands in the same spot in every frame is learned as part of
the place (user, 2026-10-08), so each view draws its own set, places and turns.

Surfaces come from a height map of the place seen from above (pyrender, orthographic, the ceiling left out): a
footprint fits where every cell under it is at one height, so the thing stands fully supported and clear of
walls and furniture, and nothing is ever below a shelf (the map only sees the top). Each asset is SAM 3D's
canonical mesh at its metric size (pose scale of flow_sam3d.py, our MoGe depth), turned about the up axis.
A placement must be seen by the view's camera (depth test) and big enough there.

Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python sim2chip/photo2scene/steps/flow_populate.py --dir data/scene3d_test/flow/user_desk

Reads <dir>/extended/isaac/{scene.json,meshes.npz}, <dir>/assets.json and <dir>/assets/sam3d/; writes
<dir>/populated/isaac/{scene.json,meshes.npz} (the place's objects plus the assets, "roles" per object) for
piece_isaac.py and view_labels.py, and <dir>/populated/populate.json.
"""
import argparse
import json
import math
import os
import shutil
from collections import Counter

import numpy as np
import pyrender
import trimesh

from env_add import simplify

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def frame(up):
    """Two horizontal axes completing up."""
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def place_meshes(scene, arr, view):
    """The place as seen in one view: its pieces (no ceiling) and its own objects, posed."""
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    meshes = []
    floor = min(float(np.mean(np.asarray(p['points']) @ up)) for p in scene['pieces'])
    for p in scene['pieces']:
        c = np.asarray(p['points'], float)
        h = c @ up
        if np.ptp(h) < 0.1 and h.mean() > floor + 1.0:  # the ceiling: the height map must see under it
            continue
        meshes.append(trimesh.Trimesh(c, [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], process=False))
    for i, t in view['objects']:
        m = trimesh.Trimesh(arr[f'obj{i}_points'], arr[f'obj{i}_faces'], process=False)
        meshes.append(m.apply_transform(np.asarray(t)))
    return meshes


class HeightMap:
    """Top surface height over a horizontal grid, from an orthographic render straight down."""

    def __init__(self, meshes, up, centre, half, res, renderer_size=1024):
        self.up = up
        self.e1, self.e2 = frame(up)
        self.centre, self.half = centre, half
        n = min(renderer_size, int(2 * half / res))
        self.n, self.res = n, 2 * half / n
        top = max(float(np.max(m.vertices @ up)) for m in meshes) + 1.0
        low = min(float(np.min(m.vertices @ up)) for m in meshes) - 1.0
        s = pyrender.Scene()
        for m in meshes:
            s.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2] = self.e1, self.e2, up  # OpenGL camera looks along -Z: down
        self.cam_h = top
        pose[:3, 3] = centre - (centre @ up) * up + top * up
        near, far = 0.01, top - low + 1
        s.add(pyrender.OrthographicCamera(xmag=half, ymag=half, znear=near, zfar=far), pose=pose)
        r = pyrender.OffscreenRenderer(n, n)
        depth = r.render(s, flags=pyrender.RenderFlags.DEPTH_ONLY)
        r.delete()
        # pyrender linearizes every camera's depth as a perspective one: undo it, an orthographic depth buffer is
        # already linear.
        hit = depth > 0
        ndc = (far + near - 2 * near * far / np.where(hit, depth, 1.0)) / (far - near)
        depth = np.where(hit, near + (ndc + 1) / 2 * (far - near), 0.0)
        self.h = np.where(depth > 0, top - depth, np.nan)

    def world(self, r, c, h):
        x = (c + 0.5) * self.res - self.half
        y = self.half - (r + 0.5) * self.res
        base = self.centre - (self.centre @ self.up) * self.up
        x, y, h = (np.asarray(t, float)[..., None] for t in (x, y, h))
        return base + x * self.e1 + y * self.e2 + h * self.up

    def footprint(self, r, c, w, d, yaw):
        """Grid cells under a w x d rectangle centred on cell (r, c), turned by yaw."""
        k = max(1, int(math.ceil(0.5 * math.hypot(w, d) / self.res)))
        rr, cc = np.mgrid[-k:k + 1, -k:k + 1]
        x, y = cc * self.res, -rr * self.res
        u = x * math.cos(yaw) + y * math.sin(yaw)
        v = -x * math.sin(yaw) + y * math.cos(yaw)
        inside = (np.abs(u) <= w / 2) & (np.abs(v) <= d / 2)
        rr, cc = rr[inside] + r, cc[inside] + c
        ok = (rr >= 0) & (rr < self.n) & (cc >= 0) & (cc < self.n)
        return (rr[ok], cc[ok]) if ok.all() else None


def load_asset(path, scale, max_faces):
    """Canonical SAM 3D mesh (y up) at metric size, base centred at the origin, simplified."""
    m = trimesh.load(path, force='mesh')
    m.visual = m.visual.to_color() if not hasattr(m.visual, 'vertex_colors') else m.visual
    m.apply_scale(scale)
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    return simplify(m, max_faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--place', default='extended', help='the place folder inside --dir')
    ap.add_argument('--out', default='populated')
    ap.add_argument('--per-view', default='2,8', help='fewest,most assets per view')
    ap.add_argument('--flat', type=float, default=0.01, help='height spread allowed under a footprint, metres')
    ap.add_argument('--min-px', type=float, default=24, help='smallest size of a placed asset in the view, pixels')
    ap.add_argument('--res', type=float, default=0.01, help='height map cell, metres (coarser for big places)')
    ap.add_argument('--max-faces', type=int, default=40000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = args.dir
    src = os.path.join(d, args.place, 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = dict(np.load(os.path.join(src, 'meshes.npz')))
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    w, h = scene['size']
    fx, fy, cx, cy = scene['fx'], scene['fy'], scene['cx'], scene['cy']

    # Assets at metric size, upright: SAM 3D's canonical y axis goes to the place's up.
    cat = json.load(open(os.path.join(d, 'assets.json')))['assets']
    e1, e2 = frame(up)
    to_world = np.eye(4)
    to_world[:3, :3] = np.stack([e1, up, np.cross(e1, up)], 1)
    assets = []
    for a in cat:
        mesh_path = os.path.join(d, a['mesh'])
        info_path = mesh_path.replace('_canon.glb', '.json')
        if not (os.path.exists(mesh_path) and os.path.exists(info_path)):
            continue
        scale = json.load(open(info_path))['pose']['scale'][0][0]
        m = load_asset(mesh_path, scale, args.max_faces)
        assets.append({**a, 'local': m, 'size': m.extents.copy()})  # x width, y height, z depth
    if not assets:
        raise SystemExit('no asset meshes')

    kept = os.path.join(d, args.place, 'views.json')  # flow_check_views.py: views the VLM passed
    if os.path.exists(kept):
        keep = set(json.load(open(kept))['kept']) | {0}  # view 0 is the photo's camera, matched to its light
        scene['views'] = [v for k, v in enumerate(scene['views']) if k in keep]
    n_place = scene['objects']
    # The place's own objects, named by env_add.py's list, which gives each mesh's object index (a place may
    # hold objects of its own before them).
    roles = [{'name': f'object {i}', 'role': 'scene'} for i in range(n_place)]
    for e in scene.get('env') or []:
        if e.get('index', n_place) < n_place:
            roles[e['index']]['name'] = e['name']
    for k, a in enumerate(assets):
        m = a['local'].copy().apply_transform(to_world)
        arr[f'obj{n_place + k}_points'] = np.asarray(m.vertices, np.float32)
        arr[f'obj{n_place + k}_faces'] = np.asarray(m.faces, np.int32)
        arr[f'obj{n_place + k}_colors'] = (m.visual.vertex_colors[:, :3] / 255).astype(np.float32)
        roles.append({'name': a['name'], 'role': a['role']})

    lo_n, hi_n = (int(x) for x in args.per_view.split(','))
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=5000)
    report = []
    for vi, view in enumerate(scene['views']):
        meshes = place_meshes(scene, arr, view)
        pose_cv = np.asarray(view['pose'])
        eye = pose_cv[:3, 3]
        # Height map over what this camera can see well: a square around the point it looks at.
        reach = max(a['size'].max() for a in assets) * fx / args.min_px
        fwd = pose_cv[:3, 2]
        centre = eye + fwd * min(reach, 8.0) / 2
        hm = HeightMap(meshes, up, centre, min(reach, 8.0) / 2 + 1.0, args.res)
        s = pyrender.Scene()
        for m in meshes:
            s.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        s.add(cam, pose=pose_cv @ CV2GL)
        renderer = pyrender.OffscreenRenderer(w, h)  # one EGL context at a time: two at once fail to switch
        depth = renderer.render(s, flags=pyrender.RenderFlags.DEPTH_ONLY)
        renderer.delete()

        want = int(rng.integers(lo_n, hi_n + 1))
        order = rng.permutation(len(assets))[:want]
        placed, why = [], Counter()
        heights = hm.h.copy()
        # Candidate cells: surface the camera sees, in the frame and not too close.
        valid = np.argwhere(np.isfinite(heights))
        pts = hm.world(valid[:, 0], valid[:, 1], heights[valid[:, 0], valid[:, 1]])
        cam_pts = (pts - eye) @ pose_cv[:3, :3]
        z = cam_pts[:, 2]
        ok = z > 0.5
        u = np.where(ok, fx * cam_pts[:, 0] / np.maximum(z, 1e-6) + cx, -1)
        v = np.where(ok, fy * cam_pts[:, 1] / np.maximum(z, 1e-6) + cy, -1)
        ok &= (u > 0.05 * w) & (u < 0.95 * w) & (v > 0.05 * h) & (v < 0.95 * h)
        seen = np.zeros(len(valid))
        seen[ok] = depth[v[ok].astype(int), u[ok].astype(int)]
        ok &= (seen == 0) | (seen > z - 0.05)
        valid = valid[ok]
        if not len(valid):
            report.append({'view': vi, 'asked': 0, 'placed': [], 'rejected': {'no surface in view': 1}})
            continue
        for k in order:
            a = assets[k]
            wd, ht, dp = a['size']
            for _ in range(200):
                r, c = valid[rng.integers(len(valid))]
                base_h = heights[r, c]
                yaw = rng.uniform(0, 2 * math.pi)
                cells = hm.footprint(r, c, wd, dp, yaw)
                if cells is None:
                    why['edge'] += 1
                    continue
                under = heights[cells]
                if not np.all(np.isfinite(under)) or np.ptp(under) > args.flat:
                    why['not flat'] += 1
                    continue
                base = hm.world(r, c, base_h)
                # Seen by this camera: the middle of the asset in the image, in front of what the place shows.
                mid = np.linalg.inv(pose_cv) @ np.append(base + up * ht / 2, 1)
                z = mid[2]
                if z < 0.5:
                    why['behind or too close'] += 1
                    continue
                u, v = fx * mid[0] / z + cx, fy * mid[1] / z + cy
                if not (0.05 * w < u < 0.95 * w and 0.05 * h < v < 0.95 * h):
                    why['out of view'] += 1
                    continue
                if max(wd, ht, dp) * fx / z < args.min_px:
                    why['too small'] += 1
                    continue
                seen = depth[int(v), int(u)]
                if seen > 0 and seen < z - 0.05:
                    why['hidden'] += 1
                    continue
                t = np.eye(4)
                t[:3, :3] = trimesh.transformations.rotation_matrix(yaw, up)[:3, :3]
                t[:3, 3] = base
                placed.append([n_place + int(k), t.tolist()])
                heights[cells] = base_h + ht  # the next one may stand on it, never inside it
                break
        view['objects'] = list(view['objects']) + placed
        report.append({'view': vi, 'asked': want, 'placed': [assets[i - n_place]['name'] for i, _ in placed],
                       'rejected': dict(why)})

    out = os.path.join(d, args.out, 'isaac')
    shutil.rmtree(out, ignore_errors=True)  # renders and labels of an earlier draw would mix with this one
    os.makedirs(out)
    scene['objects'] = n_place + len(assets)
    scene['roles'] = roles
    json.dump(scene, open(os.path.join(out, 'scene.json'), 'w'))
    np.savez_compressed(os.path.join(out, 'meshes.npz'), **arr)
    shutil.copy(os.path.join(d, args.place, 'photo.png'), os.path.join(d, args.out, 'photo.png'))
    summary = {'assets': len(assets), 'views': len(report),
               'placed_per_view': [len(x['placed']) for x in report],
               'never_placed': sorted({a['name'] for a in assets} - {n for x in report for n in x['placed']})}
    json.dump({**summary, 'per_view': report}, open(os.path.join(d, args.out, 'populate.json'), 'w'), indent=1)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
