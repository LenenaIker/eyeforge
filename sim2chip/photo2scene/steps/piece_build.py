"""Our scene from one photo, built in pieces (user's idea, 2026-10-06): nothing of the photo is used as a backdrop.
The floor, the support (a table top) and the walls are planes fitted on the photo's own depth; each takes a
material cut from the photo, rectified as if seen straight on (SAM 3 mask -> plane -> resampled), and the
objects are SAM 3D meshes posed where the real ones stand (bg_build.photo_replica, with the CCM orientation).
The room is closed with the walls the photo does not show, so the camera can go anywhere in it; outdoors (no
walls) the sky is an HDRI. Run in the `trellis2` env after bg_prepare.py, bg_depth.py, piece_surfaces.py and the
Mira mesh stages:

    PYOPENGL_PLATFORM=egl python piece_build.py --dir data/scene3d_test/ours/office_pieces \
        --meshes data/scene3d_test/photo_cases/003_home_office/003_home_office --fixed desk,bookshelf,cabinet

Writes <dir>/isaac/{scene.json, meshes.npz, tex_*.png} for piece_isaac.py and view_labels.py, <dir>/pieces.jpg
(the textures) and <dir>/pieces.json (planes, textures, objects).
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from bg_build import fit_plane, floor_height, look_at_cv, photo_replica, place, points_from_depth


def basis(n):
    """Two unit vectors spanning the plane with normal n."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def gravity(P, surf, things, vp, n, max_tilt=20.0, min_px=2000):
    """The true vertical from the walls the photo shows: buildings stand plumb, the ground need not (a street on a
    slope tilted the whole place 7.6 degrees, the house leaning in every new view; photo 24 of the pots, 2026-10-13).
    Each wall mask's points give a free plane; the vertical is the direction across all their normals. Walls of one
    direction only fix the lean towards them (n with its part along their normal taken out); no wall, or a vertical
    more than max_tilt from the ground's normal, keeps n. Returns (up, how, tilt in degrees)."""
    normals = []
    for m, kind in zip(surf['masks'], surf['kinds']):
        sel = m & ~things & vp
        if kind != 'wall' or sel.sum() < min_px:
            continue
        X = P[sel]
        X = X[np.isfinite(X).all(1)]
        X = X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)]
        tol = max(0.03, 0.01 * float(np.median(np.linalg.norm(X, axis=1))))  # depth noise grows with distance
        wm, _, inl = fit_plane(X, iters=300, tol=tol)
        if inl >= 0.5 and abs(wm @ n) < 0.5:  # a wall, not a stretch of ground or a roof
            normals.append((wm, inl * len(X)))
    if not normals:
        return n, 'ground', 0.0
    M = sum(wt * np.outer(wm, wm) for wm, wt in normals)
    ev, evec = np.linalg.eigh(M / sum(wt for _, wt in normals))
    if ev[1] > 0.1:  # two wall directions at least: the vertical is across all of them
        up, how = evec[:, 0], 'walls'
    else:  # one direction: lean towards it fixed, sideways lean left as the ground has it
        wm = evec[:, 2]
        up, how = n - (n @ wm) * wm, 'one wall direction'
    up = up / np.linalg.norm(up)
    up = up if up @ n > 0 else -up
    tilt = math.degrees(math.acos(min(1.0, float(up @ n))))
    if tilt > max_tilt:
        return n, f'ground (walls said {tilt:.1f} deg)', 0.0
    return up, how, round(tilt, 2)


GROUND_PHOTO_M = 40.0   # the photo's own ground is laid out to this distance; past it, the tiled floor
GROUND_PHOTO_PPM = 12   # its texels per metre
GROUND_PHOTO_LIFT = 0.05  # over the floor and the street bands' strips (section_fill.STRIP_LIFT)


def ground_photo(d, intr, depth, flat_at, to2, lo, hi, floor_mat, out):
    """The photo's ground seen from above (an orthophoto): every texel of the floor around the camera, up to
    GROUND_PHOTO_M away, takes the erased photo's colour where the photo sees ground there (its pixel on the ground
    masks), and the tiled floor texture elsewhere. A side lane, the cracks and the
    painted lines come out where the photo has them, which bands along a straight street cannot draw (the roads
    photo's lane up to the left, 2026-10-13). flat_at(x, y) gives floor points for arrays of floor coordinates.
    Returns the piece, or None when the photo shows too little ground."""
    w, h, fx, fy, cx, cy = intr
    bg = np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB').resize((w, h)))
    ground = np.any(np.load(os.path.join(d, 'ground.npz'))['masks'], 0)
    rays = points_from_depth(np.ones((h, w)), fx, fy, cx, cy)
    # Where the photo's ground lies, in floor coordinates: its pixels' depth points, near ones only.
    X = rays[ground] * depth[ground][:, None]
    X = X[np.isfinite(X).all(1) & (np.linalg.norm(X, axis=1) < GROUND_PHOTO_M)]
    if len(X) < 1000:
        return None
    q = np.array([to2(x) for x in X[:: max(1, len(X) // 20000)]])
    x0, y0 = np.maximum(np.percentile(q, 1, 0) - 2, lo)
    x1, y1 = np.minimum(np.percentile(q, 99, 0) + 2, hi)
    step = 1.0 / GROUND_PHOTO_PPM
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    P = flat_at(gx.ravel(), gy.ravel())
    z = P[:, 2]
    ok = z > 0.1
    u = np.where(ok, P[:, 0] / np.maximum(z, 1e-6) * fx + cx, -1)
    v = np.where(ok, P[:, 1] / np.maximum(z, 1e-6) * fy + cy, -1)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = ok & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    # Seen where the photo's pixel is ground: the ground is then the nearest thing there. A depth agreement on top
    # threw away most of a street that is not quite one plane (roads: 17 % of the texels kept).
    seen = np.zeros(len(P), bool)
    seen[inside] = ground[vi[inside], ui[inside]]
    colour = cv2.remap(bg, u.reshape(gx.shape).astype(np.float32), v.reshape(gx.shape).astype(np.float32),
                       cv2.INTER_LINEAR).reshape(-1, 3).astype(np.float32)
    tex = np.asarray(Image.open(floor_mat['texture']).convert('RGB')) if 'texture' in floor_mat else None
    if tex is not None:  # the tiled floor where the photo does not see the ground
        th, tw = tex.shape[:2]
        sx, sy = floor_mat['size_m']
        fill = tex[((gy.ravel() % sy) / sy * th).astype(int) % th, ((gx.ravel() % sx) / sx * tw).astype(int) % tw]
    else:
        fill = np.tile(np.array(floor_mat['colour']) * 255, (len(P), 1))
    a = cv2.GaussianBlur(seen.reshape(gx.shape).astype(np.float32), (9, 9), 0).reshape(-1, 1)  # soft edges
    img = (a * colour + (1 - a) * fill).reshape(*gx.shape, 3).clip(0, 255).astype(np.uint8)
    if seen.mean() < 0.05:
        return None
    path = os.path.abspath(os.path.join(out, 'tex_ground_photo.png'))
    Image.fromarray(img[::-1]).save(path)  # rows from y1 down: a texture's st origin is its bottom left
    return {'texture': path, 'corners': ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), 'seen': round(float(seen.mean()), 3),
            'colour': floor_mat['colour']}


def max_rect(valid):
    """Largest all-True axis-aligned rectangle (y0, x0, y1, x1), by the histogram method."""
    h, w = valid.shape
    heights = np.zeros(w, int)
    best = (0, 0, 0, 0, 0)
    for y in range(h):
        heights = np.where(valid[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                start, hh = stack.pop()
                area = hh * (x - start)
                if area > best[0]:
                    best = (area, y - hh + 1, start, y + 1, x)
            stack.append((start, cur))
    return best[1:]


def rectify(photo, mask, m, c, up, intr, ppm, max_side=1024):
    """The photo's pixels of plane m.X = c inside mask, resampled on a grid of the plane (straight on): returns the
    image, its valid pixels and the grid (origin, u axis, v axis, metres per pixel). u is horizontal; v runs down
    the wall or away from the camera on the floor."""
    w, h, fx, fy, cx, cy = intr
    rays = points_from_depth(np.ones((h, w)), fx, fy, cx, cy)
    t = c / (rays @ m)
    X = rays * t[..., None]
    sel = mask & (t > 0)
    # Walls: u along the image's right, v down the wall (seen from the camera side, not mirrored).
    u_ax = np.cross(up, m) if abs(up @ m) < 0.9 else basis(m)[0]
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(u_ax, m)
    pu, pv = X[sel] @ u_ax, X[sel] @ v_ax
    (u0, u1), (v0, v1) = np.percentile(pu, [1, 99]), np.percentile(pv, [1, 99])
    step = max(1 / ppm, (u1 - u0) / max_side, (v1 - v0) / max_side)
    gu, gv = np.meshgrid(np.arange(u0, u1, step), np.arange(v0, v1, step))
    P = (gu[..., None] * u_ax + gv[..., None] * v_ax) + m * c  # point on the plane (m unit, so m*c lies on it)
    P = P - m * (P @ m - c)[..., None]
    zx = (P[..., 0] / P[..., 2] * fx + cx).astype(np.float32)
    zy = (P[..., 1] / P[..., 2] * fy + cy).astype(np.float32)
    img = cv2.remap(photo, zx, zy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ok = cv2.remap(mask.astype(np.uint8), zx, zy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    ok &= (P[..., 2] > 0) & (zx >= 0) & (zx < w - 1) & (zy >= 0) & (zy < h - 1)
    return img, ok, (u0, v0, step, u_ax, v_ax)


def tileable(img, share=0.15):
    """Seamless when repeated: along each axis the last `share` is cross-faded into the start and cut off, so the
    first column continues from the last (mirrored repeats drew diamonds out of floor planks)."""
    out = img.astype(np.float32)
    for axis in (1, 0):
        size = out.shape[axis]
        k = max(2, int(size * share))
        a = np.linspace(0, 1, k).reshape((1, k, 1) if axis == 1 else (k, 1, 1))
        head = np.take(out, range(k), axis=axis)
        tail = np.take(out, range(size - k, size), axis=axis)
        out = np.take(out, range(size - k), axis=axis).copy()
        idx = [slice(None)] * 3
        idx[axis] = slice(0, k)
        out[tuple(idx)] = a * head + (1 - a) * tail
    return out.clip(0, 255).astype(np.uint8)


def material(photo, mask, m, c, up, intr, ppm, name, out, min_side_m=0.12, depth=None, near=None, max_aspect=3.0):
    """A texture for plane m.X = c: the largest clean rectangle of its rectified pixels. Returns the texture file
    and its size in metres, or the mean colour when no rectangle of min_side_m fits. Pixels much darker than the
    surface's median are cast shadows and stay out of the ground's crop; with depth and near (a percentile), only the
    nearer part of the mask is used (a street's far ground is a few blurred pixels per metre). The crop is at
    most max_aspect long: a thin strip, repeated, drew stripes over a whole street (2026-10-07)."""
    colour = (np.median(photo[mask], 0) / 255).tolist() if mask.any() else [0.6, 0.6, 0.6]
    if depth is not None and near and mask.sum() > 1000:
        mask = mask & (depth <= np.percentile(depth[mask], near))
    img, ok, (u0, v0, step, _, _) = rectify(photo, mask, m, c, up, intr, ppm)
    lum = img.astype(np.float32).mean(2)
    if near and ok.sum() > 100:  # ground only: a timber facade's dark beams are not shadows
        ok &= lum > 0.6 * np.median(lum[ok])
    ok = cv2.erode(ok.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    y0, x0, y1, x1 = max_rect(ok) if ok.any() else (0, 0, 0, 0)
    if x1 - x0 > max_aspect * (y1 - y0):
        x1 = x0 + int(max_aspect * (y1 - y0))
    if y1 - y0 > max_aspect * (x1 - x0):
        y1 = y0 + int(max_aspect * (x1 - x0))
    if min(y1 - y0, x1 - x0) * step < min_side_m:
        return {'colour': colour, 'rectified': img, 'crop': None}
    tex = tileable(img[y0:y1, x0:x1])
    y1, x1 = y0 + tex.shape[0], x0 + tex.shape[1]  # the seam blend shortens it
    path = os.path.abspath(os.path.join(out, f'tex_{name}.png'))
    Image.fromarray(tex).save(path)
    return {'texture': path, 'size_m': [(x1 - x0) * step, (y1 - y0) * step], 'colour': colour, 'rectified': img,
            'crop': (y0, x0, y1, x1)}


def quad(corners, mat, uv_axes=None):
    """A textured rectangle: 4 corners in order, UVs in texture repeats from the material's size in metres."""
    corners = np.asarray(corners, float)
    if 'size_m' in mat:
        a, b = corners[1] - corners[0], corners[3] - corners[0]
        uv = [[0, 0], [np.linalg.norm(a) / mat['size_m'][0], 0],
              [np.linalg.norm(a) / mat['size_m'][0], np.linalg.norm(b) / mat['size_m'][1]],
              [0, np.linalg.norm(b) / mat['size_m'][1]]]
    else:
        uv = [[0, 0]] * 4
    return {'points': corners.tolist(), 'uv': uv, 'texture': mat.get('texture'), 'colour': mat['colour']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--meshes', default='', help='Mira case with the object meshes; empty: the place only (its '
                    'objects are erased and come later from Replicator)')
    ap.add_argument('--fov-x', type=float, default=None)
    ap.add_argument('--camera-height', type=float, default=None)
    ap.add_argument('--views', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--solid', default='')
    ap.add_argument('--fixed', default='')
    ap.add_argument('--layout', choices=['shuffle', 'photo'], default='shuffle')
    ap.add_argument('--orbit', default='', help='comma-separated turns in degrees: instead of random views, the '
                    'photo layout seen from the photo camera turned by each (to compare with other methods)')
    ap.add_argument('--min-iou', type=float, default=0.3, help='silhouette fit needed to use an object')
    ap.add_argument('--turn', type=float, default=50, help='largest camera turn around the objects, degrees')
    ap.add_argument('--wall-height', type=float, default=2.6)
    ap.add_argument('--outdoor', action='store_true', help='walls are facades: each spans what the photo shows of '
                    'it, no room is closed and the sky is an HDRI')
    ap.add_argument('--indoor', action='store_true', help='close a room even when the photo shows no wall (a table '
                    'shot indoors); its walls take a plain plaster colour')
    ap.add_argument('--depth', choices=['photo', 'background'], default='photo',
                    help='MoGe of the photo or of the erased photo (photo2scene: the same depth SAM 3D gets)')
    args = ap.parse_args()
    d, out = args.dir, os.path.join(args.dir, 'isaac')
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    solid = {p.strip() for p in args.solid.split(',') if p.strip()}
    fixed = {p.strip() for p in args.fixed.split(',') if p.strip()}

    photo = np.asarray(Image.open(os.path.join(d, 'photo.png')).convert('RGB'))
    h, w = photo.shape[:2]
    inst = np.load(os.path.join(d, 'instances.npz'))
    masks, phrases = inst['masks'], list(inst['phrases'])
    surf = np.load(os.path.join(d, 'surfaces.npz'))
    dp = np.load(os.path.join(d, f'depth_{args.depth}.npz'))
    k = dp['intrinsics']
    fx, fy, cx, cy = k[0, 0] * w, k[1, 1] * h, k[0, 2] * w, k[1, 2] * h
    if args.fov_x:
        fx = fy = w / 2 / math.tan(math.radians(args.fov_x) / 2)
    intr = (w, h, fx, fy, cx, cy)
    P = points_from_depth(dp['depth'], fx, fy, cx, cy)
    vp = dp['mask'] > 0
    things = cv2.dilate(np.any(masks, 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0  # objects, grown

    # The support plane on the photo's depth, away from the objects.
    support = (np.asarray(Image.open(os.path.join(d, 'support_raw.png'))) > 0) & ~things & vp
    n, dd, inlier = fit_plane(P[support])
    scale = args.camera_height / dd if args.camera_height else 1.0
    P, dd, depth_p = P * scale, dd * scale, dp['depth'] * scale
    # Up is the true vertical (the walls'), the floor keeps its own normal n: on a slope it is not level.
    up, up_how, tilt = gravity(P, surf, things, vp, n)
    print(f'up from the {up_how}, {tilt} deg off the ground normal', flush=True)
    ppm = fx / float(np.median(depth_p[vp]))  # texture pixels per metre near the photo's own resolution

    # Floor: the support itself, or a plane parallel to it below (a table photo with the floor in view).
    floor_dd, floor_mask = dd, support
    for m, kind in zip(surf['masks'], surf['kinds']):
        sel = m & ~things & vp
        if kind != 'floor' or sel.sum() < 500 or (m & support).sum() > 0.5 * sel.sum():
            continue
        cand = -float(np.median(P[sel] @ n))
        if cand > dd + 0.2:  # at least 20 cm below the support
            floor_dd, floor_mask = cand, sel
            break
    on_floor = floor_dd == dd
    hz = lambda X: (X @ n + floor_dd) / (n @ up)  # noqa: E731  height above the floor, along up

    # Walls: vertical planes (normal across up) by RANSAC on each wall mask's points in level coordinates.
    e1, e2 = basis(up)
    walls = []
    for m, kind in zip(surf['masks'], surf['kinds']):
        sel = m & ~things & vp
        if kind != 'wall' or sel.sum() < 2000:
            continue
        q = np.stack([P[sel] @ e1, P[sel] @ e2], 1)
        tol = max(0.03, 0.01 * float(np.median(np.linalg.norm(P[sel], axis=1))))  # depth noise grows with distance
        best = None
        for _ in range(300):
            a, b = q[rng.choice(len(q), 2, replace=False)]
            t = b - a
            if np.linalg.norm(t) < 0.05:
                continue
            nn = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            inl = np.abs((q - a) @ nn) < tol
            if best is None or inl.sum() > best[0].sum():
                best = (inl, nn)
        inl, nn = best
        if inl.mean() < 0.5:
            continue
        wm = nn[0] * e1 + nn[1] * e2
        c = float(np.median(P[sel][inl] @ wm))
        if c > 0:  # normal towards the camera (at the origin)
            wm, c = -wm, -c
        if any(abs(wm @ o['m']) > 0.95 and abs(c - o['c']) < 0.3 for o in walls):
            continue
        walls.append({'m': wm, 'c': c, 'mask': sel, 'inliers': round(float(inl.mean()), 3)})

    # Windows: each on the wall around it, as a rectangle on that wall's plane (the glass's own depth is the view
    # through it). piece_isaac.py puts a daylight panel there facing into the room.
    windows = []
    rays = points_from_depth(np.ones((h, w)), fx, fy, cx, cy)
    for m, kind in zip(surf['masks'], surf['kinds']):
        if kind != 'window' or not walls:
            continue
        # The frame and the reveal part the window from the wall mask: a ring of 3 % of the photo (home_office,
        # 2026-10-08: with 15 px the window touched no wall).
        r_px = 2 * max(8, round(0.03 * max(h, w))) + 1
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((r_px, r_px), np.uint8)) > 0) & ~m
        wl = max(walls, key=lambda x: (ring & x['mask']).sum())
        if (ring & wl['mask']).sum() < 0.1 * ring.sum():
            continue  # not in a wall the room has
        r = rays[m]
        t = wl['c'] / (r @ wl['m'])
        X = r[t > 0] * t[t > 0][:, None]
        along = np.cross(up, wl['m'])
        a = np.percentile(X @ along, [2, 98])
        v = np.percentile(X @ up, [2, 98])
        centre = wl['m'] * wl['c'] + along * a.mean() + up * v.mean()
        centre = centre - wl['m'] * (centre @ wl['m'] - wl['c'])
        windows.append({'centre': centre + wl['m'] * 0.02, 'normal': wl['m'], 'along': along,
                        'size': [float(np.ptp(a)), float(np.ptp(v))]})

    # Objects where the photo has them.
    meas, replica = photo_replica(masks, phrases, P, vp, depth_p, n, dd, intr, args.meshes, solid,
                                  os.path.join(args.meshes, 'ccm_pose.json'), fixed) if args.meshes else ({}, [])
    good = [(j, obj, tf) for j, obj, tf in replica if meas[j]['fit_iou'] >= args.min_iou]
    print(f'{len(good)} of {len(replica)} replica objects fit (silhouette IoU >= {args.min_iou})', flush=True)
    pivot = np.mean([meas[j]['base'] for j, _, _ in good], 0) if good else -n * dd

    # Materials, cut from the photo.
    mats = {'floor': material(photo, floor_mask, n, -floor_dd, up, intr, ppm, 'floor', out, depth=depth_p,
                              near=50 if args.outdoor else None)}
    if not on_floor:
        mats['support'] = material(photo, support, n, -dd, up, intr, ppm, 'support', out)
    for i, wl in enumerate(walls):
        mats[f'wall{i}'] = material(photo, wl['mask'], wl['m'], wl['c'], up, intr, ppm, f'wall{i}', out)
    wall_mat = next((mats[f'wall{i}'] for i in range(len(walls)) if 'texture' in mats[f'wall{i}']),
                    mats['wall0'] if walls else None)

    # Pieces in floor coordinates: x along the first wall (or e1), y away from it towards the camera.
    if walls:
        ay = walls[0]['m']
        ax = np.cross(ay, up)
    else:
        ax, ay = e1, e2
    floor_o = -up * floor_dd / (n @ up)  # the floor point under the camera

    def to2(X):
        return np.array([(X - floor_o) @ ax, (X - floor_o) @ ay])

    def to3(x, y, z=0.0):  # z above the floor at (x, y), which on a slope is not floor_o's height
        X = floor_o + x * ax + y * ay
        return X + (z - hz(X)) * up
    cam2 = to2(np.zeros(3))
    reach = np.linalg.norm(pivot) * 1.5
    objs2 = np.array([to2(meas[j]['base']) for j, _, _ in good]) if good else np.array([to2(pivot)])
    lo = np.minimum(objs2.min(0), cam2) - reach * 0.6
    hi = np.maximum(objs2.max(0), cam2) + reach * 0.6
    room = None
    if args.outdoor:
        lo, hi = lo - 10, hi + 10  # open ground to the horizon
        if not good:  # the place only: the ground must reach every ground pixel, the far buildings stand on it
            g2 = np.array([to2(p) for p in P[support][:: max(1, int(support.sum()) // 20000)]])
            lo = np.minimum(lo, np.percentile(g2, 1, 0) - 15)
            hi = np.maximum(hi, np.percentile(g2, 99, 0) + 15)
    elif walls:
        # The photo's walls bound the room; the sides it does not show close it behind and beside the camera.
        lo[1] = walls[0]['c']  # on the wall m.X = c, and y = (X - floor_o).m with floor_o.m = 0
        for wl in walls[1:]:
            if abs(wl['m'] @ ax) > 0.8:
                side = wl['c'] / (wl['m'] @ ax)
                if side < objs2[:, 0].min():
                    lo[0] = side
                elif side > objs2[:, 0].max():
                    hi[0] = side
        room = (lo.copy(), hi.copy())
    elif args.indoor:
        lo, hi = lo - 0.5, hi + 0.5
        room = (lo.copy(), hi.copy())
        wall_mat = {'colour': [0.78, 0.76, 0.72]}
    pieces = []
    # Wall height: the highest wall pixel above the floor, if the photo shows the wall's top; else --wall-height.
    tops = [float(np.percentile(hz(P[wl['mask']]), 99.5)) for wl in walls]
    H = max([args.wall_height] + [t + 0.05 for t in tops])
    pieces.append(quad([to3(lo[0], lo[1]), to3(hi[0], lo[1]), to3(hi[0], hi[1]), to3(lo[0], hi[1])], mats['floor']))
    gp = None
    if args.outdoor:  # the photo's own ground, laid over the tiled floor where the photo sees it
        def flat_at(x, y):
            Xf = floor_o + np.outer(x, ax) + np.outer(y, ay)
            return Xf - np.outer(hz(Xf), up)
        gp = ground_photo(d, intr, depth_p, flat_at, to2, lo, hi, mats['floor'], out)
        if gp:
            pieces.append({'points': [to3(*c, GROUND_PHOTO_LIFT).tolist() for c in gp['corners']],
                           'uv': [[0, 0], [1, 0], [1, 1], [0, 1]], 'texture': gp['texture'], 'colour': gp['colour'],
                           'ground_photo': True})
    if room:
        for a, b in (((lo[0], lo[1]), (hi[0], lo[1])), ((hi[0], lo[1]), (hi[0], hi[1])),
                     ((hi[0], hi[1]), (lo[0], hi[1])), ((lo[0], hi[1]), (lo[0], lo[1]))):
            pieces.append(quad([to3(*a, H), to3(*b, H), to3(*b), to3(*a)], wall_mat))
        # A plain ceiling a little lighter than the walls closes the room (the sky showed above the walls).
        pieces.append(quad([to3(lo[0], lo[1], H), to3(hi[0], lo[1], H), to3(hi[0], hi[1], H), to3(lo[0], hi[1], H)],
                           {'colour': np.clip(np.array(wall_mat['colour']) * 1.15, 0, 0.95).tolist()}))
    if args.outdoor:
        # Facades: each wall over the stretch and height the photo shows, a metre wider on each side.
        for i, wl in enumerate(walls):
            t = np.cross(up, wl['m'])
            pts = P[wl['mask']]
            a0, a1 = np.percentile(pts @ t, [1, 99]) + np.array([-1.0, 1.0])
            # Its height above the floor at both ends: on a slope a low wall follows the ground (a level top made
            # the stone wall of photo 24 grow downhill, 2026-10-13).
            top = float(np.percentile(hz(pts), 99.5)) + 0.3
            base = wl['m'] * wl['c']
            base = base - t * (base @ t)
            foot = [base + t * a - hz(base + t * a) * up for a in (a0, a1)]  # each end on the floor
            mat = mats[f'wall{i}'] if 'texture' in mats[f'wall{i}'] else wall_mat
            pieces.append(quad([f + top * up for f in foot] + foot[::-1], mat))
    slab = None
    if not on_floor:
        # The support as a block from the floor to its top, its outline the support pixels and the objects on it.
        top = (np.asarray(Image.open(os.path.join(d, 'support_raw.png'))) > 0) | np.any(masks, 0)
        rays = points_from_depth(np.ones((h, w)), fx, fy, cx, cy)
        t = -dd / (rays @ n)
        S = rays[top & (t > 0)] * t[top & (t > 0)][..., None]
        q = np.stack([to2(p) for p in S[:: max(1, len(S) // 20000)]]).astype(np.float32)
        (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(q)
        box = cv2.boxPoints(((rcx, rcy), (rw, rh), ang))
        zt = floor_dd - dd  # support top above the floor
        pieces.append(quad([to3(*p, zt) for p in box], mats['support']))
        side = {'colour': [c * 0.7 for c in mats['support']['colour']]}
        for i in range(4):
            a, b = box[i], box[(i + 1) % 4]
            pieces.append(quad([to3(*a, zt), to3(*b, zt), to3(*b, 0), to3(*a, 0)], side))
        slab = (box, zt)

    # Views: cameras around the objects at the photo's distance and tilt, turned up to --turn degrees, inside the
    # room; objects shuffled on the free support (--layout photo: where the photo has them), fixed ones kept.
    lib = [(r, obj, meas[j]) for r, (j, obj, tf) in enumerate(good) if meas[j]['phrase'] not in fixed]
    anchored = [(r, obj, tf, meas[j]) for r, (j, obj, tf) in enumerate(good) if meas[j]['phrase'] in fixed]
    p2, c_h = to2(pivot), float(hz(np.zeros(3)))  # camera height above the floor
    d0 = np.linalg.norm(cam2 - p2)
    tilt0 = math.atan2(c_h - hz(pivot), d0)
    sup_z = 0.0 if on_floor else slab[1]

    def free_spot(r, placed):
        for _ in range(100):
            if slab is None:  # near the objects the cameras look at, inside the room
                x, y = np.clip(p2 + rng.uniform(-1, 1, 2) * max(1.0, d0 * 0.5), lo + 0.4, hi - 0.4)
            else:
                box = slab[0]
                a, b = rng.uniform(0.1, 0.9, 2)
                x, y = box[0] + a * (box[1] - box[0]) + b * (box[3] - box[0])
            if all(math.hypot(x - q[0], y - q[1]) > (r + q[2]) + 0.02 for q in placed):
                return x, y
        return None

    # View 0: the replica from the photo camera, to compare with the photo.
    views, tries = [{'pose': np.eye(4).tolist(), 'objects': [[r, tf.tolist()] for r, (_, _, tf) in enumerate(good)],
                     'lamp': None if room is None else to3(*((lo + hi) / 2), H - 0.4).tolist()}], 0
    orbit = [float(a) for a in args.orbit.split(',') if a.strip()]
    for deg in orbit:  # the photo's camera turned about the vertical through the objects, its framing kept
        rot = trimesh.transformations.rotation_matrix(math.radians(deg), up, pivot)
        views.append({'pose': rot.tolist(), 'objects': list(views[0]['objects']), 'lamp': views[0]['lamp']})
    while not orbit and len(views) <= args.views and tries < 50 * args.views:
        tries += 1
        objects, placed = [], []
        for r, obj, tf, mi in anchored:
            objects.append([r, tf.tolist()])
            placed.append((*to2(mi['base']), mi['width_m'] / 2))
        if args.layout == 'photo':
            objects += [[r, tf.tolist()] for r, (j, obj, tf) in enumerate(good) if meas[j]['phrase'] not in fixed]
        else:
            for r, obj, mi in [lib[i] for i in rng.permutation(len(lib))]:
                kk = rng.uniform(0.95, 1.05)
                spot = free_spot(mi['width_m'] * kk / 2, placed)
                if spot is None:
                    continue
                placed.append((*spot, mi['width_m'] * kk / 2))
                tf = place(obj, to3(*spot, sup_z), up, mi['height_m'] * kk, mi['width_m'] * kk,
                           rng.uniform(0, 2 * math.pi))
                objects.append([r, tf.tolist()])
        turn = math.radians(rng.uniform(-args.turn, args.turn))
        dist = d0 * rng.uniform(0.9, 1.6)
        tilt = tilt0 + math.radians(rng.uniform(-12, 12))
        dir2 = (cam2 - p2) / max(d0, 1e-6)
        dir2 = np.array([dir2[0] * math.cos(turn) - dir2[1] * math.sin(turn),
                         dir2[0] * math.sin(turn) + dir2[1] * math.cos(turn)])
        e2d = p2 + dir2 * dist * math.cos(tilt)
        ez = hz(pivot) + dist * math.sin(tilt)
        if ez < 0.3 or ez > H - 0.2 or (room and not ((e2d > lo + 0.3) & (e2d < hi - 0.3)).all()):
            continue
        target = pivot + rng.normal(0, 0.05 * dist, 3)
        pose = look_at_cv(to3(*e2d, ez), target, up)
        lamp = to3(*rng.uniform(lo + 0.3, hi - 0.3), H - rng.uniform(0.2, 0.6)) if room else None
        views.append({'pose': pose.tolist(), 'objects': objects, 'lamp': None if lamp is None else lamp.tolist()})
    print(f'{len(views) - 1} views out of {tries} tries', flush=True)

    # For Isaac and view_labels.py: the pieces' triangles stand in for the background mesh.
    pts, faces = [], []
    for pc in pieces:
        base_i = len(pts)
        pts += pc['points']
        faces += [[base_i, base_i + 1, base_i + 2], [base_i, base_i + 2, base_i + 3]]
    arrays = {'bg_points': np.array(pts, np.float32), 'bg_faces': np.array(faces, np.int32)}
    for i, (_, obj, _) in enumerate(good):
        arrays[f'obj{i}_points'] = obj.vertices.astype(np.float32)
        arrays[f'obj{i}_faces'] = obj.faces.astype(np.int32)
        arrays[f'obj{i}_colors'] = (obj.visual.vertex_colors[:, :3] / 255).astype(np.float32)
    np.savez_compressed(os.path.join(out, 'meshes.npz'), **arrays)
    lamp = to3(*((lo + hi) / 2), H - 0.05) if room else None
    with open(os.path.join(out, 'scene.json'), 'w') as f:
        json.dump({'size': [w, h], 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'up': up.tolist(), 'objects': len(good),
                   'floor_plane': [n.tolist(), float(floor_dd)],
                   'views': views, 'pieces': pieces, 'indoor': room is not None,
                   'lamp': None if lamp is None else lamp.tolist(),
                   'windows': [{k2: np.asarray(v).tolist() for k2, v in x.items()} for x in windows] if room else []},
                  f, default=float)

    # Sheet of the materials: each surface's rectified pixels with the crop used, then the crops.
    tiles = []
    for name, mt in mats.items():
        im = Image.fromarray(mt['rectified'])
        if mt['crop']:
            y0, x0, y1, x1 = mt['crop']
            ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
        tiles.append((f'{name} rectified', im))
        if 'texture' in mt:
            tiles.append((f"{name} {mt['size_m'][0]:.2f} x {mt['size_m'][1]:.2f} m", Image.open(mt['texture'])))
    tw = 320
    sheet = Image.new('RGB', (tw * 4, tw * math.ceil(len(tiles) / 4)), 'white')
    for i, (name, im) in enumerate(tiles):
        im = im.copy()
        im.thumbnail((tw, tw))
        sheet.paste(im, ((i % 4) * tw, (i // 4) * tw))
        ImageDraw.Draw(sheet).text(((i % 4) * tw + 4, (i // 4) * tw + 4), name, fill=(255, 0, 0))
    sheet.save(os.path.join(d, 'pieces.jpg'), quality=88)
    report = {'plane_inlier_frac': round(inlier, 3), 'up': {'from': up_how, 'tilt_deg': tilt},
              'ground_photo': gp and {'seen': gp['seen'], 'corners_m': [list(map(float, c)) for c in gp['corners']]}, 'camera_height_above_support_m': round(float(dd), 3),
              'floor_below_support_m': round(float(floor_dd - dd), 3), 'walls': [
                  {'normal': wl['m'].round(3).tolist(), 'offset_m': round(wl['c'], 3), 'inliers': wl['inliers']}
                  for wl in walls], 'room': None if room is None else [room[0].tolist(), room[1].tolist()],
              'materials': {k2: {kk: v for kk, v in mt.items() if kk in ('texture', 'size_m', 'colour')}
                            for k2, mt in mats.items()},
              'objects': [{kk: v for kk, v in meas[j].items() if kk != 'base'} for j, _, _ in good],
              'views': len(views) - 1}
    with open(os.path.join(d, 'pieces.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    print(json.dumps({k2: v for k2, v in report.items() if k2 != 'objects'}, default=float))


if __name__ == '__main__':
    main()
