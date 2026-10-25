"""How far a scene's buildings lean, measured on its images against the photo's (user, 2026-10-15: "how much more
tilted than in the original photo the simulation's buildings are", church_corner's houses leaning in views the
VLM passed).

Code, no model: every straight edge that runs near the camera's vertical (the vanishing point of up, from the
scene's intrinsics and each view's pose) is compared with that vertical at its own pixel. The photo's buildings
stand straight, so its edges give the error of the photo's camera and of the measure itself; a render's edges
leaning more than that are the scene's. Edges are pixels of strong gradient whose orientation agrees over a
neighbourhood (structure tensor coherence), so leaves and gravel add little. Pure numpy and scipy.

    python sim2chip/photo2scene/steps/lean.py --run <scene run dir>   # writes <dir>/lean.json
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

import standing

NEAR = 20.0      # degrees from the camera's vertical an edge may lean and still count as a vertical
COHERENT = 0.8   # structure tensor coherence of a straight edge (1 = one orientation in the window)
TOO_MUCH = 1.5   # degrees: a view whose buildings' median vertical leans this much more than the photo's leans
MIN_EDGES = 200  # edge pixels below which a view has no say (no building in it)
BUILDING_M, BUILDING_W = 3.0, 2.5  # a mesh this tall over a footprint this wide is a building (building_mask)


def edge_angles(img, K, up_cam, sigma=2.0, tau=0.08, mask=None):
    """Lean in degrees (signed) of every straight near-vertical edge pixel, and its strength."""
    g = np.asarray(img.convert('L'), np.float32) / 255
    gx, gy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
    jxx, jyy, jxy = (ndimage.gaussian_filter(a, sigma) for a in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-9)
    # The edge runs across the gradient: its direction is the tensor's minor eigenvector.
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    ex, ey = np.cos(theta), np.sin(theta)
    h, w = g.shape
    v = K @ up_cam  # the vertical vanishing point, homogeneous
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if abs(v[2]) > 1e-9:
        dx, dy = v[0] / v[2] - xx, v[1] / v[2] - yy
        s = np.sign(v[2])  # towards the vanishing point when it lies ahead, away from it when behind
        dx, dy = dx * s, dy * s
    else:
        dx, dy = np.full_like(xx, v[0]), np.full_like(yy, v[1])
    n = np.maximum(np.hypot(dx, dy), 1e-9)
    dx, dy = dx / n, dy / n
    lean = np.degrees(np.arctan2(ex * dy - ey * dx, ex * dx + ey * dy))
    lean = (lean + 90) % 180 - 90  # an edge has no sign of direction
    strong = (np.sqrt(tr) > tau) & (coh > COHERENT) & (np.abs(lean) < NEAR)
    if mask is not None:
        strong &= mask
    return lean[strong], np.sqrt(tr[strong])


USE_SHARE, USE_GAIN, USE_EDGES = 0.4, 0.1, 5000  # photo_vertical: when the edges' vertical is trusted


def photo_vertical(run, up0, intr, size):
    """The scene's up turned to the photo's verticals (vertical()) when they are clear: at least USE_SHARE of the
    near-vertical edge strength within 1 degree, USE_GAIN more than at up0, USE_EDGES edge pixels. Edges of the
    photo's buildings (photo_mask), of the whole photo without any. On the 2026-10-15 baseline: church_corner,
    facade_car, terrace, basque_street, straight_road and straight_whitewash turned 2 to 7 degrees (to a level
    phone, roll and pitch within 2 degrees); the roads photos, mostly trees, and home_office kept theirs."""
    f = next((os.path.join(run, x) for x in ('background.png', 'photo.png') if os.path.exists(os.path.join(run, x))))
    img = Image.open(f).convert('RGB')
    fx, fy, cx, cy = intr
    sx, sy = img.size[0] / size[0], img.size[1] / size[1]
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
    mask = photo_mask(run, img.size)
    up0 = np.asarray(up0, float) / np.linalg.norm(up0)
    e = edges(img, mask=mask)
    a = leans(e, K, up0)
    near = np.abs(a) < NEAR
    wsum = e[4][near].sum()
    before = float(e[4][near & (np.abs(a) < 1.0)].sum() / max(wsum, 1e-9))
    up, deg, share = vertical(img, K, up0, mask)
    used = share is not None and int(near.sum()) >= USE_EDGES and share >= USE_SHARE and share - before >= USE_GAIN
    return {'used': bool(used), 'up': up if used else up0, 'turned_deg': round(deg, 2), 'share': share,
            'share_before': round(before, 3), 'edges': int(near.sum()), 'buildings_mask': mask is not None}


def edges(img, sigma=2.0, tau=0.08, mask=None):
    """Straight edge pixels: (x, y, unit direction ex, ey, strength), from the structure tensor."""
    g = np.asarray(img.convert('L'), np.float32) / 255
    gx, gy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
    jxx, jyy, jxy = (ndimage.gaussian_filter(a, sigma) for a in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-9)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    ok = (np.sqrt(tr) > tau) & (coh > COHERENT)
    if mask is not None:
        ok &= mask
    y, x = np.nonzero(ok)
    return x.astype(np.float64), y.astype(np.float64), np.cos(theta[ok]), np.sin(theta[ok]), np.sqrt(tr[ok])


def leans(e, K, up_cam):
    """Signed lean in degrees of each edge from the camera's vertical at its pixel (K @ up_cam's vanishing point)."""
    x, y, ex, ey, _ = e
    v = K @ up_cam
    if abs(v[2]) > 1e-9:
        sg = np.sign(v[2])
        dx, dy = (v[0] / v[2] - x) * sg, (v[1] / v[2] - y) * sg
    else:
        dx, dy = np.full_like(x, v[0]), np.full_like(y, v[1])
    a = np.degrees(np.arctan2(ex * dy - ey * dx, ex * dx + ey * dy))
    return (a + 90) % 180 - 90


def vertical(img, K, up0, mask=None, span=10.0, max_edges=30000):
    """The photo's true vertical from its own straight edges: up0 (the camera's up, from the ground or the walls'
    depth) turned by at most `span` degrees so the most edge strength lines up with the vertical vanishing point.
    MoGe's walls put church_corner's up 7 degrees off the ground and 4 off the buildings the photo shows standing
    (2026-10-15). Returns (up, degrees turned, share of the near-vertical edges within 1 degree after)."""
    e = edges(img, mask=mask)
    a0 = leans(e, K, up0)
    near = np.abs(a0) < NEAR
    e = tuple(c[near] for c in e)
    if len(e[0]) < MIN_EDGES:
        return up0, 0.0, None
    if len(e[0]) > max_edges:
        keep = np.argsort(-e[4])[:max_edges]
        e = tuple(c[keep] for c in e)
    w = e[4]
    up0 = np.asarray(up0, float) / np.linalg.norm(up0)
    ax1 = np.cross(up0, [0, 0, 1.0])
    ax1 /= np.linalg.norm(ax1)
    ax2 = np.cross(up0, ax1)

    def turned(p, q):
        u = up0 + np.tan(np.radians(p)) * ax1 + np.tan(np.radians(q)) * ax2
        return u / np.linalg.norm(u)

    def score(u, width):
        return float((w * np.exp(-(leans(e, K, u) / width) ** 2)).sum())

    best = (0.0, 0.0)
    for step, half, width in ((1.0, span, 2.0), (0.25, 1.5, 1.0), (0.05, 0.3, 0.5)):
        grid = np.arange(-half, half + 1e-9, step)
        cands = [(best[0] + p, best[1] + q) for p in grid for q in grid]
        best = max(cands, key=lambda pq: score(turned(*pq), width))
    up = turned(*best)
    share = float(w[np.abs(leans(e, K, up)) < 1.0].sum() / w.sum())
    return up, float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))), round(share, 3)


def building_mask(arr, objects, K, pose, size, up, step=4):
    """Pixels where the view sees a building of the scene: its meshes splatted into a depth buffer at 1/step of
    the size (vertices and face centres), nearer things hiding farther ones, then grown back to full size.
    A building is a mesh at least BUILDING_M tall over a footprint at least BUILDING_W wide in both directions
    (the templates' copies carry no names); trees and shrubs are NVIDIA assets, not meshes."""
    w, h = size[0] // step, size[1] // step
    depth = np.full((h, w), np.inf, np.float32)
    owner = np.zeros((h, w), bool)
    R, t = np.asarray(pose)[:3, :3], np.asarray(pose)[:3, 3]
    up = np.asarray(up, float)
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    for i in objects:
        p = arr[f'obj{i}_points'].astype(np.float64)
        f = arr[f'obj{i}_faces']
        hgt = np.ptp(p @ up)
        wide = min(np.ptp(p @ a), np.ptp(p @ b))
        bld = hgt >= BUILDING_M and wide >= BUILDING_W
        q = np.concatenate([p, p[f].mean(1)]) if len(f) else p
        c = (q - t) @ R
        c = c[c[:, 2] > 0.3]
        if not len(c):
            continue
        u = (K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2]) / step
        v = (K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]) / step
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                x, y = np.round(u).astype(int) + du, np.round(v).astype(int) + dv
                ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                x, y, z = x[ok], y[ok], c[ok, 2].astype(np.float32)
                o = np.argsort(-z)  # far first: the nearest one written last wins
                x, y, z = x[o], y[o], z[o]
                nearer = z < depth[y, x]
                depth[y[nearer], x[nearer]] = z[nearer]
                owner[y[nearer], x[nearer]] = bld
    owner = ndimage.binary_closing(owner, iterations=2)
    big = np.asarray(Image.fromarray(owner.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    return ndimage.binary_dilation(big, iterations=6)  # the silhouette's own edges count


def view_lean(img, K, pose, up, mask=None):
    """Median absolute lean (degrees) of the view's verticals, weighted by edge strength, within mask (the
    buildings) when given; None with too few edges."""
    up_cam = np.asarray(pose)[:3, :3].T @ np.asarray(up)
    a, wgt = edge_angles(img, K, up_cam, mask=mask)
    if len(a) < MIN_EDGES:
        return None, len(a)
    o = np.argsort(np.abs(a))
    c = np.cumsum(wgt[o])
    return float(np.abs(a)[o][np.searchsorted(c, c[-1] / 2)]), int(len(a))


def scale_K(s, size):
    sx, sy = size[0] / s['size'][0], size[1] / s['size'][1]
    return np.array([[s['fx'] * sx, 0, s['cx'] * sx], [0, s['fy'] * sy, s['cy'] * sy], [0, 0, 1]])


def photo_mask(run, size):
    """The photo's buildings: the masks of its surroundings named as buildings (standing.is_building)."""
    f = os.path.join(run, 'env_masks.npz')
    if not os.path.exists(f):
        return None
    m = np.load(f)
    keep = [k for k, ph in enumerate(m['phrases']) if standing.is_building(str(ph))]
    if not keep:
        return None
    out = m['masks'][keep].any(0)
    if out.shape[::-1] != tuple(size):
        out = np.asarray(Image.fromarray(out.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    return ndimage.binary_dilation(out, iterations=6)


def measure(run):
    """How far the buildings lean in the photo (degrees, its own error), in the place from the photo's camera
    and in each extended view, and each view's excess over the photo; views over TOO_MUCH in 'leaning'."""
    out = {'views': {}}
    for part in ('place', 'extended'):
        sp = os.path.join(run, part, 'isaac', 'scene.json')
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        arr = np.load(os.path.join(run, part, 'isaac', 'meshes.npz'))
        meshes = [i for i in range(s['objects']) if f'obj{i}_points' in arr.files and f'obj{i}_faces' in arr.files
                  and f'obj{i}_usd' not in arr.files]
        if 'photo' not in out:
            bg = Image.open(os.path.join(run, 'background.png')).convert('RGB')
            out['photo'], out['photo_edges'] = view_lean(bg, scale_K(s, bg.size), np.eye(4), s['up'],
                                                         photo_mask(run, bg.size))
        for k, v in enumerate(s['views']):
            f = os.path.join(run, part, 'isaac', f'render_{k:02d}.png')
            if not os.path.exists(f):
                continue
            im = Image.open(f).convert('RGB')
            K = scale_K(s, im.size)
            mask = building_mask(arr, meshes, K, v['pose'], im.size, s['up'])
            lean, n = view_lean(im, K, v['pose'], s['up'], mask)
            if part == 'place':
                out['place'] = lean
            else:
                out['views'][k] = lean
    base = out.get('photo') or 0.0
    out['excess'] = {k: None if l is None else round(l - base, 2) for k, l in out['views'].items()}
    if out.get('place') is not None:
        out['place_excess'] = round(out['place'] - base, 2)
    out['leaning'] = [k for k, e in out['excess'].items() if e is not None and e > TOO_MUCH]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    out = measure(args.run)
    with open(os.path.join(args.run, 'lean.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ('photo', 'place', 'place_excess', 'leaning') if k in out}))


if __name__ == '__main__':
    main()
