"""How a building stands: straight up, on the ground the photo shows under it, and kept so in a level template.

SAM 3D posed terrace's row of houses 2.5 degrees off the vertical the walls give, its base a metre over the
sidewalk at the near end; env_add.py read that as a raised lot and crossroad.py, setting the lot's lowest corner
on its level ground, made a 2.6 m pedestal of it (2026-10-14). Pure numpy: env_add.py, crossroad.py and
street_template.py use it.
"""
import numpy as np

MAX_TILT = 12.0  # degrees: a building tilted more than this is another failure, left alone
WALL_COS = 0.5  # a face whose normal is this close to horizontal (|n . up| under it) is a wall


def rotation(v, up):
    """3x3 turning the unit vector v onto up (Rodrigues)."""
    axis = np.cross(v, up)
    s, c = float(np.linalg.norm(axis)), float(v @ up)
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


def tilt(pts, faces, up):
    """The building's own vertical, from its walls: the direction their normals (area weighted) leave out, which is
    the up of a box whatever its proportions (the principal axes of terrace's points said 4.4 degrees, the roof's
    slope among them; its walls 2.5). None with too few walls."""
    v = up
    for _ in range(2):  # which faces are walls depends on the vertical: once more about the new one
        cr = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
        area = np.linalg.norm(cr, axis=1)
        n = cr / np.maximum(area[:, None], 1e-12)
        w = np.abs(n @ v) < WALL_COS
        if area[w].sum() < 0.2 * area.sum():
            return None
        ev, V = np.linalg.eigh((n[w] * area[w, None]).T @ n[w])
        # The least of the wall normals' directions; with one facade only, the two least are the vertical and
        # the facade's own horizontal: the one nearer up.
        cand = V[:, :2] if ev[1] < 0.05 * ev.sum() else V[:, :1]
        v = cand[:, int(np.argmax(np.abs(cand.T @ up)))]
        v = v * np.sign(v @ up)
    return v


def straighten(pts, faces, up):
    """The building turned about its centre so its walls stand along up. Returns (points, degrees turned)."""
    v = tilt(pts, faces, up)
    if v is None:
        return pts, 0.0
    deg = float(np.degrees(np.arccos(np.clip(v @ up, -1, 1))))
    if deg < 0.5 or deg > MAX_TILT:
        return pts, 0.0
    c = pts.mean(0)
    return (pts - c) @ rotation(v, up).T + c, deg


def photo_gap(mask, points, up, floor, min_cols=20):
    """How far the building's base stands over the floor where the photo sees it: the photo's own depth at the
    lowest pixel of each column of its mask, median over the columns; None when too few columns show it. A lot is
    what the photo shows (pots' photo 27: its houses behind a retaining wall, 1 to 3 m up), not the mesh."""
    cols = np.flatnonzero(mask.any(0))
    if len(cols) < min_cols:
        return None
    rows = mask.shape[0] - 1 - np.argmax(mask[::-1, cols], axis=0)
    p = points[rows, cols]
    p = p[np.isfinite(p).all(1)]
    if len(p) < min_cols:
        return None
    return float(np.median(p @ up - floor(p)))


def flatten(p, up, level, plane):
    """Points over a sloped floor plane set over a level one at `level`, each keeping its height over the floor
    right under it: verticals stay vertical and nothing a slope raised floats (a lot's lowest corner on a slope set
    on the template's ground lifted the lot 1.5 m, terrace 2026-10-14)."""
    n, d = np.asarray(plane[0], float), float(plane[1])
    under = p @ up - (p @ n + d) / (n @ up)
    return p + np.outer(level - under, up)
