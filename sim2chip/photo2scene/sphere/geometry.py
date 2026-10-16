"""S.2 of the sphere route, geometry: the ground plane, the vertical and the scale, and the proxy mesh, from the
360 distance map (steps/depth.py) and the ground votes (steps/stuff.py). Numpy only.

The plane is fitted by RANSAC on the points SAM 3 calls ground within FIT_RANGE_M of the camera, with a tolerance
that grows with distance. Its normal is the vertical (a phone's stitcher levels the horizon, so a tilt of more than
a few degrees means a bad fit or a bad stitch) and the camera's height over it is MoGe-2's metric measure. That is
the scale unless the user gives a capture height more than HEIGHT_OVERRIDE apart from it (then the user's height
over the measured one is the scale; within it, MoGe's measure stands, user, 2026-10-16): on the flat
spheres of 2026-10-16 MoGe said 1.7-1.8 m for a phone held at face height by a 1.85 m tall person. Only the near ground fits: farther away the measured ground rises over the plane, about
linearly with distance (2-4 degrees on the flat court of 2026-10-16, an angle or depth bias of the stitch or of
MoGe's merge), and real ground is not always flat (the park's slope).

So the proxy does not replace the ground by the plane: it is the distance map itself as one mesh, every vertex on
its pixel's ray, which keeps it consistent with the panorama whatever that bias is (an object put on it lands on
the pixel the ground shows). Ground within NEAR_GROUND_M of the plane is moved onto it along its ray (clean, flat
shadows; the blur the merge leaves at the foot of cars and walls goes with it), the ground under the capture's lower
edge is filled with the plane along the rays, sky is left out, and faces across depth jumps are cut (they would hide
targets and cast false shadows), except between ground pixels, whose distances jump at grazing angles.

Frame: the sphere's (capture.py), levelled: the capture point at the origin, +y down to the plane at
y = the camera's height, metres.

Big vertical planes (walls, facades) are straightened the same way: found by RANSAC among the points whose normal
is horizontal and that SAM 3 calls building, wall or fence (never trees or hedges), seen from above as lines (the
vertical is known), and their points moved onto them along their rays.
MoGe's per-view bias, spread by the merge, bowed straight walls by 1-2 % of their distance.

Once S.3 has erased the objects, the proxy is built again from the depth of the erased panorama (depth_bg.npz),
brought to the first depth's scale by their median ratio where nothing was erased; the plane and the scale stay
the first depth's.

Writes in the place folder geometry.json, proxy.npz (vertices, faces, ground: per vertex) and proxy.ply.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import capture

GROUND_SHARE = 0.5      # share of the cuts' votes that makes a pixel ground
FIT_RANGE_M = 10.0      # the plane is fitted on ground nearer than this (MoGe's scale)
GROUND_TOL = 0.02       # RANSAC inlier distance to the plane, as a share of the point's distance
MAX_TILT_DEG = 15.0     # a plane tilted more than this from the stitcher's vertical is not taken
RANSAC_ITERS = 800
RANSAC_POINTS = 20000
MESH_STEP = 2           # proxy vertices every MESH_STEP pixels of the distance map (1024 x 512)
EDGE_RTOL = 0.06        # a face whose corners' distances differ by more than this share crosses a depth jump
NEAR_GROUND_M = 0.15    # surfaces this close to the plane (or under it) are ground, moved onto it
MAX_DISTANCE_M = 150.0  # farther surfaces are background only, no proxy
FILL_MIN_DOWN_DEG = 5.0  # uncaptured rays at least this far below the horizon are filled with the plane
HEIGHT_OVERRIDE = 0.10  # a given capture height replaces MoGe's when they differ by more than this share
# Walls: big vertical planes found in the proxy and straightened (user, 2026-10-17: straight walls came out bowed,
# 1-2 % of their distance, from MoGe's per-view bias spread by the merge).
WALL_MAX_UP = 0.3          # a point's normal with |vertical part| under this is on a vertical surface
WALL_MIN_HEIGHT_M = 0.3    # wall points stand this much over the ground (kerbs and bases are left alone)
STRUCTURE_SHARE = 0.5      # and SAM 3 calls them building, wall or fence (stuff.py's "structure" votes)
WALL_TOL = 0.03            # RANSAC inlier distance to a wall, as a share of the point's horizontal distance
WALL_TOL_MIN_M = 0.08      # but never under this
WALL_NORMAL_AGREE = 0.85   # a wall point's normal within about 30 degrees of the wall's
WALL_MIN_POINTS = 300      # fewer proxy vertices (at MESH_STEP) is not a wall worth straightening (a 1 m wall
                           # 13 m away has a few hundred; SAM 3's structure votes keep trees out)
WALL_MIN_SPAN_M = (3.0, 0.6)  # nor is one narrower or lower than this (width along it, height; a court's low
                              # surrounding wall is about 1 m)
MAX_WALLS = 12
WALL_ITERS = 400
WALL_MAX_MOVE = 0.06       # a point moves onto its wall at most this share of its distance
WALL_GAP_M = 1.5           # a wall is one stretch: a gap this wide along its line ends it
WALL_MAX_DIST_M = 40.0     # farther, the depth is too rough to tell a wall from a row of things
WALL_MIN_INCIDENCE = 0.2   # rays grazing a wall (|cos| under this) are left as they were
# Distances from the ground contact (single view metrology: Criminisi et al. 2000, Hoiem et al. 2006). MoGe
# shortens far and oblique distances (corner_cars 2026-10-17: 94-96 % of the true distance in front of the camera,
# 75-83 % at a wall's oblique ends), which bowed straight walls and lifted the far ground. Paved ground is taken as
# flat and becomes the ruler: its distance is where its ray meets the plane, and a wall, fence or facade standing on
# it takes, down its whole column, the horizontal distance of the point where its base meets it.
PAVED = ("road", "parking", "sidewalk", "floor")  # ground kinds taken as flat (grass may slope: MoGe stays there)
CONTACT_MAX_M = 25.0       # farther, one pixel of error in the base is metres: MoGe stays
CONTACT_GAP_PX = 3         # ground this close under a structure run (mesh pixels) is the run's base
CONTACT_SMOOTH = 9         # columns: running median of the contact distance along a wall (a pixel at 10 m ~ 4 %)
CONTACT_FILL = 20          # columns: shorter gaps along a wall (a lamp post, a tree) take their neighbours'
CONTACT_JUMP = 0.15        # neighbouring columns whose contacts differ more are different walls
CONTACT_RATIO = (0.7, 1.6)  # a contact distance this far from MoGe's is a wrong contact: MoGe stays
UP = np.array([0.0, -1.0, 0.0])


class GroundError(ValueError):
    pass


def fit_plane(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares plane n.p + d = 0 through points, n pointing up (towards the camera above it)."""
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    if n @ UP < 0:
        n = -n
    return n, float(-n @ c)


def fit_ground(points: np.ndarray, seed: int = 0) -> dict:
    """RANSAC plane through ground points (N, 3) seen from the origin: normal (up), the camera's height over it,
    its tilt from the sphere's vertical, inliers and their median distance to it."""
    if len(points) < 100:
        raise GroundError(f"only {len(points)} ground points")
    rng = np.random.default_rng(seed)
    sub = points[rng.choice(len(points), min(RANSAC_POINTS, len(points)), replace=False)]
    tol = GROUND_TOL * np.linalg.norm(sub, axis=1)
    best, best_count = None, 0
    cos_max = math.cos(math.radians(MAX_TILT_DEG))
    for _ in range(RANSAC_ITERS):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ UP < 0:
            n = -n
        if n @ UP < cos_max:
            continue
        count = int((np.abs(sub @ n - n @ a) < tol).sum())
        if count > best_count:
            best, best_count = (n, float(-n @ a)), count
    if best is None:
        raise GroundError("no near-level plane among the ground points")
    n, d = best
    tol_all = GROUND_TOL * np.linalg.norm(points, axis=1)
    for _ in range(3):  # refit on the inliers of the whole set
        inl = np.abs(points @ n + d) < tol_all
        n, d = fit_plane(points[inl])
    resid = np.abs(points[inl] @ n + d)
    return {"normal": n.tolist(), "height_m": d, "tilt_deg": math.degrees(math.acos(min(1.0, float(n @ UP)))),
            "inliers": round(float(inl.mean()), 3), "residual_m": float(np.median(resid)), "points": len(points)}


def levelling(n) -> np.ndarray:
    """Rotation taking the ground's normal to straight up (Rodrigues)."""
    n = np.asarray(n, float)
    v, c = np.cross(n, UP), float(n @ UP)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def grid_faces(keep: np.ndarray, distance: np.ndarray, smooth: np.ndarray | None = None) -> np.ndarray:
    """Two triangles per pixel quad of an equirectangular grid (wrapping around), where its four corners are kept
    and their distances agree within EDGE_RTOL, or all four are `smooth` (ground). Vertex index = row * width +
    column."""
    H, W = keep.shape
    idx = np.arange(H * W).reshape(H, W)
    a, b = idx[:-1], np.roll(idx, -1, 1)[:-1]       # top-left, top-right
    c, d = idx[1:], np.roll(idx, -1, 1)[1:]         # bottom-left, bottom-right
    corners = np.stack([a, b, c, d], -1)
    dd = distance.ravel()[corners]
    ok = dd.max(-1) <= dd.min(-1) * (1 + EDGE_RTOL)
    if smooth is not None:
        ok |= smooth.ravel()[corners].all(-1)
    ok &= keep.ravel()[corners].all(-1)
    q = corners[ok]
    return np.concatenate([q[:, [0, 2, 1]], q[:, [1, 2, 3]]]).astype(np.int32)


def grid_normals(P: np.ndarray) -> np.ndarray:
    """Unit normals of an equirectangular point grid (H, W, 3), from its neighbours (wrapping around); the sign
    is arbitrary."""
    dx = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    dy = np.zeros_like(P)
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dy, dx)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def fit_walls(P: np.ndarray, normals: np.ndarray, candidate: np.ndarray, seed: int = 0) -> list[dict]:
    """Vertical planes through candidate points of a levelled metric point grid, one after another, biggest
    first: each a line seen from above, n.(x, z) + c = 0. RANSAC draws a point and takes the line through it
    along its own normal; inliers lie within WALL_TOL of the line and face the same way."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(candidate.ravel())
    q = P.reshape(-1, 3)[idx][:, [0, 2]]
    nq = normals.reshape(-1, 3)[idx][:, [0, 2]]
    nq = nq / np.maximum(np.linalg.norm(nq, axis=1, keepdims=True), 1e-9)
    tol = np.maximum(WALL_TOL * np.linalg.norm(q, axis=1), WALL_TOL_MIN_M)
    y = P.reshape(-1, 3)[idx][:, 1]
    left = np.ones(len(idx), bool)
    walls = []
    for _ in range(MAX_WALLS):
        pool = np.flatnonzero(left)
        if len(pool) < WALL_MIN_POINTS:
            break
        sub = pool[rng.choice(len(pool), min(RANSAC_POINTS, len(pool)), replace=False)]
        best, best_count = None, 0
        for k in rng.choice(sub, min(WALL_ITERS, len(sub)), replace=False):
            n2 = nq[k]
            on = (np.abs((q[sub] - q[k]) @ n2) < tol[sub]) & (np.abs(nq[sub] @ n2) > WALL_NORMAL_AGREE)
            if on.sum() > best_count:
                best, best_count = (n2, float(-q[k] @ n2)), int(on.sum())
        if best is None:
            break
        n2, c = best
        for _ in range(2):  # refit on the inliers of everything left, then take them again
            on = left & (np.abs(q @ n2 + c) < tol) & (np.abs(nq @ n2) > WALL_NORMAL_AGREE)
            if on.sum() < 3:
                break
            centre = q[on].mean(0)
            t = np.linalg.svd(q[on] - centre, full_matrices=False)[2][0]
            n2 = np.array([-t[1], t[0]])
            c = float(-centre @ n2)
        on = stretch(on, q, n2)
        along = q[on] @ np.array([-n2[1], n2[0]])
        span = (float(np.ptp(along)) if on.any() else 0.0, float(np.ptp(y[on])) if on.any() else 0.0)
        left &= ~on
        if on.sum() < WALL_MIN_POINTS or span[0] < WALL_MIN_SPAN_M[0] or span[1] < WALL_MIN_SPAN_M[1]:
            continue
        t2 = np.array([-n2[1], n2[0]])
        ends = [(-c * n2 + a * t2).round(3).tolist() for a in (float(along.min()), float(along.max()))]
        walls.append({"normal": [float(n2[0]), 0.0, float(n2[1])], "offset": c, "points": idx[on], "ends_xz": ends,
                      "width_m": round(span[0], 2), "height_m": round(span[1], 2),
                      "residual_m": round(float(np.median(np.abs(q[on] @ n2 + c))), 4)})
    return walls


def stretch(on: np.ndarray, q: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """Of a line's inliers, the longest run without a gap of WALL_GAP_M along it: one wall, not every building
    and tree that happens to line up with it."""
    k = np.flatnonzero(on)
    if len(k) < 2:
        return on
    along = q[k] @ np.array([-n2[1], n2[0]])
    order = np.argsort(along)
    cuts = np.flatnonzero(np.diff(along[order]) > WALL_GAP_M) + 1
    runs = np.split(order, cuts)
    out = np.zeros_like(on)
    out[k[max(runs, key=len)]] = True
    return out


def straighten(dist_m: np.ndarray, rays_level: np.ndarray, walls: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Distances along the rays with each wall's points moved onto its plane (at most WALL_MAX_MOVE of their
    distance, never along grazing rays), and the mask of the points moved."""
    out, moved = dist_m.copy().ravel(), np.zeros(dist_m.size, bool)
    r = rays_level.reshape(-1, 3)
    for w in walls:
        k = w["points"]
        cos = r[k] @ np.array(w["normal"])
        t = -w["offset"] / np.where(np.abs(cos) > 1e-9, cos, np.nan)
        ok = (np.abs(cos) > WALL_MIN_INCIDENCE) & (t > 0) & (np.abs(t - out[k]) <= WALL_MAX_MOVE * out[k])
        n3 = np.array(w["normal"])
        w["stray_before_m"] = round(float(np.sqrt(np.mean((r[k] * out[k, None] @ n3 + w["offset"]) ** 2))), 4)
        out[k[ok]], moved[k[ok]] = t[ok], True
        w["stray_after_m"] = round(float(np.sqrt(np.mean((r[k] * out[k, None] @ n3 + w["offset"]) ** 2))), 4)
        w["moved"] = round(float(ok.mean()), 3)
    return out.reshape(dist_m.shape), moved.reshape(dist_m.shape)


def pool(m: np.ndarray, s: int) -> np.ndarray:
    """A map averaged over s x s blocks (the votes at the mesh's resolution, still soft)."""
    H, W = m.shape[0] // s * s, m.shape[1] // s * s
    return m[:H, :W].astype(np.float32).reshape(H // s, s, W // s, s).mean((1, 3))


def flat_ground(dist_m: np.ndarray, rays_level: np.ndarray, paved: np.ndarray, height: float):
    """Paved ground at the distance where its ray meets the plane (up to CONTACT_MAX_M): new distances and the
    mask of the pixels set."""
    ry = rays_level[..., 1]
    hor = np.hypot(rays_level[..., 0], rays_level[..., 2])
    use = paved & (ry > 0) & (height * hor < CONTACT_MAX_M * np.maximum(ry, 1e-9))
    return np.where(use, height / np.maximum(ry, 1e-9), dist_m), use


def runs(col: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) rows of each run of True in a column."""
    k = np.flatnonzero(np.diff(np.concatenate([[0], col.astype(np.int8), [0]])))
    return list(zip(k[::2], k[1::2] - 1))


def smooth_along(inv: np.ndarray) -> np.ndarray:
    """1/D along the columns (wrapping): gaps up to CONTACT_FILL between agreeing walls filled linearly, then a
    running median of CONTACT_SMOOTH inside each wall (a jump of CONTACT_JUMP starts another)."""
    W = len(inv)
    out = inv.copy()
    have = np.flatnonzero(~np.isnan(inv))
    if len(have) < 2:
        return out
    for a, b in zip(have, np.roll(have, -1)):
        gap = (b - a) % W
        if 1 < gap <= CONTACT_FILL and abs(inv[b] / inv[a] - 1) < CONTACT_JUMP:
            for k in range(1, gap):
                out[(a + k) % W] = inv[a] + (inv[b] - inv[a]) * k / gap
    sm = out.copy()
    r = CONTACT_SMOOTH // 2
    for x in np.flatnonzero(~np.isnan(out)):
        win = out[[(x + k) % W for k in range(-r, r + 1)]]
        win = win[~np.isnan(win) & (np.abs(win / out[x] - 1) < CONTACT_JUMP)]
        sm[x] = np.median(win)
    return sm


def contact_distances(dist_m: np.ndarray, rays_level: np.ndarray, structure: np.ndarray, paved: np.ndarray,
                      valid: np.ndarray, height: float):
    """Each column's structure standing on paved ground at the horizontal distance of its base (see PAVED). Per
    column, the lowest structure run whose base lies within CONTACT_GAP_PX of paved ground; the contact is taken
    halfway between the run's last row and the ground's first. Returns the new distances, the mask of pixels set
    and the per-column contact distances (horizontal, metres; NaN where none)."""
    H, W = dist_m.shape
    hor = np.hypot(rays_level[..., 0], rays_level[..., 2])
    inv = np.full(W, np.nan)
    span = [None] * W
    for x in range(W):
        for top, bottom in reversed(runs(structure[:, x] & valid[:, x])):
            below = np.flatnonzero(paved[bottom + 1:bottom + 1 + CONTACT_GAP_PX, x])
            if not len(below):
                continue
            g = bottom + 1 + below[0]
            r = rays_level[bottom, x] + rays_level[g, x]
            r /= np.linalg.norm(r)
            if r[1] <= 0:
                break
            D = height * math.hypot(r[0], r[2]) / r[1]
            moge = np.median(dist_m[top:bottom + 1, x] * hor[top:bottom + 1, x])
            if D < CONTACT_MAX_M and CONTACT_RATIO[0] <= D / max(moge, 1e-6) <= CONTACT_RATIO[1]:
                inv[x], span[x] = 1 / D, (top, bottom)
            break
    sm = smooth_along(inv)
    out, set_ = dist_m.copy(), np.zeros_like(valid)
    for x in np.flatnonzero(~np.isnan(sm)):
        if span[x] is None:  # a filled gap: the structure between its neighbours' rows
            near = [span[(x + k) % W] for k in range(-CONTACT_FILL, CONTACT_FILL + 1) if span[(x + k) % W]]
            if not near:
                continue
            top, bottom = min(t for t, _ in near), max(b for _, b in near)
            rows = np.arange(top, bottom + 1)
            rows = rows[structure[rows, x] & valid[rows, x]]
        else:
            rows = np.arange(span[x][0], span[x][1] + 1)
        out[rows, x] = (1 / sm[x]) / np.maximum(hor[rows, x], 1e-6)
        set_[rows, x] = True
    return out, set_, np.where(np.isnan(sm), np.nan, 1 / sm)


def far_rise(points: np.ndarray, n: np.ndarray, h: float, lo: float, hi: float) -> float | None:
    """Median height of ground points between lo and hi metres over the plane: how far the measured ground drifts
    from it (MoGe's scale)."""
    r = np.linalg.norm(points, axis=1)
    sel = (r >= lo) & (r < hi)
    return float(np.median(points[sel] @ n + h)) if sel.sum() > 100 else None


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\nproperty float x\n"
            f"property float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\n"
            "end_header\n").encode()
    f = np.zeros(len(faces), [("n", "u1"), ("v", "<i4", 3)])
    f["n"], f["v"] = 3, faces
    Path(path).write_bytes(head + vertices.astype("<f4").tobytes() + f.tobytes())


def solve(d: Path, capture_height_m: float | None = None, proxy_depth: str | None = None,
          proxy_stuff: str | None = None, straighten_walls: bool = True, contacts: bool = True) -> dict:
    """The ground, scale and proxy of a place whose depth.npz and stuff.npz exist (see the module docstring).
    capture_height_m: the height the user says the sphere was taken at, if any; it is used only when it differs
    from MoGe-2's measure by more than HEIGHT_OVERRIDE. proxy_depth: the depth file of the erased panorama, for
    the proxy (S.3); without it the proxy comes from the first depth. proxy_stuff: the ground votes on the erased
    panorama, for the proxy's ground (where the erased objects stood is ground now)."""
    d = Path(d)
    depth, stuff = np.load(d / "depth.npz"), np.load(d / "stuff.npz")
    distance, valid = depth["distance"], depth["valid"]
    H, W = distance.shape
    seen = np.asarray(Image.open(d / "seen.png").resize((W, H), Image.Resampling.NEAREST)) > 127
    ground = (stuff["ground"].astype(np.float32) > GROUND_SHARE) & valid
    rays = capture.pixel_rays(W, H)
    points = rays * distance[..., None]
    near = ground & (distance < FIT_RANGE_M)
    g = fit_ground(points[near])
    n, h = np.array(g["normal"]), g["height_m"]
    g["rise_10_20_m"] = far_rise(points[ground], n, h, 10, 20)
    given = bool(capture_height_m) and abs(capture_height_m - h) > HEIGHT_OVERRIDE * h
    height = capture_height_m if given else h
    scale = height / h
    R = levelling(n)

    ratio = 1.0
    if proxy_depth:
        bg = np.load(d / proxy_depth)
        erased = np.asarray(Image.open(d / "erased.png").resize((W, H), Image.Resampling.BILINEAR)) > 0
        same = valid & bg["valid"] & ~erased
        ratio = float(np.median(distance[same] / np.maximum(bg["distance"][same], 1e-6)))
        distance, valid = bg["distance"] * ratio, bg["valid"]
    if proxy_stuff:
        stuff = np.load(d / proxy_stuff)
        ground = (stuff["ground"].astype(np.float32) > GROUND_SHARE) & valid
    s = MESH_STEP
    ray, dist, ok, grd = rays[::s, ::s], distance[::s, ::s].copy(), valid[::s, ::s], ground[::s, ::s]
    down = -(ray @ n)  # how steeply each ray points at the plane
    plane = np.where(down > 1e-6, h / np.maximum(down, 1e-6), np.inf)  # distance along the ray to the plane
    over = (ray * dist[..., None]) @ n + h
    snap = ok & (over < NEAR_GROUND_M / scale)
    fill = ~seen[::s, ::s] & (down > math.sin(math.radians(FILL_MIN_DOWN_DEG)))
    dist = np.where(snap | fill, plane, dist)
    is_ground = grd | snap | fill
    dist_m = dist * scale
    rays_level = ray @ R.T
    contact = {}
    if contacts and all(k in stuff for k in (*PAVED, "structure")):
        paved = np.max([pool(stuff[k], s) for k in PAVED], 0) > GROUND_SHARE
        paved &= ok[:paved.shape[0], :paved.shape[1]]
        structure = pool(stuff["structure"], s) > STRUCTURE_SHARE
        dist_m, flat = flat_ground(dist_m, rays_level, paved, height)
        is_ground |= flat
        dist_m, on_contact, per_col = contact_distances(dist_m, rays_level, structure & ~is_ground, paved, ok, height)
        contact = {"paved_px": int(flat.sum()), "structure_px": int(on_contact.sum()),
                   "columns": int(np.isfinite(per_col).sum())}
    walls = []
    if straighten_walls:
        P = rays_level * dist_m[..., None]
        candidate = ok & ~is_ground & (dist_m < WALL_MAX_DIST_M) & (height - P[..., 1] > WALL_MIN_HEIGHT_M)
        normals = grid_normals(P)
        candidate &= np.abs(normals[..., 1]) < WALL_MAX_UP
        if "structure" in stuff:
            candidate &= stuff["structure"][::s, ::s].astype(np.float32) > STRUCTURE_SHARE
        walls = fit_walls(P, normals, candidate)
        dist_m, moved = straighten(dist_m, rays_level, walls)
    keep = (ok | fill) & (dist_m < MAX_DISTANCE_M)
    faces = grid_faces(keep, dist_m, smooth=is_ground)
    vertices = ((ray * dist_m[..., None]) @ R.T).reshape(-1, 3)
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    vertices, faces = vertices[used].astype(np.float32), remap[faces].astype(np.int32)
    np.savez_compressed(d / "proxy.npz", vertices=vertices, faces=faces, ground=is_ground.ravel()[used])
    # The corrected distances on the mesh grid (pano frame rays, metres), for placing objects on the same ground.
    np.savez_compressed(d / "proxy_distance.npz", distance=dist_m.astype(np.float32), valid=ok | fill,
                        ground=is_ground, step=s)
    write_ply(d / "proxy.ply", vertices, faces)
    out = {"camera_height_m": round(height, 4), "height_from": "given" if given else "moge",
           "height_given_m": capture_height_m, "proxy_from": proxy_depth or "depth.npz",
           "proxy_depth_ratio": round(ratio, 5),
           "proxy_ground_from": proxy_stuff or "stuff.npz",
           "scale": round(scale, 4), "levelling": R.round(6).tolist(),
           "ground": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()},
           "contact": contact,
           "walls": [{k: v for k, v in w.items() if k != "points"} | {"vertices": len(w["points"])}
                     for w in walls],
           "proxy": {"vertices": len(vertices), "faces": len(faces),
                     "ground_vertices": round(float(is_ground.ravel()[used].mean()), 3)}}
    (d / "geometry.json").write_text(json.dumps(out, indent=1))
    return out
