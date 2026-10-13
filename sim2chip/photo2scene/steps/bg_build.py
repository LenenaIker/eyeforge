"""Step 3: our scene from one photo. Fits the support plane, measures every object above it, turns the erased
background into a mesh coloured by the photo, checks the geometry by putting the object meshes back where the
real ones stood, and renders new compositions from moved cameras. Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python bg_build.py --dir data/scene3d_test/ours/bottles \
        --meshes data/scene3d_test/cases/bottles [--fov-x 45.9]

--camera-height (metres) rescales everything to a known camera height; MoGe alone was ~10% too large.
--meshes is flow_sam3d.py's --out folder for instances.npz (mesh NNN is instance NNN, oriented by SAM 3D's pose),
or a Mira-Scene case whose mesh/sam3d/NNN.glb match its input/mask_NNN.png (matched to our instances by mask IoU); --fov-x (degrees, horizontal) overrides MoGe's field of view, e.g. from EXIF.
--layout photo keeps every object where the photo has it and only moves the camera (a copy of the scene, not
training variety).
World frame = the photo's OpenCV camera (+X right, +Y down, +Z forward, metres).
Writes sheet.jpg, report.json and scene.glb (background + replica) into --dir.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def points_from_depth(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)


def fit_plane(pts, iters=500, tol=0.004, seed=0):
    """RANSAC plane n.p + d = 0, refined by least squares on the inliers; n points towards the camera."""
    rng = np.random.default_rng(seed)
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
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][2]
    d = -n @ c
    if d < 0:  # camera (origin) on the positive side
        n, d = -n, -d
    return n, d, float(best.mean())


def floor_height(X, up, plane):
    """The up coordinate of the floor right under (or over) each point X: plane is (n, d) with n.p + d = 0, which
    need not be level (a street on a slope, piece_build.gravity)."""
    n, d = np.asarray(plane[0], float), float(plane[1])
    return X @ up - (X @ n + d) / (n @ up)


def grid_mesh(points, colors, valid, step=2, max_jump=0.04):
    """Triangulated depth map; drops faces across depth jumps (relative) so edges do not smear."""
    pts, col, ok = points[::step, ::step], colors[::step, ::step], valid[::step, ::step]
    h, w = ok.shape
    idx = np.arange(h * w).reshape(h, w)
    z = pts[..., 2]
    faces = []
    for tri in (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))):
        ids = [idx[dy:h - 1 + dy, dx:w - 1 + dx] for dy, dx in tri]
        zs = [z[dy:h - 1 + dy, dx:w - 1 + dx] for dy, dx in tri]
        oks = [ok[dy:h - 1 + dy, dx:w - 1 + dx] for dy, dx in tri]
        keep = oks[0] & oks[1] & oks[2]
        zmax, zmin = np.maximum.reduce(zs), np.minimum.reduce(zs)
        keep &= (zmax - zmin) < max_jump * zmin
        faces.append(np.stack([i[keep] for i in ids], -1))
    m = trimesh.Trimesh(pts.reshape(-1, 3), np.concatenate(faces), process=False)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.concatenate(
        [col.reshape(-1, 3), np.full((h * w, 1), 255, np.uint8)], 1))
    return m


def smooth_fill(z, known, iters=300):
    """Fills z where not known with a smooth surface through its border (cv2.inpaint on float maps is noisy and
    tears the mesh): averages over coarser grids for a first guess, then relaxes towards the harmonic fill."""
    out, levels = z.copy(), []
    cz, cw = z * known, known.astype(np.float32)
    while min(cz.shape) > 8:
        levels.append((cz, cw))
        cz = cv2.resize(cz, (cz.shape[1] // 2, cz.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cw = cv2.resize(cw, (cw.shape[1] // 2, cw.shape[0] // 2), interpolation=cv2.INTER_AREA)
    guess = np.where(cw > 0, cz / np.maximum(cw, 1e-9), float(z[known].mean()))
    for lz, lw in reversed(levels):
        up = cv2.resize(guess, (lz.shape[1], lz.shape[0]), interpolation=cv2.INTER_LINEAR)
        guess = np.where(lw > 0.99, lz / np.maximum(lw, 1e-9), up)
    out = np.where(known, z, guess).astype(np.float32)
    for _ in range(iters):
        out = np.where(known, z, cv2.blur(out, (3, 3)))
    return out


def rotation_to(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def close_shell(m, solid=False, near=0.02, min_cover=0.4, max_lids=0.35):
    """Caps for open shells. SAM 3D on the CCM voxels only rebuilds what the photo shows, so a lying can comes
    out as a tube open at both ends. When most of the convex hull's surface lies on the mesh (the object is
    roughly convex: cans, boxes, bottles; not chairs or bikes), the hull faces far from the mesh are added as
    lids, coloured like the nearest mesh vertex. A half seen object (the back of a lying can is missing too)
    looks like a desk's open underside to these tests, so `solid` (the object is known to be a closed, roughly
    convex thing: from the phrase, later from the VLM) skips them. Returns the mesh and the hull's covered share."""
    tree = cKDTree(m.vertices)
    hull = m.convex_hull
    tol = near * m.extents.max()
    cover = float(np.mean(tree.query(trimesh.sample.sample_surface(hull, 4000, seed=0)[0])[0] < tol))
    if cover < min_cover and not solid:
        return m, cover
    centres = hull.vertices[hull.faces].mean(1)
    far = tree.query(centres)[0] > tol
    # Lids are a can's two ends at most; more missing hull than that is a desk's open underside, not a hole.
    if not far.any() or (hull.area_faces[far].sum() > max_lids * hull.area and not solid):
        return m, cover
    lids = trimesh.Trimesh(hull.vertices, hull.faces[far], process=False)
    lids.visual.vertex_colors = m.visual.vertex_colors[tree.query(lids.vertices)[1]]
    return trimesh.util.concatenate([m, lids]), cover


def load_object(path, solid=False, cap=True):
    """SAM 3D mesh (y up), base centred at the origin, height 1 (scaled later), open shells capped (not with
    cap=False: a bookshelf's open front took diagonal hull faces as a lid)."""
    m = trimesh.load(path, force='mesh')
    if cap:
        m, _ = close_shell(m, solid)
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    m.apply_scale(1 / (hi[1] - lo[1]))
    return m


def place(mesh, base, n, height, width, yaw):
    """4x4 pose: mesh y axis along the plane normal n (pointing up, away from the plane), base on the plane,
    scaled to the measured height and width (SAM 3D meshes come out too wide for their height)."""
    t = np.eye(4)
    sxz = width / max(mesh.extents[0], mesh.extents[2])
    t[:3, :3] = (rotation_to(np.array([0, 1.0, 0]), n) @ trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])[:3, :3]
                 @ np.diag([sxz, height, sxz]))
    t[:3, 3] = base
    return t


def ccm_yaw(rot, n, min_up=0.9):
    """The yaw of place() closest to a mesh rotation from Mira's CCM fit (bg_ccm_pose.py), or None when that fit
    tilts the mesh's up axis away from the plane normal (a failed fit)."""
    y, x = np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
    rot = np.asarray(rot)
    if rot @ y @ n < min_up:
        return None
    base = rotation_to(y, n)
    v = rot @ x
    u0 = base @ x
    u90 = base @ trimesh.transformations.rotation_matrix(math.pi / 2, y)[:3, :3] @ x
    return math.atan2(v @ u90, v @ u0)


def fit_pose(obj, mask, others, mi, n, intr, scale=2, yaw0=None):
    """Yaw, width and a small shift of the base that make the mesh's silhouette from the photo camera match the
    object's mask best (IoU, ignoring pixels covered by other objects). Coarse to fine, at 1/scale resolution.
    With yaw0 (from the CCM fit) the yaw stays within 30 degrees of it: a silhouette cannot tell front from back."""
    w, h, fx, fy, cx, cy = intr
    r = pyrender.OffscreenRenderer(w // scale, h // scale)
    cam = pyrender.IntrinsicsCamera(fx / scale, fy / scale, cx / scale, cy / scale, znear=0.02, zfar=100)
    small = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w // scale, h // scale))) > 127  # noqa: E731
    target, ignore = small(mask), small(others) if others is not None else np.zeros((h // scale, w // scale), bool)
    pm = pyrender.Mesh.from_trimesh(obj)
    sc = pyrender.Scene()
    node = sc.add(pm)
    sc.add(cam, pose=CV2GL)

    def iou(base, width, yaw):
        sc.set_pose(node, place(obj, base, n, mi['height_m'], width, yaw))
        sil = r.render(sc, flags=pyrender.RenderFlags.DEPTH_ONLY) > 0
        sil &= ~ignore
        return (sil & target).sum() / max((sil | target).sum(), 1)

    # Two axes on the plane for the shift: across the view and away from the camera.
    side = np.array([1.0, 0, 0]) - n[0] * n
    side /= np.linalg.norm(side)
    away = np.cross(side, n)
    away *= np.sign(away @ mi['base'])
    start = 0.0 if yaw0 is None else yaw0
    best = (iou(mi['base'], mi['width_m'], start), mi['base'], mi['width_m'], start)
    yaws = np.radians(np.arange(0, 360, 30)) if yaw0 is None else yaw0 + np.radians(np.arange(-30, 31, 10))
    for yaw in yaws:
        for k in (0.7, 1.0, 1.4):
            v = iou(mi['base'], mi['width_m'] * k, yaw)
            if v > best[0]:
                best = (v, mi['base'], mi['width_m'] * k, yaw)
    for step in (0.25, 0.1):
        _, base0, width, yaw = best
        for a in (-2, -1, 0, 1, 2):
            for b in (-2, -1, 0, 1, 2):
                base = base0 + (a * side + b * away) * step * width
                v = iou(base, width, yaw)
                if v > best[0]:
                    best = (v, base, width, yaw)
    for _ in range(2):
        _, base, width, yaw0 = best
        for dy in np.radians([-10, -5, 5, 10]):
            for k in (0.9, 1.0, 1.1):
                v = iou(base, width * k, yaw0 + dy)
                if v > best[0]:
                    best = (v, base, width * k, yaw0 + dy)
    r.delete()
    return {'fit_iou': round(float(best[0]), 3), 'base': best[1], 'width_m': round(float(best[2]), 4),
            'yaw': float(best[3])}


def estimate_sun(shadows, masks, meas, P_rays, n, dd, photo_lum, support, min_agreement=0.85):
    """Sun direction from the cast shadows (bg_prepare --shadows): each shadow pixel goes to the nearest object,
    is dropped onto the support plane, and the object's base -> shadow tip gives the sun's azimuth (opposite way)
    and height / length its elevation. Also the shadow darkness (photo luminance in shadow / lit support around
    it), which Isaac matches with the sun strength. None when no object casts a usable shadow."""
    if not shadows.any():
        return None
    objs = [k for k, mi in enumerate(meas) if 'yaw' in mi and mi['on'] is None]  # stacked bases are off the plane
    dist = np.stack([cv2.distanceTransform((~masks[k]).astype(np.uint8), cv2.DIST_L2, 5) for k in objs])
    owner = np.argmin(dist, 0)
    t = -dd / (P_rays @ n)
    ground = P_rays * t[..., None]
    dirs, elevs, weights = [], [], []
    for i, k in enumerate(objs):
        sel = shadows & (owner == i) & (t > 0)
        if sel.sum() < 200:
            continue
        v = ground[sel] - meas[k]['base']
        v -= (v @ n)[:, None] * n
        d = np.median(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6), 0)
        if np.linalg.norm(d) < 0.3:  # shadow all around the base (sun overhead or a bad mask)
            continue
        d /= np.linalg.norm(d)
        length = np.percentile(v @ d, 95)
        dirs.append(d)
        elevs.append(math.degrees(math.atan2(meas[k]['height_m'], max(length, 1e-3))))
        weights.append(sel.sum())
    if not dirs:
        return None
    wts = np.array(weights, float)
    d = (np.array(dirs) * wts[:, None]).sum(0)
    # Under one sun every shadow points the same way; soft indoor light leaves contact shadows pointing anywhere.
    agreement = float(np.linalg.norm(d) / wts.sum())
    print(f'sun from shadows: {len(dirs)} objects, direction agreement {agreement:.2f}', flush=True)
    if len(dirs) < 2 or agreement < min_agreement:
        return None
    d /= np.linalg.norm(d)
    elev = float(np.median(elevs))
    e = math.radians(elev)
    travel = math.cos(e) * d - math.sin(e) * n  # light goes down and along the shadows
    ring = (cv2.dilate(shadows.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~shadows & support
    ring &= ~np.any(masks, 0)
    darkness = float(np.median(photo_lum[shadows]) / max(np.median(photo_lum[ring]), 1e-4)) if ring.any() else None
    return {'travel': travel.tolist(), 'elevation_deg': round(elev, 1), 'objects_used': len(dirs),
            'agreement': round(agreement, 3),
            'elevations_deg': [round(x, 1) for x in elevs], 'shadow_darkness': darkness}


def look_at_cv(eye, target, up):
    """OpenCV camera-to-world pose looking from eye at target, with image 'up' along +up."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


class Renderer:
    def __init__(self, w, h, fx, fy, cx, cy):
        self.r = pyrender.OffscreenRenderer(w, h)
        self.cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.02, zfar=2000)

    def render(self, pose_cv, background, objects, n):
        """Background unlit (keeps the photo's colours), objects lit; composited by depth."""
        pose = pose_cv @ CV2GL
        s = pyrender.Scene(bg_color=[0, 0, 0, 0])
        s.add(pyrender.Mesh.from_trimesh(background, smooth=False))
        s.add(self.cam, pose=pose)
        bg_col, bg_dep = self.r.render(s, flags=pyrender.RenderFlags.FLAT)
        s = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45] * 3)
        for m, t in objects:
            s.add(pyrender.Mesh.from_trimesh(m), pose=t)
        s.add(self.cam, pose=pose)
        light = np.eye(4)
        light[:3, :3] = look_at_cv(np.zeros(3), -n + np.array([0.3, 0, 0.3]), np.array([0, 0, 1.0]))[:3, :3] @ CV2GL[:3, :3]
        s.add(pyrender.DirectionalLight(intensity=3.0), pose=light)
        ob_col, ob_dep = self.r.render(s)
        front = (ob_dep > 0) & ((bg_dep == 0) | (ob_dep < bg_dep))
        col = np.where(front[..., None], ob_col, bg_col)
        dep = np.where(front, ob_dep, bg_dep)
        return col, dep, front


def resting_on(j, mask, owner, depth, probe=4, tol=0.03):
    """Index of the object that instance j rests on, or None (it rests on the support, or its base is hidden).
    Just below the mask's bottom edge, column by column: pixels of another object that is not clearly in front
    (depth not more than tol closer) mean j sits on it; an object in front only hides j's base."""
    h, w = mask.shape
    cols = np.nonzero(mask.any(0))[0]
    bottom = h - 1 - np.argmax(mask[::-1, cols], axis=0)
    rows = np.minimum(bottom + probe, h - 1)
    below = owner[rows, cols]
    under = (below >= 0) & (below != j) & (depth[rows, cols] > depth[bottom, cols] * (1 - tol))
    if under.sum() < 0.5 * len(cols):
        return None
    return int(np.bincount(below[under]).argmax())


def flat_top(obj, band=0.03, share=0.5):
    """Whether something can rest on the mesh: the vertices in its top band span at least `share` of its
    footprint (a box lid, not a lying can's curved side or a bottle neck)."""
    v = obj.vertices
    top = v[v[:, 1] > v[:, 1].max() - band * obj.extents.max()]  # tilted lids still count
    span = lambda p: np.prod(p[:, [0, 2]].max(0) - p[:, [0, 2]].min(0))  # noqa: E731
    return len(top) > 10 and span(top) >= share * span(v)


def measure(points, hgt, n, phrase, base, on):
    """Height above the base, width across the view (spread along the camera's x axis laid on the plane)."""
    side = np.array([1.0, 0, 0]) - n[0] * n
    side /= np.linalg.norm(side)
    across = points @ side
    return {'phrase': phrase, 'height_m': round(float(np.percentile(hgt, 98)), 4),
            'width_m': round(float(np.percentile(across, 98) - np.percentile(across, 2)), 4), 'base': base,
            'distance_m': round(float(np.linalg.norm(base)), 3), 'on': on}


def photo_replica(masks, phrases, P, vp, depth_p, n, dd, intr, meshes, solid, ccm_path, fixed=()):
    """Every instance measured on the photo's own depth (P, metres, OpenCV camera frame) and its SAM 3D mesh posed
    where the real object stands. Returns the measures (one dict per instance) and the replica: (instance, mesh,
    4x4 pose) for the instances with a mesh. Fixed furniture is not capped (open fronts by design)."""
    # Every instance: height above the plane and a first base centre (refined later by fit_pose): the centroid of
    # the points touching the plane (legs, wheels), or the top points dropped onto it when none are visible.
    meas = []
    w, h, fx, fy, cx, cy = intr
    owner = np.full((h, w), -1)
    for j, m in enumerate(masks):
        owner[m & (owner < 0)] = j
    for j, (m, phrase) in enumerate(zip(masks, phrases)):
        sel = m & vp
        hgt = P[sel] @ n + dd
        on = resting_on(j, m, owner, depth_p)
        if on is not None:
            # Stacked: the base is the object's lowest visible points, at their height above the plane, and the
            # height counts from there (not from the table, which would stretch it over the object below).
            lift = float(np.percentile(hgt, 3))
            low = P[sel][hgt < lift + 0.1 * (np.percentile(hgt, 98) - lift)]
            c = low.mean(0)
            base = c - (c @ n + dd - lift) * n
            meas.append(measure(P[sel], hgt - lift, n, phrase, base, on))
            continue
        low = P[sel][hgt < 0.1 * np.percentile(hgt, 98)]
        c = low.mean(0) if len(low) >= 50 else P[sel][hgt >= np.percentile(hgt, 90)].mean(0)
        base = c - (c @ n + dd) * n
        meas.append(measure(P[sel], hgt, n, phrase, base, None))

    replica = []
    ours = os.path.exists(os.path.join(meshes, 'scene_graph.json'))
    if ours:
        # flow_sam3d.py --masks instances.npz: mesh NNN is instance NNN, its orientation from SAM 3D's own pose.
        ccm = {}
        for j in range(len(meas)):
            info = os.path.join(meshes, 'sam3d', f'{j:03d}.json')
            if os.path.exists(info) and 'rot_cv' in json.load(open(info)):
                ccm[f'{j:03d}'] = json.load(open(info))['rot_cv']
    else:
        # Mira meshes matched to our instances by IoU on Mira's 518 centre crop.
        crop = json.load(open(os.path.join(meshes, 'input', 'preprocessing.json')))['crop_xywh']
        sx = w / json.load(open(os.path.join(meshes, 'input', 'preprocessing.json')))['original_size'][0]
        l, tp, side = [round(v * sx) for v in (crop[0], crop[1], crop[2])]
        ccm = json.load(open(ccm_path)) if os.path.exists(ccm_path) else {}
    for j, mi in enumerate(meas):
        if ours:
            best = f'{j:03d}'
            path = os.path.join(meshes, 'sam3d', f'{best}_canon.glb')
        else:
            crop_mask = np.asarray(Image.fromarray(masks[j][tp:tp + side, l:l + side].astype(np.uint8) * 255).resize((518, 518))) > 127
            best, best_iou = None, 0.1
            for f in sorted(os.listdir(os.path.join(meshes, 'input'))):
                if f.startswith('mask_'):
                    mm = np.asarray(Image.open(os.path.join(meshes, 'input', f))) > 0
                    iou = (mm & crop_mask).sum() / max((mm | crop_mask).sum(), 1)
                    if iou > best_iou:
                        best, best_iou = f[5:8], iou
            path = os.path.join(meshes, 'mesh', 'sam3d', f'{best}.glb')
        if best and not os.path.exists(path):
            best = None  # SAM 3D failed on it (e.g. out of memory): the object stays out of the replica
        mi['mesh'] = best
        if best:
            obj = load_object(path, mi['phrase'] in solid, cap=mi['phrase'] not in fixed)
            others = np.any([masks[k] for k in range(len(masks)) if k != j], axis=0) if len(masks) > 1 else None
            yaw0 = ccm_yaw(ccm[best], n) if best in ccm else None
            fitted = fit_pose(obj, masks[j], others, mi, n, (w, h, fx, fy, cx, cy), yaw0=yaw0)
            fitted['ccm_yaw'] = None if yaw0 is None else round(math.degrees(yaw0), 1)
            mi.update(fitted)
            replica.append((j, obj, place(obj, mi['base'], n, mi['height_m'], mi['width_m'], mi['yaw'])))
    return meas, replica


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--meshes', required=True)
    ap.add_argument('--fov-x', type=float, default=None)
    ap.add_argument('--camera-height', type=float, default=None, help='known camera height above the support, metres')
    ap.add_argument('--views', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-jump', type=float, default=0.04,
                    help='background faces spanning a larger relative depth jump are dropped (holes) instead of '
                         'stretched; large values keep them (rubber sheet over disocclusions)')
    ap.add_argument('--max-move', type=float, default=0.4, help='largest sideways camera move, metres')
    ap.add_argument('--yaw', type=float, default=20, help='largest camera turn around the objects, degrees')
    ap.add_argument('--stack', type=float, default=0.0,
                    help='chance that a shuffled object goes on top of another (piles, cluttered tables)')
    ap.add_argument('--min-iou', type=float, default=0.15, help='replica fit needed to shuffle an object (thin '
                    'objects such as chairs fit around 0.2-0.5 even when right)')
    ap.add_argument('--min-visible', type=float, default=0.7,
                    help='visible share of an object needed to shuffle it (hidden parts come out open)')
    ap.add_argument('--solid', default='', help='comma-separated phrases of closed, roughly convex objects '
                    '(cans, boxes): their open meshes are capped with the convex hull')
    ap.add_argument('--ccm-pose', default=None, help="each mesh's orientation from Mira's CCM (bg_ccm_pose.py); "
                    'default <meshes>/ccm_pose.json when it exists')
    ap.add_argument('--fixed', default='', help='comma-separated phrases of objects that stay where the photo has '
                    'them in every view (furniture, fixtures): they cover their own erased hole, which LaMa cannot '
                    'fill well when it is large; the others are shuffled around them')
    ap.add_argument('--layout', choices=['shuffle', 'photo'], default='shuffle',
                    help='shuffle: new views move the objects around the support (training data); photo: every '
                         'view keeps them where the photo has them, only the camera moves (a copy of the scene)')
    ap.add_argument('--max-empty', type=float, default=0.08,
                    help='largest unrendered fraction of a new view (filled by inpainting)')
    ap.add_argument('--groups', default=None,
                    help='JSON file {thing name: group}: each new view shows the things of one group only (things in '
                         'no group always), for one label per image when classifying')
    ap.add_argument('--empty', type=float, default=0.0, help='with --groups: share of views with no group at all')
    args = ap.parse_args()
    d = args.dir
    rng = np.random.default_rng(args.seed)
    solid = {p.strip() for p in args.solid.split(',') if p.strip()}
    fixed = {p.strip() for p in args.fixed.split(',') if p.strip()}

    photo = np.asarray(Image.open(os.path.join(d, 'photo.png')).convert('RGB'))
    background = np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB'))
    h, w = photo.shape[:2]
    support = np.asarray(Image.open(os.path.join(d, 'support.png'))) > 0
    erased = np.asarray(Image.open(os.path.join(d, 'erased.png'))) > 0
    inst = np.load(os.path.join(d, 'instances.npz'))
    masks, phrases = inst['masks'], list(inst['phrases'])
    dp, db = np.load(os.path.join(d, 'depth_photo.npz')), np.load(os.path.join(d, 'depth_background.npz'))

    k = dp['intrinsics']
    fx, fy, cx, cy = k[0, 0] * w, k[1, 1] * h, k[0, 2] * w, k[1, 2] * h
    moge_fov = math.degrees(2 * math.atan(w / 2 / fx))
    if args.fov_x:
        fx = fy = w / 2 / math.tan(math.radians(args.fov_x) / 2)
    P = points_from_depth(dp['depth'], fx, fy, cx, cy)
    B = points_from_depth(db['depth'], fx, fy, cx, cy)
    vp, vb = dp['mask'] > 0, db['mask'] > 0

    # Support plane from the photo, on pixels that were never under an object.
    plane_px = support & ~erased & vp
    n, dd, inlier = fit_plane(P[plane_px])
    moge_height = float(dd)
    # MoGe's metric scale is off by ~10% (bottles test); one known length (camera height from the interview) fixes it.
    scale = args.camera_height / dd if args.camera_height else 1.0
    P, dd, depth_p = P * scale, dd * scale, dp['depth'] * scale
    cam_height = float(dd)
    pitch = math.degrees(math.asin(abs(n[2])))  # camera axis vs plane: 90 = looking straight down
    # Background depth to the photo's scale (median over untouched pixels), then the support snapped to the plane.
    # Untouched pixels keep the photo's own depth: MoGe on a background with large erased holes can bend the whole
    # room (home office: walls off by more than 5%), and the replica is placed against the photo's depth.
    common = vp & vb & ~erased
    bg_scale = float(np.median(dp['depth'][common] / db['depth'][common]))
    B = B * bg_scale * scale
    B = np.where((vp & ~erased)[..., None], P, B)
    vb = vb | vp & ~erased
    rays = points_from_depth(np.ones((h, w)), fx, fy, cx, cy)  # unit-depth ray per pixel (B is inf where MoGe has no depth)
    t = -dd / (rays @ n)
    on_plane = rays * t[..., None]
    # Snap the support and the holes inside it (erased objects), not erased pixels beyond its edge.
    raw = (np.asarray(Image.open(os.path.join(d, 'support_raw.png'))) > 0).astype(np.uint8)
    outside = np.pad(1 - raw, 1, constant_values=1)
    cv2.floodFill(outside, None, (0, 0), 2)
    snap = (outside[1:-1, 1:-1] != 2) & (t > 0)
    # Erased pixels the support mask leaves open to the outside (a pile cutting its outline) still snap when they
    # lie inside the support's convex outline; left alone they meet the snapped plane with a jump that tears the
    # mesh. Erased pixels past a table's straight edge stay out.
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(raw)), 1)
    snap |= erased & (hull > 0) & (t > 0)
    B = np.where(snap[..., None], on_plane, B)
    # Erased pixels off the support (furniture against a wall): MoGe reads LaMa's fill of a large hole as a surface
    # about a metre away, so the background stood in front of the replica and hid it (home office: desk at 2.5 m,
    # its hole at 0.8 m). They take the depth of the background around the hole (the wall behind), filled in at 1/4
    # resolution. Pushing them behind each replica mesh instead left steps at the mask edges that tore the mesh.
    loose = erased & ~snap
    fill = None
    if loose.any():
        z = np.where(vb & ~loose, B[..., 2], 0).astype(np.float32)
        q = 4
        zs = cv2.resize(z, (w // q, h // q), interpolation=cv2.INTER_NEAREST)
        holes = (zs <= 0) | ~np.isfinite(zs)
        zs = smooth_fill(np.where(holes, 0, zs).astype(np.float32), ~holes)
        fill = cv2.resize(zs, (w, h), interpolation=cv2.INTER_LINEAR)
        vb = vb | loose
    # Sky and anything MoGe leaves without depth: a backdrop at the farthest valid depth, so moved cameras see it
    # instead of a hole.
    far = np.percentile(B[vb & ~loose][:, 2], 99)
    B = np.where((vb | snap)[..., None], B, rays * far)

    meas, replica = photo_replica(masks, phrases, P, vp, depth_p, n, dd, (w, h, fx, fy, cx, cy), args.meshes, solid,
                                  args.ccm_pose or os.path.join(args.meshes, 'ccm_pose.json'), fixed)

    if fill is not None:
        # The fill interpolates the floor below a tall bookshelf too, so it can still stand in front of the replica:
        # inside each object's mask it goes behind the whole mesh, eased to zero across the erased margin (a step
        # at the mask edge tore the mesh).
        push = np.zeros((h, w), np.float32)
        for j, obj, tf in replica:
            back = 1.02 * float(trimesh.transform_points(obj.vertices, tf)[:, 2].max())
            sel = loose & masks[j]
            push[sel] = np.maximum(push[sel], back - fill[sel])
        if push.any():
            core = push > 0
            known = cv2.resize((core | ~loose).astype(np.uint8), (w // q, h // q), interpolation=cv2.INTER_NEAREST) > 0
            ps = cv2.resize(push, (w // q, h // q), interpolation=cv2.INTER_NEAREST)
            eased = cv2.resize(smooth_fill(np.where(known, ps, 0).astype(np.float32), known), (w, h),
                               interpolation=cv2.INTER_LINEAR)
            fill = fill + np.where(core, push, np.where(loose, eased, 0))
        B = np.where(loose[..., None], rays * fill[..., None], B)
    bg_mesh = grid_mesh(B, background, np.ones_like(vb), max_jump=args.max_jump)

    shadow_path = os.path.join(d, 'shadows.png')
    sun = None
    if os.path.exists(shadow_path):
        photo_lum = (photo.astype(np.float32) / 255) ** 2.2 @ np.array([0.2126, 0.7152, 0.0722])
        sun = estimate_sun(np.asarray(Image.open(shadow_path)) > 0, masks, meas,
                           points_from_depth(np.ones((h, w)), fx, fy, cx, cy), n, dd, photo_lum, support)
    rend = Renderer(w, h, fx, fy, cx, cy)
    eye = np.eye(4)
    rep_col, rep_dep, rep_front = rend.render(eye, bg_mesh, [(o, t) for _, o, t in replica], n)
    checks, front = [], []
    for j, obj, tf in replica:
        _, alone_dep, alone = rend.render(eye, bg_mesh, [(obj, tf)], n)
        front.append(alone & (alone_dep <= rep_dep + 1e-4))  # pixels where this replica object is the front one
        m = masks[j] & vp
        iou = float((alone & m).sum() / max((alone | m).sum(), 1))
        both = alone & m
        rel = float(np.median(np.abs(rep_dep[both] - depth_p[both]) / depth_p[both])) if both.any() else None
        # Share of the replica's silhouette not hidden by other objects in the photo: SAM 3D only rebuilds what it
        # sees, so a mostly hidden object comes out partial. (Its own mask would penalise thin chairs and bikes,
        # whose legs SAM 3 masks loosely.)
        others = np.any([masks[k] for k in range(len(masks)) if k != j], axis=0) if len(masks) > 1 else np.zeros_like(m)
        visible = float(1 - (alone & others).sum() / max(alone.sum(), 1))
        checks.append({'phrase': meas[j]['phrase'], 'mesh': meas[j]['mesh'], 'iou': round(iou, 3),
                       'visible': round(visible, 3), 'on': meas[j]['on'],
                       'depth_rel_err': None if rel is None else round(rel, 4)})

    # New compositions: the replica objects shuffled on the free part of the support (--layout photo: left where
    # they are), cameras moved around.
    # Candidate spots: support pixels away from its border and from the image border. Only where SAM 3 saw the
    # support (its holes filled: the objects stood there): an erased bottle top seen against the floor behind the
    # table is in `support` too, and a bottle placed there stood in the air past the table's edge (2026-10-11).
    seen = ndimage.binary_fill_holes(raw > 0)
    free = cv2.erode((support & seen & vb).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    free[:h // 10], free[-h // 10:], free[:, :w // 10], free[:, -w // 10:] = False, False, False, False
    ys, xs = np.nonzero(free[::4, ::4])
    cand = on_plane[ys * 4, xs * 4]
    # Only well rebuilt objects are shuffled; the rest stay in the photo-camera replica.
    good = [c['iou'] >= args.min_iou and c['visible'] >= args.min_visible for c in checks]
    lib = [(r, obj, meas[j]['height_m'], meas[j]['width_m']) for r, ((j, obj, _), ok) in enumerate(zip(replica, good))
           if ok and meas[j]['phrase'] not in fixed]  # r: index in the replica (Isaac's meshes)
    # Fixed objects keep their photo pose in every view (left out if their replica fit failed: a wrong mesh would
    # show in every view, and their hole then shows instead).
    anchored = [(r, obj, tf, meas[j]) for r, ((j, obj, tf), ok) in enumerate(zip(replica, good))
                if ok and meas[j]['phrase'] in fixed]
    if fixed:
        print(f'{len(anchored)} fixed objects kept in place', flush=True)
    if args.layout == 'shuffle':
        print(f'{len(lib)} of {len(replica)} objects shuffled in new views', flush=True)
    # One group per view (--groups): each replica object's group by its name in flow_assets.py's catalogue.
    group_of = {}
    if args.groups:
        groups = json.load(open(args.groups))
        cat = os.path.join(d, 'assets.json')
        rows = json.load(open(cat))['assets'] if os.path.exists(cat) else []
        group_of = {r: groups.get(rows[j]['name'] if j < len(rows) else str(phrases[j]))
                    for r, (j, _, _) in enumerate(replica)}
        shown = sorted({group_of[e[0]] for e in lib + anchored} - {None})
        print(f'one group per view among {shown}, {args.empty:.0%} with none', flush=True)
    # For Isaac (bg_isaac.py): view 0 is the replica from the photo camera, then every accepted new view.
    bg_eye, dep_eye, _ = rend.render(eye, bg_mesh, [], n)
    isaac_views = [{'pose': eye.tolist(), 'objects': [[i, tf.tolist()] for i, (_, _, tf) in enumerate(replica)],
                    'background': cv2.inpaint(bg_eye, (dep_eye == 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)}]
    tiles = [('photo', photo), ('background (erased)', background), ('replica, photo camera', rep_col)]
    err = np.zeros((h, w, 3), np.uint8)
    rel_map = np.abs(rep_dep - depth_p) / np.maximum(depth_p, 1e-6)
    err[..., 0] = np.clip(rel_map / 0.05 * 255, 0, 255).astype(np.uint8)  # red = 5% depth error or more
    err[rep_dep == 0] = 0
    tiles.append(('depth error (red: 5% or more)', err))
    # Moved cameras: the photo's camera turned about the vertical through the objects' centre (its tilt kept), then
    # moved closer or further and up or down. The optical axis' ground point would be tens of metres away in
    # near-horizontal street photos.
    up = n
    pivot = np.mean([mi['base'] for mi in meas], 0)
    views = tries = 0
    print('new views', flush=True)
    while views < args.views and tries < 50 * args.views:
        tries += 1
        # Turns are capped so the camera moves at most --max-move metres sideways: a few degrees in a street
        # (objects 5 m away), the full --yaw on a table. Parallax beyond that tears or smears the background.
        yaw_cap = min(args.yaw, math.degrees(args.max_move / np.linalg.norm(pivot)))
        step = args.max_move / np.linalg.norm(pivot)  # the same metric cap for moving in/out and up/down
        yaw = rng.uniform(-yaw_cap, yaw_cap)
        lift = rng.uniform(-0.5 * min(step, 0.1), min(step, 0.2))
        dist = rng.uniform(max(0.75, 1 - step), min(1.05, 1 + 0.25 * step))
        rot = trimesh.transformations.rotation_matrix(math.radians(yaw), up)[:3, :3]
        reach = np.linalg.norm(pivot)
        pose = np.eye(4)
        pose[:3, :3] = rot @ trimesh.transformations.rotation_matrix(rng.normal(0, 0.03), up)[:3, :3]
        pose[:3, 3] = pivot + rot @ (-pivot) * dist + up * lift * reach
        # Objects stand where this camera sees them whole: placed anywhere on a big table, a lone bottle was cut by
        # the frame or out of it in most views (sun cream, 2026-10-11).
        R, T = pose[:3, :3], pose[:3, 3]

        def whole(p, hh, margin=0.03):
            c = (np.stack([p, p + n * hh]) - T) @ R
            if (c[:, 2] <= 0).any():
                return False
            u, v = fx * c[:, 0] / c[:, 2] + cx, fy * c[:, 1] / c[:, 2] + cy
            return bool(((u > margin * w) & (u < (1 - margin) * w) & (v > margin * h) & (v < (1 - margin) * h)).all())
        placed, objs, ids = [], [], []
        if args.groups:
            pick = None if not shown or rng.random() < args.empty else shown[rng.integers(len(shown))]
            keep = lambda r: group_of.get(r) in (None, pick)  # noqa: E731
        else:
            keep = lambda r: True  # noqa: E731
        if args.layout == 'photo':  # the replica as it is; the loop below then has nothing to place
            objs, ids = [(obj, tf) for _, obj, tf in replica], list(range(len(replica)))
        else:  # fixed objects first, so the shuffled ones keep clear of their footprint
            for r, obj, tf, mi in anchored:
                if not keep(r):
                    continue
                placed.append([mi['base'], mi['width_m'], mi['height_m'], False, False])
                objs.append((obj, tf))
                ids.append(r)
        for i in rng.permutation(len(lib)) if args.layout == 'shuffle' else []:
            r, obj, hh, ww = lib[i]
            if not keep(r):
                continue
            # Piles: some objects go on top of a placed one at least as wide, which then carries nothing else.
            tops = [q for q in placed if q[3] and q[4] and q[1] >= 0.8 * ww]
            if tops and rng.random() < args.stack:
                q = tops[rng.integers(len(tops))]
                q[3] = False
                v = rng.normal(size=3)
                v -= (v @ n) * n
                p = q[0] + n * q[2] + v / np.linalg.norm(v) * rng.uniform(0, 0.2) * q[1]
                k = rng.uniform(0.95, 1.05)
                placed.append([p, ww * k, hh * k, False, False])
                objs.append((obj, place(obj, p, n, hh * k, ww * k, rng.uniform(0, 2 * math.pi))))
                ids.append(r)
                continue
            for _ in range(50):
                p = cand[rng.integers(len(cand))]
                if whole(p, hh) and all(np.linalg.norm(p - q[0]) > (ww + q[1]) / 2 + 0.01 for q in placed if q[0] @ n + dd < 1e-3):
                    k = rng.uniform(0.95, 1.05)
                    placed.append([p, ww * k, hh * k, True, flat_top(obj)])
                    objs.append((obj, place(obj, p, n, hh * k, ww * k, rng.uniform(0, 2 * math.pi))))
                    ids.append(r)
                    break
        col, dep, _ = rend.render(pose, bg_mesh, objs, n)
        empty = dep == 0
        if empty.mean() > args.max_empty:  # camera left the reconstructed area
            continue
        col = cv2.inpaint(col, empty.astype(np.uint8), 5, cv2.INPAINT_TELEA)  # cracks and thin borders
        bg_only = rend.render(pose, bg_mesh, [], n)[0]
        isaac_views.append({'pose': pose.tolist(), 'objects': [[i, t.tolist()] for i, (_, t) in zip(ids, objs)],
                            'background': cv2.inpaint(bg_only, empty.astype(np.uint8), 5, cv2.INPAINT_TELEA)})
        views += 1
        tiles.append((f'empty {empty.mean():.0%}, yaw {yaw:+.0f}, lift {lift:+.2f}, dist x{dist:.2f}', col))

    tw, th = 300, round(300 * h / w)
    cols = 5
    sheet = Image.new('RGB', (tw * cols, th * math.ceil(len(tiles) / cols)), 'white')
    for i, (name, im) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(Image.fromarray(im).resize((tw, th)), (x, y))
        ImageDraw.Draw(sheet).text((x + 5, y + 5), name, fill=(255, 40, 40))
    sheet.save(os.path.join(d, 'sheet.jpg'), quality=88)

    # Each replica object's photo pixels where it is the front object of the replica and agrees with the photo depth
    # (within 5%): Isaac fits light and colour on them.
    agree = np.abs(rep_dep - depth_p) <= 0.05 * depth_p
    # flow_assets.py's catalogue (asked instances first, in instances.npz order): which Isaac object is one of the
    # user's classes and which a distractor, so view_labels.py boxes only what the student must learn.
    cat = os.path.join(d, 'assets.json')
    roles = None
    if os.path.exists(cat):
        rows = json.load(open(cat))['assets']
        roles = [{'name': rows[j]['name'], 'role': rows[j]['role']} if j < len(rows) else
                 {'name': str(phrases[j]), 'role': 'distractor'} for j, _, _ in replica]
    export_isaac(os.path.join(d, 'isaac'), bg_mesh, background, [o for _, o, _ in replica], isaac_views,
                 (w, h, fx, fy, cx, cy), n, [masks[j] & agree & f for (j, _, _), f in zip(replica, front)], sun,
                 roles)

    scene = trimesh.Scene()
    scene.add_geometry(bg_mesh, node_name='background')
    for j, obj, tf in replica:
        scene.add_geometry(obj, node_name=f'object_{j:02d}', transform=tf)
    scene.export(os.path.join(d, 'scene.glb'))
    report = {'fov_x_moge_deg': round(moge_fov, 1), 'fov_x_used_deg': round(math.degrees(2 * math.atan(w / 2 / fx)), 1),
              'camera_height_above_support_m': round(cam_height, 3),
              'camera_height_moge_m': round(moge_height, 3), 'metric_scale_applied': round(scale, 4), 'camera_pitch_to_support_deg': round(pitch, 1),
              'plane_inlier_frac': round(inlier, 3), 'sun_from_shadows': sun, 'background_depth_scale': round(bg_scale, 4),
              'objects': [{k2: v for k2, v in mi.items() if k2 != 'base'} for mi in meas], 'replica_checks': checks}
    with open(os.path.join(d, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))
    print(f'{views} views accepted out of {tries} tries')


def export_isaac(out, bg_mesh, background, objects, views, intr, n, photo_masks, sun=None, roles=None):
    """Plain arrays and JSON for bg_isaac.py (Isaac's python has no trimesh): the background mesh with UVs into
    background.png, every object mesh with vertex colours (in its own frame), and per view the camera pose, the
    object poses and the background seen unlit from that camera."""
    os.makedirs(out, exist_ok=True)
    w, h = intr[:2]
    v = bg_mesh.vertices
    # The background grid comes from the image, so each vertex projects back to its own pixel: that is its UV.
    fx, fy, cx, cy = intr[2:]
    uv = np.stack([(v[:, 0] / v[:, 2] * fx + cx) / w, 1 - (v[:, 1] / v[:, 2] * fy + cy) / h], 1)
    arrays = {'bg_points': v.astype(np.float32), 'bg_faces': bg_mesh.faces.astype(np.int32), 'bg_uv': uv.astype(np.float32)}
    for i, o in enumerate(objects):
        arrays[f'obj{i}_points'] = o.vertices.astype(np.float32)
        arrays[f'obj{i}_faces'] = o.faces.astype(np.int32)
        arrays[f'obj{i}_colors'] = (o.visual.vertex_colors[:, :3] / 255).astype(np.float32)
    np.savez_compressed(os.path.join(out, 'meshes.npz'), **arrays)
    Image.fromarray(background).save(os.path.join(out, 'background.png'))
    for i, m in enumerate(photo_masks):  # object i's pixels in the photo, to fit light and colour on view 0
        Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(out, f'photo_mask_{i:02d}.png'))
    for k, view in enumerate(views):
        Image.fromarray(view.pop('background')).save(os.path.join(out, f'bg_unlit_{k:02d}.png'))
    with open(os.path.join(out, 'scene.json'), 'w') as f:
        json.dump({'size': [int(w), int(h)], 'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy), 'up': n.tolist(),
                   'objects': len(objects), 'views': views, 'sun': sun, **({'roles': roles} if roles else {})}, f)


if __name__ == '__main__':
    main()
