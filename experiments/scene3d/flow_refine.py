"""Step 1.5b of photo2scene: each surroundings mesh's pose, refined on our own depth. SAM 3D's layout head
(flow_sam3d.py) places each mesh from one look at the photo; here it is fitted to MoGe's points inside its mask by
ICP with scale (Umeyama on closest points, the mesh seen from the camera), and a step is kept only if the mesh's
silhouette IoU on its mask goes up. (With Mira-Scene's gravity placement it took a house from IoU 0.008 to 0.65,
basque_street 2026-10-07.) Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python experiments/scene3d/flow_refine.py --dir data/street_test/flow/basque_street

Writes <dir>/meshes/scene_refined.glb (each mesh in the OpenGL photo camera frame, named object_NNN by its mask)
and <dir>/refine.json (IoU per mesh: SAM 3D's pose, refined).
"""
import argparse
import glob
import json
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageFilter
from scipy.spatial import cKDTree


def umeyama(src, dst):
    """Similarity (s, R, t) with s R src + t ~ dst, least squares."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.sign(np.linalg.det(u @ vt))
    S = np.diag([1, 1, d])
    R = u @ S @ vt
    s = (sig * np.diag(S)).sum() / max((a ** 2).sum() / len(src), 1e-12)
    return s, R, md - s * R @ ms


class View:
    """The photo's camera at a reduced size: pixel intrinsics, OpenGL frame."""

    def __init__(self, K, w, h):
        self.K, self.w, self.h = K, w, h
        self.r = pyrender.OffscreenRenderer(w, h)

    def depth(self, m):
        sc = pyrender.Scene()
        sc.add(pyrender.Mesh.from_trimesh(m))
        K = self.K
        sc.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=2000), pose=np.eye(4))
        return self.r.render(sc, flags=pyrender.RenderFlags.DEPTH_ONLY)

    def points(self, depth):
        """OpenGL camera points of the rendered pixels."""
        ys, xs = np.nonzero(depth > 0)
        z = depth[ys, xs]
        K = self.K
        return np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, -(ys + 0.5 - K[1, 2]) / K[1, 1] * z, -z], 1)


def axis_fit(src, dst):
    """Per-axis scale and shift (s_j a_j + t_j ~ b_j), least squares on each axis alone."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    s = (a * b).sum(0) / np.maximum((a ** 2).sum(0), 1e-12)
    return s, dst.mean(0) - s * src.mean(0)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def load(path):
    m = trimesh.load(path, force='mesh')
    if m.visual.kind == 'texture':
        m.visual = m.visual.to_color()
    return m


def resize(a, w, h):
    return np.asarray(Image.fromarray(a).resize((w, h), Image.Resampling.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--size', type=int, default=640, help='longest side of the IoU renders')
    ap.add_argument('--axis-scale', action='store_true',
                    help='after the similarity ICP, also scale each of the mesh\'s own axes (Mira-CCM shapes are not '
                         'a similarity of the MoGe points)')
    args = ap.parse_args()
    d = args.dir
    case = os.path.join(d, 'meshes')
    em = np.load(os.path.join(d, 'env_masks.npz'))
    dp = np.load(os.path.join(d, 'depth_background.npz'))
    H, W = dp['depth'].shape
    f = args.size / max(W, H)
    w, h = round(W * f), round(H * f)
    K = dp['intrinsics'] * np.array([[w], [h], [1]])  # normalised to pixels of the small render
    P = np.stack([resize(dp['points'][..., i].astype(np.float32), w, h) for i in range(3)], -1)
    P = P * np.array([1, -1, -1], np.float32)  # OpenCV to OpenGL
    V = resize((dp['mask'] > 0).astype(np.uint8), w, h).astype(bool) & np.isfinite(P).all(-1)
    view = View(K, w, h)
    out_scene = trimesh.Scene()
    report = {}
    for path in sorted(glob.glob(os.path.join(case, 'sam3d', '[0-9][0-9][0-9].glb'))):
        k = int(os.path.basename(path)[:3])
        oid = f'object_{k:03d}'
        mask = resize(em['masks'][k].astype(np.uint8) * 255, w, h) > 0
        # Inner points only: MoGe smears depth across object edges.
        inner = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
        target = P[inner & V]
        m = load(path)
        scores = {'sam3d': iou(view.depth(m) > 0, mask)}
        best, best_iou = m.copy(), scores['sam3d']
        if len(target) > 200:
            tree = cKDTree(target)
            for _ in range(args.iters):
                z = view.depth(m)
                if not (z > 0).any():
                    break
                Q = view.points(z)
                if len(Q) > 20000:
                    Q = Q[np.random.default_rng(0).choice(len(Q), 20000, replace=False)]
                dist, idx = tree.query(Q)
                keep = dist < np.percentile(dist, 80)  # trim the worst pairs: the hidden back, stray depth
                s, R, t = umeyama(Q[keep], target[idx[keep]])
                if not 0.7 < s < 1.4:  # one step never rescales much; a jump means the pairs went wrong
                    break
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = s * R, t
                m.apply_transform(T)
                score = iou(view.depth(m) > 0, mask)
                if score > best_iou:
                    best, best_iou = m.copy(), score
            if args.axis_scale:
                # From the best similarity pose: stretch along the mesh's oriented box axes, total 0.5-2x per axis.
                m = best.copy()
                F = m.bounding_box_oriented.primitive.transform  # box frame to camera
                Fi = np.linalg.inv(F)
                total = np.ones(3)
                for _ in range(args.iters):
                    z = view.depth(m)
                    if not (z > 0).any():
                        break
                    Q = view.points(z)
                    if len(Q) > 20000:
                        Q = Q[np.random.default_rng(0).choice(len(Q), 20000, replace=False)]
                    dist, idx = tree.query(Q)
                    keep = dist < np.percentile(dist, 80)
                    a = Q[keep] @ Fi[:3, :3].T + Fi[:3, 3]
                    b = target[idx[keep]] @ Fi[:3, :3].T + Fi[:3, 3]
                    sc, t = axis_fit(a, b)
                    sc = np.clip(sc, 0.8, 1.25)
                    sc = np.clip(total * sc, 0.5, 2.0) / total
                    t = (b - a * sc).mean(0)
                    total *= sc
                    A = np.eye(4)
                    A[:3, :3], A[:3, 3] = np.diag(sc), t
                    m.apply_transform(F @ A @ Fi)
                    score = iou(view.depth(m) > 0, mask)
                    if score > best_iou:
                        best, best_iou = m.copy(), score
                # Then the silhouette itself: coordinate search on the box-axis scales and the shift across the view
                # (x, y at the mesh's depth), halving the step when no move helps.
                F = best.bounding_box_oriented.primitive.transform
                Fi = np.linalg.inv(F)
                step = 0.08
                while step > 0.005:
                    moved = False
                    for j in range(5):
                        for sign in (1, -1):
                            A = np.eye(4)
                            if j < 3:
                                A[j, j] = 1 + sign * step
                                T = F @ A @ Fi
                            else:
                                T = np.eye(4)
                                T[j - 3, 3] = sign * step * np.abs(best.centroid[2])
                            cand = best.copy()
                            cand.apply_transform(T)
                            score = iou(view.depth(cand) > 0, mask)
                            if score > best_iou + 1e-4:
                                best, best_iou, moved = cand, score, True
                    if not moved:
                        step /= 2
        scores['refined'] = best_iou
        report[oid] = {k2: round(v, 3) for k2, v in scores.items()}
        out_scene.add_geometry(best, node_name=oid, geom_name=oid)
    out_scene.export(os.path.join(case, 'scene_refined.glb'))
    with open(os.path.join(args.dir, 'refine.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
