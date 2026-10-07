"""Step 1.5e2 of the scene flow: each surroundings mesh's pose, refined on our own depth. On the street test
Mira's placement (stage 5, gravity_joint) put several houses off their masks (silhouette IoU 0.0-0.3 on
basque_street), and even its initial pose only reached 0.3-0.6. Here every mesh starts from the best of Mira's
two poses and is fitted to MoGe's points inside its mask by ICP with scale (Umeyama on closest points, the mesh
seen from the camera), kept only if its silhouette IoU on the mask goes up. Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python experiments/scene3d/flow_refine.py --dir data/street_test/flow/basque_street

Writes <dir>/mira/env/scene_refined.glb (each mesh in Mira's OpenGL camera frame, named by its object id) and
<dir>/refine.json (IoU per mesh: gravity, initial, refined, chosen).
"""
import argparse
import glob
import json
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image
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
    def __init__(self, K, size=518):
        self.K, self.size = K, size
        self.r = pyrender.OffscreenRenderer(size, size)

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


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def meshes_of(path):
    sc = trimesh.load(path)
    out = {}
    for node in sc.graph.nodes_geometry:
        tf, name = sc.graph[node]
        if 'object_' not in node:
            continue
        m = sc.geometry[name].copy()
        m.apply_transform(tf)
        out[node[node.index('object_'):][:10]] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--iters', type=int, default=30)
    args = ap.parse_args()
    case = os.path.join(args.dir, 'mira', 'env')
    sdir = glob.glob(os.path.join(case, 'scene', '*', '*', 'scene.glb'))[0]
    gravity = meshes_of(sdir)
    initial = meshes_of(os.path.join(os.path.dirname(sdir), 'scene_initial.glb'))
    dep = glob.glob(os.path.join(case, 'depth', '*'))[0]
    K = np.load(os.path.join(dep, 'intrinsics.npy'))
    P = np.load(os.path.join(dep, 'camera_pts_map.npy'))
    V = np.load(os.path.join(dep, 'valid_mask.npy'))
    view = View(K)
    out_scene = trimesh.Scene()
    report = {}
    for oid in sorted(gravity):
        k = int(oid[7:10])
        mask = np.asarray(Image.open(os.path.join(case, 'input', f'mask_{k:03d}.png'))) > 0
        # Inner points only: MoGe smears depth across object edges.
        inner = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
            __import__('PIL.ImageFilter', fromlist=['MinFilter']).MinFilter(5))) > 0
        target = P[inner & V]
        cands = {'gravity': gravity[oid], 'initial': initial.get(oid, gravity[oid])}
        scores = {n: iou(view.depth(m) > 0, mask) for n, m in cands.items()}
        start = max(scores, key=scores.get)
        m = cands[start].copy()
        best, best_iou = m.copy(), scores[start]
        if len(target) > 200:
            tree = cKDTree(target)
            for _ in range(args.iters):
                d = view.depth(m)
                if not (d > 0).any():
                    break
                Q = view.points(d)
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
        scores['refined'] = best_iou
        report[oid] = {k2: round(v, 3) for k2, v in scores.items()} | {'start': start}
        out_scene.add_geometry(best, node_name=oid, geom_name=oid)
    out_scene.export(os.path.join(case, 'scene_refined.glb'))
    with open(os.path.join(args.dir, 'refine.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
