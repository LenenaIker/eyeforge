"""Step 1.5a of photo2scene: one SAM 3D Objects mesh per surroundings mask that passed the check
(flow_check_masks.py), posed in the photo's camera. SAM 3D runs whole (its stage 1 invents the hidden back, its
layout head gives the pose) on the erased photo with our own MoGe depth as its point map, so every mesh lands in
the frame and scale of our floor and planes. This replaces Mira-Scene's stages 2-5 (CCM voxels, SAM 3D stage 2,
floor, gravity placement): flow_refine.py fits the poses to our depth afterwards. Run in the `eyeforge-sam3d` env:

    python experiments/scene3d/flow_sam3d.py --dir data/street_test/flow/basque_street

Writes <dir>/meshes/sam3d/NNN.glb (mesh NNN of mask NNN, OpenGL camera frame: +X right, +Y up, -Z forward,
metres, vertex colours), NNN.json (SAM 3D's pose, seconds, GPU peak) and <dir>/meshes/scene_graph.json (names).
A mesh already on disk is kept, so a fresh process after a GPU memory failure only fills the gaps.

The support route meshes the photo's movable objects instead (bg_build.py puts them back and shuffles them):

    python experiments/scene3d/flow_sam3d.py --dir <dir> --masks instances.npz --image photo --out meshes_objects

which also writes NNN_canon.glb (SAM 3D's own y-up frame) and the rotation from it to the OpenCV camera
("rot_cv" in NNN.json), the orientation bg_build.py starts its silhouette fit from.
"""
import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['LIDRA_SKIP_INIT'] = 'true'
os.environ.setdefault('CONDA_PREFIX', sys.prefix)  # SAM 3D's notebook module reads it

import numpy as np
import torch
import trimesh
from PIL import Image

SAM3D_DIR = os.environ.get('SIM2CHIP_SAM3D_DIR', '/media/lenena-iker/Crucial/tools/sam-3d-objects')
# SAM 3D's glb is y-up; its pose maps the z-up canonical frame: glb (x, y, z) is canonical (x, -z, y).
GLB_TO_CANON = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
# PyTorch3D camera (+X left, +Y up, +Z forward) to OpenGL (+X right, +Y up, -Z forward).
P3D_TO_GL = np.diag([-1.0, 1.0, -1.0, 1.0])


def load_pipeline():
    sys.path.insert(0, os.path.join(SAM3D_DIR, 'notebook'))
    from inference import Inference
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils, render_utils

    # Licence-clean path (as experiments/img2mesh/run_sam3d.py): gsplat instead of the Inria rasteriser, no hole
    # filling (it goes through nvdiffrast).
    render_frames = render_utils.render_frames
    render_utils.render_frames = lambda s, e, i, options={}, **kw: render_frames(s, e, i, {**options,
                                                                                          'backend': 'gsplat'}, **kw)
    postprocess_mesh = postprocessing_utils.postprocess_mesh
    postprocessing_utils.postprocess_mesh = lambda *a, **kw: postprocess_mesh(*a, **{**kw, 'fill_holes': False})
    return Inference(os.path.join(SAM3D_DIR, 'checkpoints', 'hf', 'pipeline.yaml'), compile=False)


def canon_to_p3d(out):
    """SAM 3D's pose (PyTorch3D, row vectors) as a 4x4 on column vectors: voxel frame to PyTorch3D camera."""
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    tf = compose_transform(scale=out['scale'], rotation=quaternion_to_matrix(out['rotation']),
                           translation=out['translation'])
    return tf.get_matrix()[0].detach().cpu().numpy().T  # PyTorch3D keeps the transpose


def pose_matrix(M):
    """Voxel-frame-to-PyTorch3D-camera pose to a glb-to-OpenGL-camera one."""
    return P3D_TO_GL @ M @ GLB_TO_CANON


def umeyama(a, b):
    """Similarity (s, R, t) with b ~ s R a + t, as a 4x4."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A ** 2).sum(1).mean()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, mb - s * R @ ma
    return T


def fit_ccm_pose(ccm, pts, mask, rng, iters=300):
    """CCM canonical coordinates to PyTorch3D camera points of the same pixels, robust similarity."""
    ok = mask & np.isfinite(pts).all(-1) & (np.abs(ccm).sum(-1) > 1e-4)
    a, b = ccm[ok].astype(np.float64), pts[ok].astype(np.float64)
    if len(a) > 20000:
        keep = rng.choice(len(a), 20000, replace=False)
        a, b = a[keep], b[keep]
    tol = 0.05 * np.linalg.norm(np.ptp(b, 0))
    best = None
    for _ in range(iters):
        T = umeyama(*(x[rng.choice(len(a), 4, replace=False)] for x in (a, b)))
        inl = np.linalg.norm(a @ T[:3, :3].T + T[:3, 3] - b, axis=1) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    T = umeyama(a[best], b[best])
    err = np.linalg.norm(a @ T[:3, :3].T + T[:3, 3] - b, axis=1)
    return T, {'inliers': round(float(best.mean()), 3), 'median_err_m': round(float(np.median(err)), 3)}


def to_pixels(x, K, w, h):
    """PyTorch3D camera points to pixel (u, v) and depth z."""
    z = x[:, 2]
    u = K[0, 0] * w * (-x[:, 0]) / z + K[0, 2] * w
    v = K[1, 1] * h * (-x[:, 1]) / z + K[1, 2] * h
    return u, v, z


def hybrid_voxels(sam_coords, M_sam, ccm_vox, M_ccm, depth, K, w, h, margin=2.0):
    """CCM voxels (seen side, pixel aligned) plus SAM 3D stage-1 voxels moved into the CCM frame, minus those that
    sit in front of the surface the photo shows (free space). Returns int (K, 3) indices and counts."""
    v = sam_coords / 64.0 - 0.5
    x = v @ M_sam[:3, :3].T + M_sam[:3, 3]
    u, vv, z = to_pixels(x, K, w, h)
    ui, vi = np.round(u).astype(int), np.round(vv).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 0)
    seen = np.full(len(x), np.inf)
    seen[inside] = depth[vi[inside], ui[inside]]
    step = np.linalg.norm(M_ccm[:3, 0]) / 64.0  # one CCM voxel in metres
    free = np.isfinite(seen) & (z < seen - margin * step)
    c = x[~free] @ np.linalg.inv(M_ccm[:3, :3]).T - np.linalg.inv(M_ccm[:3, :3]) @ M_ccm[:3, 3]
    idx = np.floor((c + 0.5) * 64).astype(int)
    idx = idx[((idx >= 0) & (idx < 64)).all(1)]
    both = np.unique(np.concatenate([ccm_vox, idx]), axis=0)
    return both.astype(np.int32), {'ccm': len(ccm_vox), 'sam3d': len(sam_coords), 'carved': int(free.sum()),
                                   'added': len(both) - len(ccm_vox)}


def run_one(pm, rgba, pointmap, seed, coords_fn):
    """SAM 3D whole as InferencePipelinePointMap.run (no post-processing), with a hook that may swap the stage-1
    voxels and the pose before stage 2."""
    image = pm.merge_image_and_mask(rgba, None)
    with pm.device:
        pdict = pm.compute_pointmap(image, pointmap)
        ss_in = pm.preprocess_image(image, pm.ss_preprocessor, pointmap=pdict['pointmap'])
        slat_in = pm.preprocess_image(image, pm.slat_preprocessor)
        torch.manual_seed(seed)
        ss = pm.sample_sparse_structure(ss_in, inference_steps=None, use_distillation=False)
        ss.update(pm.pose_decoder(ss, scene_scale=ss_in.get('pointmap_scale'),
                                  scene_shift=ss_in.get('pointmap_shift')))
        ss['scale'] = ss['scale'] * ss['downsample_factor']
        M = canon_to_p3d(ss)
        coords, M, extra = coords_fn(ss['coords'], M)
        slat = pm.sample_slat(slat_in, coords, inference_steps=None, use_distillation=False)
        outs = pm.decode_slat(slat, pm.decode_formats)
        outs = pm.postprocess_slat_output(outs, False, False, True)
    return outs['glb'], M, ss, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fresh', action='store_true', help='start from no meshes (the masks changed)')
    ap.add_argument('--voxels', choices=['sam3d', 'ccm', 'hybrid'], default='sam3d',
                    help='stage-2 voxels: SAM 3D stage 1 (invents the back), Mira-CCM (flow_ccm.py, pixel aligned, '
                         'seen side only) or both, SAM 3D carved by the photo depth')
    ap.add_argument('--masks', default='env_masks.npz', help='npz (masks, phrases) in --dir')
    ap.add_argument('--image', choices=['background', 'photo'], default='background',
                    help='the picture SAM 3D sees, with depth_<image>.npz as its point map')
    ap.add_argument('--out', default='meshes', help='folder in --dir; meshes go to <out>/sam3d')
    args = ap.parse_args()
    d = args.dir
    root = os.path.join(d, args.out)
    out_dir = os.path.join(root, 'sam3d')
    if args.fresh:  # everything but flow_ccm.py's output
        for name in os.listdir(root) if os.path.isdir(root) else []:
            if name != 'ccm':
                path = os.path.join(root, name)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(out_dir, exist_ok=True)

    em = np.load(os.path.join(d, args.masks))
    phrases = [str(p) for p in em['phrases']]
    graph = {'nodes': [{'id': f'object_{k:03d}', 'kind': 'object', 'mask_index': k, 'name': p}
                       for k, p in enumerate(phrases)]}
    with open(os.path.join(root, 'scene_graph.json'), 'w') as f:
        json.dump(graph, f, indent=1)
    todo = [k for k in range(len(phrases)) if not os.path.exists(os.path.join(out_dir, f'{k:03d}.glb'))]
    if not todo:
        print(json.dumps({'meshes': len(phrases), 'made': 0}))
        return

    image = np.asarray(Image.open(os.path.join(d, f'{args.image}.png')).convert('RGB'))
    h, w = image.shape[:2]
    dp = np.load(os.path.join(d, f'depth_{args.image}.npz'))
    pts = dp['points'].astype(np.float32)  # OpenCV camera
    if pts.shape[:2] != (h, w):
        raise ValueError(f'depth {pts.shape[:2]} does not match the photo {(h, w)}')
    bad = ~((dp['mask'] > 0) & np.isfinite(pts).all(-1))
    pts = pts * np.array([-1, -1, 1], np.float32)  # OpenCV to PyTorch3D camera
    pts[bad] = np.nan
    pointmap = torch.from_numpy(pts)

    inference = load_pipeline()
    pm = inference._pipeline
    K = dp['intrinsics']
    depth = np.where(bad, np.nan, pts[..., 2])
    rng = np.random.default_rng(args.seed)
    made, failed = [], []
    for k in todo:
        mask = em['masks'][k].astype(bool)
        if mask.shape != (h, w):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        def coords_fn(sam, M_sam):
            if args.voxels == 'sam3d':
                return sam, M_sam, {}
            z = np.load(os.path.join(d, 'meshes', 'ccm', f'{k:03d}.npz'))
            M_ccm, fit = fit_ccm_pose(z['ccm'].astype(np.float32), pts, mask, rng)
            if args.voxels == 'ccm':
                vox, counts = z['voxels'], {'ccm': len(z['voxels'])}
            else:
                vox, counts = hybrid_voxels(sam[:, 1:].cpu().numpy(), M_sam, z['voxels'], M_ccm, depth, K, w, h)
            c = torch.from_numpy(np.concatenate([np.zeros((len(vox), 1), np.int32), vox], 1)).to(sam)
            return c, M_ccm, {'ccm_fit': fit, 'voxels': counts}

        try:
            glb, M, out, extra = run_one(pm, inference.merge_mask_to_rgba(image, mask), pointmap, args.seed,
                                         coords_fn)
        except torch.OutOfMemoryError:
            failed.append((k, phrases[k], 'GPU memory'))
            print(f'{k:03d} {phrases[k]}: out of GPU memory', flush=True)
            torch.cuda.empty_cache()
            continue
        mesh = glb
        rot_cv = None
        if args.out != 'meshes':
            mesh.export(os.path.join(out_dir, f'{k:03d}_canon.glb'))
            R = (np.diag([1.0, -1.0, -1.0, 1.0]) @ pose_matrix(M))[:3, :3]  # OpenGL to OpenCV camera
            rot_cv = (R / np.cbrt(np.linalg.det(R))).round(5).tolist()
        mesh.apply_transform(pose_matrix(M))
        mesh.export(os.path.join(out_dir, f'{k:03d}.glb'))
        info = {'phrase': phrases[k], 'seconds': round(time.time() - t0, 1),
                'peak_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), 'faces': int(len(mesh.faces)),
                'extent_m': np.ptp(mesh.vertices, 0).round(2).tolist(),
                'pose': {n: out[n].detach().cpu().numpy().round(4).tolist() for n in ('rotation', 'translation', 'scale')},
                'voxels': args.voxels, **({'rot_cv': rot_cv} if rot_cv else {}), **extra}
        with open(os.path.join(out_dir, f'{k:03d}.json'), 'w') as f:
            json.dump(info, f, indent=1)
        made.append(k)
        print(f'{k:03d} {phrases[k]}: {info["seconds"]} s, {info["peak_gib"]} GiB, {info["extent_m"]} m {extra}',
              flush=True)
        del out, glb
        torch.cuda.empty_cache()
    print(json.dumps({'meshes': len(phrases), 'made': made, 'failed': failed}))


if __name__ == '__main__':
    main()
