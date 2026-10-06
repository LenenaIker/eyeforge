"""Each object's orientation from Mira-Scene's canonical coordinate maps (CCM), as its step 5 solves it: every
pixel of an object says which point of its canonical mesh it shows, and a RANSAC similarity fit against the
depth points gives the mesh's rotation in the camera. bg_build.py --ccm-pose uses it to start its silhouette fit
from the right turn: a silhouette alone cannot tell a bookshelf's front from its back (home office, 2026-10-06).
Run in the mira-geometry env with Mira's infer_scripts on PYTHONPATH (as run_photo.sh sets it):

    python bg_ccm_pose.py --case data/scene3d_test/photo_cases/003_home_office/003_home_office

Reads <case>/CCM (stage 2) and <case>/depth/<method> (stage 1); writes <case>/ccm_pose.json:
{"NNN": 3x3 rotation taking the raw SAM 3D mesh NNN.glb (y up) into the OpenCV camera (+X right, +Y down,
+Z forward), scale removed; objects the fit failed on (too few points) are left out}.
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

GL2CV = np.diag([1.0, -1.0, -1.0])


def mira_step5():
    """Mira's 5_construct_scene.py as a module (its file name starts with a digit)."""
    import utils.solve_transform as st  # fails early with a clear message when PYTHONPATH misses infer_scripts
    path = os.path.join(os.path.dirname(st.__file__), '..', '5_construct_scene.py')
    spec = importlib.util.spec_from_file_location('mira_step5', os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, help='a Mira case directory after stages 1-3')
    args = ap.parse_args()
    case = os.path.abspath(args.case)
    m5 = mira_step5()
    depth, method = m5.load_precomputed_depth(os.path.dirname(case), os.path.basename(case))
    ccm = m5.load_ccm_outputs(os.path.join(case, 'CCM'))
    if depth is None or ccm is None:
        raise SystemExit(f'missing depth or CCM in {case}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = depth.camera_pts_map.shape[:2]
    maps = []
    for c in ccm['ccm_list']:
        t = torch.from_numpy(c).unsqueeze(0).float().to(device)
        if t.shape[-2:] != (h, w):
            t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
        maps.append(t)
    maps = torch.cat(maps, 0)
    masks = (maps.abs().sum(1, keepdim=True) > 1e-6).float()
    n = maps.shape[0]
    pts = torch.from_numpy(depth.camera_pts_map).to(device).unsqueeze(0).expand(n, -1, -1, -1)
    valid = torch.from_numpy(depth.valid_mask).to(device).unsqueeze(0).expand(n, -1, -1)
    fits = m5.solve_similarity_transforms(maps, pts, valid, masks)
    out = {'depth_method': method}
    for i, f in enumerate(fits):
        mat = m5.transform_to_matrix(f) @ m5.CANONICAL_MESH_ALIGNMENT  # raw mesh -> canonical -> GL camera
        rot = mat[:3, :3] / np.cbrt(np.linalg.det(mat[:3, :3]))
        u, _, vt = np.linalg.svd(GL2CV @ rot)
        if np.allclose(mat[:3, :3], m5.CANONICAL_MESH_ALIGNMENT[:3, :3]):  # the solver's identity: no fit
            continue
        out[f'{i:03d}'] = (u @ vt).tolist()
    with open(os.path.join(case, 'ccm_pose.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'{len(out) - 1} of {n} objects oriented ({method} depth)')


if __name__ == '__main__':
    main()
