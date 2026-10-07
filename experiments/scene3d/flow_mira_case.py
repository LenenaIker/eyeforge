"""Step 1.5 of the scene flow, input side: a Mira-Scene case for the surroundings that passed the mask check
(flow_check_masks.py), ready for Mira's stages 2-5 (CCM, SAM 3D, floor, placement).

Mira works on a 518 px square. Its own case maker crops the middle square, which on a wide street photo throws
away the houses on both sides; here the erased photo is padded to a square instead. Mira's depth stage is
skipped: our MoGe-2 depth of the whole erased photo (bg_depth.py) is written in its place, padded the same way
and turned to Mira's OpenGL camera, so its meshes land in the same frame and scale as our planes (env_add.py
then needs no rescale). Any env with numpy and Pillow:

    python experiments/scene3d/flow_mira_case.py --dir data/street_test/flow/basque_street

Writes <dir>/mira/env/{input/..., depth/moge2/..., scene_graph.json, case.json} and <dir>/mira/pad.json.
"""
import argparse
import hashlib
import json
import math
import os
import shutil

import numpy as np
from PIL import Image

SIZE = 518


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    d = args.dir
    case = os.path.join(d, 'mira', 'env')
    inp, dep = os.path.join(case, 'input'), os.path.join(case, 'depth', 'moge2')
    # Mira's stages skip what exists: a case left from other masks would mix old meshes with new masks.
    shutil.rmtree(os.path.join(d, 'mira'), ignore_errors=True)
    os.makedirs(inp, exist_ok=True)
    os.makedirs(dep, exist_ok=True)
    os.makedirs(os.path.join(case, 'logs'), exist_ok=True)

    bg = Image.open(os.path.join(d, 'background.png')).convert('RGB')
    w, h = bg.size
    side = max(w, h)
    left, top = (side - w) // 2, (side - h) // 2
    s = SIZE / side

    def pad(a, fill=0):
        out = np.full((side, side) + a.shape[2:], fill, a.dtype)
        out[top:top + h, left:left + w] = a
        return out

    def small(a, nearest=True):
        if a.dtype == bool:
            return np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 127
        return np.asarray(Image.fromarray(a).resize((SIZE, SIZE), Image.Resampling.NEAREST if nearest
                                                    else Image.Resampling.LANCZOS))

    rgb = pad(np.asarray(bg), 0)
    rgb[:top], rgb[top + h:] = 127, 127
    rgb[:, :left], rgb[:, left + w:] = 127, 127
    Image.fromarray(rgb).save(os.path.join(inp, 'source.png'))
    scene = Image.fromarray(rgb).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    scene.save(os.path.join(inp, 'scene.png'))

    em = np.load(os.path.join(d, 'env_masks.npz'))
    masks = [small(pad(m)) for m in em['masks']]
    phrases = list(em['phrases'])
    nodes, edges = [], []
    for k, (m, p) in enumerate(zip(masks, phrases)):
        Image.fromarray(m.astype(np.uint8) * 255, 'L').save(os.path.join(inp, f'mask_{k:03d}.png'))
        oid = f'object_{k:03d}'
        nodes.append({'id': oid, 'kind': 'object', 'mask_index': k, 'mask_file': f'input/mask_{k:03d}.png',
                      'name': p, 'short_name': p, 'caption': p, 'caption_source': 'flow', 'name_locked': True,
                      'caption_locked': True, 'motion': 'fixed', 'fixed': True, 'dynamic': False,
                      'upright_mode': 'auto', 'confidence': 1.0})
        edges.append({'child': oid, 'parent': 'floor', 'relation': 'rests_on', 'confidence': 1.0,
                      'operational': True})
    ground = np.any(np.load(os.path.join(d, 'ground.npz'))['masks'], 0)
    floor = small(pad(ground)) & ~(np.any(masks, 0) if masks else False)
    Image.fromarray(floor.astype(np.uint8) * 255, 'L').save(os.path.join(inp, 'floor_mask.png'))
    fg = np.asarray(scene).copy()
    fg[~(np.any(masks, 0) if masks else np.zeros((SIZE, SIZE), bool))] = 0
    Image.fromarray(fg).save(os.path.join(inp, 'scene_fg.png'))

    # Our depth in Mira's frame: padded, 518 px, OpenGL (+Y up, -Z forward), pixel intrinsics.
    dp = np.load(os.path.join(d, 'depth_background.npz'))
    pts = dp['points'].astype(np.float32) * np.array([1, -1, -1], np.float32)
    valid = (dp['mask'] > 0) & np.isfinite(pts).all(-1)
    pts[~valid] = 0
    depth = np.where(valid, dp['depth'], 0).astype(np.float32)
    P = np.stack([small(pad(pts[..., i])) for i in range(3)], -1).astype(np.float32)
    D = small(pad(depth)).astype(np.float32)
    V = small(pad(valid)) & (D > 0)
    K = dp['intrinsics']
    fx, fy = K[0, 0] * w * s, K[1, 1] * h * s
    cx, cy = (K[0, 2] * w + left) * s, (K[1, 2] * h + top) * s
    np.save(os.path.join(dep, 'depth.npy'), D)
    np.save(os.path.join(dep, 'camera_pts_map.npy'), P)
    np.save(os.path.join(dep, 'valid_mask.npy'), V)
    np.save(os.path.join(dep, 'intrinsics.npy'), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32))
    with open(os.path.join(dep, 'fov_x_rad.txt'), 'w') as f:
        f.write(f'{2 * math.atan(SIZE / 2 / fx):.10f}\n')
    with open(os.path.join(dep, 'metric_alignment.json'), 'w') as f:
        json.dump({'schema': 'mira_depth_to_moge2_metric_scale_v1', 'case': 'env', 'source_method': 'moge2',
                   'target_method': 'moge2', 'scale': 1.0, 'rotation': 'identity', 'translation': [0.0, 0.0, 0.0],
                   'note': 'our bg_depth.py MoGe-2 depth of the whole photo, padded to the square',
                   'coordinate_convention': 'OpenGL +X right, +Y up, -Z forward'}, f, indent=1)

    src = os.path.abspath(os.path.join(d, 'background.png'))
    digest = hashlib.sha256(open(src, 'rb').read()).hexdigest()
    with open(os.path.join(inp, 'preprocessing.json'), 'w') as f:
        json.dump({'schema': 'mira_preprocessing_v1', 'source': src, 'source_filename': 'background.png',
                   'source_sha256': digest, 'original_size': [w, h], 'crop_xywh': [-left, -top, side, side],
                   'output_size': [SIZE, SIZE], 'exif_transposed': True, 'padded': True}, f, indent=2)
    with open(os.path.join(case, 'case.json'), 'w') as f:
        json.dump({'schema': 'mira_case_v1', 'case_id': 'env', 'source': src, 'source_filename': 'background.png',
                   'source_sha256': digest}, f, indent=2)
    graph = {'schema': 'mira_scene_graph_v1', 'case': 'env', 'edge_direction': 'child_to_direct_supporter',
             'nodes': [{'id': 'floor', 'kind': 'static_environment', 'name': 'floor', 'motion': 'fixed',
                        'fixed': True, 'dynamic': False},
                       {'id': 'world_anchor', 'kind': 'virtual_static_anchor', 'name': 'wall_or_ceiling',
                        'motion': 'fixed', 'fixed': True, 'dynamic': False}] + nodes,
             'edges': edges}
    with open(os.path.join(case, 'scene_graph.json'), 'w') as f:
        json.dump(graph, f, indent=2)
    with open(os.path.join(d, 'mira', 'pad.json'), 'w') as f:
        json.dump({'size': [w, h], 'side': side, 'left': left, 'top': top, 'mira': SIZE}, f)
    print(json.dumps({'instances': phrases, 'floor_frac': round(float(floor.mean()), 3),
                      'pad': [left, top], 'fov_x_deg': round(math.degrees(2 * math.atan(SIZE / 2 / fx)), 1)}))


if __name__ == '__main__':
    main()
