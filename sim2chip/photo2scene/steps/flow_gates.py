"""Step 2.2 of the scene flow, the geometric gates of the imitation (catalog entries 1 and 3): the place built
from the photo (piece_build.py + env_add.py) is rendered from the photo's own camera with pyrender, without
light, and compared in code with the photo before any VLM looks at it.

Per surroundings mesh: its lowest points against the ground (floating or sunk), its silhouette against its own
mask in the photo (IoU) and its depth against MoGe's on that mask. For the whole view: ground depth against
MoGe's. A mesh that fails is listed to be dropped; the flow rebuilds without it (a dropped mesh is a valid fix).
Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python sim2chip/photo2scene/steps/flow_gates.py --dir data/street_test/flow/corner_timber

Reads <dir>/place/isaac/{scene.json,meshes.npz}, <dir>/env_masks.npz, <dir>/depth_background.npz,
<dir>/ground.npz; writes <dir>/gates.json and <dir>/gates.jpg (render silhouettes over the photo).
"""
import argparse
import json
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def render(scene, arr, items, w, h):
    """Depth and per-item id (0 = nothing, k+1 = item k) from view 0, the photo's camera (OpenCV pose)."""
    sc = pyrender.Scene(bg_color=[0, 0, 0, 0])
    for k, (pts, faces) in enumerate(items):
        m = trimesh.Trimesh(pts, faces, process=False)
        col = np.array([(k + 1) % 256, (k + 1) // 256, 0, 255], np.uint8)
        m.visual.vertex_colors = np.tile(col, (len(pts), 1))
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.IntrinsicsCamera(scene['fx'], scene['fy'], scene['cx'], scene['cy'], znear=0.05, zfar=3000)
    pose = np.asarray(scene['views'][0]['pose']) @ GL2CV
    sc.add(cam, pose=pose)
    r = pyrender.OffscreenRenderer(w, h)
    col, depth = r.render(sc, flags=pyrender.RenderFlags.FLAT)
    r.delete()
    ids = col[..., 0].astype(int) + 256 * col[..., 1].astype(int)
    return depth, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--min-iou', type=float, default=0.45)
    ap.add_argument('--max-gap', type=float, default=0.06, help='largest gap or sink at the base, share of the '
                    'mesh height')
    ap.add_argument('--min-cover', type=float, default=0.9, help='share of the photo ground the floor covers')
    ap.add_argument('--max-depth-err', type=float, default=0.2, help='median relative depth error allowed')
    args = ap.parse_args()
    d = args.dir
    src = os.path.join(d, 'place', 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = np.load(os.path.join(src, 'meshes.npz'))
    w, h = scene['size']
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    floor = scene['pieces'][0]
    level = float(np.mean(np.asarray(floor['points']) @ up))

    env = scene.get('env', [])
    em = np.load(os.path.join(d, 'env_masks.npz'))
    masks = em['masks']
    items = [(np.asarray(floor['points']), np.array([[0, 1, 2], [0, 2, 3]]))]
    for e in env:  # env_add.py leaves the surroundings in the photo camera frame (identity in every view)
        items.append((arr[f"obj{e['index']}_points"], arr[f"obj{e['index']}_faces"]))
    depth, ids = render(scene, arr, items, w, h)
    dp = np.load(os.path.join(d, 'depth_background.npz'))
    moge = np.where(dp['mask'] > 0, dp['depth'], np.nan)

    def rel_err(sel):
        sel = sel & np.isfinite(moge) & (depth > 0)
        if sel.sum() < 50:
            return None
        return round(float(np.median(np.abs(depth[sel] - moge[sel]) / moge[sel])), 3)

    ground = np.any(np.load(os.path.join(d, 'ground.npz'))['masks'], 0)
    erased = np.asarray(Image.open(os.path.join(d, 'erased.png'))) > 0
    # Within 60 m: a straight road runs to the horizon, past any floor piece, and nothing is placed that far.
    seen = ground & ~erased & (dp['mask'] > 0) & (dp['depth'] < 60)
    cover = float((seen & (ids >= 1)).sum() / max(seen.sum(), 1))  # floor or a mesh in front of it, not sky
    report = {'ground_depth_err': rel_err(ground & ~erased & (ids == 1)), 'ground_cover': round(cover, 3),
              'meshes': []}
    drop = []
    for k, e in enumerate(env):
        pts = arr[f"obj{e['index']}_points"]
        hgt = float(np.ptp(pts @ up))
        # On the surface, by area: a skirt (env_add.py --fill-ground) has few vertices but large faces.
        surf = trimesh.sample.sample_surface(trimesh.Trimesh(pts, arr[f"obj{e['index']}_faces"], process=False),
                                             20000, seed=0)[0]
        base = float(np.percentile(surf @ up, 0.5)) - level
        mi = int(e['name'][:3])
        mask = masks[mi] if mi < len(masks) else np.zeros((h, w), bool)
        sil = ids == k + 2
        iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
        err = rel_err(mask & sil)
        checks = {
            # Sunk is hidden under the floor and the silhouette check still holds it to the photo; floating shows.
            'stands on the ground': base <= args.max_gap * max(hgt, 1.0),
            'silhouette on its mask': iou >= args.min_iou,
            'depth like the photo': err is None or err <= args.max_depth_err,
        }
        why = [c for c, ok in checks.items() if not ok]
        report['meshes'].append({'index': e['index'], 'name': e['name'], 'height_m': round(hgt, 2),
                                 'base_gap_m': round(base, 2), 'iou': round(iou, 3), 'depth_err': err,
                                 'checks': checks, 'why': why})
        if why:
            drop.append(mi)
    report['drop'] = drop
    # The floor piece must lie under every ground pixel of the photo: on 2026-10-07 a floor of a few metres left
    # the house floating over the HDRI's own ground.
    report['ok'] = ((report['ground_depth_err'] is None or report['ground_depth_err'] <= args.max_depth_err)
                    and cover >= args.min_cover)
    with open(os.path.join(d, 'gates.json'), 'w') as f:
        json.dump(report, f, indent=1)

    # Picture: the photo with each mesh's silhouette outlined (green kept, red dropped) and its mask tinted.
    photo = np.asarray(Image.open(os.path.join(d, 'photo.png')).convert('RGB')).astype(np.float32)
    out = photo * 0.6
    for k, mr in enumerate(report['meshes']):
        sil = ids == k + 2
        mi = int(mr['name'][:3])
        if mi < len(masks):
            out[masks[mi]] = out[masks[mi]] * 0.7 + np.array([60, 60, 255]) * 0.3
        edge = sil & ~(np.roll(sil, 1, 0) & np.roll(sil, -1, 0) & np.roll(sil, 1, 1) & np.roll(sil, -1, 1))
        for dy in (-1, 0, 1):
            out[np.roll(edge, dy, 0)] = [255, 40, 40] if mr['why'] else [40, 255, 40]
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(os.path.join(d, 'gates.jpg'), quality=88)
    print(json.dumps({'ground_depth_err': report['ground_depth_err'], 'ground_cover': report['ground_cover'], 'drop': drop,
                      'meshes': [(m['name'], m['iou'], m['base_gap_m'], m['depth_err']) for m in report['meshes']]}))


if __name__ == '__main__':
    main()
