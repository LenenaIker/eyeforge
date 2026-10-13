"""Labels for the views that bg_isaac.py rendered: per view, an instance map of the object meshes (occluded by the
background mesh and by each other) and one box per visible object, like BOP's mask_visib boxes. Run in the
`trellis2` env:

    PYOPENGL_PLATFORM=egl python view_labels.py --dir data/scene3d_test/ours/hope_val_000001 [--min-pixels 30]

Reads <dir>/isaac/{scene.json,meshes.npz}; writes <dir>/isaac/labels.json
({"render_NN.png": [[x0, y0, x1, y1], ...]}, pixels) and inst_NN.png (instance id + 1, 0 = none). For a scene
with "roles" (flow_populate.py) each label is {"box", "name", "role"} and the place's own objects get none: the
student trains on the targets, the distractors' boxes are kept to count how often they show.
"""
import argparse
import json
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image
from scipy import ndimage

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera (+Y down, +Z forward) to OpenGL (+Y up, -Z forward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--min-pixels', type=int, default=30, help='smallest visible area that gets a box')
    args = ap.parse_args()
    d = os.path.join(args.dir, 'isaac')
    scene_def = json.load(open(os.path.join(d, 'scene.json')))
    arrays = np.load(os.path.join(d, 'meshes.npz'))
    roles = scene_def.get('roles')  # flow_populate.py: name and role (target, distractor, scene) per object
    w, h = scene_def['size']
    if 'bg_points' in arrays.files:
        bg = trimesh.Trimesh(arrays['bg_points'], arrays['bg_faces'], process=False)
    else:  # a place built in pieces (piece_build.py): its quads hide what stands behind them
        quads = [np.asarray(p['points'], float) for p in scene_def['pieces']]
        bg = trimesh.Trimesh(np.concatenate(quads), np.concatenate(
            [np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]]) + 4 * i for i in range(len(quads))]), process=False)
    objs = [trimesh.Trimesh(arrays[f'obj{i}_points'], arrays[f'obj{i}_faces'], process=False)
            for i in range(scene_def['objects'])]
    cam = pyrender.IntrinsicsCamera(scene_def['fx'], scene_def['fy'], scene_def['cx'], scene_def['cy'],
                                    znear=0.01, zfar=5000)
    renderer = pyrender.OffscreenRenderer(w, h)
    labels = {}
    for k, view in enumerate(scene_def['views']):
        scene = pyrender.Scene()
        seg = {}
        seg[scene.add(pyrender.Mesh.from_trimesh(bg, smooth=False))] = (0, 0, 0)
        ids = []
        for j, (i, tf) in enumerate(view['objects']):
            node = scene.add(pyrender.Mesh.from_trimesh(objs[i], smooth=False), pose=np.asarray(tf))
            # Two channels: a street with its trees holds hundreds of objects, and with one the ids wrapped past
            # 255 and the cars lost their boxes (roads, 2026-10-13).
            seg[node] = ((j + 1) % 256, (j + 1) // 256, 0)
            ids.append(i)
        scene.add(cam, pose=np.asarray(view['pose']) @ CV2GL)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg)
        inst = color[..., 0].astype(np.int32) + 256 * color[..., 1].astype(np.int32)
        boxes = []
        for j in range(1, len(ids) + 1):
            # Stray mesh fragments far from the object would stretch the box: keep the blobs at least a fifth
            # of the largest one (an occluder can still split an object in two).
            parts, n = ndimage.label(inst == j)
            if n == 0:
                continue
            sizes = ndimage.sum(np.ones_like(parts), parts, range(1, n + 1))
            keep = np.isin(parts, 1 + np.nonzero(sizes >= sizes.max() / 5)[0])
            ys, xs = np.nonzero(keep)
            if len(xs) >= args.min_pixels:
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                if roles is None:
                    boxes.append(box)
                elif roles[ids[j - 1]]['role'] != 'scene':  # the place's own furniture is background
                    boxes.append({'box': box, **roles[ids[j - 1]]})
        labels[f'render_{k:02d}.png'] = boxes
        Image.fromarray(inst.astype(np.uint16)).save(os.path.join(d, f'inst_{k:02d}.png'))
    renderer.delete()
    with open(os.path.join(d, 'labels.json'), 'w') as f:
        json.dump(labels, f)
    print({k: len(v) for k, v in labels.items()})


if __name__ == '__main__':
    main()
