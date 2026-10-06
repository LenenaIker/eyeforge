"""Mira-Scene's reconstruction (scene_with_floor.glb, step 5) seen from the photo camera and from it turned about
the vertical through the objects, as piece_build.py --orbit does with ours, plus a side-by-side sheet: photo, then
Mira and ours per angle. Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python mira_views.py --case data/scene3d_test/cases/006_street_corner \
        --ours data/scene3d_test/ours/street_orbit --orbit=-40,-20,20,40 --out data/scene3d_test/compare/street.jpg
"""
import argparse
import glob
import json
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--ours', required=True, help='a piece_build.py --orbit scene rendered by piece_isaac.py')
    ap.add_argument('--orbit', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    angles = [float(a) for a in args.orbit.split(',')]
    opt = json.load(open(glob.glob(os.path.join(args.case, 'scene', '*', '*', 'scene_optimization.json'))[0]))
    scene_path = os.path.join(os.path.dirname(glob.glob(os.path.join(args.case, 'scene', '*', '*',
                                                                      'scene_optimization.json'))[0]),
                              'scene_with_floor.glb')
    to_cam = np.asarray(opt['floor_to_camera_transform'])  # floor frame -> OpenGL camera of the photo
    k = np.load(glob.glob(os.path.join(args.case, 'depth', '*', 'intrinsics.npy'))[0])
    photo = Image.open(os.path.join(args.case, 'input', 'scene.png')).convert('RGB')
    w, h = photo.size
    fx, fy, cx, cy = (k[0, 0], k[1, 1], k[0, 2], k[1, 2]) if k[0, 0] > 10 else (k[0, 0] * w, k[1, 1] * h,
                                                                                  k[0, 2] * w, k[1, 2] * h)
    tscene = trimesh.load(scene_path)
    sc = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[0.5] * 3)
    centres = []
    for name, geom in tscene.geometry.items():
        node = [n for n in tscene.graph.nodes_geometry if tscene.graph[n][1] == name][0]
        tf = to_cam @ tscene.graph[node][0]
        sc.add(pyrender.Mesh.from_trimesh(geom), pose=tf)
        if 'floor' not in name and 'floor' not in node:
            centres.append(trimesh.transform_points(geom.bounds.mean(0, keepdims=True), tf)[0])
    pivot = np.mean(centres, 0)
    up = to_cam[:3, 1]  # floor +Y in the camera frame
    sc.add(pyrender.DirectionalLight(intensity=3.0), pose=trimesh.transformations.rotation_matrix(-0.8, [1, 0, 0]))
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.05, zfar=500)
    cam_node = sc.add(cam, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(w, h)
    mira = []
    for deg in [0.0] + angles:
        sc.set_pose(cam_node, trimesh.transformations.rotation_matrix(np.radians(deg), up, pivot))
        mira.append(Image.fromarray(r.render(sc)[0]))
    r.delete()
    ours = [Image.open(os.path.join(args.ours, 'isaac', f'render_{i:02d}.png')).convert('RGB')
            for i in range(len(angles) + 1)]
    t = 320
    sheet = Image.new('RGB', (t * (len(angles) + 2), t * 2), 'white')
    sheet.paste(photo.resize((t, t)), (0, 0))
    ImageDraw.Draw(sheet).text((5, 5), 'photo', fill=(255, 0, 0))
    for i, deg in enumerate([0.0] + angles):
        for row, (name, im) in enumerate((('Mira-Scene', mira[i]), ('ours, in pieces', ours[i]))):
            x, y = t * (i + 1), t * row
            sheet.paste(im.resize((t, round(t * im.height / im.width))).crop((0, 0, t, t)), (x, y))
            ImageDraw.Draw(sheet).text((x + 5, y + 5), f'{name}, turn {deg:+.0f}', fill=(255, 0, 0))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sheet.save(args.out, quality=90)
    print(args.out)


if __name__ == '__main__':
    main()
