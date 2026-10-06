"""Contact sheet per Mira-Scene case: the photo, the rebuilt scene seen from the photo's camera, and the scene
from new viewpoints (orbit left and right, higher, further back), to judge how far the camera can move.
Run in any env with pyrender (e.g. `trellis2`):

    PYOPENGL_PLATFORM=egl python render_scenes.py --cases <dir> [--out <dir>]

Reads <case>/scene/sam3d/moge2_depth/scene_with_floor.glb (floor frame: +Y up, floor at y=0),
<case>/floor/floor_alignment.json (camera <-> floor transforms) and <case>/depth/moge2/fov_x_rad.txt.
"""
import argparse
import json
import math
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw

SIZE = 384


def look_at(eye, target, up=(0, 1, 0)):
    """OpenGL camera pose (camera looks down -Z) at eye towards target."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def render(scene_mesh, pose, yfov, renderer):
    scene = pyrender.Scene.from_trimesh_scene(scene_mesh, ambient_light=[0.55, 0.55, 0.55], bg_color=[1, 1, 1, 1])
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, znear=0.01, zfar=500), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=2.5), pose=look_at(pose[:3, 3] + [2, 5, 1], pose[:3, 3]))
    color, _ = renderer.render(scene)
    return Image.fromarray(color)


def sheet_for(case_dir, renderer):
    glb = os.path.join(case_dir, 'scene', 'sam3d', 'moge2_depth', 'scene_with_floor.glb')
    if not os.path.exists(glb):
        return None
    scene_mesh = trimesh.load(glb, force='scene')
    with open(os.path.join(case_dir, 'floor', 'floor_alignment.json')) as f:
        align = json.load(f)
    cam_to_floor = np.asarray(align['camera_to_floor_transform'])
    fov = float(open(os.path.join(case_dir, 'depth', 'moge2', 'fov_x_rad.txt')).read().split()[0])

    photo_pose = cam_to_floor  # the photo's camera expressed in the floor frame
    eye = photo_pose[:3, 3]
    # Orbit target: centre of the objects (all geometry but the floor), on the floor.
    objs = [g.bounds for n, g in scene_mesh.geometry.items() if 'floor' not in n.lower()]
    bounds = scene_mesh.bounds if not objs else np.array([np.min([b[0] for b in objs], 0), np.max([b[1] for b in objs], 0)])
    target = (bounds[0] + bounds[1]) / 2
    offset = eye - target
    dist = np.linalg.norm(offset[[0, 2]])

    def orbit(yaw_deg, height_gain=0.0, scale=1.0):
        a = math.radians(yaw_deg)
        x, z = offset[0], offset[2]
        rx, rz = x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)
        e = target + np.array([rx * scale, offset[1] * scale + height_gain * dist, rz * scale])
        return look_at(e, target)

    views = [('photo camera', photo_pose), ('orbit -30', orbit(-30)), ('orbit +30', orbit(30)),
             ('orbit +90', orbit(90)), ('higher', orbit(0, 0.6)), ('back x1.6', orbit(0, 0, 1.6))]
    photo = Image.open(os.path.join(case_dir, 'input', 'scene.png')).convert('RGB').resize((SIZE, SIZE))
    tiles = [('photo', photo)] + [(name, render(scene_mesh, pose, fov, renderer)) for name, pose in views]
    sheet = Image.new('RGB', (SIZE * 4, SIZE * 2), 'white')
    for i, (name, im) in enumerate(tiles):
        x, y = (i % 4) * SIZE, (i // 4) * SIZE
        sheet.paste(im, (x, y))
        ImageDraw.Draw(sheet).text((x + 6, y + 6), name, fill=(220, 0, 0))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', default=None, help='default: <cases>/../sheets')
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.cases)), 'sheets')
    os.makedirs(out, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(SIZE, SIZE)
    for name in sorted(os.listdir(args.cases)):
        sheet = sheet_for(os.path.join(args.cases, name), renderer)
        if sheet is not None:
            sheet.save(os.path.join(out, f'{name}.jpg'), quality=88)
            print('ok', name, flush=True)
        else:
            print('no scene', name, flush=True)


if __name__ == '__main__':
    main()
