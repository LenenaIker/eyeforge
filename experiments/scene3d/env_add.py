"""The surroundings of a photo (house, garden wall, hedge) as SAM 3D meshes, added to a scene built in pieces.

Mira-Scene's stages 1-5 run on a case whose "objects" are the surroundings (mira_prepare.py with phrases like
"house" or "hedge"); its scene_with_floor.glb places each mesh where the photo shows it. This script moves those
meshes into the photo camera frame of a piece_build.py scene (OpenCV, our own MoGe depth, rescaled to it), adds
them as static objects to every view and writes a copy of the scene to render with piece_isaac.py. It also saves
a sheet of each mesh alone, turned, to judge what SAM 3D makes of things that are not objects. Run in the
`trellis2` env:

    PYOPENGL_PLATFORM=egl python env_add.py --env data/scene3d_test/env_cases/006_street_env \
        --scene data/scene3d_test/ours/street_orbit --out data/scene3d_test/ours/street_env [--drop-walls]
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import open3d as o3d
import pyrender
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, cKDTree

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def simplify(m, max_faces):
    """Quadric decimation keeping the vertex colours: SAM 3D gives 0.2-1.2 M triangles per thing, far more than
    a background seen at a few hundred pixels needs, and the scene has to travel to a web service."""
    if len(m.faces) <= max_faces:
        return m
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    o.vertex_colors = o3d.utility.Vector3dVector(m.visual.vertex_colors[:, :3] / 255)
    o = o.simplify_quadric_decimation(max_faces)
    colours = (np.clip(np.asarray(o.vertex_colors), 0, 1) * 255).astype(np.uint8)
    return trimesh.Trimesh(np.asarray(o.vertices), np.asarray(o.triangles), vertex_colors=colours, process=False)


def env_meshes(case, max_faces):
    """Each surroundings mesh in Mira's OpenGL photo camera frame, with vertex colours, by node name. A flow
    case's poses refined on our depth (flow_refine.py, scene_refined.glb) take the place of Mira's."""
    refined = os.path.join(case, 'scene_refined.glb')
    if os.path.exists(refined):
        scene = trimesh.load(refined)
        out = {}
        for node in scene.graph.nodes_geometry:
            tf, name = scene.graph[node]
            m = scene.geometry[name].copy()
            if m.visual.kind == "texture":
                m.visual = m.visual.to_color()
            m.apply_transform(tf)
            out[node] = simplify(m, max_faces)
        return out
    opt_path = glob.glob(os.path.join(case, 'scene', '*', '*', 'scene_optimization.json'))[0]
    to_cam = np.asarray(json.load(open(opt_path))['floor_to_camera_transform'])
    scene = trimesh.load(os.path.join(os.path.dirname(opt_path), 'scene_with_floor.glb'))
    out = {}
    for node in scene.graph.nodes_geometry:
        tf, name = scene.graph[node]
        if 'floor' in node or 'floor' in name:
            continue
        m = scene.geometry[name].copy()
        if m.visual.kind == "texture":
            m.visual = m.visual.to_color()
        m.apply_transform(to_cam @ tf)
        out[node] = simplify(m, max_faces)
    return out


def skirt(pts, faces, cols, up, level, min_gap=0.1):
    """A floating mesh (its base hidden behind a hedge or a wall in the photo, so SAM 3D left out the lower
    floors) gets a prism from its lowest band down to the floor: the band's convex outline, the band's colour.
    Moving it down instead put a house metres off its own photo pixels (basque_street, 2026-10-07)."""
    h = pts @ up
    base = float(np.percentile(h, 0.5))
    gap = base - level
    if gap < min_gap:
        return pts, faces, cols
    band = h <= base + 0.1 * max(float(np.ptp(h)), 0.5)
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    q = np.stack([pts[band] @ e1, pts[band] @ e2], 1)
    try:
        hull = q[ConvexHull(q).vertices]
    except Exception:
        return pts, faces, cols
    top = [a * e1 + b * e2 + base * up for a, b in hull]
    bottom = [a * e1 + b * e2 + level * up for a, b in hull]
    n0, k = len(pts), len(hull)
    new = np.array(top + bottom)
    f = []
    for j in range(k):
        a, b = j, (j + 1) % k
        f += [[n0 + a, n0 + b, n0 + k + b], [n0 + a, n0 + k + b, n0 + k + a]]
    colour = np.median(cols[band], 0)
    return (np.concatenate([pts, new]), np.concatenate([faces, np.array(f)]),
            np.concatenate([cols, np.tile(colour, (2 * k, 1))]))


def depth_scale(case, ours, masks):
    """Our depth over Mira's on the surroundings' pixels: both are metric MoGe, but on different resolutions."""
    mira = np.load(os.path.join(glob.glob(os.path.join(case, 'depth', '*'))[0], 'depth.npy'))
    z = np.load(os.path.join(ours, 'depth_photo.npz'))['depth']
    z = np.asarray(Image.fromarray(z.astype(np.float32)).resize(mira.shape[::-1], Image.Resampling.NEAREST))
    sel = masks & (mira > 0) & np.isfinite(z) & (z > 0)
    return float(np.median(z[sel] / mira[sel]))


def turntable(meshes, out, t=256, angles=(0, 45, 90, 180, 270)):
    """Each mesh alone from the front (the photo's side) and turned about its vertical axis."""
    sheet = Image.new('RGB', (t * len(angles), t * len(meshes)), 'white')
    r = pyrender.OffscreenRenderer(t, t)
    for row, (name, m) in enumerate(meshes.items()):
        g = m.copy()
        g.apply_translation(-g.bounds.mean(0))
        g.apply_scale(1 / g.extents.max())
        for col, deg in enumerate(angles):
            sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.6] * 3)
            # Mira's frame is OpenGL: the photo camera looks down -Z from the origin, so +Z faces the camera.
            pose = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0])
            sc.add(pyrender.Mesh.from_trimesh(g), pose=pose)
            cam = np.eye(4)
            cam[2, 3] = 1.6
            sc.add(pyrender.PerspectiveCamera(yfov=np.radians(40)), pose=cam)
            sc.add(pyrender.DirectionalLight(intensity=2.5), pose=cam)
            img = Image.fromarray(r.render(sc)[0])
            sheet.paste(img, (col * t, row * t))
            ImageDraw.Draw(sheet).text((col * t + 4, row * t + 4), f'{name} {deg}', fill=(255, 0, 0))
    r.delete()
    sheet.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', required=True, help='Mira case of the surroundings, after stage 5')
    ap.add_argument('--scene', required=True, help='a piece_build.py scene directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--drop-walls', action='store_true', help='leave out the facade planes the meshes replace')
    ap.add_argument('--drop-covered-walls', action='store_true', help='leave out only the facade planes a mesh '
                    'covers (half their area within 1.5 m of it)')
    ap.add_argument('--max-faces', type=int, default=20000, help='triangles per mesh at most')
    ap.add_argument('--snap-ground', action='store_true', help='set each mesh on the floor by its lowest points')
    ap.add_argument('--fill-ground', action='store_true', help='a mesh above the floor gets a skirt down to it '
                    '(its base was hidden in the photo, so SAM 3D left it out); keeps where the photo put it')
    ap.add_argument('--keep', default=None, help='comma-separated mask indices of the meshes to add (the ones '
                    'flow_check_meshes.py passed); default all')
    args = ap.parse_args()
    meshes = env_meshes(args.env, args.max_faces)
    graph = json.load(open(os.path.join(args.env, 'scene_graph.json')))
    names = {n['id']: f"{n['mask_index']:03d} {n['name']}" for n in graph['nodes'] if n['kind'] == 'object'}
    meshes = {next((v for k, v in names.items() if k in node or v[:3] in node), node): m
              for node, m in sorted(meshes.items())}
    if args.keep is not None:
        keep = {int(i) for i in args.keep.split(',') if i.strip()}
        meshes = {k: m for k, m in meshes.items() if k[:3].isdigit() and int(k[:3]) in keep}
    os.makedirs(os.path.join(args.out, 'isaac'), exist_ok=True)
    if meshes:
        turntable(meshes, os.path.join(args.out, 'env_meshes.jpg'))

    masks = np.any([np.asarray(Image.open(p)) > 0 for p in glob.glob(os.path.join(args.env, 'input', 'mask_*.png'))], 0)
    # A flow case (flow_mira_case.py) already carries our own depth, padded: same frame and scale.
    padded = os.path.exists(os.path.join(os.path.dirname(os.path.normpath(args.env)), 'pad.json'))
    s = 1.0 if padded else depth_scale(args.env, args.scene, masks)
    src = os.path.join(args.scene, 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = dict(np.load(os.path.join(src, 'meshes.npz')))
    first = scene['objects']
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    level = float(np.mean(np.asarray(scene['pieces'][0]['points']) @ up))  # piece_build's floor comes first
    for k, (name, m) in enumerate(meshes.items()):
        i = first + k
        pts = trimesh.transform_points(m.vertices, GL2CV) * s
        cols = m.visual.vertex_colors[:, :3] / 255
        faces = m.faces
        if args.snap_ground:  # catalog entry 1: a base hidden in the photo comes out floating or sunk
            pts = pts + (level - np.percentile(pts @ up, 0.5)) * up
        elif args.fill_ground:
            pts, faces, cols = skirt(pts, faces, cols, up, level)
        arr[f'obj{i}_points'] = pts.astype(np.float32)
        arr[f'obj{i}_faces'] = np.asarray(faces).astype(np.int32)
        arr[f'obj{i}_colors'] = np.asarray(cols).astype(np.float32)
        for v in scene['views']:
            v['objects'].append([i, np.eye(4).tolist()])
        print(f'{name}: {len(m.faces)} faces, {np.ptp(pts, 0).round(2).tolist()} m, at {pts.mean(0).round(1).tolist()}')
    scene['objects'] = first + len(meshes)
    scene['env'] = [{'index': first + k, 'name': name} for k, name in enumerate(meshes)]
    walls = tuple(f'tex_wall{j}.png' for j in range(9))
    if args.drop_walls:
        scene['pieces'] = [p for p in scene['pieces'] if not (p['texture'] or '').endswith(walls)]
    elif args.drop_covered_walls and len(meshes):
        # A facade plane stays as the fallback where no kept mesh stands: SAM 3D fails on facades seen at a
        # grazing angle along a street (straight_whitewash, 2026-10-07), the photo's plane does not.
        env_pts = np.concatenate([arr[f'obj{first + k}_points'] for k in range(len(meshes))])
        tree = cKDTree(env_pts[:: max(1, len(env_pts) // 200000)])
        kept = []
        for p in scene['pieces']:
            if not (p['texture'] or '').endswith(walls):
                kept.append(p)
                continue
            c = np.asarray(p['points'])
            uv = np.random.default_rng(0).uniform(0, 1, (400, 2))
            q = c[0] + uv[:, :1] * (c[1] - c[0]) + uv[:, 1:] * (c[3] - c[0])
            if (tree.query(q)[0] < 1.5).mean() < 0.5:
                kept.append(p)
        scene['pieces'] = kept
    for stale in ('orig_scene.json', 'orig_meshes.npz'):  # flow_photometry.py's originals belong to the old place
        if os.path.exists(os.path.join(args.out, 'isaac', stale)):
            os.remove(os.path.join(args.out, 'isaac', stale))
    json.dump(scene, open(os.path.join(args.out, 'isaac', 'scene.json'), 'w'))
    np.savez_compressed(os.path.join(args.out, 'isaac', 'meshes.npz'), **arr)
    shutil.copy(os.path.join(args.scene, 'photo.png'), args.out)
    print(json.dumps({'scale': round(s, 3), 'meshes': len(meshes), 'pieces': len(scene['pieces'])}))


if __name__ == '__main__':
    main()
