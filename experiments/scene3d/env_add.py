"""The surroundings of a photo (house, garden wall, hedge) as SAM 3D meshes, added to a scene built in pieces.

Mira-Scene's stages 1-5 run on a case whose "objects" are the surroundings (mira_prepare.py with phrases like
"house" or "hedge"); its scene_with_floor.glb places each mesh where the photo shows it. This script moves those
meshes into the photo camera frame of a piece_build.py scene (OpenCV, our own MoGe depth, rescaled to it), adds
them as static objects to every view and writes a copy of the scene to render with piece_isaac.py. It also saves
a sheet of each mesh alone, turned, to judge what SAM 3D makes of things that are not objects. Run in the
`trellis2` env:

    PYOPENGL_PLATFORM=egl python env_add.py --env data/scene3d_test/env_cases/006_street_env \
        --scene data/scene3d_test/ours/street_orbit --out data/scene3d_test/ours/street_env [--drop-walls]

With --photo-texture (photo2scene), every mesh face the photo's camera sees takes the photo itself as its texture:
each vertex is projected into the erased photo (background.png) and keeps that position as its texture
coordinate, so a flat facade of a few large triangles still shows every window at the photo's resolution. Faces
the camera does not see (sides, back, the skirts) keep SAM 3D's vertex colours, toned to the photo by the median
ratio on the faces it sees. Per object in meshes.npz: obj{i}_uv, obj{i}_photo_faces (bool per face) and
obj{i}_texture (the image).
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
from PIL import Image, ImageDraw, ImageFilter
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
    """Each surroundings mesh in the OpenGL photo camera frame, with vertex colours, by node name: photo2scene's
    refined poses (flow_refine.py, scene_refined.glb), or an older Mira-Scene case's own placement."""
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


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def project(v, K):
    """Continuous pixel coordinates (pixel centres at +0.5) and depth of OpenGL camera points."""
    z = -v[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return K[0, 2] + K[0, 0] * v[:, 0] / z, K[1, 2] - K[1, 1] * v[:, 1] / z, z


def split_facing(m, K, W, H, max_px=16.0, rounds=6):
    """Faces turned to the camera split until none spans more than max_px on the photo: decimation leaves a flat
    facade as a few large triangles, and one corner past the mask's edge would cost the whole triangle its photo
    (basque_street's main house: photo on 6 % of its faces, 2026-10-07)."""
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    col = m.visual.vertex_colors[:, :3].astype(float)
    for _ in range(rounds):
        u, w, z = project(v, K)
        uv = np.stack([u, w], 1)[f]
        span = np.max([np.linalg.norm(uv[:, a] - uv[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 0)
        cen = v[f].mean(1)
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        facing = np.abs((n * cen).sum(1)) > 0.25 * np.linalg.norm(n, axis=1) * np.linalg.norm(cen, axis=1)
        on = (z[f] > 0.05).all(1) & ((uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)).any(1)
        big = np.nonzero(facing & on & (span > max_px))[0]
        if not len(big):
            break
        v, f, attr = trimesh.remesh.subdivide(v, f, face_index=big, vertex_attributes={'col': col})
        col = attr['col']
    return trimesh.Trimesh(v, f, vertex_colors=np.clip(col, 0, 255).astype(np.uint8), process=False)


def photo_texture(meshes, d, tol=0.03, min_cos=0.25):
    """Per mesh (OpenGL photo camera frame, named "NNN phrase" by its mask): the mesh with its faces turned to
    the camera split small (split_facing), texture coordinates of every vertex in the photo, which faces show the
    photo (all three vertices seen: in front of every other mesh, inside the mesh's own mask, the face not
    grazing), and its vertex colours toned to the photo."""
    photo = np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB'))
    H, W = photo.shape[:2]
    K = np.load(os.path.join(d, 'depth_background.npz'))['intrinsics'] * np.array([[W], [H], [1]])
    em = np.load(os.path.join(d, 'env_masks.npz'))['masks']
    meshes = {name: split_facing(m, K, W, H) for name, m in meshes.items()}
    sc = pyrender.Scene()
    for m in meshes.values():
        sc.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(m.vertices, m.faces, process=False)))
    sc.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], znear=0.05, zfar=2000), pose=np.eye(4))
    r = pyrender.OffscreenRenderer(W, H)
    depth = r.render(sc, flags=pyrender.RenderFlags.DEPTH_ONLY)
    r.delete()
    lin = to_linear(photo / 255.0)
    out = {}
    for name, m in meshes.items():
        mask = em[int(name[:3])].astype(np.uint8) * 255
        # A pixel short of the mask's edge: the edge mixes the thing with what stands behind it.
        mask = np.asarray(Image.fromarray(mask).resize((W, H), Image.Resampling.NEAREST)
                          .filter(ImageFilter.MinFilter(5))) > 0
        v = np.asarray(m.vertices)
        u, w, z = project(v, K)
        inside = (z > 0.05) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
        xi = np.clip(np.nan_to_num(u), 0, W - 1).astype(int)
        yi = np.clip(np.nan_to_num(w), 0, H - 1).astype(int)
        dz = depth[yi, xi]
        seen = inside & (dz > 0) & (np.abs(dz - z) < tol * z + 0.05) & mask[yi, xi]
        cen = v[m.faces].mean(1)
        cos = np.abs((m.face_normals * cen).sum(1)) / np.maximum(np.linalg.norm(cen, axis=1), 1e-6)
        faces = seen[m.faces].all(1) & (cos > min_cos)
        uv = np.where(inside[:, None], np.stack([u / W, 1 - w / H], 1), 0).astype(np.float32)
        cols = m.visual.vertex_colors[:, :3] / 255.0
        if seen.sum() > 50:
            gain = np.median(lin[yi, xi][seen], 0) / np.maximum(np.median(to_linear(cols)[seen], 0), 1e-4)
            cols = to_srgb(to_linear(cols) * np.clip(gain, 0.5, 2.0))
        out[name] = (m, uv, faces, cols)
        print(f'{name}: photo on {faces.mean():.0%} of the faces')
    return out


def skirt(pts, faces, cols, up, level, min_gap=0.1):
    """A floating mesh (its base hidden behind a hedge or a wall in the photo, so SAM 3D left out the lower
    floors) gets a prism from its lowest band down to the floor: the band's convex outline, the band's colour.
    Moving it down instead put a house metres off its own photo pixels (basque_street, 2026-10-07)."""
    h = pts @ up
    base = float(np.percentile(h, 0.5))
    gap = base - level
    if gap < min_gap:  # a few centimetres: set it down, it stays on its photo pixels (sideboard 7 cm, home_office)
        return pts - max(gap, 0.0) * up, faces, cols
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
    ap.add_argument('--photo-texture', action='store_true', help='faces the photo sees take the photo as texture')
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

    # photo2scene's meshes (flow_sam3d.py) and a padded Mira flow case are posed on our own depth: same frame and
    # scale. An older Mira case had its own depth.
    same = os.path.isdir(os.path.join(args.env, 'sam3d')) or \
        os.path.exists(os.path.join(os.path.dirname(os.path.normpath(args.env)), 'pad.json'))
    if same:
        s = 1.0
    else:
        masks = np.any([np.asarray(Image.open(p)) > 0
                        for p in glob.glob(os.path.join(args.env, 'input', 'mask_*.png'))], 0)
        s = depth_scale(args.env, args.scene, masks)
    src = os.path.join(args.scene, 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = dict(np.load(os.path.join(src, 'meshes.npz')))
    first = scene['objects']
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    level = float(np.mean(np.asarray(scene['pieces'][0]['points']) @ up))  # piece_build's floor comes first
    tex = photo_texture(meshes, args.scene) if args.photo_texture and meshes else {}
    if tex:
        tex_path = os.path.abspath(os.path.join(args.out, 'isaac', 'tex_photo.png'))
        shutil.copy(os.path.join(args.scene, 'background.png'), tex_path)
    for k, (name, m) in enumerate(meshes.items()):
        i = first + k
        pts = trimesh.transform_points(m.vertices, GL2CV) * s
        cols = m.visual.vertex_colors[:, :3] / 255
        faces = m.faces
        if name in tex:
            m, uv, photo_faces, cols = tex[name]
            pts = trimesh.transform_points(m.vertices, GL2CV) * s
            faces = m.faces
        if args.snap_ground:  # catalog entry 1: a base hidden in the photo comes out floating or sunk
            pts = pts + (level - np.percentile(pts @ up, 0.5)) * up
        elif args.fill_ground:
            pts, faces, cols = skirt(pts, faces, cols, up, level)
        if name in tex:  # the skirt's new vertices and faces take no photo
            arr[f'obj{i}_uv'] = np.concatenate([uv, np.zeros((len(pts) - len(uv), 2), np.float32)])
            arr[f'obj{i}_photo_faces'] = np.concatenate([photo_faces, np.zeros(len(faces) - len(photo_faces), bool)])
            arr[f'obj{i}_texture'] = np.array(tex_path)
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
