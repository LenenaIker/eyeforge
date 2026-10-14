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

from bg_build import floor_height

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


def skirt(pts, faces, cols, up, floor, min_gap=0.1):
    """A floating mesh (its base hidden behind a hedge or a wall in the photo, so SAM 3D left out the lower
    floors) gets a prism from its lowest band down to the floor: the band's convex outline, the band's colour.
    Moving it down instead put a house metres off its own photo pixels (basque_street, 2026-10-07). floor(X): the
    up coordinate of the floor under each point (a slope is not level)."""
    h = pts @ up - floor(pts)  # above the floor right under each point
    base = float(np.percentile(h, 0.5))
    if base < min_gap:  # a few centimetres: set it down, it stays on its photo pixels (sideboard 7 cm, home_office)
        # Measured on the surface by area, as flow_gates.py does: by vertices a sideboard was still 6 cm up.
        surf = trimesh.sample.sample_surface(trimesh.Trimesh(pts, faces, process=False), 20000, seed=0)[0]
        return pts - max(float(np.percentile(surf @ up - floor(surf), 0.5)), 0.0) * up, faces, cols
    band = h <= base + 0.1 * max(float(np.ptp(h)), 0.5)
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    q = np.stack([pts[band] @ e1, pts[band] @ e2], 1)
    try:
        hull = q[ConvexHull(q).vertices]
    except Exception:
        return pts, faces, cols
    flat = np.array([a * e1 + b * e2 for a, b in hull])
    under = floor(flat)
    top = list(flat + (under + base)[:, None] * up)
    bottom = list(flat + under[:, None] * up)
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


def vegetation(args, scene, arr, up, floor):
    """flow_vegetation.py's NVIDIA trees and shrubs into the place: each where its mask's depth points stand, its
    height above the floor the mask's, turned at random. meshes.npz gets the asset's proxy (veg_catalog.py) for the
    pyrender steps and its USD URL with the pose from the USD's own units and axes (obj<i>_usd, obj<i>_usd_pose)
    for Isaac. Returns the scene's new env entries."""
    import veg_catalog
    picks = json.load(open(args.vegetation))['assets']
    vz = np.load(os.path.join(os.path.dirname(args.vegetation), 'vegetation.npz'))
    dp = np.load(os.path.join(args.scene, 'depth_background.npz'))
    from section_fill import pose
    cat = {c['name']: c for c in veg_catalog.catalog(args.veg_dir)}
    out = []
    for a in picks:
        m = vz['masks'][a['index']] & (dp['mask'] > 0)
        Q = dp['points'][m]
        Q = Q[np.isfinite(Q).all(1) & (np.linalg.norm(Q, axis=1) < 80)]
        if len(Q) < 50:
            print(f"{a['phrase']} ({a['name']}): too few depth points, left out")
            continue
        height = float(np.percentile(Q @ up - floor(Q), 98))
        # The photo sees the crown's front: the trunk stands about half a crown behind its median point.
        flat = Q - np.outer(Q @ up, up)
        centre = np.median(flat, 0)
        away = centre / max(np.linalg.norm(centre), 1e-6)
        side = np.cross(up, away)
        width = float(np.ptp(np.percentile(Q @ side, [5, 95])))
        centre = centre + away * 0.25 * width
        base = centre + (floor(centre)[0] - centre @ up) * up
        z = np.load(veg_catalog.proxy(args.veg_dir, cat[a['name']]))
        yaw = np.random.default_rng(a['index']).uniform(0, 2 * np.pi)
        i = add_asset(scene, arr, a['url'], pose(base, height, z, up, yaw), z)
        print(f"{a['phrase']} -> {a['name']}: {height:.1f} m (asset {float(z['height_m']):.1f} m), "
              f"at {base.round(1).tolist()}")
        out.append({'index': i, 'name': f"{a['index']:03d} {a['phrase']}", 'vegetation': a['name']})
    return out


def add_asset(scene, arr, url, place, proxy):
    """An NVIDIA asset as the scene's next object: its proxy posed for the pyrender steps, its USD and pose (from
    the USD's own units and axes) for Isaac. Returns its index."""
    i = scene['objects']
    arr[f'obj{i}_points'] = (proxy['points'] @ place[:3, :3].T + place[:3, 3]).astype(np.float32)
    arr[f'obj{i}_faces'] = proxy['faces'].astype(np.int32)
    arr[f'obj{i}_colors'] = proxy['colors'].astype(np.float32)
    arr[f'obj{i}_usd'] = np.array(url)
    arr[f'obj{i}_usd_pose'] = (place @ proxy['to_proxy']).astype(np.float64)
    scene['objects'] = i + 1
    return i


BUILDING_WORDS = ('house', 'building', 'apartment', 'block', 'tower', 'facade')
LOT_MIN, LOT_MAX = 0.6, 6.0  # a building's base this far over the street stands on a lot raised to it
# Where a building's base is: its lowest 5 % of points, not 0.5 % (a stray vertex below it left the building
# floating a metre over its lot, photo 27).
LOT_BASE_PCT = 5


def raised(name, pts, up, floor):
    """How far a building's base stands over the floor, when it stands on a raised lot (LOT_MIN to LOT_MAX), else
    None: the photo's terraces behind a retaining wall (the pots' photo 27, 2026-10-13), which a skirt drew as a
    dark pedestal."""
    if not any(w in name for w in BUILDING_WORDS):
        return None
    gap = float(np.percentile(pts @ up - floor(pts), LOT_BASE_PCT))
    return gap if LOT_MIN < gap < LOT_MAX else None


TILE_ODD = 0.05  # a wall's crop with more of its pixels this far from its median shade has something before it


def wall_tile(arr, i, up, path):
    """A crop of the photo where object i (a wall) shows in it, saved to path for tiling, and its size in metres
    (along, up); None when the wall has no photo texture or something stands before it (a parked car tiled over
    every lot, photo 27, 2026-10-14): over TILE_ODD of the crop's pixels 0.25 off its median luminance (its
    stones 0.002, the car 0.12)."""
    if i is None or f'obj{i}_texture' not in arr:
        return None
    on = np.asarray(arr[f'obj{i}_photo_faces'], bool)
    if on.sum() < 4:
        return None
    vid = np.unique(np.asarray(arr[f'obj{i}_faces'])[on])
    uv, X = np.asarray(arr[f'obj{i}_uv'])[vid], np.asarray(arr[f'obj{i}_points'])[vid]
    img = Image.open(str(arr[f'obj{i}_texture'])).convert('RGB')
    W, H = img.size
    # The middle of what the photo shows of it: its ends are often cut by a car or the frame.
    u0, u1 = np.percentile(uv[:, 0], [15, 85]) * W
    t0, t1 = (1 - np.percentile(uv[:, 1], [85, 15])) * H
    if u1 - u0 < 16 or t1 - t0 < 16:
        return None
    h = X @ up
    flat = X - np.outer(h, up)
    along = float(np.ptp(flat @ np.linalg.eigh(np.cov((flat - flat.mean(0)).T))[1][:, -1]))
    size = (max(along * 0.7, 0.5), max(float(np.ptp(h)) * 0.7, 0.3))
    crop = img.crop((int(u0), int(t0), int(u1), int(t1)))
    lum = np.asarray(crop.convert('L').filter(ImageFilter.GaussianBlur(2)), float) / 255
    if np.mean(np.abs(lum - np.median(lum)) > 0.25) > TILE_ODD:
        return None
    crop.save(path)
    return path, size


def lots(args, scene, arr, up, floor, high):
    """Lots raised under the buildings that stand over the street (high: {env name: object index}), laid as a
    scene builder does, between the sidewalk and the building: a block from the building's front, or the
    retaining wall in front of it, to a metre behind it, its top at the building's base, its sides the wall's
    colour. The building's env entry gets 'on' (its lot's index), so the templates move them together. Returns the
    lots' env entries."""
    if not high:
        return []
    flat = lambda X: X - np.outer(np.atleast_2d(X) @ up, up)  # noqa: E731
    env = {e['name']: e for e in scene['env']}
    sec = json.load(open(args.section)) if args.section and os.path.exists(args.section) else None
    if sec:
        c, a, b = (np.asarray(sec['road'][k], float) for k in ('centre', 'along', 'across'))
    else:  # the buildings' own long axis
        q = flat(np.concatenate([arr[f'obj{i}_points'] for i in high.values()]))
        c = q.mean(0)
        a = np.linalg.eigh(np.cov((q - c).T))[1][:, -1]
        a -= (a @ up) * up
        a /= np.linalg.norm(a)
        b = np.cross(a, up)
    walls = [e for e in scene['env'] if 'wall' in e['name'] or 'fence' in e['name']]
    top_colour = np.asarray(scene['pieces'][0]['colour'], float) * 0.9
    out = []
    def footprint(i):
        """A building's base: its points, top, side of the street, across span and along span."""
        pts = arr[f'obj{i}_points']
        h = pts @ up - floor(pts)
        base = pts[h < np.percentile(h, LOT_BASE_PCT) + 0.5]
        top = float(np.percentile(pts @ up, LOT_BASE_PCT))
        U, V = (flat(base) - c) @ b, (flat(base) - c) @ a
        side = 1.0 if np.median(U) >= 0 else -1.0
        inner, outer = float((side * U).min()), float((side * U).max()) + 1.0
        return top, side, inner, outer, float(V.min()) - 1.0, float(V.max()) + 1.0

    def in_front(side, inner, v0, v1):
        """The retaining wall in front of a building (the lot's front) and where it stands across, or None."""
        best = None
        for e in walls:
            W = arr[f"obj{e['index']}_points"]
            wu, wv = side * ((flat(W) - c) @ b), (flat(W) - c) @ a
            if wv.max() > v0 and wv.min() < v1 and inner - 6.0 < np.median(wu) < inner + 0.5:
                if best is None or np.median(wu) < best[0]:
                    best = (float(np.median(wu)), e['index'])
        return best

    # A lot with no wall in front of it takes the stones of another lot's front wall: the nearest wall of any kind
    # was a stretch of the photo with cars parked before it, and painted them on the lots (photo 27, 2026-10-14).
    fronts = [w[1] for i in high.values() if (w := in_front(*[footprint(i)[k] for k in (1, 2, 4, 5)]))]
    shared = max(set(fronts), key=fronts.count) if fronts else None
    tiles = {w: t for w in dict.fromkeys(fronts)
             if (t := wall_tile(arr, w, up, os.path.join(args.out, 'isaac', f'tex_lot_wall{w}.png')))}
    for name, i in high.items():
        top, side, inner, outer, v0, v1 = footprint(i)
        front, wall = inner - 0.5, shared
        if (w := in_front(side, inner, v0, v1)):
            front, wall = min(front, w[0]), w[1]
        wall_colour = np.median(arr[f'obj{wall}_colors'], 0) if wall is not None else np.array([0.55, 0.53, 0.5])
        ring = [c + side * u * b + v * a for u, v in ((front, v0), (outer, v0), (outer, v1), (front, v1))]
        tops = [x + (top - x @ up) * up for x in ring]
        bottoms = [x + (float(floor(x)[0]) - 0.2 - x @ up) * up for x in ring]
        # The top, then each side on its own four corners (its own texture coordinates), in another colour.
        P = list(tops)
        F = [[0, 1, 2], [0, 2, 3]]
        for j in range(4):
            k = (j + 1) % 4
            n = len(P)
            P += [tops[j], tops[k], bottoms[k], bottoms[j]]
            F += [[n, n + 1, n + 2], [n, n + 2, n + 3]]
        P = np.array(P)
        F = F + [f[::-1] for f in F]  # both sides: seen from the street and from above
        C = np.array([top_colour] * 4 + [wall_colour] * 16)
        tile = tiles.get(wall) or next(iter(tiles.values()), None)
        li = scene['objects']
        arr[f'obj{li}_points'] = P.astype(np.float32)
        arr[f'obj{li}_faces'] = np.asarray(F, np.int32)
        arr[f'obj{li}_colors'] = np.clip(C, 0, 1).astype(np.float32)
        if tile:  # the sides tiled with the retaining wall's own stones, metres over the crop's size
            path, (tw, th) = tile
            uv = np.zeros((len(P), 2))
            for j in range(4):
                n = 4 + 4 * j
                w_m = float(np.linalg.norm(P[n + 1] - P[n]))
                h_m = float(np.linalg.norm(P[n] - P[n + 3]))
                uv[n:n + 4] = [[0, h_m / th], [w_m / tw, h_m / th], [w_m / tw, 0], [0, 0]]
            arr[f'obj{li}_uv'] = uv.astype(np.float32)
            arr[f'obj{li}_photo_faces'] = np.array([False, False] + [True] * 8 + [False, False] + [True] * 8)
            arr[f'obj{li}_texture'] = np.array(path)
            arr[f'obj{li}_tile'] = np.array(True)
        scene['objects'] = li + 1
        env[name]['on'] = li
        out.append({'index': li, 'name': f'{name[:3]} lot', 'lot': True})
        print(f'{name}: on a lot {top - float(floor(ring[0])[0]):.1f} m over the street, '
              f'{outer - front:.1f} x {v1 - v0:.1f} m')
    return out


def off_photo_plants(args, scene, arr, fixed, objects, min_show=0.003, min_on=0.5):
    """The filled plants without those the photo's camera sees (at least min_show of the picture) mostly where the
    photo shows no plant (under min_on of their pixels on flow_section.py's plants.npz): the photo is the truth for
    what it sees. A row laid along the camera's own path put a tree in front of it, 8 to 12 % of the picture with
    18 to 34 % on plants, against 74 to 78 % for the photo's own tree (roads photo 3, 2026-10-14)."""
    from flow_gates import render
    path = os.path.join(os.path.dirname(args.section), 'plants.npz')
    if not objects or not os.path.exists(path):
        return objects
    plants = np.load(path)['mask']
    w, h = scene['size']
    items = fixed + [(o['proxy']['points'] @ o['place'][:3, :3].T + o['place'][:3, 3], o['proxy']['faces'])
                     for o in objects]
    _, ids = render(scene, arr, items, w, h)
    if plants.shape != ids.shape:
        return objects
    keep = []
    for k, o in enumerate(objects):
        sil = ids == len(fixed) + k + 1
        n = int(sil.sum())
        on = (sil & plants).sum() / max(n, 1)
        if n >= min_show * sil.size and on < min_on:
            print(f"left out a {o['band']} ({o['name']}): {n / sil.size:.1%} of the photo, {on:.0%} on its plants",
                  flush=True)
        else:
            keep.append(o)
    return keep


def bands(args, scene, arr, up, floor, rounds=6, tolerance=0.15):
    """flow_section.py's bands filled over the street the photo sees (section_fill.py): a front row of trees at the
    photo's height and spacing, a back row taller and leaning over the road, hedges and shrubs, kept off the
    road, the meshes and the photo's own trees; grass and sidewalk strips. Then tuned to the photo, seen from its
    camera (proxies rendered with pyrender, a few rounds): the back row grows or shrinks until the trees cover as
    much of the picture as the photo's, it leans more or less until they cover as much of its top third (the
    canopy over the road), and the shrubs thicken until theirs match. Returns the new env entries."""
    import section_fill
    from flow_gates import render
    section = json.load(open(args.section))
    target = section.get('image_cover', {})
    env = scene.get('env', [])
    # Kept away from every surroundings object already there: a mesh by all its points, a tree by its trunk (its
    # lowest metre; by the whole crown, a row of trees left no room for one more, 2026-10-13).
    avoid = []
    for e in env:
        p = arr[f"obj{e['index']}_points"]
        if e.get('vegetation'):
            hgt = p @ up
            p = p[hgt < hgt.min() + 1.0]
        avoid.append(p[:: max(1, len(p) // 400)])
    avoid = np.concatenate(avoid) if avoid else np.zeros((0, 3))
    # Beyond what the road's own pixels show: the photo's trees go on down the street.
    vis = section['road']['visible']
    along = (vis[0], max(vis[1], 40.0))
    w, h = scene['size']
    fixed = [(arr[f"obj{e['index']}_points"], arr[f"obj{e['index']}_faces"]) for e in env]
    fixed_group = ['tree' if e.get('vegetation') else None for e in env]
    look, objects, strips = dict(section_fill.LOOK), [], []
    for rnd in range(rounds):
        objects, strips = section_fill.fill(section, up, floor, along, args.veg_dir, seed=7, avoid=avoid, look=look)
        items = fixed + [(o['proxy']['points'] @ o['place'][:3, :3].T + o['place'][:3, 3], o['proxy']['faces'])
                         for o in objects]
        groups = fixed_group + [section_fill.GROUP[o['band']] for o in objects]
        _, ids = render(scene, arr, items, w, h)
        is_tree = np.isin(ids, [k + 1 for k, g in enumerate(groups) if g == 'tree'])
        cover = {'tree': float(is_tree.mean()), 'tree_top': float(is_tree[: h // 3].mean()),
                 'shrub': float(np.isin(ids, [k + 1 for k, g in enumerate(groups) if g == 'shrub']).mean())}
        off = {g: cover[g] / target[g] for g in cover if target.get(g, 0) > 0.01}
        print(f"round {rnd}: {len(objects)} plants, cover {({g: round(v, 3) for g, v in cover.items()})} for "
              f"{target}, look {({g: round(v, 2) for g, v in look.items()})}", flush=True)
        if all(abs(x - 1) <= tolerance for x in off.values()):
            break
        if 'tree' in off:
            look['back'] = float(np.clip(look['back'] * np.clip(off['tree'] ** -0.5, 0.85, 1.25), 1.0, 2.2))
        if 'tree_top' in off and abs(off['tree_top'] - 1) > tolerance:
            look['tilt'] = float(np.clip(look['tilt'] + (3.0 if off['tree_top'] < 1 else -3.0), 0.0, 12.0))
        if 'shrub' in off:
            look['shrub'] = float(np.clip(look['shrub'] / max(off['shrub'], 1e-3), 0.3, 3.0))
    objects = off_photo_plants(args, scene, arr, fixed, objects)
    scene['pieces'] += strips
    out = []
    for k, o in enumerate(objects):
        i = add_asset(scene, arr, o['url'], o['place'], o['proxy'])
        out.append({'index': i, 'name': f"{900 + k:03d} {o['band']}", 'vegetation': o['name'], 'filled': True})
    rows = {r: sum(o['row'] == r for o in objects) for r in ('front', 'back', 'hedge', 'shrub')}
    print(f"bands: {len(objects)} plants {rows} ({', '.join(sorted({o['name'] for o in objects}))}), "
          f"{len(strips)} ground strips, look {look}")
    return out


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
    ap.add_argument('--vegetation', default=None, help="flow_vegetation.py's picks: NVIDIA trees and shrubs "
                    'placed where the photo has them (with --veg-dir, veg_catalog.py --dir)')
    ap.add_argument('--veg-dir', default=None)
    ap.add_argument('--section', default=None, help="flow_section.py's street bands, filled over the stretch the "
                    'photo sees')
    args = ap.parse_args()
    graph_path = os.path.join(args.env, 'scene_graph.json')
    if os.path.exists(graph_path):
        meshes = env_meshes(args.env, args.max_faces)
        graph = json.load(open(graph_path))
    else:  # every surroundings mask was vegetation: SAM 3D made nothing
        meshes, graph = {}, {'nodes': []}
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
    if scene.get('floor_plane'):  # piece_build's floor, which on a slope is not level
        def floor(X):
            return floor_height(np.atleast_2d(X), up, scene['floor_plane'])
    else:  # older scenes: a level floor, the first piece
        level = float(np.mean(np.asarray(scene['pieces'][0]['points']) @ up))

        def floor(X):
            return np.full(len(np.atleast_2d(X)), level)
    tex = photo_texture(meshes, args.scene) if args.photo_texture and meshes else {}
    if tex:
        tex_path = os.path.abspath(os.path.join(args.out, 'isaac', 'tex_photo.png'))
        shutil.copy(os.path.join(args.scene, 'background.png'), tex_path)
    high = {}  # buildings standing over the street: env name -> object index
    for k, (name, m) in enumerate(meshes.items()):
        i = first + k
        pts = trimesh.transform_points(m.vertices, GL2CV) * s
        cols = m.visual.vertex_colors[:, :3] / 255
        faces = m.faces
        if name in tex:
            m, uv, photo_faces, cols = tex[name]
            pts = trimesh.transform_points(m.vertices, GL2CV) * s
            faces = m.faces
        if raised(name, pts, up, floor) is not None:  # on a lot (lots()): neither moved down nor skirted
            high[name] = i
        elif args.snap_ground:  # catalog entry 1: a base hidden in the photo comes out floating or sunk
            pts = pts - np.percentile(pts @ up - floor(pts), 0.5) * up
        elif args.fill_ground:
            pts, faces, cols = skirt(pts, faces, cols, up, floor)
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
    for e in lots(args, scene, arr, up, floor, high):
        for v in scene['views']:
            v['objects'].append([e['index'], np.eye(4).tolist()])
        scene['env'].append(e)
    if args.vegetation and os.path.exists(args.vegetation):
        for e in vegetation(args, scene, arr, up, floor):
            for v in scene['views']:
                v['objects'].append([e['index'], np.eye(4).tolist()])
            scene['env'].append(e)
    if args.section and os.path.exists(args.section) and args.veg_dir:
        for e in bands(args, scene, arr, up, floor):
            for v in scene['views']:
                v['objects'].append([e['index'], np.eye(4).tolist()])
            scene['env'].append(e)
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
