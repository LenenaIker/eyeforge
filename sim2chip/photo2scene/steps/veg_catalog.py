"""Trees and shrubs for photo2scene from NVIDIA's public asset bucket (Assets/Vegetation; the user's choice over
SAM 3D meshes, which came out as green blocks, and over cards cut from the photo, 2026-10-13). Lists the catalog
with each asset's thumbnail (the bucket's own .thumbs) and, for the assets a scene uses, a light proxy: the USD's
meshes, point instancers expanded, simplified, in metres with Z up and coloured from its thumbnail. The proxies
stand in for the trees in the pyrender steps (gates, labels, height maps); Isaac renders the USD itself by its URL.
Run in the mesh env (pxr, open3d):

    python sim2chip/photo2scene/steps/veg_catalog.py --dir <cache>/vegetation [--proxies Black_Oak,Boxwood]

Writes <dir>/catalog.json ([{"name", "group", "url", "thumb"}]), <dir>/thumbs/<name>.png and
<dir>/proxy/<name>.npz (points, faces, colors, height_m, to_proxy: the USD's units and axes to the
proxy's).
"""
import argparse
import json
import os
import re
import time
import urllib.request

import numpy as np

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
GROUPS = {"Trees": "tree", "Shrub": "shrub"}
MAX_FACES = 20000  # as env_add.py simplifies the SAM 3D meshes


def fetch(url, path, tries=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(tries):
        try:
            tmp = path + '.part'
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            return path
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def keys(prefix):
    xml = urllib.request.urlopen(f"{BUCKET}/?prefix={prefix}&max-keys=1000", timeout=60).read().decode()
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def catalog(d):
    path = os.path.join(d, 'catalog.json')
    if os.path.exists(path):
        return json.load(open(path))
    out = []
    for folder, group in GROUPS.items():
        thumbs = {os.path.basename(k).split('.usd')[0]: k for k in keys(f"Assets/Vegetation/{folder}/.thumbs/")
                  if k.endswith('.png')}
        for k in keys(f"Assets/Vegetation/{folder}/"):
            if '/' in k[len(f"Assets/Vegetation/{folder}/"):] or not k.endswith('.usd'):
                continue  # materials/ and .thumbs/
            name = os.path.basename(k)[:-4]
            if name not in thumbs:
                continue
            fetch(f"{BUCKET}/{thumbs[name]}", os.path.join(d, 'thumbs', f'{name}.png'))
            out.append({'name': name, 'group': group, 'url': f"{BUCKET}/{k}",
                        'thumb': os.path.join('thumbs', f'{name}.png')})
    json.dump(out, open(path, 'w'), indent=1)
    return out


def thumb_colours(path):
    """The canopy's and the trunk's colour: mean of the thumbnail's opaque pixels in its upper and lower fifth."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert('RGBA')).astype(np.float32) / 255
    rows = np.nonzero(a[..., 3].max(1) > 0.5)[0]
    if not len(rows):
        return np.array([0.25, 0.35, 0.18]), np.array([0.3, 0.24, 0.18])
    top, bottom = rows.min(), rows.max()
    span = bottom - top + 1
    out = []
    for y0, y1 in ((top, top + int(0.8 * span)), (bottom - max(1, span // 5), bottom + 1)):
        px = a[y0:y1][a[y0:y1, ..., 3] > 0.5][:, :3]
        out.append(px.mean(0) if len(px) else np.array([0.3, 0.3, 0.3]))
    return out[0], out[1]


def proxy(d, item):
    """The asset's geometry in metres, Z up, simplified to MAX_FACES, coloured canopy above a fifth of its height
    and trunk below."""
    import open3d as o3d
    from pxr import Usd, UsdGeom
    out = os.path.join(d, 'proxy', f"{item['name']}.npz")
    if os.path.exists(out):
        return out
    usd = fetch(item['url'], os.path.join(d, 'usd', f"{item['name']}.usd"))
    st = Usd.Stage.Open(usd)
    mpu = UsdGeom.GetStageMetersPerUnit(st) or 0.01
    zup = UsdGeom.GetStageUpAxis(st) == 'Z'
    xc = UsdGeom.XformCache()
    pts, faces = [], []

    def add(mesh_prim, to_world):
        """A mesh's triangles (faces fanned) with its points through the 4x4 to_world."""
        m = UsdGeom.Mesh(mesh_prim)
        p = np.asarray(m.GetPointsAttr().Get() or [], float)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get() or [], int)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get() or [], int)
        if not len(p) or not len(counts):
            return
        tris, k = [], 0
        for c in counts:
            tris += [[idx[k], idx[k + j], idx[k + j + 1]] for j in range(1, c - 1)]
            k += c
        faces.append(np.asarray(tris) + sum(len(x) for x in pts))
        pts.append((np.c_[p, np.ones(len(p))] @ to_world.T)[:, :3])

    def world(prim):  # USD matrices are row-major with row vectors: transposed to act on column vectors
        return np.asarray(xc.GetLocalToWorldTransform(prim)).T

    for prim in st.Traverse():
        if prim.IsA(UsdGeom.PointInstancer):
            inst = UsdGeom.PointInstancer(prim)
            protos = inst.GetPrototypesRel().GetTargets()
            ids = np.asarray(inst.GetProtoIndicesAttr().Get() or [], int)
            tfs = inst.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
            for j, tf in zip(ids, tfs):
                root = st.GetPrimAtPath(protos[j])
                for sub in Usd.PrimRange(root):
                    if sub.IsA(UsdGeom.Mesh):
                        # inside the prototype, then the instance's transform, then the instancer's place
                        inner = np.linalg.inv(world(root)) @ world(sub)
                        add(sub, world(prim) @ np.asarray(tf).T @ inner)
        elif prim.IsA(UsdGeom.Mesh) and not any(a.IsA(UsdGeom.PointInstancer) for a in _ancestors(prim)):
            add(prim, world(prim))
    # to_proxy: the asset's own units and axes to the proxy's (metres, Z up, trunk base at the origin), so
    # Isaac can pose the USD exactly where the proxy stands.
    A = np.eye(4) * mpu
    A[3, 3] = 1
    if not zup:  # Y up to Z up: (x, y, z) -> (x, -z, y)
        A = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]]) @ A
    P = np.concatenate(pts) @ A[:3, :3].T
    F = np.concatenate(faces)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(P), o3d.utility.Vector3iVector(F))
    if len(F) > MAX_FACES:
        mesh = mesh.simplify_quadric_decimation(MAX_FACES)
    P, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    shift = np.zeros(3)
    shift[:2] = -np.median(P[P[:, 2] < np.percentile(P[:, 2], 5), :2], 0)  # trunk base at the origin
    shift[2] = -P[:, 2].min()
    P = P + shift
    A[:3, 3] += shift
    h = float(P[:, 2].max())
    canopy, trunk = thumb_colours(os.path.join(d, item['thumb']))
    cols = np.where((P[:, 2] > 0.2 * h)[:, None], canopy, trunk)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, points=P.astype(np.float32), faces=F.astype(np.int32), colors=cols.astype(np.float32),
                        height_m=h, to_proxy=A)
    return out


def _ancestors(prim):
    p = prim.GetParent()
    while p and p.GetPath().pathString != '/':
        yield p
        p = p.GetParent()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--proxies', default='', help='comma-separated asset names to make proxies for')
    args = ap.parse_args()
    cat = catalog(args.dir)
    by = {c['name']: c for c in cat}
    for name in [n for n in args.proxies.split(',') if n]:
        path = proxy(args.dir, by[name])
        z = np.load(path)
        print(f"{name}: {len(z['faces'])} faces, {float(z['height_m']):.1f} m", flush=True)
    print(json.dumps({'assets': len(cat), 'trees': sum(c['group'] == 'tree' for c in cat),
                      'shrubs': sum(c['group'] == 'shrub' for c in cat)}))


if __name__ == '__main__':
    main()
