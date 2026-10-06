"""Light, textured GLB from a dense vertex-coloured mesh, on the CPU only (no nvdiffrast, no GPU rasteriser).

SAM 3D Objects' own texture baking goes through nvdiffrast (non-commercial licence). Its licence-clean output
is a dense mesh (70k-1.5M faces) with vertex colours. This turns it into what Isaac Sim wants:

1. Decimate to --faces with pyvista/VTK (BSD / MIT).
2. Unwrap UVs with xatlas (MIT).
3. Rasterise the UV triangles with PIL (triangle id per texel), get each texel's 3D point by barycentric
   interpolation and colour it from the dense mesh: inverse-distance mix of its 4 nearest dense vertices
   (KD-tree). The dense vertices are far closer together than the texels, so this matches the exact closest
   point on the surface by eye at a fraction of the time (trimesh's closest_point took ~20 s a model).
4. Pad the texture past the UV islands (nearest filled texel) so mip-mapping does not bleed the background.

Usage:
    python bake_vertex_colors.py in.glb out.glb [--faces 20000] [--texture 1024]
    python bake_vertex_colors.py --root ~/work/eyeforge/data/img2mesh_test/out_sam3d   # every */model.glb
"""
import argparse
import os
import time

import numpy as np
import pyvista as pv
import trimesh
import xatlas
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree


def decimate(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= faces:
        return trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    poly = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64))
    poly = poly.decimate(1 - faces / len(mesh.faces))
    return trimesh.Trimesh(np.asarray(poly.points), poly.faces.reshape(-1, 4)[:, 1:], process=False)


def rasterise_uv(uvs: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """Triangle id (+1) per texel, 0 where no triangle covers it. Row 0 is v = 1 (image convention)."""
    img = Image.new('I', (size, size), 0)
    draw = ImageDraw.Draw(img)
    pix = np.stack([uvs[:, 0] * size - 0.5, (1 - uvs[:, 1]) * size - 0.5], axis=1)
    for i, f in enumerate(faces):
        draw.polygon([tuple(p) for p in pix[f]], fill=i + 1, outline=i + 1)
    return np.asarray(img, dtype=np.int64)


def barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of 2D points p in triangles abc (all N x 2), clipped into the triangle."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.clip(np.stack([1 - v - w, v, w], axis=1), 0, None)  # outline texels fall just outside
    return bc / bc.sum(1, keepdims=True)


def bake(dense: trimesh.Trimesh, faces: int = 20000, size: int = 1024) -> trimesh.Trimesh:
    colors = np.asarray(dense.visual.vertex_colors, dtype=np.float32)[:, :3]
    low = decimate(dense, faces)
    vmap, ind, uvs = xatlas.parametrize(low.vertices, low.faces)
    verts = low.vertices[vmap]

    tri = rasterise_uv(uvs, ind, size)
    ys, xs = np.nonzero(tri)
    fid = tri[ys, xs] - 1
    p = np.stack([(xs + 0.5) / size, 1 - (ys + 0.5) / size], axis=1)
    f = ind[fid]
    bc = barycentric(p, uvs[f[:, 0]], uvs[f[:, 1]], uvs[f[:, 2]])
    points = (verts[f] * bc[:, :, None]).sum(1)

    # Colour from the nearest dense vertices, weighted by inverse distance.
    dist, nn = cKDTree(dense.vertices).query(points, k=4, workers=-1)
    wts = 1 / np.maximum(dist, 1e-9)
    rgb = (colors[nn] * wts[:, :, None]).sum(1) / wts.sum(1, keepdims=True)

    tex = np.zeros((size, size, 3), np.float32)
    tex[ys, xs] = rgb
    # Pad: every empty texel takes its nearest filled one.
    _, (iy, ix) = ndimage.distance_transform_edt(tri == 0, return_indices=True)
    tex = tex[iy, ix]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8)),
        metallicFactor=0.0, roughnessFactor=1.0)
    return trimesh.Trimesh(verts, ind, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material), process=False)


def convert(src: str, dst: str, faces: int, size: int) -> dict:
    t0 = time.time()
    dense = trimesh.load(src, force='mesh', process=False)
    out = bake(dense, faces, size)
    out.export(dst)
    return {'src': src, 'faces_in': len(dense.faces), 'faces_out': len(out.faces), 'seconds': round(time.time() - t0, 1),
            'mb': round(os.path.getsize(dst) / 2**20, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', nargs='?')
    ap.add_argument('dst', nargs='?')
    ap.add_argument('--root', help='bake every <root>/*/model.glb to model_baked.glb')
    ap.add_argument('--faces', type=int, default=20000)
    ap.add_argument('--texture', type=int, default=1024)
    args = ap.parse_args()
    if args.root:
        for d in sorted(os.listdir(args.root)):
            src = os.path.join(args.root, d, 'model.glb')
            if os.path.exists(src):
                print(convert(src, os.path.join(args.root, d, 'model_baked.glb'), args.faces, args.texture), flush=True)
    else:
        print(convert(args.src, args.dst, args.faces, args.texture))


if __name__ == '__main__':
    main()
