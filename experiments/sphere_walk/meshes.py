"""Shared by the walk test's layer arms: an equirectangular distance map as a textured mesh for render_walk.py,
and LaMa."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sim2chip.photo2scene.sphere import render_plan

LAMA = "/media/lenena-iker/Crucial/tools/lama/big-lama.pt"
TEAR = 0.08  # relative depth jump across a quad that tears it (a splat edge)


def lama_inpaint(model, image, mask):
    """LaMa on the whole image (as bg_prepare.inpaint): sides padded to a multiple of 8."""
    img = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)[None].float().cuda() / 255
    msk = torch.from_numpy((mask > 0).astype(np.float32))[None, None].cuda()
    h, w = img.shape[2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    img, msk = F.pad(img, (0, pw, 0, ph), mode="reflect"), F.pad(msk, (0, pw, 0, ph))
    with torch.no_grad():
        out = model(img, msk)[0, :, :h, :w]
    return Image.fromarray((out.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


def grid_mesh(dist, keep, R, h, tear=TEAR):
    """A mesh with one vertex per pixel of an equirectangular distance map, in Isaac's world; quads kept where all
    four corners are kept and their depths within `tear` of each other. UVs into the same panorama."""
    H, W = dist.shape
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d_pano = np.stack([np.cos(LA) * np.sin(LO), -np.sin(LA), np.cos(LA) * np.cos(LO)], -1)
    pts = (d_pano @ R.T) * dist[..., None]  # d_level = R d_pano
    uv = np.stack([np.broadcast_to((np.arange(W) + 0.5) / W, (H, W)),
                   np.broadcast_to(1 - (np.arange(H) + 0.5)[:, None] / H, (H, W))], -1)
    # wrap the seam: one more column, the first one again with u + 1
    pts = np.concatenate([pts, pts[:, :1]], 1)
    uv = np.concatenate([uv, uv[:, :1] + np.array([1.0, 0])], 1)
    dist = np.concatenate([dist, dist[:, :1]], 1)
    keep = np.concatenate([keep, keep[:, :1]], 1)
    W1 = W + 1
    idx = np.arange(H * W1).reshape(H, W1)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    q = np.stack([dist[:-1, :-1], dist[:-1, 1:], dist[1:, 1:], dist[1:, :-1]], -1)
    ok = keep[:-1, :-1] & keep[:-1, 1:] & keep[1:, 1:] & keep[1:, :-1]
    ok &= (q.max(-1) - q.min(-1)) < tear * q.min(-1)
    faces = np.concatenate([np.stack([a, b, c], -1)[ok], np.stack([a, c, e], -1)[ok]])
    used = np.unique(faces)
    remap = np.full(H * W1, -1)
    remap[used] = np.arange(len(used))
    world = render_plan.to_world(pts.reshape(-1, 3)[used], h).astype(np.float32)
    fuv = uv.reshape(-1, 2)[used][remap[faces]].reshape(-1, 2).astype(np.float32)
    return world, remap[faces].astype(np.int32), fuv
