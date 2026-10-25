"""Inside the WorldStereo job: a 3D cache that grows leg by leg, as WorldStereo's paper does with WorldMirror
("we incrementally reconstruct point clouds for each generated video"). The released code never does it: every
leg's guide (render.mp4, render_mask.mp4) is fixed before the run, from the panorama's point cloud alone, so two
legs that look at the same unseen place each invent it on their own (tests C1/C2 of
docs/pruebas_reconstruccion_worldstereo.md).

A scene with cache_init.npz (ws_cache.py stage: the panorama's guide cloud) gets:
- before each leg: its guide rendered again from the cache with upstream's renderer and settings, so what earlier
  legs invented is no longer a hole; frame 0 stays as staged (the start image). The guide used is copied next to
  the leg's result (guide_render.mp4, guide_mask.mp4) with the hole shares before and after (guide.json).
- after each leg: MapAnything (Apache 2.0 weights) on its frames with the cameras we asked for; every FRAME_STEP-th
  frame's pixels that were holes in its guide join the cache (depth under MAX_DEPTH, MapAnything's mask). A leg that
  moves less than ANCHOR_PATH_M has no parallax to fix the scale: its depth is scaled to the cache's on the pixels
  the guide covered (ws_depth.py anchor's rule).
A scene with order.txt runs its legs in that order (C2).
A scene with cache_align.txt (U1, ws_upstream.py) places a leg's points as the paper does instead: MapAnything's own
cameras, then one similarity (Umeyama, trimmed) that lays the leg's points on the cache where its guide covered the
frame. C1 used the cameras we asked for and fixed no scale on walks, while the videos drift 7-118 cm from them and
MapAnything's scale strays 0-40 % from the panorama's (docs/pruebas_reconstruccion_worldstereo.md).

ws_cache.py run ships this file into the job and calls install()."""
import json
import os

import numpy as np
import torch

CKPT = "facebook/map-anything-apache"
MAX_DEPTH = 40.0
FRAME_STEP = 2
UPSAMPLE = 2           # MapAnything's depth (about 518 px wide) is unprojected at twice its resolution, colours from
                       # the frame itself: at its own density a later leg that comes closer sees rows of points
VOXEL_M = 0.015        # new points closer than this to each other are merged (the frames of a leg overlap)
ANCHOR_PATH_M = 0.5
RADIUS, PER_PIXEL = 0.008, 20  # upstream's guide rendering (ws_test.render)
ALIGN_MAX_DEPTH = 20.0  # metres: only cache points this near anchor the similarity (no sky, no contracted far)
ALIGN_PAIRS, ALIGN_KEEP, ALIGN_ROUNDS = 60000, 0.8, 4  # sampled pixel pairs; each round refits on the best 80 %
ALIGN_MIN_PAIRS = 2000
ALIGN_SCALE = (0.6, 1.6)  # a fit outside this is taken as failed: the leg falls back to C1's placement
ALIGN_SKIP_GAP = 1.0      # metres: ... unless C1's placement is this far off too, then the leg adds no points

_cache = {}   # scene path -> [points (N, 3), colours (N, 3) in -1..1], CPU float32
_model = []


def enabled(scene: str) -> bool:
    return os.path.exists(f"{scene}/cache_init.npz")


def cloud(scene: str) -> list:
    if scene not in _cache:
        z = np.load(f"{scene}/cache_init.npz")
        _cache[scene] = [torch.from_numpy(z["points"].astype(np.float32)),
                         torch.from_numpy(z["colors"].astype(np.float32))]
        print(f"cache {scene}: {len(_cache[scene][0])} panorama points", flush=True)
    return _cache[scene]


def render(scene: str, w2cs: torch.Tensor, Ks: torch.Tensor, h: int, w: int, depth: bool = False):
    """Upstream's point renderer on the cache: RGB uint8 (F, h, w, 3) and holes (F, h, w) bool, or z-depth."""
    from src.pointcloud import point_rendering
    P, C = cloud(scene)
    out_a, out_b = [], []
    for i in range(0, len(w2cs), 4):
        r, m = point_rendering(K=Ks[i:i + 4].float(), w2cs=w2cs[i:i + 4].float(), points=P, colors=C,
                               device="cuda", h=h, w=w, render_radius=RADIUS, points_per_pixel=PER_PIXEL,
                               background_color=[0, 0, 0], return_depth=depth)
        if depth:
            out_b.append(m[:, 0].cpu())  # -1 where no point
        else:
            out_a.append(((r.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy())
            out_b.append(m[:, 0].cpu().numpy() > 0.5)
    return torch.cat(out_b) if depth else (np.concatenate(out_a), np.concatenate(out_b))


def read_video(path: str) -> np.ndarray:
    from src.general_utils import load_video
    return np.stack([np.asarray(f) for f in load_video(path)])


def write_video(path: str, frames: np.ndarray):
    import imageio
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, -1)
    imageio.mimwrite(path, frames, fps=16, codec="libx264", quality=9, macro_block_size=1)


def reguide(scene: str, output: str, view_id: str, traj_id: str):
    """The leg's guide from the cache, written over its staged render.mp4 / render_mask.mp4 (frame 0 kept)."""
    leg = f"{scene}/{view_id}/{traj_id}"
    cam = json.load(open(f"{leg}/camera.json"))
    w2cs, Ks = torch.tensor(cam["extrinsic"]), torch.tensor(cam["intrinsic"])
    old_rgb, old_mask = read_video(f"{leg}/render.mp4"), read_video(f"{leg}/render_mask.mp4")[..., 0] > 127
    rgb, holes = render(scene, w2cs, Ks, cam["height"], cam["width"])
    rgb[0], holes[0] = old_rgb[0], old_mask[0]
    write_video(f"{leg}/render.mp4", rgb)
    write_video(f"{leg}/render_mask.mp4", holes.astype(np.uint8) * 255)
    dst = f"{output}/{view_id}/{traj_id}"
    os.makedirs(dst, exist_ok=True)
    write_video(f"{dst}/guide_render.mp4", rgb)
    write_video(f"{dst}/guide_mask.mp4", holes.astype(np.uint8) * 255)
    info = {"holes_staged": round(float(old_mask[1:].mean()), 4), "holes_cache": round(float(holes[1:].mean()), 4),
            "cache_points": int(len(cloud(scene)[0]))}
    json.dump(info, open(f"{dst}/guide.json", "w"))
    print(f"{view_id}/{traj_id}: holes {info['holes_staged']:.3f} staged -> {info['holes_cache']:.3f} from the cache "
          f"({info['cache_points']} points)", flush=True)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) with dst ~ s R src + t (ws_colmap.py's)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, dg, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.eye(3)
    sgn[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ sgn @ vt
    s = np.trace(np.diag(dg) @ sgn) / a.var(0).sum()
    return s, r, md - s * r @ ms


def unproject(z: np.ndarray, K: np.ndarray, c2w: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    cam = np.stack([(xx + 0.5 - K[0, 2]) / K[0, 0] * z, (yy + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def align(d, seen, Ko, poses, zc, c2w_asked):
    """The similarity from MapAnything's world (its depth d with its cameras poses) onto the cache, from the pixels
    the leg's guide covered: there the video copies the guide, so the cache's point rendered on the asked camera is
    the same surface. Trimmed: each round refits on the ALIGN_KEEP nearest pairs (the video does not copy exactly).
    None when too few pairs or the scale is implausible."""
    ok = seen & (zc > 0) & (zc < ALIGN_MAX_DEPTH)
    f, yy, xx = np.nonzero(ok)
    if len(f) < ALIGN_MIN_PAIRS:
        print(f"align: {len(f)} pairs, too few", flush=True)
        return None
    pick = np.random.default_rng(0).choice(len(f), min(len(f), ALIGN_PAIRS), replace=False)
    f, yy, xx = f[pick], yy[pick], xx[pick]
    src = np.concatenate([unproject(d[i, yy[f == i], xx[f == i]], Ko[i], poses[i], yy[f == i], xx[f == i])
                          for i in np.unique(f)])
    dst = np.concatenate([unproject(zc[i, yy[f == i], xx[f == i]], Ko[i], c2w_asked[i], yy[f == i], xx[f == i])
                          for i in np.unique(f)])
    # C1's placement of the same pixels (asked cameras, MapAnything's depth unscaled), for the log and the check;
    # MapAnything's own world is not ours (it starts at its first camera), so src - dst says nothing before the fit
    c1 = np.concatenate([unproject(d[i, yy[f == i], xx[f == i]], Ko[i], c2w_asked[i], yy[f == i], xx[f == i])
                         for i in np.unique(f)])
    before = float(np.median(np.linalg.norm(c1 - dst, axis=1)))
    keep = np.ones(len(src), bool)
    for _ in range(ALIGN_ROUNDS):
        s, r, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm(src @ r.T * s + t - dst, axis=1)
        keep = res <= np.quantile(res, ALIGN_KEEP)
    after = float(np.median(res))
    print(f"align: {len(src)} pairs, scale {s:.3f}, median gap to the cache {after:.3f} m (C1 {before:.3f})", flush=True)
    if not ALIGN_SCALE[0] <= s <= ALIGN_SCALE[1] or after > before:
        if before > ALIGN_SKIP_GAP:
            print("align: rejected and C1's placement is off too: no points from this leg", flush=True)
            return "skip"
        print("align: rejected, C1's placement instead", flush=True)
        return None
    return s, r, t


def sim_pose(sim, c2w: np.ndarray) -> np.ndarray:
    """A camera-to-world pose carried by the similarity; its rotation block holds the scale."""
    s, r, t = sim
    out = np.eye(4, dtype=np.float32)
    out[:3, :3], out[:3, 3] = s * r @ c2w[:3, :3], s * r @ c2w[:3, 3] + t
    return out


def grow(scene: str, gen_frames: list, w2cs: torch.Tensor, Ks: torch.Tensor, view_id: str, traj_id: str):
    """MapAnything depth of a generated leg; its hole pixels join the cache."""
    import cv2
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    if not _model:
        _model.append(MapAnything.from_pretrained(CKPT).cuda().eval())
    leg = f"{scene}/{view_id}/{traj_id}"
    frames = np.stack([np.asarray(f) for f in gen_frames])
    holes = read_video(f"{leg}/render_mask.mp4")[..., 0] > 127
    w2c = w2cs.cpu().numpy().astype(np.float32)
    c2w = np.linalg.inv(w2c)
    K = Ks[0].cpu().numpy().astype(np.float32)
    views = [{"img": torch.from_numpy(f), "intrinsics": torch.from_numpy(K), "camera_poses": torch.from_numpy(m),
              "is_metric_scale": torch.tensor([True])} for f, m in zip(frames, c2w)]
    with torch.no_grad():
        pred = _model[0].infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True,
                               amp_dtype="bf16", apply_mask=True, mask_edges=True)
    d = np.stack([p["depth_z"][0, ..., 0].float().cpu().numpy() for p in pred])
    ok = np.stack([p["mask"][0, ..., 0].cpu().numpy() for p in pred]).astype(bool) & (d > 0) & (d < MAX_DEPTH)
    Ko = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in pred])
    h, w = d.shape[1:]
    hole = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in holes]) > 0
    scale, path = 1.0, float(np.linalg.norm(c2w[:, :3, 3] - c2w[0, :3, 3], axis=1).max())
    sim = None
    if os.path.exists(f"{scene}/cache_align.txt"):
        zc = render(scene, w2cs.cpu(), torch.as_tensor(Ko), h, w, depth=True).numpy()
        poses = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in pred])
        sim = align(d, ok & ~hole, Ko, poses, zc, c2w)
        if sim == "skip":
            return
        if sim is not None:
            c2w = np.stack([sim_pose(sim, m) for m in poses])  # MapAnything's cameras laid on the cache
    if sim is None and path < ANCHOR_PATH_M:
        zc = render(scene, w2cs.cpu(), torch.as_tensor(Ko), h, w, depth=True).numpy()
        seen = ok & ~hole & (zc > 0)
        if seen.sum() > 2000:
            scale = float(np.median(zc[seen] / d[seen]))
    P, C = [], []
    fh, fw = frames.shape[1:3]
    for f in range(1, len(d), FRAME_STEP):
        hu, wu = h * UPSAMPLE, w * UPSAMPLE
        du_ = cv2.resize(d[f] * scale, (wu, hu), interpolation=cv2.INTER_NEAREST)
        keep = cv2.resize((ok[f] & hole[f]).astype(np.uint8), (wu, hu), interpolation=cv2.INTER_NEAREST) > 0
        yy, xx = np.nonzero(keep)
        z = du_[yy, xx]
        k = Ko[f]
        cam = np.stack([((xx + 0.5) / UPSAMPLE - k[0, 2]) / k[0, 0] * z,
                        ((yy + 0.5) / UPSAMPLE - k[1, 2]) / k[1, 1] * z, z], -1)
        u = np.clip((cam[:, 0] / z * K[0, 0] + K[0, 2]).astype(int), 0, fw - 1)  # the frame's own pixel
        v = np.clip((cam[:, 1] / z * K[1, 1] + K[1, 2]).astype(int), 0, fh - 1)
        P.append(cam @ c2w[f, :3, :3].T + c2w[f, :3, 3])
        C.append(frames[f][v, u] / 127.5 - 1)
    P, C = np.concatenate(P).astype(np.float32), np.concatenate(C).astype(np.float32)
    _, first = np.unique(np.floor(P / VOXEL_M).astype(np.int64), axis=0, return_index=True)
    P, C = P[first], C[first]
    c = cloud(scene)
    c[0], c[1] = torch.cat([c[0], torch.from_numpy(P)]), torch.cat([c[1], torch.from_numpy(C)])
    how = f"aligned, scale {sim[0]:.2f}" if sim is not None else f"scale {scale:.2f}"
    print(f"{view_id}/{traj_id}: +{len(P)} points from its holes ({how}, path {path:.1f} m), "
          f"cache {len(c[0])}", flush=True)


def install(du, rwm):
    """Wrap run_ws.py's leg order and loader and the memory bank's update."""
    sort, load = du.sort_trajs, du.load_mutli_traj_dataset

    def sort_trajs(scene_path):
        legs = sort(scene_path)
        f = f"{scene_path}/order.txt"
        if os.path.exists(f):
            want = [ln.strip() for ln in open(f) if ln.strip()]
            legs = sorted(legs, key=lambda p: want.index("/".join(p.split("/")[-3:-1])))
            print("order.txt:", ["/".join(p.split("/")[-3:-1]) for p in legs], flush=True)
        return legs

    def load_mutli_traj_dataset(cfg, input_path, output_path, view_id, traj_id, *a, **k):
        if enabled(input_path):
            reguide(input_path, output_path, view_id, traj_id)
        return load(cfg, input_path, output_path, view_id, traj_id, *a, **k)

    update = rwm.SimpleMemoryBank.update_memory

    def update_memory(self, gen_frames, tar_w2cs_full, tar_Ks_full, view_id=None, traj_id=None):
        if enabled(self.root_path):
            grow(self.root_path, gen_frames, tar_w2cs_full, tar_Ks_full, view_id, traj_id)
        return update(self, gen_frames, tar_w2cs_full, tar_Ks_full, view_id, traj_id)

    du.sort_trajs, du.load_mutli_traj_dataset = sort_trajs, load_mutli_traj_dataset
    rwm.SimpleMemoryBank.update_memory = update_memory
