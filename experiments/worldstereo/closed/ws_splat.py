"""One 3D Gaussian splat (gsplat) of a WorldStereo run (ws_test.py): the panorama's 27 memory cuts plus every frame of
every generated leg, so the scene can be walked in Isaac Sim like the FlashWorld and photosphere splats.

The panorama is the truth: its cuts weigh PANO_WEIGHT with our MoGe-2 depth of the panorama. A generated frame
teaches fully where the guide render had a hole (what WorldStereo invented) and weighs SEEN_WEIGHT where the
panorama's point cloud already covered it, with that point cloud's depth; in the holes, MoGe-2's depth of the
frame, scaled to the point cloud's on the seen pixels, only keeps floaters away. Sky pixels push the splat clear
(Isaac's dome shows the sky). Each generated frame gets its own colour correction. Frame k of a leg with
k % HOLDOUT == HOLDOUT // 2 is left out and scored.

    <eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_splat.py views --case photo21   # MoGe-2, depths
    <gsplat env>/bin/python experiments/worldstereo/ws_splat.py train --case photo21
    <gsplat env>/bin/python experiments/worldstereo/ws_splat.py train --case photo21 --arm G3   # a test arm
    <gsplat env>/bin/python experiments/worldstereo/ws_splat.py arms --case photo21             # arms side by side
    (cd <3dgrut> && source .venv/bin/activate && python <repo>/experiments/sphere_walk/gs_usd.py <splat.ply>)

Writes data/worldstereo_test/<case>/splat/: views.npz, splat.ply, splat.usdz, sheet.jpg, info.json. A test arm
(ARMS) writes splat.ply, info.json, sheet.jpg and novel.npz to splat/arms/<arm>/ and is scored like the others:
held-out PSNR (seen, invented), SSIM and LPIPS on the non-sky pixels, sharpness of the novel views."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

PANO_WEIGHT = 4.0
SEEN_WEIGHT = 0.3      # a generated pixel the panorama already covered: it must not compete with the photo
HOLE_WEIGHT = 1.0      # a generated pixel the guide had no point for: what WorldStereo brings
DEPTH_SEEN = 0.5       # relative depth error against the panorama's points
DEPTH_HOLE = 0.05      # against MoGe's scaled depth in the holes: only against floaters
SKY_ALPHA = 1.0
SSIM_SHARE = 0.2
AFFINE_REG = 1.0
PANO_SHARE = 0.25      # share of steps on a panorama cut
HOLDOUT = 7
MAX_DEPTH = 40.0       # metres; farther depth targets are left out (the guide squashed them, see ws_test.prep)
STEPS = 12000
SCENE_SCALE = 10.0
GROW_GRAD = 0.0006
INIT_PANO_PTS = 400000
INIT_HOLE_PTS = 4000   # MoGe points per generated frame in its holes (every 3rd frame)
INIT_OPACITY = 0.7
C0 = 0.28209479177387814
SEED = 0
# Robust fit (RobustNeRF's trimmed loss): a generated pixel whose error is above the running ROBUST_Q quantile of
# all generated pixels is something the other legs contradict (a car one leg invented, a smeared hedge) and stops
# teaching; 3x3 smoothing and 16x16 patches that are mostly inliers keep fine texture in.
ROBUST_START = 1500
ROBUST_Q = 0.8
ROBUST_PATCH = 16
ROBUST_PATCH_OK = 0.6
MAX_RATIO = 10.0       # anisotropy cap: needles along the walk's rays streak when seen from the side
RATIO_WEIGHT = 0.1
ARMS = {"G0": {}, "G3": {"robust": True}, "G1": {"steps": 30000, "grow": 0.0003},
        "G5": {"aniso": True}, "G2": {"steps": 30000, "mcmc": 4000000},
        "P2": {"steps": 30000, "mcmc": 4000000, "views": "views_p2.npz"},  # G2 on phase 1 + ws_nbv.py's phase 2
        "S1": {"steps": 30000, "mcmc": 4000000, "views": "views_s1.npz"},  # G2 on phase 1 + the 360s (spin_*)
        "S3": {"steps": 30000, "mcmc": 4000000, "views": "views_s3.npz"},  # the 360s guided by G2's splat (spg_*)
        # Why the spins' detail does not reach the splat (2026-10-22): S3 with one change each.
        "R1": {"steps": 30000, "mcmc": 4000000, "views": "views_s3.npz", "weights": (1.0, 1.0, 1.0)},  # seen, hole, pano
        "R2": {"steps": 30000, "mcmc": 4000000, "views": "views_s3.npz", "pose": True},
        "R3": {"steps": 30000, "mcmc": 4000000, "views": "views_s3.npz", "keep": 0.75},  # each clip's first 75 %
        # source per zone: pixels most other legs contradict stop teaching (ws_depth.py consensus, MapAnything)
        "R5": {"steps": 30000, "mcmc": 4000000, "views": "views_s3.npz", "consensus": "consensus_views_s3.npz"},
        "R6": {"steps": 30000, "mcmc": 4000000, "views": "views_s3.npz",  # R5, the nearer view rules
               "consensus": "consensus_views_s3_closest.npz"},
        "R7": {"steps": 30000, "mcmc": 4000000, "views": "views_s3.npz",  # R6, one generation per object
               "consensus": "select_views_s3.npz"},  # ws_objects.py
        # G2, each place taught only by the first leg that saw it (ws_depth.py consensus --rule first): invent once
        "A1": {"steps": 30000, "mcmc": 4000000, "consensus": "consensus_views_first.npz"},
        "NS": {"steps": 30000, "mcmc": 4000000, "views": "views_s.npz"},  # phase 1 + the 360s guided by G2 (spg_*)
        "NSF": {"steps": 30000, "mcmc": 4000000, "views": "views_s.npz",  # NS, each place from the first leg that saw it
                "consensus": "consensus_views_s_first.npz"},
        # G2 plus G2's renders at pseudo views towards the student cameras, cleaned by NVIDIA Fixer (ws_fixer.py)
        "F1": {"steps": 30000, "mcmc": 4000000, "views": "views_f1.npz"},
        "F3a": {"steps": 30000, "mcmc": 4000000, "views": "views_f3a.npz"},  # ws_fixer.py F3: masked, progressive
        "F3b": {"steps": 30000, "mcmc": 4000000, "views": "views_f3b.npz"},
        "F3c": {"steps": 30000, "mcmc": 4000000, "views": "views_f3c.npz"},
        # G2 plus pseudo views whose far zone (beyond the walks' reach + 1 m) was painted (ws_farfill.py):
        # FF holes by Qwen then Fixer, FX Fixer alone
        "FF": {"steps": 30000, "mcmc": 4000000, "views": "views_ff.npz"},
        "FX": {"steps": 30000, "mcmc": 4000000, "views": "views_fx.npz"},
        # F1 with GenFusion's video repair (colour and disparity, 16-frame clips) instead of Fixer (ws_genfusion.py)
        "GF1": {"steps": 30000, "mcmc": 4000000, "views": "views_gf1.npz"},
        # G2 as 2D Gaussians (gsplat's 2DGS, Apache 2.0): flat discs that lie on the surface, for a mesh from their
        # median depth (ws_mesh.py --arm D1); the official 2DGS, PGSR, GOF and MILo code is Inria non-commercial
        "D1": {"steps": 30000, "mcmc": 4000000, "twod": True}}
MCMC_REG = 0.01        # opacity and scale pull of the MCMC strategy (its paper's values)
# 2DGS regularisers (gsplat's simple_trainer_2dgs defaults): rendered normals agree with the depth's, and each ray's
# weight gathers at one depth (distortion)
NORMAL_W, NORMAL_START = 0.05, 7000
DIST_W, DIST_START = 0.01, 3000
# Pose refinement (gsplat's CameraOptModule: 3 translation + 6D rotation deltas per camera): a generated frame lands
# 2 to 15 px off the camera it was asked for, so each one gets its own correction; the panorama cuts stay fixed and
# hold the scene in place. A frame is drawn about 70 times in a run, hence a larger rate than gsplat's 1e-5 (meant
# for COLMAP poses), and it starts once the splat has a shape.
POSE_LR = 1e-3
POSE_REG = 1e-6        # weight decay
POSE_START = 1000
# codes per pixel
SKY, SEEN, HOLE, PANO = 0, 1, 2, 3
PSEUDO, NONE = 4, 5    # a Fixer-cleaned pseudo view's pixel (ws_fixer.py), and one that teaches nothing
PSEUDO_WEIGHT = 0.3    # like a seen pixel: it must not compete with the frames


def legs(case: str, skip: tuple = ()) -> list[str]:
    """Every generated leg, but those whose folder starts with one of skip (another test's legs)."""
    res = ws.DATA / case / "out" / "out" / case
    return sorted(f"{p.parts[-3]}/{p.parts[-2]}" for p in res.glob(f"*/traj*/{ws.MODEL}_result.mp4")
                  if not p.parts[-3].startswith(tuple(skip)))


def panorama_points(case: str):
    """The metric point cloud of the panorama (ws_test.prep's, without the far contraction) and its colours."""
    import cv2
    meta = json.loads((ws.DATA / case / "prep.json").read_text())
    full = np.asarray(Image.open(ws.PANOS / case / meta["pano"]).convert("RGB"))
    pano = cv2.resize(full, (ws.PANO_W, ws.PANO_W // 2), interpolation=cv2.INTER_AREA)
    z = np.load(ws.DATA / case / "distance.npz")
    dist, valid = z["distance"] * meta["scale"], z["mask"]
    keep = valid & ~ws.edge_mask(dist) & np.isfinite(dist) & (dist > 0)
    return ws.pano_dirs(*dist.shape)[keep] * dist[keep][:, None], pano[keep], full, dist, valid


def zbuffer(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Depth (camera z) of the nearest point per pixel, 0 where none lands; a 3x3 minimum closes the gaps between
    points near the camera."""
    from scipy import ndimage
    p = points @ m[:3, :3].T + m[:3, 3]
    p = p[p[:, 2] > 0.1]
    u = np.floor(p[:, 0] / p[:, 2] * ws.F + ws.W / 2).astype(int)
    v = np.floor(p[:, 1] / p[:, 2] * ws.F + ws.H / 2).astype(int)
    ok = (u >= 0) & (u < ws.W) & (v >= 0) & (v < ws.H)
    z = np.full(ws.H * ws.W, np.inf, np.float32)
    np.minimum.at(z, v[ok] * ws.W + u[ok], p[ok, 2])
    z = ndimage.minimum_filter(z.reshape(ws.H, ws.W), 3)
    return np.where(np.isfinite(z), z, 0).astype(np.float32)


def views(case: str, name: str = "views.npz", skip: tuple = ()):
    """views.npz: every training and held-out view (image, per-pixel code, depth target, world-to-camera)."""
    import cv2
    import imageio.v3 as iio
    import torch
    sys.path[:0] = [str(ws.TOOLS / "MoGe"), str(ws.TOOLS / "moge_deps")]
    from moge.model.v2 import MoGeModel
    t0 = time.time()
    pts, _, full, dist, valid = panorama_points(case)
    scene, res = ws.DATA / case / "scene", ws.DATA / case / "out" / "out" / case
    imgs, codes, depths, w2cs, names = [], [], [], [], []

    # Panorama cuts: the photo, with the panorama's distance turned into camera depth.
    j, i = np.meshgrid(np.arange(ws.W) + 0.5, np.arange(ws.H) + 0.5)
    ray_len = np.sqrt(((j - ws.W / 2) / ws.F) ** 2 + ((i - ws.H / 2) / ws.F) ** 2 + 1).astype(np.float32)
    dist_full = cv2.resize(np.where(valid, dist, 0).astype(np.float32), (full.shape[1], full.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    for k, m in enumerate(ws.bank_cameras()):
        d = crop_nearest(dist_full, m) / ray_len
        imgs.append(np.asarray(Image.open(scene / f"pano_bank/images/{k:04d}.png").convert("RGB")))
        codes.append(np.where(d > 0, PANO, SKY).astype(np.uint8))
        depths.append(np.where(d < MAX_DEPTH, d, 0).astype(np.float16))
        w2cs.append(m)
        names.append(f"pano/{k:04d}")

    model = MoGeModel.from_pretrained(ws.MOGE_CKPT).cuda().eval()
    fov = math.degrees(2 * math.atan(ws.W / 2 / ws.F))
    scales = {}
    for leg in legs(case, skip):
        frames = iio.imread(res / leg / f"{ws.MODEL}_result.mp4", plugin="pyav")
        holes = iio.imread(scene / leg / "render_mask.mp4", plugin="pyav")[..., 0] > 127
        cams = json.loads((scene / leg / "camera.json").read_text())["extrinsic"]
        leg_k = []
        for f, (img, hole, m) in enumerate(zip(frames, holes, cams)):
            m = np.array(m)
            x = torch.tensor(img / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None]
            with torch.no_grad():
                out = model.infer(x, fov_x=torch.tensor([fov], device="cuda"), apply_mask=True)
            mg = out["depth"][0].cpu().numpy()
            sky = ~out["mask"][0].cpu().numpy()
            mg = np.where(np.isfinite(mg) & ~sky, mg, 0)
            prox = zbuffer(pts, m)
            ok = ~hole & (prox > 0) & (mg > 0) & (prox < MAX_DEPTH)
            k = float(np.median(prox[ok] / mg[ok])) if ok.sum() > 500 else np.nan
            leg_k.append(k)
            imgs.append(img)
            codes.append(np.where(sky, SKY, np.where(hole, HOLE, SEEN)).astype(np.uint8))
            depths.append(np.stack([np.where(prox < MAX_DEPTH, prox, 0), mg], 0).astype(np.float16))
            w2cs.append(m)
            names.append(f"{leg}/{f:02d}")
        scales[leg] = leg_k
        print(f"{leg}: MoGe scale {np.nanmedian(leg_k) if np.isfinite(leg_k).any() else float('nan'):.2f}, "
              f"holes {np.mean([h.mean() for h in holes]):.2f}, {time.time() - t0:.0f} s", flush=True)
    # A frame without enough seen pixels (deep into a leg) takes its leg's median scale, else the run's.
    every = np.concatenate([np.array(v) for v in scales.values()])
    glob_k = float(np.nanmedian(every))
    n = len(ws.bank_cameras())
    for leg, ks in scales.items():
        lk = float(np.nanmedian(ks)) if np.isfinite(ks).any() else glob_k
        for k in ks:
            prox, mg = depths[n].astype(np.float32)
            code = codes[n]
            mg = mg * (k if np.isfinite(k) else lk)
            depths[n] = np.where(code == SEEN, prox, np.where((code == HOLE) & (mg < MAX_DEPTH), mg, 0)).astype(np.float16)
            n += 1
    out = ws.DATA / case / "splat"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / name, img=np.stack(imgs), code=np.stack(codes), depth=np.stack(depths),
                        w2c=np.stack(w2cs).astype(np.float32), name=np.array(names))
    (out / "moge_scales.json").write_text(json.dumps({"global": glob_k, "legs": scales}, default=float))
    print(f"{len(imgs)} views, MoGe scale {glob_k:.2f}, {time.time() - t0:.0f} s -> {out / name}")


def crop_nearest(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """ws_test.crop without interpolation (for distances)."""
    import cv2
    ph, pw = img.shape[:2]
    j, i = np.meshgrid(np.arange(ws.W) + 0.5, np.arange(ws.H) + 0.5)
    rays = np.stack([(j - ws.W / 2) / ws.F, (i - ws.H / 2) / ws.F, np.ones_like(j)], -1) @ m[:3, :3]
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    yaw, pitch = np.arctan2(rays[..., 1], rays[..., 0]), np.arcsin(np.clip(rays[..., 2], -1, 1))
    u = ((0.5 - yaw / (2 * np.pi)) % 1.0) * pw - 0.5
    v = (0.5 - pitch / np.pi) * ph - 0.5
    return cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)


def robust_mask(r: "torch.Tensor", thr: float) -> "torch.Tensor":
    """Inlier weight (0/1) per pixel from its error r (H, W)."""
    import torch.nn.functional as F
    inl = (r < thr).float()[None, None]
    smooth = F.avg_pool2d(inl, 3, 1, 1) > 0.5
    patch = F.avg_pool2d(inl, ROBUST_PATCH, ROBUST_PATCH, ceil_mode=True) >= ROBUST_PATCH_OK
    patch = patch.repeat_interleave(ROBUST_PATCH, 2).repeat_interleave(ROBUST_PATCH, 3)[..., :r.shape[0], :r.shape[1]]
    return (smooth | patch)[0, 0].float()


def rotation_6d(d6: "torch.Tensor") -> "torch.Tensor":
    """Rotation matrix from Zhou et al.'s 6D representation (Gram-Schmidt), as gsplat's examples."""
    import torch
    import torch.nn.functional as F
    b1 = F.normalize(d6[:3], dim=-1)
    b2 = F.normalize(d6[3:] - (b1 * d6[3:]).sum() * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), 0)


def sharpness(im: np.ndarray, alpha: np.ndarray) -> float:
    """Variance of the Laplacian on the grey image, over the pixels the splat covers."""
    import cv2
    lap = cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)
    m = alpha > 0.5
    return float(lap[m].var()) if m.sum() > 1000 else float("nan")


def train(case: str, steps: int, arm: str | None = None):
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization, rasterization_2dgs
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
    from plyfile import PlyData, PlyElement
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(ws.ROOT / "experiments" / "sphere_walk"))
    from gs_train import ssim_map
    t0 = time.time()
    dev = torch.device("cuda")
    cfg = ARMS[arm] if arm else {}
    steps = cfg.get("steps", steps)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    out = ws.DATA / case / "splat"
    z = np.load(out / cfg.get("views", "views.npz"))
    if arm:
        out = out / "arms" / arm
        out.mkdir(parents=True, exist_ok=True)
    img, code, depth, w2c, name = z["img"], z["code"], z["depth"], z["w2c"], z["name"]
    n_pano = sum(str(s).startswith("pano/") for s in name)
    frame = np.array([int(str(s).rsplit("/", 1)[1]) for s in name])
    held = np.array([i >= n_pano and frame[i] % HOLDOUT == HOLDOUT // 2 and not str(name[i]).startswith("fixer_")
                     for i in range(len(name))])
    clip = {str(s).rsplit("/", 1)[0]: f for s, f in zip(name, frame)}  # a leg's last frame index
    gen = [i for i in range(n_pano, len(name)) if not held[i]
           and frame[i] < round((clip[str(name[i]).rsplit("/", 1)[0]] + 1) * cfg.get("keep", 1.0))]
    gen_idx = {i: k for k, i in enumerate(gen)}
    keep_px = {}
    if cfg.get("consensus"):
        cz = np.load(ws.DATA / case / "depth_ma" / cfg["consensus"])
        at = {str(n): k for k, n in enumerate(cz["name"])}
        keep = cz["keep"]  # read once: every cz["keep"] decompresses the whole array, and a view of it keeps it alive
        keep_px = {i: keep[at[str(name[i])]].copy() for i in gen if str(name[i]) in at}
        del keep
    K = torch.tensor(ws.intrinsic(), dtype=torch.float32, device=dev)

    # Start: the panorama's points plus each 3rd generated frame's hole pixels at their scaled MoGe depth.
    pts, cols = [], []
    p, c, *_ = panorama_points(case)
    s = np.random.default_rng(0).choice(len(p), min(INIT_PANO_PTS, len(p)), replace=False)
    pts.append(p[s])
    cols.append(c[s] / 255)
    for i in gen[::3]:
        yy, xx = np.nonzero((code[i] == HOLE) & (depth[i] > 0))
        if not len(yy):
            continue
        s = np.random.default_rng(i).choice(len(yy), min(INIT_HOLE_PTS, len(yy)), replace=False)
        yy, xx = yy[s], xx[s]
        d = depth[i][yy, xx].astype(np.float32)
        cam = np.stack([(xx + 0.5 - ws.W / 2) / ws.F * d, (yy + 0.5 - ws.H / 2) / ws.F * d, d], -1)
        c2w = np.linalg.inv(w2c[i])
        pts.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
        cols.append(img[i][yy, xx] / 255)
    P, C = np.concatenate(pts).astype(np.float32), np.concatenate(cols).astype(np.float32)
    nn, _ = cKDTree(P).query(P, k=4)
    N = len(P)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.as_tensor(P, device=dev)),
        "scales": torch.nn.Parameter(torch.as_tensor(np.log(np.clip(nn[:, 1:].mean(1), 0.003, 0.5)),
                                                     dtype=torch.float32, device=dev)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.full((N,), math.log(INIT_OPACITY / (1 - INIT_OPACITY)), device=dev)),
        "sh0": torch.nn.Parameter(torch.as_tensor((C - 0.5) / C0, device=dev)[:, None, :]),
    })
    lrs = {"means": 1.6e-4 * SCENE_SCALE, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / steps))
    if cfg.get("mcmc"):
        strategy = MCMCStrategy(cap_max=cfg["mcmc"], refine_stop_iter=int(steps * 0.83))
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state()
    else:
        strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(steps * 0.66), refine_every=100,
                                   reset_every=10 ** 9, grow_grad2d=cfg.get("grow", GROW_GRAD), verbose=False)
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=SCENE_SCALE)
    affine = torch.nn.Parameter(torch.eye(3, 4, device=dev).repeat(len(gen), 1, 1))
    aff_opt = torch.optim.Adam([affine], lr=1e-3)
    weight_of = torch.tensor([0.0, *cfg.get("weights", (SEEN_WEIGHT, HOLE_WEIGHT, PANO_WEIGHT)), PSEUDO_WEIGHT, 0.0],
                             device=dev)
    pose = torch.nn.Parameter(torch.zeros(len(gen), 9, device=dev))
    pose_opt = torch.optim.Adam([pose], lr=POSE_LR, weight_decay=POSE_REG)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_opt, gamma=0.01 ** (1.0 / steps))
    thr, kept, masks = None, [], {}
    print(f"{n_pano} cuts, {len(gen)} generated frames, {held.sum()} held out, {N} Gaussians to start", flush=True)

    def adjusted(i):
        """World-to-camera of view i with its learnt correction (a rigid move in the camera's frame)."""
        m = torch.as_tensor(w2c[i], device=dev)
        if i not in gen_idx:
            return m
        d = pose[gen_idx[i]]
        t = torch.eye(4, device=dev)
        t[:3, :3] = rotation_6d(d[3:] + torch.tensor([1.0, 0, 0, 0, 1, 0], device=dev))
        t[:3, 3] = d[:3]
        return t @ m

    twod_out = {}  # the last 2DGS render's normals, normals from depth, distortion and median depth

    def render(m):
        if cfg.get("twod"):
            rgbd, alpha, nrm, surf, dist, med, info = rasterization_2dgs(
                params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), params["sh0"],
                torch.as_tensor(m, device=dev)[None], K[None], ws.W, ws.H, sh_degree=0, render_mode="RGB+ED",
                near_plane=0.05, far_plane=500, packed=False, distloss=True, depth_mode="expected")
            twod_out.update(normals=nrm[0], surf=surf[0], distort=dist[0, ..., 0], median=med[0, ..., 0])
            return rgbd[0, ..., :3], rgbd[0, ..., 3], alpha[0, ..., 0], info
        rgbd, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), params["sh0"],
            torch.as_tensor(m, device=dev)[None], K[None], ws.W, ws.H, sh_degree=0, render_mode="RGB+ED",
            near_plane=0.05, far_plane=500, packed=False)
        return rgbd[0, ..., :3], rgbd[0, ..., 3], alpha[0, ..., 0], info

    for step in range(steps):
        i = np.random.randint(n_pano) if np.random.rand() < PANO_SHARE else gen[np.random.randint(len(gen))]
        gt = torch.as_tensor(img[i], device=dev).float().div(255).permute(2, 0, 1)
        cd = torch.as_tensor(code[i], device=dev).long()
        wpx = weight_of[cd]
        if i in keep_px:
            wpx = wpx * torch.as_tensor(keep_px[i], device=dev)
        rgb, dep, alpha, info = render(adjusted(i) if cfg.get("pose") and step >= POSE_START else w2c[i])
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = rgb.permute(2, 0, 1)
        if i in gen_idx:
            a = affine[gen_idx[i]]
            pred = torch.einsum("ij,jhw->ihw", a[:, :3], pred) + a[:, 3, None, None]
            if cfg.get("robust"):
                r = (pred - gt).abs().mean(0).detach()
                q = torch.quantile(r[cd != SKY][::7], ROBUST_Q).item() if (cd != SKY).any() else None
                if q is not None:
                    thr = q if thr is None else 0.99 * thr + 0.01 * q
                if step >= ROBUST_START and thr is not None:
                    w = robust_mask(r, thr)
                    wpx = wpx * w
                    kept.append(w[cd != SKY].mean().item())
                    masks[i] = w.byte().cpu().numpy()
        l1 = (wpx * (pred - gt).abs().mean(0)).mean()
        ss = (wpx * (1 - ssim_map(pred, gt).mean(0))).mean()
        loss = (1 - SSIM_SHARE) * l1 + SSIM_SHARE * ss + SKY_ALPHA * (alpha * (cd == SKY)).mean()
        tgt = torch.as_tensor(depth[i].astype(np.float32), device=dev)
        dw = torch.where((cd == SEEN) | (cd == PANO), DEPTH_SEEN, torch.where(cd == HOLE, DEPTH_HOLE, 0.0)) * (tgt > 0)
        loss = loss + (dw * (dep - tgt).abs() / tgt.clamp(min=0.1)).mean()
        loss = loss + AFFINE_REG * ((affine - torch.eye(3, 4, device=dev)) ** 2).mean()
        if cfg.get("twod"):
            if step >= NORMAL_START:
                agree = (twod_out["normals"] * twod_out["surf"] * alpha.detach()[..., None]).sum(-1)
                loss = loss + NORMAL_W * (1 - agree).mean()
            if step >= DIST_START:
                loss = loss + DIST_W * twod_out["distort"].mean()
        if cfg.get("mcmc"):
            loss = loss + MCMC_REG * (torch.sigmoid(params["opacities"]).mean() + torch.exp(params["scales"]).mean())
        if cfg.get("aniso"):
            sc = params["scales"]
            loss = loss + RATIO_WEIGHT * F.relu(sc.max(1).values - sc.min(1).values - math.log(MAX_RATIO)).mean()
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        aff_opt.step()
        aff_opt.zero_grad(set_to_none=True)
        if cfg.get("pose"):
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
            pose_sched.step()
        sched.step()
        if cfg.get("mcmc"):
            strategy.step_post_backward(params, optimizers, state, step, info, lr=sched.get_last_lr()[0])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 1000 == 0 or step == steps - 1:
            extra = f" inliers {np.mean(kept[-500:]):.2f}" if kept else ""
            print(f"step {step} loss {loss.item():.4f} n {len(params['means'])}{extra} {time.time() - t0:.0f} s",
                  flush=True)

    import lpips
    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    with torch.no_grad():
        # Held-out frames: PSNR on the non-sky pixels, seen and invented apart; SSIM and LPIPS on the non-sky pixels.
        score = {"seen": [], "hole": [], "ssim": [], "lpips": []}
        shots = []
        for i in np.nonzero(held)[0]:
            if str(name[i]).startswith(("nbv_", "spin_", "spg_")):
                continue  # scored on phase 1's held-out frames only, the same for every arm
            rgb, _, _, _ = render(w2c[i])
            pr = rgb.clamp(0, 1).cpu().numpy()
            gt = img[i] / 255
            for key, cval in (("seen", SEEN), ("hole", HOLE)):
                msk = code[i] == cval
                if msk.sum() > 1000:
                    score[key].append(float(-10 * np.log10(((pr - gt) ** 2)[msk].mean())))
            ground = torch.as_tensor(code[i] != SKY, device=dev).float()
            pt, gtt = rgb.clamp(0, 1).permute(2, 0, 1), torch.as_tensor(gt, device=dev).float().permute(2, 0, 1)
            score["ssim"].append(float((ssim_map(pt, gtt).mean(0) * ground).sum() / ground.sum()))
            dmap = lp(pt[None] * ground, gtt[None] * ground, normalize=True)[0, 0]  # sky blacked out in both
            score["lpips"].append(float((dmap * ground).sum() / ground.sum()))
            shots.append((str(name[i]), img[i], (pr * 255).astype(np.uint8)))
        # Novel views: between the walks, raised and turned, never seen by any frame.
        novel = []
        for yaw in range(0, 360, 45):
            for pos, pitch in (([0, 0, 1.4], -25.0), ([2.5 * math.cos(math.radians(yaw + 20)),
                                                      2.5 * math.sin(math.radians(yaw + 20)), 0], -5.0)):
                rgb, _, alpha, _ = render(ws.w2c(pos, yaw, pitch).astype(np.float32))
                novel.append((f"yaw {yaw} {'up 1.4 m' if pos[2] else '2.5 m off, beside'}",
                              (rgb.clamp(0, 1) * 255).byte().cpu().numpy()))
                score.setdefault("sharp", []).append(sharpness(novel[-1][1], alpha.cpu().numpy()))
        m = params["means"].cpu().numpy()
        c = np.clip(params["sh0"][:, 0].cpu().numpy() * C0 + 0.5, 0, 1)
        dt = [(k, "f4") for k in ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
              + [f"f_rest_{j}" for j in range(45)] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                      "rot_0", "rot_1", "rot_2", "rot_3"]]
        a = np.zeros(len(m), dt)
        a["x"], a["y"], a["z"] = m.T
        a["f_dc_0"], a["f_dc_1"], a["f_dc_2"] = ((c - 0.5) / C0).T
        a["opacity"] = params["opacities"].cpu().numpy()
        for j in range(3):
            a[f"scale_{j}"] = params["scales"][:, j].cpu().numpy()
        q = F.normalize(params["quats"], dim=-1).cpu().numpy()
        for j in range(4):
            a[f"rot_{j}"] = q[:, j]
        PlyData([PlyElement.describe(a, "vertex")]).write(out / "splat.ply")
    if cfg.get("twod"):  # every view's colour and median depth, for ws_mesh.py's TSDF
        with torch.no_grad():
            rr, dd = [], []
            for i in range(len(name)):
                rgb, _, alpha, _ = render(w2c[i])
                rr.append((rgb.clamp(0, 1) * 255).byte().cpu().numpy())
                dd.append(torch.where(alpha > 0.5, twod_out["median"], 0).half().cpu().numpy())
        np.savez_compressed(out / "renders.npz", img=np.stack(rr), depth=np.stack(dd), code=code, w2c=w2c, name=name)
    pose_info = {}
    if cfg.get("pose"):
        with torch.no_grad():
            fixed = np.stack([adjusted(i).cpu().numpy() for i in gen])
        rel = fixed @ np.linalg.inv(w2c[gen])  # the correction, camera frame
        deg = np.degrees(np.arccos(np.clip((np.trace(rel[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
        cm = np.linalg.norm(rel[:, :3, 3], axis=1) * 100
        pose_info = {"pose_deg_median": float(np.median(deg)), "pose_deg_p90": float(np.percentile(deg, 90)),
                     "pose_cm_median": float(np.median(cm)), "pose_cm_p90": float(np.percentile(cm, 90))}
        np.savez_compressed(out / "poses.npz", name=name[gen], w2c=fixed.astype(np.float32))
    info = {"arm": arm, "config": cfg, "steps": steps, **pose_info, "gaussians": len(m), "cuts": n_pano,
            "generated_frames": len(gen), "held_out": int(held.sum()),
            "psnr_seen": float(np.mean(score["seen"])) if score["seen"] else None,
            "psnr_invented": float(np.mean(score["hole"])) if score["hole"] else None,
            "ssim": float(np.mean(score["ssim"])), "lpips": float(np.mean(score["lpips"])),
            "novel_sharpness": float(np.nanmean(score["sharp"])),
            "inlier_share": float(np.mean(kept)) if kept else None,
            "seconds": round(time.time() - t0), "units": "metres, z up, camera 1.6 m above the ground"}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    if arm:
        np.savez_compressed(out / "novel.npz", label=np.array([n for n, _ in novel]), img=np.stack([v for _, v in novel]),
                            held_name=np.array([n for n, _, _ in shots]), held_gt=np.stack([g for _, g, _ in shots]),
                            held=np.stack([p for _, _, p in shots]))
        if masks:
            few = sorted(masks)[::max(1, len(masks) // 4)][:4]
            np.savez_compressed(out / "masks.npz", name=np.array([str(name[i]) for i in few]),
                                img=np.stack([img[i] for i in few]), mask=np.stack([masks[i] for i in few]))
    print(info, flush=True)


def sheet(out: Path, case: str, shots, novel, info):
    """Held-out frames (WorldStereo against the splat) and novel views of the splat."""
    from PIL import ImageFont
    font = ImageFont.load_default(size=22)
    cw, ch = 400, 231
    held_rows = []
    for k in range(0, len(shots[::2]), 4):
        a = shots[::2][k:k + 4]
        held_rows.append([(f"{n}: WorldStereo", g) for n, g, _ in a])
        held_rows.append([(f"{n}: splat (held out)", p) for n, _, p in a])
    rows = held_rows + [novel[k:k + 4] for k in range(0, len(novel), 4)]
    title = (f"{case}: WorldStereo splat, {info['gaussians']} Gaussians; held-out PSNR seen {info['psnr_seen']:.1f}, "
             f"invented {info['psnr_invented']:.1f} dB. Top: held-out frames; bottom: novel views")
    sh = Image.new("RGB", (cw * 4, 40 + ch * len(rows)), (0, 0, 0))
    ImageDraw.Draw(sh).text((8, 8), title, fill=(255, 255, 255), font=font)
    for r, row in enumerate(rows):
        for k, (label, im) in enumerate(row):
            t = Image.fromarray(im).resize((cw, ch))
            ImageDraw.Draw(t).text((6, 4), label, fill=(255, 255, 0), font=ImageFont.load_default(size=16))
            sh.paste(t, (k * cw, 40 + r * ch))
    sh.save(out / "sheet.jpg", quality=88)
    print(out / "sheet.jpg")


def video_frames(path: Path, idx: list[float]) -> list[np.ndarray]:
    """Frames at fractions idx of a video (RGB), with OpenCV (the gsplat env has no PyAV)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for t in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * (n - 1)))
        out.append(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
    return out


def compare(case: str):
    """The same walk forward from the photo by each route: Matrix-3D's panorama video (cut forward, no 3D),
    FlashWorld's forward move (from its sheet: its splats were not kept), WorldStereo's generated frames and the
    splat fitted to them, plus the splat from where no route walked."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from plyfile import PlyData
    from PIL import ImageFont
    font = ImageFont.load_default(size=22)
    dev = torch.device("cuda")
    out = ws.DATA / case / "splat"
    v = PlyData.read(out / "splat.ply")["vertex"]
    g = lambda *k: torch.as_tensor(np.stack([np.asarray(v[x], np.float32) for x in k], -1), device=dev)  # noqa: E731
    sp = dict(means=g("x", "y", "z"), quats=F.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1),
              scales=torch.exp(g("scale_0", "scale_1", "scale_2")), opacities=torch.sigmoid(g("opacity")[:, 0]),
              colors=g("f_dc_0", "f_dc_1", "f_dc_2")[:, None])
    K = torch.tensor(ws.intrinsic(), dtype=torch.float32, device=dev)

    def splat(pos, yaw, pitch=ws.PITCH):
        with torch.no_grad():
            rgb, _, _ = rasterization(**sp, viewmats=torch.as_tensor(ws.w2c(pos, yaw, pitch), dtype=torch.float32,
                                                                    device=dev)[None],
                                      Ks=K[None], width=ws.W, height=ws.H, sh_degree=0, packed=False)
        return (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()

    fr = [0.0, 0.25, 0.5, 0.75, 1.0]
    meta = json.loads((ws.DATA / case / "prep.json").read_text())
    yaw0, len0 = meta["trajs"]["reconstruct_0/traj0"]["yaw"], meta["trajs"]["reconstruct_0/traj0"]["length"]
    m3d = video_frames(ws.PANOS / case / "m3d_straight" / "pano_video.mp4", fr)
    m3d = [ws.crop(f, ws.w2c([0, 0, 0], 0.0, ws.PITCH)) for f in m3d]
    fw = Image.open(ws.DATA.parent / "flashworld_test" / f"sheet_{case}.jpg")
    tw = fw.width / 6  # its last row: orbit, left, right, up, back, forward at the end of each move
    fw_tiles = [np.asarray(fw.crop((round(k * tw), 680, round((k + 1) * tw), 880))) for k in (5, 1, 2, 3)]
    res = ws.DATA / case / "out" / "out" / case
    gen = video_frames(res / "reconstruct_0/traj0" / f"{ws.MODEL}_result.mp4", fr)
    yr = math.radians(yaw0)
    fwd = [splat([math.cos(yr) * len0 * t, math.sin(yr) * len0 * t, 0], yaw0) for t in fr]
    off = [splat([1.5, 2.0, 0], -60.0), splat([3.0, -1.5, 0.8], 120.0, -15.0), splat([0, 0, 3.0], 0.0, -45.0),
           splat([-2.0, 0.5, 0], 30.0), splat([6.0, 0.0, 0], 180.0)]
    rows = [("Matrix-3D (panorama walk, cut forward; video only, no 3D)", m3d),
            ("FlashWorld (forward, left, right, up; ~1 unit moves, its splats were not kept)", fw_tiles),
            (f"WorldStereo generated frames, walk 0 ({len0:.1f} m, then {meta['trajs']['reconstruct_0/traj1']['length']:.1f} m more)", gen),
            ("WorldStereo splat, same walk (metric, Isaac USD)", fwd),
            ("WorldStereo splat, where nobody walked: beside the road, raised, overhead, behind, looking back", off)]
    cw, ch = 400, 231
    sh = Image.new("RGB", (cw * 5, len(rows) * (ch + 30)), (0, 0, 0))
    for r, (label, tiles) in enumerate(rows):
        ImageDraw.Draw(sh).text((8, r * (ch + 30) + 4), label, fill=(255, 255, 0), font=font)
        for k, t in enumerate(tiles):
            sh.paste(Image.fromarray(t).resize((cw, ch)), (k * cw, r * (ch + 30) + 30))
    sh.save(ws.DATA / case / "sheet_compare.jpg", quality=88)
    print(ws.DATA / case / "sheet_compare.jpg")


def score(case: str, arm: str):
    """An arm's splat scored on phase 1's held-out frames (views.npz), like train does: the same frames for every
    arm whatever it was fitted to. Writes score.json next to the splat."""
    import lpips
    import torch
    sys.path.insert(0, str(ws.ROOT / "experiments" / "sphere_walk"))
    from gs_train import ssim_map
    from ws_nbv import load_splat, render
    dev = torch.device("cuda")
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    img, code, w2c, name = z["img"], z["code"], z["w2c"], z["name"]
    frame = np.array([int(str(s).rsplit("/", 1)[1]) for s in name])
    held = [i for i in range(len(name)) if not str(name[i]).startswith("pano/") and frame[i] % HOLDOUT == HOLDOUT // 2]
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", dev)
    rgb, _, _ = render(sp, w2c[held], np.array(ws.intrinsic(), np.float32), ws.W, ws.H)
    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    out = {"seen": [], "hole": [], "ssim": [], "lpips": []}
    with torch.no_grad():
        for k, i in enumerate(held):
            pt = rgb[k].float().div(255).permute(2, 0, 1)
            gt = torch.as_tensor(img[i], device=dev).float().div(255).permute(2, 0, 1)
            pr, g = pt.permute(1, 2, 0).cpu().numpy(), img[i] / 255
            for key, cval in (("seen", SEEN), ("hole", HOLE)):
                m = code[i] == cval
                if m.sum() > 1000:
                    out[key].append(float(-10 * np.log10(((pr - g) ** 2)[m].mean())))
            ground = torch.as_tensor(code[i] != SKY, device=dev).float()
            out["ssim"].append(float((ssim_map(pt, gt).mean(0) * ground).sum() / ground.sum()))
            d = lp(pt[None] * ground, gt[None] * ground, normalize=True)[0, 0]
            out["lpips"].append(float((d * ground).sum() / ground.sum()))
    res = {"arm": arm, "held_out": len(held), **{k: round(float(np.mean(v)), 4) for k, v in out.items()}}
    (ws.DATA / case / "splat" / "arms" / arm / "score.json").write_text(json.dumps(res, indent=1))
    print(res, flush=True)


def arms(case: str):
    """splat/arms/sheet.jpg: every trained arm side by side on the same novel and held-out views, with its scores;
    the robust arm's inlier masks at the bottom (black: pixels that stopped teaching)."""
    from PIL import ImageFont
    base = ws.DATA / case / "splat" / "arms"
    names = [a for a in ARMS if (base / a / "novel.npz").exists()]
    infos = {a: json.loads((base / a / "info.json").read_text()) for a in names}
    nov = {a: np.load(base / a / "novel.npz") for a in names}
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    cw, ch = 560, 323
    first = nov[names[0]]
    pick_novel = [k for k, lab in enumerate(first["label"]) if "beside" in str(lab)][::1][:8]
    pick_held = [0, len(first["held_name"]) // 2, len(first["held_name"]) - 1]
    rows = []
    for k in pick_novel:
        rows.append((f"novel: {first['label'][k]}", [(a, nov[a]["img"][k]) for a in names]))
    for k in pick_held:
        rows.append((f"held out: {first['held_name'][k]}",
                     [("WorldStereo", first["held_gt"][k])] + [(a, nov[a]["held"][k]) for a in names]))
    for a in names:
        if (base / a / "masks.npz").exists():
            mk = np.load(base / a / "masks.npz")
            rows.append((f"{a}: inlier masks (black = left out)",
                         [(str(n), (im * (0.25 + 0.75 * m[..., None])).astype(np.uint8))
                          for n, im, m in zip(mk["name"], mk["img"], mk["mask"])]))
    cols = max(len(r[1]) for r in rows)
    head = 40 + 30 * len(names)
    sh = Image.new("RGB", (cw * cols, head + len(rows) * (ch + 28)), (0, 0, 0))
    d = ImageDraw.Draw(sh)
    d.text((8, 8), f"{case}: splat test arms (higher PSNR/SSIM/sharpness, lower LPIPS = better)", fill=(255, 255, 255),
           font=font)
    for j, a in enumerate(names):
        f = infos[a]
        d.text((8, 40 + 30 * j), f"{a} {f['config']}: PSNR seen {f['psnr_seen']:.2f} invented {f['psnr_invented']:.2f}"
               f"  SSIM {f['ssim']:.3f}  LPIPS {f['lpips']:.3f}  sharpness {f['novel_sharpness']:.0f}"
               f"  {f['gaussians'] / 1e6:.2f}M Gaussians  {f['seconds']} s", fill=(255, 255, 0), font=font)
    for r, (label, tiles) in enumerate(rows):
        y = head + r * (ch + 28)
        d.text((8, y + 2), label, fill=(255, 255, 0), font=font)
        for k, (tl, im) in enumerate(tiles):
            t = Image.fromarray(im).resize((cw, ch))
            ImageDraw.Draw(t).text((6, 4), tl, fill=(0, 255, 255), font=small)
            sh.paste(t, (k * cw, y + 28))
    sh.save(base / "sheet.jpg", quality=88)
    print(base / "sheet.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["views", "train", "compare", "arms", "score"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--views", default="views.npz", help="views: the file to write")
    ap.add_argument("--skip", nargs="*", default=[], help="views: leave out legs starting with these (nbv_, spin_)")
    ap.add_argument("--case", default="photo21")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    {"views": lambda: views(a.case, a.views, tuple(a.skip)), "train": lambda: train(a.case, a.steps, a.arm),
     "compare": lambda: compare(a.case), "arms": lambda: arms(a.case), "score": lambda: score(a.case, a.arm)}[a.cmd]()
