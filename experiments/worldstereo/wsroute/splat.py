"""One 3D Gaussian splat (gsplat) of a WorldStereo run: the panorama's 27 memory cuts plus every frame of every
generated leg, so the scene can be walked in Isaac Sim.

The panorama is the truth: its cuts weigh PANO_WEIGHT with our MoGe-2 depth of the panorama. A generated frame
teaches fully where the guide render had a hole (what WorldStereo invented) and weighs SEEN_WEIGHT where the
panorama's point cloud already covered it, with that point cloud's depth; in the holes, MoGe-2's depth of the
frame, scaled to the point cloud's on the seen pixels, only keeps floaters away. Sky pixels push the splat clear
(Isaac's dome shows the sky). Each generated frame gets its own colour correction. Fixer's pseudo views (fixer.py)
teach their target pixels like a seen pixel, without depth. Frame k of a leg with k % HOLDOUT == HOLDOUT // 2 is
left out and scored: held-out PSNR (seen, invented), SSIM and LPIPS on the non-sky pixels, sharpness of novel views.

    views  SIM2CHIP_SCENE_PYTHON   splat/<views file>: every training and held-out view
    train  SIM2CHIP_GSPLAT_PYTHON  splat/arms/<arm>/: splat.ply, info.json, novel.npz

Every arm is recipe G2 (30k steps, MCMC capped at 4M Gaussians); they differ in their views file (arm_views)."""
from __future__ import annotations

import json
import math
import time

import numpy as np
from PIL import Image

from . import camera
from .camera import F, H, W
from .config import arm_views
from .panorama import Panorama
from .paths import Case, add_moge, moge_ckpt

PANO_WEIGHT = 4.0
SEEN_WEIGHT = 0.3      # a generated pixel the panorama already covered: it must not compete with the photo
HOLE_WEIGHT = 1.0      # a generated pixel the guide had no point for: what WorldStereo brings
PSEUDO_WEIGHT = 0.3    # a Fixer pseudo view's target pixel: like a seen pixel, it must not compete with the frames
DEPTH_SEEN = 0.5       # relative depth error against the panorama's points
DEPTH_HOLE = 0.05      # against MoGe's scaled depth in the holes: only against floaters
SKY_ALPHA = 1.0
SSIM_SHARE = 0.2
AFFINE_REG = 1.0
PANO_SHARE = 0.25      # share of steps on a panorama cut
HOLDOUT = 7
MAX_DEPTH = 40.0       # metres; farther depth targets are left out (the guide squashed them, see walks.prep)
STEPS = 30000
MCMC_CAP = 4_000_000
MCMC_REG = 0.01        # opacity and scale pull of the MCMC strategy (its paper's values)
SCENE_SCALE = 10.0     # sets the means' learning rate
INIT_PANO_PTS = 400000
INIT_HOLE_PTS = 4000   # MoGe points per generated frame in its holes (every 3rd frame)
INIT_OPACITY = 0.7
C0 = 0.28209479177387814
SEED = 0
SCORE_SKIP = ("nbv_", "spin_", "spg_")  # held-out frames of these legs are not scored (the same frames for every arm)
# codes per pixel
SKY, SEEN, HOLE, PANO = 0, 1, 2, 3
PSEUDO, NONE = 4, 5    # a Fixer pseudo view's pixel that teaches, and one that teaches nothing


def frame_index(names) -> np.ndarray:
    """Frame number of each view name (<leg>/<frame>)."""
    return np.array([int(str(s).rsplit("/", 1)[1]) for s in names])


def is_held(name: str, frame: int) -> bool:
    """A generated frame left out of the fit and scored."""
    return not name.startswith(("pano/", "fixer_")) and frame % HOLDOUT == HOLDOUT // 2


def views(case_name: str, name: str = "views.npz", skip: tuple = ()):
    """splat/<name>: every training and held-out view (image, per-pixel code, depth target, world-to-camera)."""
    import cv2
    import imageio.v3 as iio
    import torch
    add_moge()
    from moge.model.v2 import MoGeModel
    case = Case(case_name)
    t0 = time.time()
    pano = Panorama(case)
    pts = pano.points()[0]
    imgs, codes, depths, w2cs, names = [], [], [], [], []

    # Panorama cuts: the photo, with the panorama's distance turned into camera depth.
    j, i = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    ray_len = np.sqrt(((j - W / 2) / F) ** 2 + ((i - H / 2) / F) ** 2 + 1).astype(np.float32)
    dist_full = cv2.resize(np.where(pano.valid, pano.dist, 0).astype(np.float32),
                           (pano.full.shape[1], pano.full.shape[0]), interpolation=cv2.INTER_NEAREST)
    for k, m in enumerate(camera.bank_cameras()):
        d = camera.crop(dist_full, m, nearest=True) / ray_len
        imgs.append(np.asarray(Image.open(case.scene / f"pano_bank/images/{k:04d}.png").convert("RGB")))
        codes.append(np.where(d > 0, PANO, SKY).astype(np.uint8))
        depths.append(np.where(d < MAX_DEPTH, d, 0).astype(np.float16))
        w2cs.append(m)
        names.append(f"pano/{k:04d}")

    model = MoGeModel.from_pretrained(moge_ckpt()).cuda().eval()
    fov = math.degrees(2 * math.atan(W / 2 / F))
    scales = {}
    for leg in case.legs(skip):
        frames = iio.imread(case.leg_video(leg), plugin="pyav")
        holes = iio.imread(case.scene / leg / "render_mask.mp4", plugin="pyav")[..., 0] > 127
        cams = json.loads((case.scene / leg / "camera.json").read_text())["extrinsic"]
        leg_k = []
        for f, (img, hole, m) in enumerate(zip(frames, holes, cams)):
            m = np.array(m)
            x = torch.tensor(img / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None]
            with torch.no_grad():
                out = model.infer(x, fov_x=torch.tensor([fov], device="cuda"), apply_mask=True)
            mg = out["depth"][0].cpu().numpy()
            sky = ~out["mask"][0].cpu().numpy()
            mg = np.where(np.isfinite(mg) & ~sky, mg, 0)
            prox = camera.zbuffer(pts, m)
            ok = ~hole & (prox > 0) & (mg > 0) & (prox < MAX_DEPTH)
            leg_k.append(float(np.median(prox[ok] / mg[ok])) if ok.sum() > 500 else np.nan)
            imgs.append(img)
            codes.append(np.where(sky, SKY, np.where(hole, HOLE, SEEN)).astype(np.uint8))
            depths.append(np.stack([np.where(prox < MAX_DEPTH, prox, 0), mg], 0).astype(np.float16))
            w2cs.append(m)
            names.append(f"{leg}/{f:02d}")
        scales[leg] = leg_k
        print(f"{leg}: MoGe scale {np.nanmedian(leg_k) if np.isfinite(leg_k).any() else float('nan'):.2f}, "
              f"holes {np.mean([h.mean() for h in holes]):.2f}, {time.time() - t0:.0f} s", flush=True)
    # A frame without enough seen pixels (deep into a leg) takes its leg's median scale, else the run's.
    glob_k = float(np.nanmedian(np.concatenate([np.array(v) for v in scales.values()])))
    n = len(camera.bank_cameras())
    for ks in scales.values():
        lk = float(np.nanmedian(ks)) if np.isfinite(ks).any() else glob_k
        for k in ks:
            prox, mg = depths[n].astype(np.float32)
            code = codes[n]
            mg = mg * (k if np.isfinite(k) else lk)
            invented = np.where((code == HOLE) & (mg < MAX_DEPTH), mg, 0)
            depths[n] = np.where(code == SEEN, prox, invented).astype(np.float16)
            n += 1
    case.splat.mkdir(exist_ok=True)
    np.savez_compressed(case.splat / name, img=np.stack(imgs), code=np.stack(codes), depth=np.stack(depths),
                        w2c=np.stack(w2cs).astype(np.float32), name=np.array(names))
    (case.splat / "moge_scales.json").write_text(json.dumps({"global": glob_k, "legs": scales}, default=float))
    print(f"{len(imgs)} views, MoGe scale {glob_k:.2f}, {time.time() - t0:.0f} s -> {case.splat / name}")


def sharpness(im: np.ndarray, alpha: np.ndarray) -> float:
    """Variance of the Laplacian on the grey image, over the pixels the splat covers."""
    import cv2
    lap = cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)
    m = alpha > 0.5
    return float(lap[m].var()) if m.sum() > 1000 else float("nan")


def novel_poses() -> list[tuple[str, np.ndarray]]:
    """Views no frame saw: raised 1.4 m at the centre and 2.5 m beside it, every 45 degrees."""
    out = []
    for yaw in range(0, 360, 45):
        for pos, pitch in (([0, 0, 1.4], -25.0), ([2.5 * math.cos(math.radians(yaw + 20)),
                                                  2.5 * math.sin(math.radians(yaw + 20)), 0], -5.0)):
            out.append((f"yaw {yaw} {'up 1.4 m' if pos[2] else '2.5 m off, beside'}",
                        camera.w2c(pos, yaw, pitch).astype(np.float32)))
    return out


def write_ply(params, path):
    """The splat as a 3DGS .ply (degree-0 colour, the rest of the harmonics zero)."""
    import torch.nn.functional as tF
    from plyfile import PlyData, PlyElement
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
    q = tF.normalize(params["quats"], dim=-1).cpu().numpy()
    for j in range(4):
        a[f"rot_{j}"] = q[:, j]
    PlyData([PlyElement.describe(a, "vertex")]).write(path)


def init_points(case: Case, img, code, depth, w2c, gen) -> tuple[np.ndarray, np.ndarray]:
    """The start: the panorama's points plus each 3rd generated frame's hole pixels at their scaled MoGe depth."""
    pts, cols = [], []
    p, c = Panorama(case).points()
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
        cam = np.stack([(xx + 0.5 - W / 2) / F * d, (yy + 0.5 - H / 2) / F * d, d], -1)
        c2w = np.linalg.inv(w2c[i])
        pts.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
        cols.append(img[i][yy, xx] / 255)
    return np.concatenate(pts).astype(np.float32), np.concatenate(cols).astype(np.float32)


def ssim_map(a, b):
    """SSIM per pixel of two (3, H, W) images, 11-pixel Gaussian window (was sphere_walk/gs_train.py's, whose
    module needs the retired photo2scene package)."""
    import torch
    import torch.nn.functional as tF
    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5 ** 2))
    win = (g[:, None] * g[None]) / g.sum() ** 2
    win = win.expand(3, 1, 11, 11)
    f = lambda x: tF.conv2d(x[None], win, padding=5, groups=3)[0]  # noqa: E731
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def train(case_name: str, arm: str, steps: int = STEPS):
    import torch
    import torch.nn.functional as tF
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy
    from scipy.spatial import cKDTree
    case = Case(case_name)
    t0 = time.time()
    dev = torch.device("cuda")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    z = np.load(case.splat / arm_views(arm))
    out = case.arm(arm)
    out.mkdir(parents=True, exist_ok=True)
    img, code, depth, w2c, name = z["img"], z["code"], z["depth"], z["w2c"], z["name"]
    n_pano = sum(str(s).startswith("pano/") for s in name)
    frame = frame_index(name)
    held = np.array([i >= n_pano and is_held(str(name[i]), frame[i]) for i in range(len(name))])
    gen = [i for i in range(n_pano, len(name)) if not held[i]]
    gen_idx = {i: k for k, i in enumerate(gen)}
    K = torch.tensor(camera.intrinsic(), dtype=torch.float32, device=dev)

    P, C = init_points(case, img, code, depth, w2c, gen)
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
    strategy = MCMCStrategy(cap_max=MCMC_CAP, refine_stop_iter=int(steps * 0.83))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    affine = torch.nn.Parameter(torch.eye(3, 4, device=dev).repeat(len(gen), 1, 1))
    aff_opt = torch.optim.Adam([affine], lr=1e-3)
    weight_of = torch.tensor([0.0, SEEN_WEIGHT, HOLE_WEIGHT, PANO_WEIGHT, PSEUDO_WEIGHT, 0.0], device=dev)
    print(f"{n_pano} cuts, {len(gen)} generated frames, {held.sum()} held out, {N} Gaussians to start", flush=True)

    def render(m):
        rgbd, alpha, info = rasterization(
            params["means"], tF.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), params["sh0"],
            torch.as_tensor(m, device=dev)[None], K[None], W, H, sh_degree=0, render_mode="RGB+ED",
            near_plane=0.05, far_plane=500, packed=False)
        return rgbd[0, ..., :3], rgbd[0, ..., 3], alpha[0, ..., 0], info

    for step in range(steps):
        i = np.random.randint(n_pano) if np.random.rand() < PANO_SHARE else gen[np.random.randint(len(gen))]
        gt = torch.as_tensor(img[i], device=dev).float().div(255).permute(2, 0, 1)
        cd = torch.as_tensor(code[i], device=dev).long()
        wpx = weight_of[cd]
        rgb, dep, alpha, info = render(w2c[i])
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = rgb.permute(2, 0, 1)
        if i in gen_idx:
            a = affine[gen_idx[i]]
            pred = torch.einsum("ij,jhw->ihw", a[:, :3], pred) + a[:, 3, None, None]
        l1 = (wpx * (pred - gt).abs().mean(0)).mean()
        ss = (wpx * (1 - ssim_map(pred, gt).mean(0))).mean()
        loss = (1 - SSIM_SHARE) * l1 + SSIM_SHARE * ss + SKY_ALPHA * (alpha * (cd == SKY)).mean()
        tgt = torch.as_tensor(depth[i].astype(np.float32), device=dev)
        dw = torch.where((cd == SEEN) | (cd == PANO), DEPTH_SEEN, torch.where(cd == HOLE, DEPTH_HOLE, 0.0)) * (tgt > 0)
        loss = loss + (dw * (dep - tgt).abs() / tgt.clamp(min=0.1)).mean()
        loss = loss + AFFINE_REG * ((affine - torch.eye(3, 4, device=dev)) ** 2).mean()
        loss = loss + MCMC_REG * (torch.sigmoid(params["opacities"]).mean() + torch.exp(params["scales"]).mean())
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        aff_opt.step()
        aff_opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, lr=sched.get_last_lr()[0])
        if step % 1000 == 0 or step == steps - 1:
            print(f"step {step} loss {loss.item():.4f} n {len(params['means'])} {time.time() - t0:.0f} s", flush=True)

    import lpips
    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    with torch.no_grad():
        # Held-out frames: PSNR on the non-sky pixels, seen and invented apart; SSIM and LPIPS on the non-sky pixels.
        score = {"seen": [], "hole": [], "ssim": [], "lpips": [], "sharp": []}
        shots = []
        for i in np.nonzero(held)[0]:
            if str(name[i]).startswith(SCORE_SKIP):
                continue
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
        novel = []
        for label, m in novel_poses():
            rgb, _, alpha, _ = render(m)
            novel.append((label, (rgb.clamp(0, 1) * 255).byte().cpu().numpy()))
            score["sharp"].append(sharpness(novel[-1][1], alpha.cpu().numpy()))
        write_ply(params, out / "splat.ply")
    info = {"arm": arm, "config": {"steps": steps, "mcmc": MCMC_CAP, "views": arm_views(arm)}, "steps": steps,
            "gaussians": len(params["means"]), "cuts": n_pano, "generated_frames": len(gen),
            "held_out": int(held.sum()),
            "psnr_seen": float(np.mean(score["seen"])) if score["seen"] else None,
            "psnr_invented": float(np.mean(score["hole"])) if score["hole"] else None,
            "ssim": float(np.mean(score["ssim"])), "lpips": float(np.mean(score["lpips"])),
            "novel_sharpness": float(np.nanmean(score["sharp"])),
            "seconds": round(time.time() - t0), "units": "metres, z up, camera 1.6 m above the ground"}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    np.savez_compressed(out / "novel.npz", label=np.array([n for n, _ in novel]), img=np.stack([v for _, v in novel]),
                        held_name=np.array([n for n, _, _ in shots]), held_gt=np.stack([g for _, g, _ in shots]),
                        held=np.stack([p for _, _, p in shots]))
    print(info, flush=True)
