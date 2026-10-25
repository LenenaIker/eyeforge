"""One 3D Gaussian splat fitted to the panorama and to every frame of a site's GEN3C clips at once (gsplat), in place
of tsdf_gen3c.py's per-frame depths fused in a TSDF (2026-10-20, docs/plan_gs_gen3c_2026-10-20.md).

The panorama is the truth: its cuts at the capture point weigh PANO_WEIGHT per pixel with our own depth. A GEN3C
frame teaches fully only where its cache had a hole (black in <clip>_buffer.mp4: what GEN3C had to invent) and
weighs SEEN_WEIGHT where the panorama already saw; each frame gets its own colour correction (GEN3C's exposure
drifts). Sky pixels only push the splat clear (Isaac's dome shows the sky). --plain (arm GS0) drops all of that: every pixel
weighs 1, no depth, no colour correction. One frame in HOLDOUT of each clip is left out for gs_eval.py.

Run in the gsplat env (/media/lenena-iker/Crucial/tools/envs/gsplat) after gs_moge.py:

    python experiments/sphere_walk/gs_train.py --dir data/sphere_test/<place> --out GS1 [--plain]

Writes <dir>/walk/<out>/: splat.ply (sRGB colours, for gs_usd.py), layers.json, holdout.json (the left-out
frames as views for render_walk.py --views), depth/NAME.npy (the splat's depth of every view of views.json, 0 where
it is not opaque, for post.py) and info.json."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

C0 = 0.28209479177387814
PANO_WEIGHT = 4.0      # per pixel of a panorama cut (tsdf_gen3c.py's REAL_WEIGHT)
SEEN_WEIGHT = 0.1      # a GEN3C pixel the panorama already saw: it must not compete with the photo
HOLE_WEIGHT = 1.0      # a GEN3C pixel its cache did not have: what GEN3C brings
DEPTH_SEEN = 0.5       # relative depth error against our proxy where the panorama saw
DEPTH_HOLE = 0.05      # against MoGe's depth, scaled to our proxy, in the holes: only against floaters
SKY_ALPHA = 1.0        # the splat must stay clear where the sky is (Isaac's dome shows it): unchecked, Gaussians
                       # at the clouds' edges stretched into white streaks (2026-10-20)
SSIM_SHARE = 0.2
AFFINE_REG = 1.0
PANO_SHARE = 0.25      # share of steps on a panorama cut
PSEUDO_SHARE = 0.25    # share of steps on a pseudo view (--pseudo)
PSEUDO_WEIGHT = 1.0    # a pseudo view's pixel the panorama saw (B_layers' front, off its depth edges)
SEEN_TOL = 0.04        # a pseudo pixel's surface within this share of the panorama's distance: the one it saw
EDGE_RATIO = 1.15      # the panorama's distance jumping more than this within EDGE_WIN: a depth edge (a stretched
EDGE_WIN = 7           # face of the proxy, a guess), never a pseudo truth
HOLDOUT = 10           # frame k of a clip is left out when k % HOLDOUT == HOLDOUT // 2
BLACK = 12             # a buffer pixel darker than this (0-255, mp4) is a cache hole
STEPS = 15000
SCENE_SCALE = 10.0     # metres, for gsplat's split and prune sizes
GROW_GRAD = 0.0006     # gsplat's densify threshold (its default 0.0002 grew GS1 to 7.6M Gaussians, 1.8 GB: the
                       # panorama's weight 4 multiplies the gradients)
HEADINGS = tuple(range(0, 360, 30))
PITCHES = (-35.0, 0.0, 40.0, 75.0)
CUT = {"fov_deg": 80.0, "size": [1280, 704]}
INIT_OPACITY = 0.7     # high: what no view reaches (B_layers' back layer) must stay opaque, not fade to the dome
INIT_HOLE_PTS = 4000   # MoGe points per trained frame in its holes (every 10th frame)
OPAQUE = 0.5           # alpha over which a view's pixel has the splat's depth


def ssim_map(a, b):
    """SSIM per pixel of two (3, H, W) images, 11-pixel Gaussian window."""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5 ** 2))
    win = (g[:, None] * g[None]) / g.sum() ** 2
    win = win.expand(3, 1, 11, 11)
    f = lambda x: F.conv2d(x[None], win, padding=5, groups=3)[0]  # noqa: E731
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


class Site:
    def __init__(self, d: Path):
        geo = json.loads((d / "geometry.json").read_text())
        self.R, self.h = np.array(geo["levelling"]), geo["camera_height_m"]
        pd = np.load(d / "proxy_distance.npz")
        self.pdist = np.where(pd["valid"], pd["distance"], 0).astype(np.float32)
        ph, pw = pd["valid"].shape
        seen = np.asarray(Image.open(d / "seen.png").convert("L").resize((pw, ph))) > 127
        self.sky = (seen & ~pd["valid"]).astype(np.uint8)  # post.py's rule

    def pano_lookup(self, img, d_level, nearest=True):
        d = render_plan.pano_rays(d_level, self.R)
        if not nearest:
            return capture.sample(img, d)
        x, y = capture.to_pixel(d, img.shape[1], img.shape[0])
        return img[np.clip(np.round(y).astype(int), 0, img.shape[0] - 1), np.round(x).astype(int) % img.shape[1]]

    def world_rays_level(self, c2w, W, H, f):
        u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
        d = np.stack([u / f, v / f, np.ones_like(u)], -1) @ np.asarray(c2w)[:3, :3].T
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return d @ render_plan.LEVEL_TO_WORLD


def vertex_colours(pts, faces, uv, tex):
    """Each vertex's colour from its layer's texture (sRGB 0-1), through one of its faces' UVs."""
    t = np.asarray(Image.open(tex).convert("RGB")).astype(np.float32) / 255
    per = np.zeros((len(pts), 2), np.float32)
    per[faces.reshape(-1)] = uv.reshape(-1, 2)
    H, W = t.shape[:2]
    return t[np.clip(((1 - per[:, 1]) * H).astype(int), 0, H - 1), np.clip((per[:, 0] * W).astype(int), 0, W - 1)]


def cover(params, render, holdout, ims, out):
    """--cover: each held-out frame rendered against GEN3C's, after a per-channel gain and offset fitted where the
    panorama saw (training fits each frame's colour too). sharp: the render's mean |Laplacian| over GEN3C's, the
    blur; clear: the share of its pixels the splat leaves under OPAQUE alpha."""
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False).cuda()
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device="cuda")[None, None]

    def sharpness(x, m):
        e = F.conv2d(x.mean(0)[None, None], lap, padding=1)[0, 0].abs()
        return float(e[m].mean())

    def psnr(a, b, m):
        return float(-10 * torch.log10(((a - b) ** 2).mean(0)[m].mean().clamp(min=1e-10))) if m.sum() > 2000 else None

    rows = []
    for h in holdout:
        rgb, dep, alpha, _ = render(params, h)
        pred = rgb.permute(2, 0, 1).clamp(0, 1)
        z = dep[(alpha > OPAQUE) & (torch.as_tensor(h["code"], device="cuda") > 0)]
        z = z[torch.randperm(len(z), device="cuda")[:100000]]  # quantile's input limit
        zq = torch.quantile(z, torch.tensor([0.1, 0.5], device="cuda"))
        gt = torch.as_tensor(h["img"], device="cuda").float().div(255).permute(2, 0, 1)
        code = torch.as_tensor(h["code"], device="cuda")
        seen, hole, solid = code == 1, code == 2, code > 0
        if seen.sum() > 500:
            for c in range(3):
                A = torch.stack([pred[c][seen], torch.ones_like(pred[c][seen])], -1)
                sol = torch.linalg.lstsq(A, gt[c][seen][:, None]).solution[:, 0]
                pred[c] = (pred[c] * sol[0] + sol[1]).clamp(0, 1)
        inner = torch.as_tensor(ndimage.binary_erosion(h["code"] > 0, iterations=3), device="cuda")
        t = lambda x: (torch.where(solid, x, 0) * 2 - 1)[None]  # noqa: E731
        rows.append({"name": h["name"], "span": h["span"], "c2w": h["c2w"].tolist(), "fov_deg": h["fov_deg"],
                     "hole_share": float(hole.float().mean()), "psnr_all": psnr(pred, gt, solid),
                     "psnr_hole": psnr(pred, gt, hole), "psnr_seen": psnr(pred, gt, seen),
                     "lpips": float(net(t(pred), t(gt)).item() / max(float(solid.float().mean()), 1e-3)),
                     "sharp": sharpness(pred, inner) / max(sharpness(gt, inner), 1e-6),
                     "clear": float((alpha[solid] < OPAQUE).float().mean()),
                     "depth_near": float(zq[0]), "depth_med": float(zq[1])})
    train = [{"kind": im["kind"], "c2w": np.asarray(im["c2w"]).tolist()} for im in ims]
    (out / "cover.json").write_text(json.dumps({"held": rows, "train": train}))
    print(f"cover: {len(rows)} held-out frames, {sum(r['span'] for r in rows)} in spans", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--clips", nargs="+", default=["N01"], help="folders under walk with paths.json and the videos")
    ap.add_argument("--out", default="GS1")
    ap.add_argument("--plain", action="store_true", help="arm GS0: every pixel weighs 1, no depth, no colour fix")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--grow", type=float, default=GROW_GRAD)
    ap.add_argument("--pseudo", help="arm GS2: a folder under walk with B_layers' Isaac renders of gs_pseudo.py's "
                    "views (render_walk.py --views pseudo.json); their pixels the panorama saw teach the splat")
    ap.add_argument("--hold", action="append", default=[], help="also leave out clip I (its index over --clips' "
                    "paths) whole, or I:A-B its frames A to B: how far from its training cameras the splat holds")
    ap.add_argument("--cover", action="store_true", help="measure the held-out frames in gsplat (cover.json for "
                    "cover_report.py) instead of writing the splat and the walk depths")
    args = ap.parse_args()
    hold = {}
    for s in args.hold:
        ci, _, span = s.partition(":")
        a, _, b = span.partition("-")
        hold.setdefault(int(ci), []).append((int(a), int(b)) if span else (0, 10 ** 6))
    torch.manual_seed(0)
    np.random.seed(0)
    d = Path(args.dir)
    w = d / "walk"
    out = w / args.out
    (out / "depth").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    site = Site(d)
    dev = "cuda"

    # Training images. code: 0 sky (never counts), 1 seen by the panorama, 2 a cache hole, 3 a panorama cut's pixel,
    # 4 a pseudo view's pixel the panorama saw, 5 another pseudo pixel (no truth, no sky: never counts).
    ims = []
    tex = np.asarray(Image.open(w / "BL_scene" / "tex_front.png").convert("RGB"))
    for hd in HEADINGS:
        for pitch in PITCHES:
            v = {**CUT, "yaw_deg": float(hd), "pitch_deg": pitch}
            W, H = v["size"]
            rays = capture.view_rays(v)
            img = np.clip(site.pano_lookup(tex, rays, nearest=False) + 0.5, 0, 255).astype(np.uint8)
            sky = site.pano_lookup(site.sky, rays) > 0
            z = site.pano_lookup(site.pdist, rays) * (rays @ capture.rotation(v["yaw_deg"], v["pitch_deg"]))[..., 2]
            ims.append({"kind": "pano", "img": img, "c2w": np.array(render_plan.camera_pose(v, site.h), np.float32),
                        "f": capture.focal(W, v["fov_deg"]), "code": np.where(sky, 0, 3).astype(np.uint8),
                        "depth": z.astype(np.float32), "name": f"pano_y{hd:03d}_p{pitch:+.0f}"})
    n_pano = len(ims)
    holdout = []
    ci = -1
    for folder in args.clips:
        for p in json.loads((w / folder / "paths.json").read_text())["paths"]:
            vid = w / folder / f"{p['name']}.mp4"
            if not vid.exists():
                continue
            ci += 1
            W, H = p["view"]["size"]
            f = capture.focal(W, p["view"]["fov_deg"])
            frames = iio.imread(vid)
            buf = iio.imread(vid.with_name(p["name"] + "_buffer.mp4"))
            moge = np.load(vid.with_name(p["name"] + "_moge.npz"))["depth"].astype(np.float32)
            for k in range(1, len(frames)):  # frame 0 is our own render
                c2w = np.array(p["poses"][k], np.float32)
                name = f"{p['name']}_f{k:03d}"
                hole = ndimage.binary_dilation(buf[k].max(-1) < BLACK, iterations=2)
                sky = site.pano_lookup(site.sky, site.world_rays_level(c2w, W, H, f)) > 0
                code = np.where(sky, 0, np.where(hole, 2, 1)).astype(np.uint8)
                span = any(a <= k <= b for a, b in hold.get(ci, []))
                if k % HOLDOUT == HOLDOUT // 2 or span:
                    holdout.append({"name": name, "img": frames[k], "code": code, "c2w": c2w, "f": f,
                                    "fov_deg": p["view"]["fov_deg"], "size": [W, H], "span": span})
                    continue
                ims.append({"kind": "gen3c", "img": frames[k], "c2w": c2w, "f": f, "code": code,
                            "moge": moge[k], "name": name})
    n_gen = len(ims) - n_pano
    print(f"{n_pano} cuts, {n_gen} GEN3C frames, {len(holdout)} held out, {time.time() - t0:.0f} s", flush=True)

    # Initial Gaussians: B_layers' front (the panorama's colours) and back (LaMa) vertices.
    sc = np.load(w / "BL_scene" / "scene.npz")
    pts, cols = [], []
    for layer in ("front", "back"):
        pts.append(sc[f"{layer}_pts"])
        cols.append(vertex_colours(sc[f"{layer}_pts"], sc[f"{layer}_faces"], sc[f"{layer}_uv"],
                                   w / "BL_scene" / f"tex_{layer}.png"))

    def K_of(im):
        H, W = im["img"].shape[:2]
        return torch.tensor([[im["f"], 0, W / 2], [0, im["f"], H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)

    def render(params, im, colours=None):
        H, W = im["img"].shape[:2]
        viewmat = torch.linalg.inv(torch.as_tensor(im["c2w"], device=dev))[None]
        rgbd, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), params["sh0"] if colours is None else colours, viewmat, K_of(im)[None],
            W, H, sh_degree=0 if colours is None else None, render_mode="RGB+ED", near_plane=0.05, far_plane=500,
            packed=False)
        return rgbd[0, ..., :3], rgbd[0, ..., 3], alpha[0, ..., 0], info

    # Our proxy's depth in each GEN3C frame (the front layer as opaque splats), for the depth loss and MoGe's scale.
    fp = torch.as_tensor(pts[0], dtype=torch.float32, device=dev)
    dist, _ = cKDTree(pts[0]).query(pts[0], k=4)
    fs = torch.as_tensor(np.log(np.clip(dist[:, 1:].mean(1), 0.003, 0.5)), dtype=torch.float32, device=dev)
    front = {"means": fp, "quats": torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(len(fp), 4),
             "scales": fs[:, None].expand(len(fp), 3), "opacities": torch.full((len(fp),), 5.0, device=dev),
             "sh0": torch.zeros(len(fp), 1, 3, device=dev)}
    hole_pts, hole_cols, scales_k = [], [], []
    for i, im in enumerate(ims[n_pano:]):
        with torch.no_grad():
            _, dep, alpha, _ = render(front, im)
        prox = torch.where(alpha > 0.9, dep, torch.zeros_like(dep)).cpu().numpy()
        H, W = prox.shape
        mg = np.asarray(Image.fromarray(im["moge"]).resize((W, H), Image.Resampling.NEAREST))
        ok = (im["code"] == 1) & (prox > 0) & (mg > 0) & (prox < 40)
        k = float(np.median(prox[ok] / mg[ok])) if ok.sum() > 500 else 1.0
        scales_k.append(k)
        im["depth"] = np.where(im["code"] == 1, prox, np.where(im["code"] == 2, mg * k, 0)).astype(np.float32)
        del im["moge"]
        if i % 10 == 0:  # GEN3C's hole pixels as points, where B_layers has nothing to offer
            yy, xx = np.nonzero((im["code"] == 2) & (im["depth"] > 0))
            if len(yy):
                s = np.random.choice(len(yy), min(INIT_HOLE_PTS, len(yy)), replace=False)
                yy, xx = yy[s], xx[s]
                z = im["depth"][yy, xx]
                cam = np.stack([(xx + 0.5 - W / 2) / im["f"] * z, (yy + 0.5 - H / 2) / im["f"] * z, z], -1)
                hole_pts.append(cam @ im["c2w"][:3, :3].T + im["c2w"][:3, 3])
                hole_cols.append(im["img"][yy, xx].astype(np.float32) / 255)
    del front
    torch.cuda.empty_cache()
    n_pseudo = 0
    if args.pseudo:
        gain = np.array(json.loads((w / "post_B_layers.json").read_text())["gain"])
        lo, hi = ndimage.minimum_filter(site.pdist, EDGE_WIN, mode="wrap"), ndimage.maximum_filter(site.pdist, EDGE_WIN,
                                                                                                   mode="wrap")
        edge = ((hi > lo * EDGE_RATIO) | (lo <= 0)).astype(np.uint8)
        cpt = np.array([0.0, 0.0, site.h])
        for v in json.loads((w / "pseudo.json").read_text())["views"]:
            raw = w / args.pseudo
            lin = np.load(raw / f"lin_{v['name']}.npy").astype(np.float32)
            z = np.load(raw / f"depth_{v['name']}.npy").astype(np.float32)
            W, H = v["size"]
            f = capture.focal(W, v["fov_deg"])
            c2w = np.array(v["pose"], np.float32)
            u, vv = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
            P = np.stack([u / f * z, vv / f * z, z], -1) @ c2w[:3, :3].T + c2w[:3, 3]
            ray = P - cpt
            rng_ = np.linalg.norm(ray, axis=-1)
            d_level = (ray / np.maximum(rng_[..., None], 1e-6)) @ render_plan.LEVEL_TO_WORLD
            meas = site.pano_lookup(site.pdist, d_level)
            ok = (z > 0) & (np.abs(rng_ - meas) < SEEN_TOL * meas) & (site.pano_lookup(edge, d_level) == 0)
            srgb = np.clip(lin * gain, 0, 1)
            srgb = np.where(srgb <= 0.0031308, 12.92 * srgb, 1.055 * srgb ** (1 / 2.4) - 0.055)
            sky = (z <= 0) & (site.pano_lookup(site.sky, site.world_rays_level(c2w, W, H, f)) > 0)
            ims.append({"kind": "pseudo", "img": (srgb * 255 + 0.5).astype(np.uint8), "c2w": c2w, "f": f,
                        "code": np.where(ok, 4, np.where(sky, 0, 5)).astype(np.uint8), "depth": np.where(ok, z, 0).astype(np.float16),
                        "name": v["name"]})
            n_pseudo += 1
        print(f"{n_pseudo} pseudo views, {np.mean([(im['code'] == 4).mean() for im in ims[-n_pseudo:]]):.2f} of "
              f"their pixels trusted", flush=True)
    print(f"proxy depths, MoGe scale median {np.median(scales_k):.2f} ({np.min(scales_k):.2f}-{np.max(scales_k):.2f}), "
          f"{time.time() - t0:.0f} s", flush=True)
    P = np.concatenate(pts + hole_pts).astype(np.float32)
    C = np.concatenate(cols + hole_cols).astype(np.float32)
    dist, _ = cKDTree(P).query(P, k=4)
    N = len(P)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.as_tensor(P, device=dev)),
        "scales": torch.nn.Parameter(torch.as_tensor(np.log(np.clip(dist[:, 1:].mean(1), 0.003, 0.5)),
                                                     dtype=torch.float32, device=dev)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.full((N,), math.log(INIT_OPACITY / (1 - INIT_OPACITY)), device=dev)),
        "sh0": torch.nn.Parameter(torch.as_tensor((C - 0.5) / C0, device=dev)[:, None, :]),
    })
    lrs = {"means": 1.6e-4 * SCENE_SCALE, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.steps))
    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(args.steps * 0.66), refine_every=100,
                               reset_every=10 ** 9, grow_grad2d=args.grow, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=SCENE_SCALE)
    affine = torch.nn.Parameter(torch.eye(3, 4, device=dev).repeat(n_gen, 1, 1))
    aff_opt = torch.optim.Adam([affine], lr=1e-3)
    print(f"{N} Gaussians to start", flush=True)

    weight_of = torch.tensor([0.0, SEEN_WEIGHT, HOLE_WEIGHT, PANO_WEIGHT, PSEUDO_WEIGHT, 0.0], device=dev)
    if args.plain:
        weight_of = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], device=dev)
    for step in range(args.steps):
        r = np.random.rand()
        if r < PANO_SHARE:
            i = np.random.randint(n_pano)
        elif r < PANO_SHARE + PSEUDO_SHARE and n_pseudo:
            i = n_pano + n_gen + np.random.randint(n_pseudo)
        else:
            i = n_pano + np.random.randint(n_gen)
        im = ims[i]
        gt = torch.as_tensor(im["img"], device=dev).float().div(255).permute(2, 0, 1)
        code = torch.as_tensor(im["code"], device=dev).long()
        wpx = weight_of[code]
        rgb, dep, alpha, info = render(params, im)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = rgb.permute(2, 0, 1)
        if im["kind"] == "gen3c" and not args.plain:
            a = affine[i - n_pano]
            pred = torch.einsum("ij,jhw->ihw", a[:, :3], pred) + a[:, 3, None, None]
        l1 = (wpx * (pred - gt).abs().mean(0)).mean()
        ss = (wpx * (1 - ssim_map(pred, gt).mean(0))).mean()
        loss = (1 - SSIM_SHARE) * l1 + SSIM_SHARE * ss + SKY_ALPHA * (alpha * (code == 0)).mean()
        if not args.plain:
            tgt = torch.as_tensor(im["depth"], device=dev).float()
            dw = torch.where((code == 1) | (code == 3) | (code == 4), DEPTH_SEEN, torch.where(code == 2, DEPTH_HOLE, 0.0)) * (tgt > 0)
            loss = loss + (dw * (dep - tgt).abs() / tgt.clamp(min=0.1)).mean()
            loss = loss + AFFINE_REG * ((affine - torch.eye(3, 4, device=dev)) ** 2).mean()
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        aff_opt.step()
        aff_opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"step {step} loss {loss.item():.4f} n {len(params['means'])} {time.time() - t0:.0f} s", flush=True)

    if args.cover:
        with torch.no_grad():
            cover(params, render, holdout, ims, out)
        return
    with torch.no_grad():
        # Every view of the walk test: the splat's depth for post.py (Isaac's NuRec gives colour only).
        plan = json.loads((w / "views.json").read_text())
        for v in plan["views"]:
            W, H = v["size"]
            vim = {"img": np.zeros((H, W, 3), np.uint8), "c2w": np.array(v["pose"], np.float32),
                   "f": capture.focal(W, v["fov_deg"])}
            rgb, dep, alpha, _ = render(params, vim)
            np.save(out / "depth" / f"{v['name']}.npy", torch.where(alpha > OPAQUE, dep, 0).cpu().numpy()
                    .astype(np.float16))
            if v["move"] in ("0", "f3", "f5", "l3", "r3", "b3"):
                Image.fromarray((rgb.clamp(0, 1) * 255).byte().cpu().numpy()).save(out / "depth" / f"{v['name']}.jpg")
        # The splat for Isaac: degree 0, its colour as sRGB (NuRec linearises it itself: linear colours came out
        # with crushed shadows and burnt lights, 2026-10-20).
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
    hv = [{"name": h["name"], "size": h["size"], "fov_deg": h["fov_deg"], "pose": h["c2w"].tolist(), "move": "H",
           "heading": 0, "clip_frame": h["name"]} for h in holdout]
    (out / "holdout.json").write_text(json.dumps({"camera_height_m": site.h, "views": hv}))
    (out / "layers.json").write_text(json.dumps([{"name": "splat", "usdz": "splat.usdz"}]))
    info = {"arm": "plain" if args.plain else "photosphere rules" + (" + pseudo views" if args.pseudo else ""),
            "pseudo_views": n_pseudo, "steps": args.steps, "gaussians": len(m),
            "cuts": n_pano, "gen3c_frames": n_gen, "held_out": len(holdout), "moge_scale": np.median(scales_k),
            "seconds": round(time.time() - t0)}
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
    print(info, flush=True)


if __name__ == "__main__":
    main()
