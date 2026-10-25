"""Next-best-view for a WorldStereo scene (the sphere's nbv.py, ported): phase 1 walks the paths (ws_test.py), its
frames are fitted into a splat (ws_splat.py, arm G2), and phase 2 plans the legs that look where phase 1 never did.

- audit: the student's cameras (a 1 m grid within STUDENT_RADIUS_M, 8 headings) render the splat; a pixel is known
  when some view WorldStereo made (or a panorama cut) saw its surface point unoccluded from within PARALLAX_MAX_DEG
  of the student's direction, or when it lies on the ground. The rest are holes: kept as (voxel, direction bin)
  keys weighted by how many student pixels need them, and grouped from above into clusters (what one car hides).
- plan: candidate legs start on a frame WorldStereo already made (its image is the leg's start frame: sharp and
  consistent) and move sideways, diagonally or back, or arc round a cluster, turning at most MAX_TURN_DEG over the
  leg. A leg must stay in the area, clear of obstacles, and no frame may be mostly unknown. Each is scored by the
  hole keys its frames see; a greedy pick by gain keeps --legs of them. The picked legs are written as WorldStereo
  inputs: start_frame.png, camera.json, the splat's render as guide with the unknown pixels as holes (render_mask),
  the caption of the leg it starts from.

    <gsplat env>/bin/python experiments/worldstereo/ws_nbv.py audit --case photo21 --arm G2
    <gsplat env>/bin/python experiments/worldstereo/ws_nbv.py plan  --case photo21 --arm G2 [--legs 8]
    <gsplat env>/bin/python experiments/worldstereo/ws_nbv.py stage --case photo21   # legs into the scene
    <gsplat env>/bin/python experiments/worldstereo/ws_nbv.py guide --case photo21 --arm G2 --src spin_ --dst spg_
        (the same legs, renamed, guided by the splat's render: hole only where it has nothing)
    then ws_test.py launch --resume (only the new legs are generated, after phase 1's are in memory)

Writes data/worldstereo_test/<case>/nbv/: audit.npz, audit.json, sheet_audit.jpg, plan.json, sheet_plan.jpg and
legs/<leg>/traj0/ (WorldStereo's input layout; the runner must generate them after phase 1's legs)."""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_splat as wsp  # noqa: E402
import ws_test as ws  # noqa: E402

# The student's cameras: where a synthetic camera may stand (the spec's area), at the walk's height.
STUDENT_RADIUS_M = 5.0
STUDENT_GRID_M = 1.0
STUDENT_HEADINGS = tuple(range(0, 360, 45))
STUDENT_PITCH_DEG = -15.0
STUDENT_FOV_DEG = 68.0
STUDENT_SIZE = (128, 96)
CLEARANCE_M = 0.5      # a camera this near an obstacle point is inside a hedge or a car
MIN_VIEW_DEPTH_M = 2.0  # a student view whose median depth is nearer faces a hedge from arm's length: left out
# Known: seen by a made view from within PARALLAX_MAX_DEG (nbv.py's guess), not hidden behind its depth.
PARALLAX_MAX_DEG = 20.0
OCCLUSION_REL = 0.05
OCCLUSION_ABS_M = 0.1
GROUND_M = 0.15        # this near the ground plane holds from anywhere (nbv.py, 2026-10-18)
TRAIN_SCALE = 4        # made views' depth maps at 1/4 for the occlusion test
ALPHA_MIN = 0.5
VOXEL_M = 0.25
BIN_DEG = 20.0         # hole keys: voxel and the student's direction to it, in BIN_DEG azimuth bins
CLUSTER_M = 0.5
# Legs: 21 frames like phase 1's, from a made frame.
START_EVERY = 2        # every 2nd frame of each phase-1 leg is a start
MOVES_DEG = (90, -90, 45, -45, 135, -135, 180)  # direction of travel against the start's heading
MOVE_M = (2.0, 3.5)
TOP_CLUSTERS = 6
ARC_M = 3.5
ARC_RANGE_M = (2.0, 9.0)
MAX_TURN_DEG = 60.0    # over the leg: about 3 deg per frame, the user found phase 1's 9 too fast (2026-10-22)
AREA_M = 6.0
SCORE_FRAMES = (0, 5, 10, 15, 20)
SCORE_SCALE = 4
FRAME_MAX_UNKNOWN = 0.6
MIN_GAIN = 0.002
LEGS = 8
SPOTS = None          # [(x, y)]: only the student cameras at these grid spots (every heading) set the targets
TARGETS = "parallax"   # or "conflict": holes and no-data pixels of ws_conflict.py, guide holes only there


def load_splat(path: Path, dev):
    import torch
    import torch.nn.functional as F
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    g = lambda *k: torch.as_tensor(np.stack([np.asarray(v[x], np.float32) for x in k], -1), device=dev)  # noqa: E731
    return dict(means=g("x", "y", "z"), quats=F.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1),
                scales=torch.exp(g("scale_0", "scale_1", "scale_2")), opacities=torch.sigmoid(g("opacity")[:, 0]),
                colors=g("f_dc_0", "f_dc_1", "f_dc_2")[:, None])


def render(sp, w2cs: np.ndarray, K: np.ndarray, w: int, h: int, batch: int = 16):
    """RGB (uint8), z-depth and alpha of the splat for each world-to-camera."""
    import torch
    dev = sp["means"].device
    rgbs, deps, alphas = [], [], []
    for i in range(0, len(w2cs), batch):
        m = torch.as_tensor(np.asarray(w2cs[i:i + batch]), dtype=torch.float32, device=dev)
        k = torch.as_tensor(K, dtype=torch.float32, device=dev)[None].repeat(len(m), 1, 1)
        with torch.no_grad():
            out, alpha, _ = wsp_rasterize(sp, m, k, w, h)
        rgbs.append((out[..., :3].clamp(0, 1) * 255).byte())
        deps.append(out[..., 3])
        alphas.append(alpha[..., 0])
    return torch.cat(rgbs), torch.cat(deps), torch.cat(alphas)


def wsp_rasterize(sp, m, k, w, h):
    from gsplat import rasterization
    return rasterization(**sp, viewmats=m, Ks=k, width=w, height=h, sh_degree=0, render_mode="RGB+ED",
                         near_plane=0.05, far_plane=500, packed=False)


def intrinsic(w: int, h: int, fov_deg: float) -> np.ndarray:
    f = w / 2 / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def pose_of(m: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Position, yaw and pitch (degrees, ws_test.w2c's convention) of a world-to-camera."""
    c2w = np.linalg.inv(m)
    fwd = m[2, :3]
    return c2w[:3, 3], math.degrees(math.atan2(fwd[1], fwd[0])), math.degrees(math.asin(np.clip(fwd[2], -1, 1)))


def lift(dep, K: np.ndarray, w2cs: np.ndarray):
    """World points (N, H, W, 3) of z-depth maps."""
    import torch
    dev = dep.device
    n, h, w = dep.shape
    j, i = torch.meshgrid(torch.arange(w, device=dev) + 0.5, torch.arange(h, device=dev) + 0.5, indexing="xy")
    cam = torch.stack([(j - K[0, 2]) / K[0, 0] * dep, (i - K[1, 2]) / K[1, 1] * dep, dep], -1)
    c2w = torch.as_tensor(np.linalg.inv(np.asarray(w2cs)), dtype=torch.float32, device=dev)
    return torch.einsum("nij,nhwj->nhwi", c2w[:, :3, :3], cam) + c2w[:, None, None, :3, 3]


class Made:
    """The views WorldStereo made (and the panorama's cuts): centres, world-to-cameras and splat depth maps for the
    known test."""

    def __init__(self, sp, case: str, arm: str):
        import torch
        z = np.load(ws.DATA / case / "splat" / wsp.ARMS[arm].get("views", "views.npz"))
        self.w2c, self.name = z["w2c"], [str(s) for s in z["name"]]
        self.img = z["img"]
        self.w, self.h = ws.W // TRAIN_SCALE, ws.H // TRAIN_SCALE
        self.K = np.array(ws.intrinsic(), np.float32)
        self.K[:2] /= TRAIN_SCALE
        _, dep, alpha = render(sp, self.w2c, self.K, self.w, self.h)
        self.dep = torch.where(alpha > ALPHA_MIN, dep, torch.full_like(dep, 1e6))
        dev = dep.device
        self.R = torch.as_tensor(self.w2c[:, :3, :3], device=dev)
        self.t = torch.as_tensor(self.w2c[:, :3, 3], device=dev)
        self.C = torch.as_tensor(np.linalg.inv(self.w2c)[:, :3, 3], device=dev)

    def known(self, X, from_pos, chunk: int = 4096):
        """(N,) True where a made view saw point X (N, 3) unoccluded within PARALLAX_MAX_DEG of the direction
        from from_pos (N, 3)."""
        import torch
        cos_max = math.cos(math.radians(PARALLAX_MAX_DEG))
        out = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        Kx, Ky, cx, cy = float(self.K[0, 0]), float(self.K[1, 1]), float(self.K[0, 2]), float(self.K[1, 2])
        for s in range(0, len(X), chunk):
            x, p = X[s:s + chunk], from_pos[s:s + chunk]
            v = torch.nn.functional.normalize(p - x, dim=-1)
            cam = torch.einsum("vij,nj->vni", self.R, x) + self.t[:, None]
            z = cam[..., 2]
            u = (cam[..., 0] / z.clamp(min=1e-3) * Kx + cx).long()
            q = (cam[..., 1] / z.clamp(min=1e-3) * Ky + cy).long()
            ok = (z > 0.1) & (u >= 0) & (u < self.w) & (q >= 0) & (q < self.h)
            d = self.dep[torch.arange(len(self.R), device=X.device)[:, None], q.clamp(0, self.h - 1),
                         u.clamp(0, self.w - 1)]
            vis = ok & (z <= d * (1 + OCCLUSION_REL) + OCCLUSION_ABS_M)
            vm = torch.nn.functional.normalize(self.C[:, None] - x[None], dim=-1)
            out[s:s + chunk] = (vis & ((vm * v[None]).sum(-1) >= cos_max)).any(0)
        return out


def obstacles(case: str, sp) -> np.ndarray:
    """Points (x, y) standing between ws_test.OBSTACLE_Z above the ground: the panorama's and the splat's opaque
    Gaussians (what phase 1 added)."""
    pts, *_ = wsp.panorama_points(case)
    m = sp["means"].cpu().numpy()
    m = m[sp["opacities"].cpu().numpy() > 0.5]
    every = np.concatenate([pts, m[::5]])
    hgt = every[:, 2] + ws.CAMERA_HEIGHT
    keep = (hgt > ws.OBSTACLE_Z[0]) & (hgt < ws.OBSTACLE_Z[1])
    return every[keep][:, :2]


def keys_of(X, from_pos):
    """Hole keys: voxel of X and the azimuth bin of the direction from from_pos to X, packed in one int64."""
    import torch
    vox = torch.floor(X / VOXEL_M).long() + 2 ** 15
    d = X - from_pos
    b = (torch.rad2deg(torch.atan2(d[:, 1], d[:, 0])) % 360 // BIN_DEG).long()
    return ((vox[:, 0] * 2 ** 16 + vox[:, 1]) * 2 ** 16 + vox[:, 2]) * 32 + b


def unpack(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = keys % 32
    v = keys // 32
    vox = np.stack([v // 2 ** 32, (v // 2 ** 16) % 2 ** 16, v % 2 ** 16], -1) - 2 ** 15
    return (vox + 0.5) * VOXEL_M, b


def classify(sp, made: Made, w2cs: np.ndarray, K: np.ndarray, w: int, h: int):
    """Per pixel of each view: 0 sky, 1 ground, 2 known, 3 hole (a surface no made view explains, or nothing below
    the horizon); plus the hole keys and the RGB."""
    import torch
    if not isinstance(made, Made):  # ws_conflict.Evidence: targets by WorldStereo's own frames
        import ws_conflict
        return ws_conflict.nbv_classify(sp, made, w2cs, K, w, h)
    rgb, dep, alpha = render(sp, w2cs, K, w, h)
    X = lift(dep, K, w2cs)
    pos = torch.as_tensor(np.linalg.inv(np.asarray(w2cs))[:, :3, 3], dtype=torch.float32, device=dep.device)
    P = pos[:, None, None].expand_as(X)
    ray = X - P
    below = ray[..., 2] < 0
    surf = alpha > ALPHA_MIN
    ground = surf & (X[..., 2] < -ws.CAMERA_HEIGHT + GROUND_M)
    code = torch.zeros(dep.shape, dtype=torch.uint8, device=dep.device)
    code[~surf & below] = 3
    code[ground] = 1
    test = surf & ~ground
    kn = made.known(X[test], P[test])
    sub = torch.full((int(test.sum()),), 3, dtype=torch.uint8, device=dep.device)
    sub[kn] = 2
    code[test] = sub
    hole = surf & (code == 3)
    keys = keys_of(X[hole], P[hole])
    return code, keys, rgb, X, surf


def free_space(obs: np.ndarray, cell: float = 0.25, span: float = 12.0):
    """Where a camera can stand: cells CLEARANCE_M clear of obstacle points and reachable from the capture point
    without crossing them (a hedge's far side is not the street). Returns a test function on (x, y)."""
    n = int(2 * span / cell)
    occ = np.zeros((n, n), bool)
    ij = np.floor((obs + span) / cell).astype(int)
    ij = ij[(ij >= 0).all(1) & (ij < n).all(1)]
    occ[ij[:, 0], ij[:, 1]] = True
    r = int(math.ceil(CLEARANCE_M / cell))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    occ = ndimage.binary_dilation(occ, structure=(xx ** 2 + yy ** 2) <= r * r)
    lab, _ = ndimage.label(~occ)
    c = int(span / cell)
    free = lab == lab[c, c] if lab[c, c] else np.zeros_like(occ)

    def ok(xy) -> np.ndarray:
        q = np.floor((np.atleast_2d(xy) + span) / cell).astype(int)
        inside = (q >= 0).all(1) & (q < n).all(1)
        out = np.zeros(len(q), bool)
        out[inside] = free[q[inside, 0], q[inside, 1]]
        return out
    return ok


def student_poses(obs: np.ndarray) -> tuple[np.ndarray, list]:
    free = free_space(obs)
    r = np.arange(-STUDENT_RADIUS_M, STUDENT_RADIUS_M + 1e-6, STUDENT_GRID_M)
    where = [(float(x), float(y)) for x in r for y in r if math.hypot(x, y) <= STUDENT_RADIUS_M + 1e-6
             and free([x, y])[0]]
    poses, out = [], []
    for x, y in where:
        for hd in STUDENT_HEADINGS:
            poses.append(ws.w2c([x, y, 0.0], hd, STUDENT_PITCH_DEG).astype(np.float32))
            out.append((x, y, hd))
    return np.stack(poses), out


def nbv_dir(case: str, arm: str = "G2") -> Path:
    d = ws.DATA / case / (("nbv" if arm == "G2" else f"nbv_{arm}") + ("_cf" if TARGETS == "conflict" else ""))
    d.mkdir(exist_ok=True)
    return d


def made_for(sp, case: str, arm: str):
    if TARGETS == "conflict":
        import ws_conflict
        return ws_conflict.Evidence(case, sp)
    return Made(sp, case, arm)


def audit(case: str, arm: str):
    import torch
    t0 = time.time()
    dev = torch.device("cuda")
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", dev)
    made = made_for(sp, case, arm)
    obs = obstacles(case, sp)
    poses, where = student_poses(obs)
    if SPOTS:
        sel = [k for k, (x, y, _) in enumerate(where) if (x, y) in SPOTS]
        poses, where = poses[sel], [where[k] for k in sel]
    K = intrinsic(*STUDENT_SIZE, STUDENT_FOV_DEG)
    _, dep, alpha = render(sp, poses, K, *STUDENT_SIZE)
    med = torch.stack([d[a > ALPHA_MIN].median() if (a > ALPHA_MIN).any() else d.new_tensor(1e6)
                       for d, a in zip(dep, alpha)]).cpu().numpy()
    keep = med >= MIN_VIEW_DEPTH_M
    poses, where = poses[keep], [w for w, k in zip(where, keep) if k]
    shares, keys, codes, rgbs = [], [], [], []
    for s in range(0, len(poses), 32):
        code, k, rgb, _, _ = classify(sp, made, poses[s:s + 32], K, *STUDENT_SIZE)
        shares += (code == 3).float().mean((1, 2)).tolist()
        keys.append(k.cpu().numpy())
        codes.append(code.cpu().numpy())
        rgbs.append(rgb.cpu().numpy())
    keys = np.concatenate(keys)
    total = len(poses) * STUDENT_SIZE[0] * STUDENT_SIZE[1]
    uk, cnt = np.unique(keys, return_counts=True)
    codes, rgbs = np.concatenate(codes), np.concatenate(rgbs)
    d = nbv_dir(case, arm)
    np.savez_compressed(d / "audit.npz", keys=uk, weight=cnt / total, poses=poses, where=np.array(where),
                        shares=np.array(shares), obstacles=obs.astype(np.float32))
    cl = clusters(uk, cnt / total)
    info = {"arm": arm, "views": len(poses), "facing_hedge": int((~keep).sum()),
            "positions": len({(x, y) for x, y, _ in where}),
            "hole": float(np.mean(shares)), "sky": float((codes == 0).mean()), "ground": float((codes == 1).mean()),
            "known": float((codes == 2).mean()), "keys": int(len(uk)), "clusters": cl[:10],
            "seconds": round(time.time() - t0)}
    (d / "audit.json").write_text(json.dumps(info, indent=1))
    audit_sheet(d, case, info, made, obs, uk, cnt / total, cl, where, shares, codes, rgbs)
    print(json.dumps({k: v for k, v in info.items() if k != "clusters"}), flush=True)


def clusters(keys: np.ndarray, weight: np.ndarray) -> list[dict]:
    """Hole voxels grouped from above (CLUSTER_M, 8-connected), heaviest first (nbv.py's)."""
    if not len(keys):
        return []
    c, _ = unpack(keys)
    ij = np.floor(c[:, :2] / CLUSTER_M).astype(int)
    lo = ij.min(0)
    grid = np.zeros(ij.max(0) - lo + 1, bool)
    grid[tuple((ij - lo).T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    which = lab[tuple((ij - lo).T)]
    out = []
    for k in range(1, n + 1):
        sel = which == k
        w = weight[sel]
        p = (c[sel] * w[:, None]).sum(0) / w.sum()
        out.append({"centre": p.round(2).tolist(), "weight": float(w.sum()), "keys": int(sel.sum())})
    return sorted(out, key=lambda q: -q["weight"])


def top_view(obs: np.ndarray, paths: list, keys=None, weight=None, cl=None, picked=None, size: int = 800,
             span: float = 20.0):
    """20 m square from above: x (the panorama's centre) up, y (left) left. Obstacles grey, phase-1 cameras blue,
    hole weight red, clusters numbered, picked legs coloured."""
    im = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(im)
    px = lambda x, y: (size / 2 - y / span * size, size / 2 - x / span * size)  # noqa: E731
    r = STUDENT_RADIUS_M / span * size
    d.ellipse([size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r], outline=(90, 90, 90))
    for x, y in obs[::3]:
        d.point(px(x, y), fill=(130, 130, 130))
    if keys is not None:
        c, _ = unpack(keys)
        acc = {}
        for (x, y, _), w in zip(c, weight):
            k = (math.floor(x / VOXEL_M), math.floor(y / VOXEL_M))
            acc[k] = acc.get(k, 0) + w
        top = max(acc.values()) if acc else 1
        for (i, j), w in acc.items():
            a = min(1.0, (w / top) ** 0.5)
            x, y = px((i + 0.5) * VOXEL_M, (j + 0.5) * VOXEL_M)
            d.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(int(80 + 175 * a), int(40 * (1 - a)), int(40 * (1 - a))))
    for p in paths:
        d.line([px(*q) for q in p], fill=(80, 140, 255), width=2)
    for k, q in enumerate((cl or [])[:TOP_CLUSTERS]):
        x, y = px(*q["centre"][:2])
        d.text((x + 4, y - 8), str(k), fill=(255, 255, 0))
    cols = [(0, 255, 120), (255, 160, 0), (255, 0, 255), (0, 220, 255), (255, 255, 0), (160, 255, 0),
            (255, 90, 90), (200, 160, 255)]
    for k, c in enumerate(picked or []):
        col = cols[k % len(cols)]
        pts = [px(*q) for q in c["xy"]]
        d.line(pts, fill=col, width=3)
        d.ellipse([pts[0][0] - 4, pts[0][1] - 4, pts[0][0] + 4, pts[0][1] + 4], fill=col)
        for f in (0, len(c["xy"]) - 1):
            x, y = pts[f]
            a = math.radians(c["yaw"][f])
            d.line([(x, y), (x - 18 * math.sin(a), y - 18 * math.cos(a))], fill=col, width=2)
        d.text((pts[-1][0] + 5, pts[-1][1]), str(k), fill=col)
    return im


def phase1_paths(case: str, arm: str = "G2") -> list:
    """Camera paths of the legs the arm was fitted to (its views file)."""
    names = np.load(ws.DATA / case / "splat" / wsp.ARMS[arm].get("views", "views.npz"))["name"]
    out = []
    for leg in sorted({str(n).rsplit("/", 1)[0] for n in names if not str(n).startswith("pano/")}):
        cam = ws.DATA / case / "scene" / leg / "camera.json"
        cam = cam if cam.exists() else nbv_dir(case) / "legs" / leg / "camera.json"
        cams = json.loads(cam.read_text())["extrinsic"]
        out.append([tuple(pose_of(np.array(m))[0][:2]) for m in cams])
    return out


def overlay(rgb: np.ndarray, code: np.ndarray) -> np.ndarray:
    o = rgb.astype(np.float32)
    o[code == 3] = o[code == 3] * 0.35 + np.array([255, 0, 0]) * 0.65
    o[code == 1] = o[code == 1] * 0.8 + np.array([0, 0, 255]) * 0.2
    return o.astype(np.uint8)


def audit_sheet(d, case, info, made, obs, keys, weight, cl, where, shares, codes, rgbs):
    from PIL import ImageFont
    font = ImageFont.load_default(size=20)
    tv = top_view(obs, phase1_paths(case, info["arm"]), keys, weight, cl)
    order = np.argsort(shares)[::-1]
    pick = list(order[:8]) + list(order[len(order) // 2:len(order) // 2 + 4])
    tw, th = 384, 288
    sh = Image.new("RGB", (tv.width + 3 * tw, max(tv.height, 4 * th) + 40), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    dr.text((8, 8), f"{case} audit on {info['arm']}: student holes {info['hole']:.1%} of pixels (known "
            f"{info['known']:.1%}, ground {info['ground']:.1%}, sky {info['sky']:.1%}); red = no made view explains it",
            fill=(255, 255, 255), font=font)
    sh.paste(tv, (0, 40))
    for k, i in enumerate(pick):
        t = Image.fromarray(overlay(rgbs[i], codes[i])).resize((tw, th), Image.NEAREST)
        x, y, hd = where[i]
        ImageDraw.Draw(t).text((4, 4), f"({x:+.0f},{y:+.0f}) heading {hd:.0f}: {shares[i]:.0%}", fill=(255, 255, 0),
                               font=ImageFont.load_default(size=16))
        sh.paste(t, (tv.width + (k % 3) * tw, 40 + (k // 3) * th))
    sh.save(d / "sheet_audit.jpg", quality=88)
    print(d / "sheet_audit.jpg", flush=True)


# --- plan ------------------------------------------------------------------------------------------------------


def smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def starts(case: str, arm: str = "G2") -> list[dict]:
    """Every START_EVERY-th frame of each phase-1 leg the arm was fitted to: its pose, image and caption."""
    out = []
    res = ws.DATA / case / "out" / "out" / case
    fitted = {str(n).rsplit("/", 1)[0] for n in np.load(ws.DATA / case / "splat" / wsp.ARMS[arm].get("views", "views.npz"))["name"]}
    for leg in [g for g in wsp.legs(case, ("nbv_", "spin_")) if g in fitted]:
        cams = json.loads((ws.DATA / case / "scene" / leg / "camera.json").read_text())["extrinsic"]
        frames = wsp.video_frames(res / leg / f"{ws.MODEL}_result.mp4", [k / (len(cams) - 1) for k in range(len(cams))])
        cap = ws.DATA / case / "scene" / leg / "traj_caption.json"
        for f in range(0, len(cams), START_EVERY):
            p, yaw, pitch = pose_of(np.array(cams[f]))
            out.append({"from": f"{leg}/{f:02d}", "pos": p, "yaw": yaw, "pitch": pitch, "image": frames[f],
                        "caption": json.loads(cap.read_text())["prompt"] if cap.exists() else ""})
    return out


def candidates(st: list[dict], cl: list[dict]) -> list[dict]:
    t = smooth(np.linspace(0, 1, ws.NFRAME))
    out = []
    for s in st:
        p0 = s["pos"]
        for mv in MOVES_DEG:
            a = math.radians(s["yaw"] + mv)
            for dist in MOVE_M:
                xy = p0[:2] + t[:, None] * dist * np.array([math.cos(a), math.sin(a)])
                out.append({"name": f"{s['from']}_move{mv:+d}_{dist:g}", "kind": "move", "start": s, "xy": xy,
                            "yaw": np.full(ws.NFRAME, s["yaw"])})
        for ci, c in enumerate(cl[:TOP_CLUSTERS]):
            ctr = np.array(c["centre"][:2])
            r = float(np.hypot(*(p0[:2] - ctr)))
            if not ARC_RANGE_M[0] <= r <= ARC_RANGE_M[1]:
                continue
            a0 = math.atan2(*(p0[:2] - ctr)[::-1])
            for sgn in (1, -1):
                ang = a0 + sgn * t * ARC_M / r
                xy = ctr + r * np.stack([np.cos(ang), np.sin(ang)], -1)
                look = np.degrees(np.arctan2(ctr[1] - xy[:, 1], ctr[0] - xy[:, 0]))
                off = wrap(s["yaw"] - look[0])
                yaw = look + (1 - t) * off
                if np.abs(wrap(yaw - s["yaw"])).max() > MAX_TURN_DEG:
                    continue
                out.append({"name": f"{s['from']}_arc{ci}{'+' if sgn > 0 else '-'}", "kind": "arc", "start": s,
                            "xy": xy, "yaw": yaw, "cluster": ci})
    return out


def leg_w2cs(c: dict) -> np.ndarray:
    z = float(c["start"]["pos"][2])
    return np.stack([ws.w2c([x, y, z], yw, c["start"]["pitch"]) for (x, y), yw in zip(c["xy"], c["yaw"])]
                    ).astype(np.float32)


def plan(case: str, arm: str, legs: int):
    import torch
    t0 = time.time()
    dev = torch.device("cuda")
    d = nbv_dir(case, arm)
    a = np.load(d / "audit.npz")
    hk, hw, obs = a["keys"], a["weight"], a["obstacles"]
    cl = json.loads((d / "audit.json").read_text())["clusters"]
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", dev)
    made = made_for(sp, case, arm)
    free = free_space(obs)
    st = starts(case, arm)
    cands = candidates(st, cl)
    K = np.array(ws.intrinsic(), np.float32)
    K[:2] /= SCORE_SCALE
    w, h = ws.W // SCORE_SCALE, ws.H // SCORE_SCALE
    ok = []
    for c in cands:
        # The start is a phase-1 frame: already walked, so only the rest must keep clear.
        if np.hypot(*c["xy"].T).max() > AREA_M or not free(c["xy"][1:]).all():
            continue
        code, _, _, X, surf = classify(sp, made, leg_w2cs(c)[list(SCORE_FRAMES)], K, w, h)
        unk = (code == 3).float().mean((1, 2))
        c["unknown"] = unk.tolist()
        if unk.max().item() > FRAME_MAX_UNKNOWN:
            continue
        # What its frames see counts, hole for them or not: the leg looks at it from its own direction.
        P = torch.as_tensor(np.linalg.inv(leg_w2cs(c)[list(SCORE_FRAMES)])[:, :3, 3], device=dev)
        P = P[:, None, None].expand_as(X)
        seen = keys_of(X[surf], P[surf]).unique().cpu().numpy()
        near = np.unique(np.concatenate([bin_shift(seen, s) for s in (-1, 0, 1)]))
        i = np.clip(np.searchsorted(hk, near), 0, len(hk) - 1)
        c["sees"] = i[hk[i] == near]
        ok.append(c)
    print(f"{len(cands)} candidates, {len(ok)} feasible, {time.time() - t0:.0f} s", flush=True)
    picked = greedy(ok, hw, legs)
    write_legs(case, d, sp, made, picked)
    total = float(hw.sum())
    info = {"arm": arm, "candidates": len(cands), "feasible": len(ok), "hole_weight": total,
            "covered": float(sum(c["gain"] for c in picked)), "seconds": round(time.time() - t0),
            "legs": [{"name": f"nbv_{k:02d}", "from": c["start"]["from"], "kind": c["kind"], "candidate": c["name"],
                      "gain": round(c["gain"], 5), "max_unknown": round(max(c["unknown"]), 3),
                      "length_m": round(float(np.linalg.norm(np.diff(c["xy"], axis=0), axis=1).sum()), 2)}
                     for k, c in enumerate(picked)]}
    (d / "plan.json").write_text(json.dumps(info, indent=1))
    plan_sheet(case, d, info, obs, hk, hw, cl, picked)
    print(json.dumps({k: v for k, v in info.items() if k != "legs"}), flush=True)
    for leg in info["legs"]:
        print(leg, flush=True)


def bin_shift(keys: np.ndarray, s: int) -> np.ndarray:
    """The same voxels in the neighbouring direction bin (a view within PARALLAX_MAX_DEG may sit in the next bin)."""
    n = int(360 // BIN_DEG)
    return keys - keys % 32 + (keys % 32 + s) % n


def greedy(cands: list[dict], weight: np.ndarray, budget: int) -> list[dict]:
    covered = np.zeros(len(weight), bool)
    picked = []
    while len(picked) < budget:
        best, gain = None, MIN_GAIN
        for c in cands:
            if any(c is q for q in picked):
                continue
            s = c["sees"]
            g = float(weight[s[~covered[s]]].sum())
            if g > gain:
                best, gain = c, g
        if best is None:
            break
        best["gain"] = gain
        covered[best["sees"]] = True
        picked.append(best)
    return picked


def write_legs(case: str, d: Path, sp, made: Made, picked: list[dict]):
    """WorldStereo inputs per picked leg: the splat's render as guide, the unknown pixels as holes."""
    import imageio.v2 as iio2
    out = d / "legs"
    if out.exists():
        shutil.rmtree(out)
    K = np.array(ws.intrinsic(), np.float32)
    for k, c in enumerate(picked):
        t = out / f"nbv_{k:02d}" / "traj0"
        t.mkdir(parents=True)
        m = leg_w2cs(c)
        code, _, rgb, _, _ = classify(sp, made, m, K, ws.W, ws.H)
        rgb = rgb.cpu().numpy()
        if isinstance(made, Made):
            hole = ((code == 0) | (code == 3)).cpu().numpy()
        else:  # sky from the panorama, so the guide leaves only the targets to invent
            hole, sky = (code == 3).cpu().numpy(), (code == 0).cpu().numpy()
            for f in range(len(m)):
                rgb[f][sky[f]] = made.sky_colour(m[f], K, ws.W, ws.H)[sky[f]]
        rgb[hole] = 0
        rgb[0], hole[0] = c["start"]["image"], False
        Image.fromarray(c["start"]["image"]).save(t.parent / "start_frame.png")
        iio2.mimwrite(t / "render.mp4", list(rgb), fps=16, codec="libx264", quality=8, macro_block_size=16)
        iio2.mimwrite(t / "render_mask.mp4", list(np.repeat((hole * 255).astype(np.uint8)[..., None], 3, -1)), fps=16,
                      codec="libx264", quality=8, macro_block_size=16)
        (t / "camera.json").write_text(json.dumps({
            "extrinsic": [p.tolist() for p in m.astype(np.float64)], "intrinsic": [ws.intrinsic()] * ws.NFRAME,
            "width": ws.W, "height": ws.H, "type": c["kind"], "rotation_deg": 0}))
        (t / "traj_caption.json").write_text(json.dumps({"prompt": c["start"]["caption"]}))
        c["guide"] = [rgb[f] for f in (0, ws.NFRAME // 2, ws.NFRAME - 1)]
        c["holes"] = float(hole[1:].mean())


def plan_sheet(case, d, info, obs, hk, hw, cl, picked):
    from PIL import ImageFont
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    tv = top_view(obs, phase1_paths(case, info["arm"]), hk, hw, cl, picked)
    tw, th = 320, 185
    rows = len(picked)
    sh = Image.new("RGB", (tv.width + 3 * tw, max(tv.height, rows * (th + 22)) + 40), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    dr.text((8, 8), f"{case} phase 2 plan on {info['arm']}: {len(picked)} legs cover {info['covered']:.1%} of "
            f"{info['hole_weight']:.1%} hole pixels. Guide: start (phase-1 frame), middle, end; black = to invent",
            fill=(255, 255, 255), font=font)
    sh.paste(tv, (0, 40))
    for k, c in enumerate(picked):
        y = 40 + k * (th + 22)
        dr.text((tv.width + 4, y), f"{k}: {c['kind']} from {c['start']['from']} ({c['name'].split('_', 2)[-1]}), "
                f"gain {c['gain']:.2%}, to invent {c['holes']:.0%}", fill=(255, 255, 0), font=small)
        for j, g in enumerate(c["guide"]):
            sh.paste(Image.fromarray(g).resize((tw, th)), (tv.width + j * tw, y + 20))
    sh.save(d / "sheet_plan.jpg", quality=88)
    print(d / "sheet_plan.jpg", flush=True)


def guide(case: str, arm: str, src: str, dst: str):
    """Copies of the scene's legs named src* as dst*: same cameras, start frames, chains and captions, but the guide
    is the splat's render with holes only where it renders nothing (the conservative mask). The src legs move to
    legsets/<src> so the next run neither regenerates nor remembers them."""
    import imageio.v2 as iio2
    import torch
    scene = ws.DATA / case / "scene"
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", torch.device("cuda"))
    K = np.array(ws.intrinsic(), np.float32)
    for leg in sorted(scene.glob(f"{src}*")):
        new = scene / leg.name.replace(src, dst, 1)
        if new.exists():
            shutil.rmtree(new)
        (new / "traj0").mkdir(parents=True)
        cams = json.loads((leg / "traj0" / "camera.json").read_text())
        rgb, _, alpha = render(sp, np.array(cams["extrinsic"], np.float32), K, ws.W, ws.H)
        rgb, hole = rgb.cpu().numpy(), (alpha < ALPHA_MIN).cpu().numpy()
        rgb[hole] = 0
        if (leg / "start_frame.png").exists():
            shutil.copy(leg / "start_frame.png", new / "start_frame.png")
            rgb[0], hole[0] = np.asarray(Image.open(new / "start_frame.png").convert("RGB")), False
        if (leg / "chain_from.txt").exists():
            (new / "chain_from.txt").write_text((leg / "chain_from.txt").read_text().replace(src, dst, 1))
        iio2.mimwrite(new / "traj0" / "render.mp4", list(rgb), fps=16, codec="libx264", quality=8, macro_block_size=16)
        iio2.mimwrite(new / "traj0" / "render_mask.mp4", list(np.repeat((hole * 255).astype(np.uint8)[..., None], 3, -1)),
                      fps=16, codec="libx264", quality=8, macro_block_size=16)
        for f in ("camera.json", "traj_caption.json"):
            shutil.copy(leg / "traj0" / f, new / "traj0" / f)
        print(new.name, f"holes {hole[1:].mean():.2f}", flush=True)
    keep = ws.DATA / case / "legsets" / src.rstrip("_")
    keep.mkdir(parents=True, exist_ok=True)
    for leg in sorted(scene.glob(f"{src}*")):
        shutil.move(str(leg), keep / leg.name)


def stage(case: str):
    """The planned legs into WorldStereo's scene folder (earlier nbv legs there are replaced)."""
    scene = ws.DATA / case / "scene"
    for old in scene.glob("nbv_*"):
        shutil.rmtree(old)
    for leg in sorted((nbv_dir(case) / "legs").glob("nbv_*")):
        shutil.copytree(leg, scene / leg.name)
        print(scene / leg.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "plan", "stage", "guide"])
    ap.add_argument("--src", default="spin_")
    ap.add_argument("--dst", default="spg_")
    ap.add_argument("--case", default="photo21")
    ap.add_argument("--arm", default="G2")
    ap.add_argument("--legs", type=int, default=LEGS)
    ap.add_argument("--targets", default=TARGETS, choices=["parallax", "conflict"])
    ap.add_argument("--max-unknown", type=float, default=FRAME_MAX_UNKNOWN, help="guide holes allowed per frame")
    ap.add_argument("--radius", type=float, default=STUDENT_RADIUS_M, help="student area; legs may reach 1 m past it")
    ap.add_argument("--spots", default="", help='"x,y;x,y": audit only these grid spots (every heading)')
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN, help="share of student pixels a leg must add")
    a = ap.parse_args()
    TARGETS, FRAME_MAX_UNKNOWN, MIN_GAIN = a.targets, a.max_unknown, a.min_gain
    STUDENT_RADIUS_M, AREA_M = a.radius, max(AREA_M, a.radius + 1.0)
    SPOTS = [tuple(float(v) for v in t.split(",")) for t in a.spots.split(";")] if a.spots else None
    {"audit": lambda: audit(a.case, a.arm), "plan": lambda: plan(a.case, a.arm, a.legs),
     "stage": lambda: stage(a.case), "guide": lambda: guide(a.case, a.arm, a.src, a.dst)}[a.cmd]()
