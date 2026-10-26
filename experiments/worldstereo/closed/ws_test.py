"""WorldStereo 2.0 (github.com/FuchengSu/WorldStereo, code Apache 2.0, weights MIT, base Wan2.1-I2V-14B Apache 2.0)
on a WorldGen panorama: several walks out of the photosphere along the walkable directions, each one remembering the
frames the earlier walks generated (the model's spatial-stereo memory), to see how well it invents what the
panorama never saw. Upstream has not released its panorama preprocessing (HY-World 2.0's WorldNav does it, under a
licence that excludes the EU), so this script builds the scene folder run_multi_traj.py reads:

    pano_bank/images/0000.png + cameras.json   27 perspective crops of the panorama (the initial memory)
    reconstruct_<k>/start_frame.png            the panorama seen from the walk's first camera
    reconstruct_<k>/traj0/                     walk k, leg 1: camera.json, render.mp4, render_mask.mp4,
                                               traj_caption.json (point cloud of the panorama rendered on the path)
    reconstruct_<k>/traj1/                     leg 2, on from the end of leg 1 (starts on leg 1's last frame)
    wonder_0/traj0/                            along walk 0 turning round to face the start (memory check)
    spin_<k>_<j>/traj0/                        (spin, after a run) a slow 360 in place at the end of walk k, in
                                               SPIN_LEGS quarter turns; each starts on the one before's last frame

Conventions are upstream's (read from its example scene): world z up, camera at the origin, OpenCV w2c, 832x480
frames with fx = fy = 240 (120 degree field of view), 21 frames per leg. Metres: the panorama's distances are
scaled so the ground lies CAMERA_HEIGHT below the camera.

    <eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_test.py prep --case photo21   # pytorch3d, MoGe-2
    python experiments/worldstereo/ws_test.py caption --case photo21                          # teacher-sam3
    python experiments/worldstereo/ws_test.py launch --case photo21                           # edge-sdg
    python experiments/worldstereo/ws_test.py fetch --case photo21                            # edge-sdg
    <eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_test.py sheet --case photo21
    <eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_test.py spin --case photo21    # then launch --resume

launch needs the image (experiments/worldstereo/build.sh) and `set -a; . ./.env; set +a` first."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "worldstereo_test"
PANOS = ROOT / "data" / "panogen_test" / "out"
TOOLS = Path(os.environ.get("TOOLS", "/media/lenena-iker/Crucial/tools"))
WORLDSTEREO = Path(os.environ.get("WORLDSTEREO_REPO", TOOLS / "WorldStereo"))
MOGE_CKPT = os.environ.get("MOGE_CKPT", str(TOOLS / "moge_ckpt/moge-2-vitl/model.pt"))
BUCKET = os.environ.get("SIM2CHIP_CACHE_BUCKET", "sim2chip-cache")
PREFIX = "worldstereo_test"
MODEL = "worldstereo-memory-dmd"

W, H, F = 832, 480, 240.0  # upstream's frame and focal length
NFRAME = 21
CAMERA_HEIGHT = 1.6  # metres; the panorama is shot from a hand-held phone
PITCH = -5.0  # degrees; walks look slightly down, like upstream's start elevation
PANO_W = 2048  # the point cloud's resolution (2M points, about upstream's density per view)
# Walkability: an obstacle is a point between these heights above the ground inside a corridor this wide.
OBSTACLE_Z = (0.25, 2.2)
CORRIDOR = 0.5  # half width, metres
MARGIN = 0.8  # stop this far before an obstacle
MAX_FREE = 10.0
LEG = 4.0  # metres per leg (upstream's exploration walks are about 4.3)
MIN_FREE = 5.5  # a path, not a gap that ends in a hedge after one leg
MIN_SEPARATION = 25.0  # degrees between chosen directions
MAX_WALKS = 4
CENTRE_SEARCH = 3.0  # metres: the walks start from the clearest point this close to the photo's (the middle of the
# street, not the pavement the photo was taken from; user, 2026-10-24: photo21's walks hugged the left side)
CENTRE_CLEAR = 4.0  # metres of clearance that count as fully clear (a wide square keeps the photo's spot)
RAISE = 2.5         # metres the raised leg climbs (to ~4 m above the ground), looking down to RAISE_PITCH
RAISE_PITCH = -25.0
SPIN_LEGS = 4  # quarter turns: 90 degrees in 21 frames, 4.5 per frame (the turn back's 9-13 was too fast, 2026-10-22)


# ── geometry ────────────────────────────────────────────────────────────────────────────────────────────────

def pano_dirs(h: int, w: int) -> np.ndarray:
    """Unit direction of each equirectangular pixel: x at the panorama's centre, y to its left, z up."""
    yaw = (0.5 - (np.arange(w) + 0.5) / w) * 2 * np.pi
    pitch = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    yaw, pitch = np.meshgrid(yaw, pitch)
    return np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], -1)


def w2c(pos, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """OpenCV world-to-camera matrix for a camera at pos looking at yaw (0 = panorama centre, + = left), pitch."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    down = np.cross(fwd, right)
    r = np.stack([right, down, fwd], 0)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, -r @ np.asarray(pos, float)
    return m


def intrinsic() -> list:
    return [[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]]


def crop(pano: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The panorama seen by a camera at the origin with world-to-camera m (only its rotation counts)."""
    import cv2
    ph, pw = pano.shape[:2]
    j, i = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(j - W / 2) / F, (i - H / 2) / F, np.ones_like(j)], -1) @ m[:3, :3]  # camera -> world
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    yaw, pitch = np.arctan2(rays[..., 1], rays[..., 0]), np.arcsin(np.clip(rays[..., 2], -1, 1))
    u = ((0.5 - yaw / (2 * np.pi)) % 1.0) * pw - 0.5
    v = (0.5 - pitch / np.pi) * ph - 0.5
    return cv2.remap(pano, u.astype(np.float32), v.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


# ── depth and walkable directions ───────────────────────────────────────────────────────────────────────────

def pano_distance(pano: np.ndarray, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """MoGe-2 on 20 perspective views of the panorama, merged into one distance map (MoGe's own panorama
    recipe, with the metric v2 model). Returns distance (PANO_W/2 x PANO_W) and its valid mask (False = sky)."""
    if cache.exists():
        z = np.load(cache)
        return z["distance"], z["mask"]
    import cv2
    import torch
    sys.path[:0] = [str(TOOLS / "MoGe"), str(TOOLS / "moge_deps")]
    from moge.model.v2 import MoGeModel
    from moge.utils.panorama import get_panorama_cameras, merge_panorama_depth, split_panorama_image
    try:
        import utils3d_moge as utils3d
    except ImportError:
        import utils3d
    model = MoGeModel.from_pretrained(MOGE_CKPT).cuda().eval()
    ext, ints = get_panorama_cameras()
    views = split_panorama_image(pano, ext, ints, 512)
    dists, masks = [], []
    for i in range(0, len(views), 4):
        x = torch.tensor(np.stack(views[i:i + 4]) / 255, dtype=torch.float32, device="cuda").permute(0, 3, 1, 2)
        fov = torch.tensor(np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(ints[i:i + 4])))[0], device="cuda")
        with torch.no_grad():
            out = model.infer(x, fov_x=fov, apply_mask=False)
        dists += list(out["points"].norm(dim=-1).cpu().numpy())
        masks += list(out["mask"].cpu().numpy())
    d, m = merge_panorama_depth(1920, 960, dists, masks, ext, ints)
    ph, pw = pano.shape[:2]
    d = cv2.resize(d.astype(np.float32), (pw, ph), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(m.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
    np.savez_compressed(cache, distance=d, mask=m)
    return d, m


def edge_mask(dist: np.ndarray, ratio: float = 1.04) -> np.ndarray:
    """True on depth discontinuities (a 3x3 neighbour more than 4 % nearer or farther): flying points."""
    import cv2
    k = np.ones((3, 3), np.uint8)
    hi, lo = cv2.dilate(dist, k), cv2.erode(dist, k)
    return hi > lo * ratio


def walkable(points: np.ndarray, step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Free distance along each azimuth: how far a CORRIDOR-wide walker goes before an obstacle point."""
    hgt = points[:, 2] + CAMERA_HEIGHT
    obs = points[(hgt > OBSTACLE_Z[0]) & (hgt < OBSTACLE_Z[1])][:, :2]
    obs = obs[np.linalg.norm(obs, axis=1) < MAX_FREE + MARGIN + 1]
    yaws = np.arange(-180, 180, step)
    free = np.full(len(yaws), MAX_FREE)
    for i, y in enumerate(np.radians(yaws)):
        d, n = np.array([math.cos(y), math.sin(y)]), np.array([-math.sin(y), math.cos(y)])
        along, lat = obs @ d, np.abs(obs @ n)
        hit = along[(along > 0) & (lat < CORRIDOR)]
        if len(hit):
            free[i] = min(MAX_FREE, max(0.0, hit.min() - MARGIN))
    return yaws, free


def centre(points: np.ndarray, step: float = 0.25) -> np.ndarray:
    """The clearest ground point within CENTRE_SEARCH metres of the photo's: farthest from every obstacle point
    (clearance capped at CENTRE_CLEAR), the nearer one on ties."""
    from scipy.spatial import cKDTree
    hgt = points[:, 2] + CAMERA_HEIGHT
    obs = points[(hgt > OBSTACLE_Z[0]) & (hgt < OBSTACLE_Z[1])][:, :2]
    obs = obs[np.linalg.norm(obs, axis=1) < CENTRE_SEARCH + CENTRE_CLEAR + 1]
    if not len(obs):
        return np.zeros(3)
    g = np.arange(-CENTRE_SEARCH, CENTRE_SEARCH + step / 2, step)
    cand = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    cand = cand[np.linalg.norm(cand, axis=1) <= CENTRE_SEARCH]
    clear = np.minimum(cKDTree(obs).query(cand)[0], CENTRE_CLEAR)
    best = cand[np.argmax(clear - 0.05 * np.linalg.norm(cand, axis=1))]
    return np.array([best[0], best[1], 0.0])


def pick_directions(yaws: np.ndarray, free: np.ndarray) -> list[tuple[float, float]]:
    """Up to MAX_WALKS directions free for at least MIN_FREE metres, as far apart as possible: first the freest
    (ties to the photo's own direction, yaw 0, where the pixels are real), then each time the open direction
    farthest from those already chosen, at least MIN_SEPARATION from them. A street gets forward and back, a square
    gets four spread ways instead of four neighbours (photo3, photo27: every way 10 m free, 2026-10-24)."""
    sm = free.copy()
    for k in (1, 2):  # a direction counts only as far as its neighbours within 4 degrees are free too
        sm = np.minimum(sm, np.minimum(np.roll(free, k), np.roll(free, -k)))
    wide = np.mean([np.roll(sm, k) for k in range(-10, 11)], 0)  # the middle of an opening beats its edge
    open_ = sm >= MIN_FREE
    if not open_.any():
        return []
    gap = lambda y, ys: min(abs((y - c + 180) % 360 - 180) for c in ys)  # noqa: E731
    first = max(np.nonzero(open_)[0], key=lambda i: (sm[i] >= sm.max() - 0.5, -abs(yaws[i])))
    chosen = [(float(yaws[first]), float(sm[first]))]
    while len(chosen) < MAX_WALKS:
        cand = [i for i in np.nonzero(open_)[0] if gap(yaws[i], [c for c, _ in chosen]) >= MIN_SEPARATION]
        if not cand:
            break
        i = max(cand, key=lambda i: (round(gap(yaws[i], [c for c, _ in chosen]) / 10), sm[i], wide[i]))
        chosen.append((float(yaws[i]), float(sm[i])))
    return chosen


# ── trajectories ────────────────────────────────────────────────────────────────────────────────────────────

def between(start, end, yaw: float, z=(0.0, 0.0), pitch=(PITCH, PITCH)) -> list[np.ndarray]:
    """From start to end facing yaw (a sideways drift when the line is not along yaw), rising z[0] -> z[1] metres
    and tilting pitch[0] -> pitch[1], eased (smoothstep)."""
    out = []
    for t in np.linspace(0, 1, NFRAME):
        e = t * t * (3 - 2 * t)
        pos = np.asarray(start, float) + (np.asarray(end, float) - np.asarray(start, float)) * t
        out.append(w2c(pos + np.array([0, 0, z[0] + (z[1] - z[0]) * e]), yaw, pitch[0] + (pitch[1] - pitch[0]) * e))
    return out


def straight(start, yaw: float, length: float) -> list[np.ndarray]:
    y = math.radians(yaw)
    d = np.array([math.cos(y), math.sin(y), 0.0])
    return [w2c(np.asarray(start) + d * length * t, yaw, PITCH) for t in np.linspace(0, 1, NFRAME)]


def spin_poses(pos, yaw0: float, j: int) -> list[np.ndarray]:
    """Quarter turn j of a 360 in place, turning left at a steady rate."""
    return [w2c(pos, yaw0 + 90 * (j + t), PITCH) for t in np.linspace(0, 1, NFRAME)]


def plan(walks: list[tuple[float, float]], c=(0.0, 0.0, 0.0)) -> dict[str, dict]:
    """Folder -> {yaw, poses, kind}. reconstruct_<k>: walk k from the centre c (its free distance measured from
    there): leg 1 starts on the photo's spot (its first frame is the panorama's) and drifts onto the line through c,
    leg 2 goes on when there is room; wonder_0: the freest walk again, turning round to face the start; wonder_1:
    the freest walk's first half climbing RAISE metres while looking down to RAISE_PITCH."""
    c = np.asarray(c, float)
    trajs = {}
    for k, (yaw, free) in enumerate(walks):
        y = math.radians(yaw)
        d = np.array([math.cos(y), math.sin(y), 0.0])
        l1 = min(LEG, free)
        trajs[f"reconstruct_{k}/traj0"] = dict(yaw=yaw, poses=between([0, 0, 0], c + d * l1, yaw), kind="walk",
                                               length=float(np.linalg.norm(c + d * l1)))
        if free >= l1 + 2.0:
            l2 = min(LEG, free - l1)
            trajs[f"reconstruct_{k}/traj1"] = dict(yaw=yaw, poses=straight(c + d * l1, yaw, l2), kind="walk on",
                                                   length=l2)
    yaw, free = walks[0]
    y = math.radians(yaw)
    d = np.array([math.cos(y), math.sin(y), 0.0])
    lb = min(LEG, free) * 0.75
    back = [w2c((c + d * lb) * t, yaw + 180 * (t * t * (3 - 2 * t)), PITCH) for t in np.linspace(0, 1, NFRAME)]
    trajs["wonder_0/traj0"] = dict(yaw=yaw, poses=back, kind="turn back", length=lb)
    lr = min(LEG, free) * 0.5
    trajs["wonder_1/traj0"] = dict(yaw=yaw, poses=between([0, 0, 0], c + d * lr, yaw, (0.0, RAISE),
                                                          (PITCH, RAISE_PITCH)), kind="raise", length=lr)
    return trajs


def bank_cameras() -> list[np.ndarray]:
    """Upstream's panorama memory: 9 yaws 40 degrees apart (from the back) at pitch 0, +27 and -27."""
    yaws = [180 - 40 * i for i in range(9)]
    return [w2c([0, 0, 0], y, p) for p in (0, 27, -27) for y in yaws]


# ── rendering ───────────────────────────────────────────────────────────────────────────────────────────────

def renderer():
    """Upstream's point renderer (pytorch3d, radius 0.008, 20 points per pixel, grey where no point lands);
    kornia is only imported for an edge filter this script does not use."""
    sys.modules.setdefault("kornia", types.ModuleType("kornia"))
    sys.path.insert(0, str(WORLDSTEREO))
    from src.pointcloud import point_rendering
    return point_rendering


def render(point_rendering, points, colors, poses) -> tuple[np.ndarray, np.ndarray]:
    import torch
    k = torch.tensor([intrinsic()] * len(poses), dtype=torch.float32)
    rgb, holes = [], []
    for i in range(0, len(poses), 4):
        r, m = point_rendering(K=k[i:i + 4], w2cs=torch.tensor(np.stack(poses[i:i + 4]), dtype=torch.float32),
                               points=points, colors=colors, device="cuda", h=H, w=W, render_radius=0.008,
                               points_per_pixel=20, background_color=[0, 0, 0])
        rgb.append(((r.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy())
        holes.append((m[:, 0] * 255).byte().cpu().numpy())
    return np.concatenate(rgb), np.concatenate(holes)


def guide_cloud(case: str):
    """prep's point cloud of the panorama (upstream's depth contraction), from the saved distances."""
    import cv2
    import torch
    meta = json.loads((DATA / case / "prep.json").read_text())
    full = np.asarray(Image.open(PANOS / case / meta["pano"]).convert("RGB"))
    pano = cv2.resize(full, (PANO_W, PANO_W // 2), interpolation=cv2.INTER_AREA)
    z = np.load(DATA / case / "distance.npz")
    dist, valid = z["distance"] * meta["scale"], z["mask"]
    keep = valid & ~edge_mask(dist) & np.isfinite(dist) & (dist > 0)
    mid = np.median(dist[keep]) * 8
    d = dist[keep]
    d = np.where(d > mid, 2 * mid - mid ** 2 / (d + 1e-6), d)
    points = torch.tensor(pano_dirs(*dist.shape)[keep] * d[:, None], dtype=torch.float32)
    return points, torch.tensor(pano[keep] / 127.5 - 1, dtype=torch.float32), meta


def spin(case: str):
    """A slow 360 in place at the end of each walk (its last leg), guided like the walks (the panorama's point
    cloud). The first quarter starts on the walk's last generated frame; the others name the quarter before in
    chain_from.txt and the job's runner starts them on its last frame."""
    import shutil
    import cv2
    points, colors, meta = guide_cloud(case)
    point_rendering = renderer()
    scene, res = DATA / case / "scene", DATA / case / "out" / "out" / case
    for old in scene.glob("spin_*"):
        shutil.rmtree(old)
    walks = sorted({n.split("/")[0] for n in meta["trajs"] if n.startswith("reconstruct_")})
    for k, walk in enumerate(walks):
        last = f"{walk}/traj1" if f"{walk}/traj1" in meta["trajs"] else f"{walk}/traj0"
        end = np.array(json.loads((scene / last / "camera.json").read_text())["extrinsic"][-1])
        pos, yaw0 = np.linalg.inv(end)[:3, 3], math.degrees(math.atan2(end[2, 1], end[2, 0]))
        for j in range(SPIN_LEGS):
            d = scene / f"spin_{k}_{j}" / "traj0"
            d.mkdir(parents=True)
            poses = spin_poses(pos, yaw0, j)
            rgb, holes = render(point_rendering, points, colors, poses)
            if j == 0:
                cap = cv2.VideoCapture(str(res / last / f"{MODEL}_result.mp4"))
                frames = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frames.append(f)
                Image.fromarray(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)).save(d.parent / "start_frame.png")
                rgb[0], holes[0] = np.asarray(Image.open(d.parent / "start_frame.png")), 0
            else:
                (d.parent / "chain_from.txt").write_text(f"spin_{k}_{j - 1}/traj0")
            write_video(d / "render.mp4", rgb)
            write_video(d / "render_mask.mp4", holes)
            (d / "camera.json").write_text(json.dumps({
                "extrinsic": [p.tolist() for p in poses], "intrinsic": [intrinsic()] * NFRAME,
                "width": W, "height": H, "type": "spin", "rotation_deg": 0}))
            (d / "traj_caption.json").write_text((scene / last / "traj_caption.json").read_text())
            print(d, f"yaw {yaw0 + 90 * j:.0f} -> {yaw0 + 90 * (j + 1):.0f}, holes {np.mean(holes[1:] > 127):.2f}",
                  flush=True)


def write_video(path: Path, frames: np.ndarray):
    import imageio.v3 as iio
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, -1)
    iio.imwrite(path, frames, plugin="pyav", codec="libx264", fps=16)


def prep(case: str, pano_name: str):
    import cv2
    import torch
    out = DATA / case / "scene"
    out.mkdir(parents=True, exist_ok=True)
    full = np.asarray(Image.open(PANOS / case / pano_name).convert("RGB"))
    pano = cv2.resize(full, (PANO_W, PANO_W // 2), interpolation=cv2.INTER_AREA)
    dist, valid = pano_distance(pano, DATA / case / "distance.npz")
    dirs = pano_dirs(*dist.shape)
    keep = valid & ~edge_mask(dist) & np.isfinite(dist) & (dist > 0)

    # Scale: the ground (median height of what lies 30+ degrees below the horizon, within 15 m) sits
    # CAMERA_HEIGHT below the camera.
    pts = dirs * dist[..., None]
    low = keep & (dirs[..., 2] < -0.5) & (dist < 15)
    scale = CAMERA_HEIGHT / -np.median(pts[low][:, 2])
    pts *= scale
    dist = dist * scale

    c = centre(pts[keep])
    yaws, free = walkable(pts[keep] - c)
    walks = pick_directions(yaws, free)
    print(f"{case}: scale {scale:.3f}, centre {c[:2].round(2).tolist()}, walks (yaw, free m): "
          f"{[(round(y), round(f, 1)) for y, f in walks]}")
    if not walks:
        raise SystemExit("no walkable direction")
    trajs = plan(walks, c)

    # The renders use upstream's depth contraction (beyond 8x the median distance, squashed to 16x).
    mid = np.median(dist[keep]) * 8
    d = dist[keep]
    d = np.where(d > mid, 2 * mid - mid ** 2 / (d + 1e-6), d)
    points = torch.tensor(dirs[keep] * d[:, None], dtype=torch.float32)
    colors = torch.tensor(pano[keep] / 127.5 - 1, dtype=torch.float32)
    point_rendering = renderer()

    bank = out / "pano_bank" / "images"
    bank.mkdir(parents=True, exist_ok=True)
    cams = {}
    for i, m in enumerate(bank_cameras()):
        Image.fromarray(crop(full, m)).save(bank / f"{i:04d}.png")
        cams[f"{i:04d}"] = {"extrinsic": m.tolist(), "intrinsic": intrinsic()}
    (out / "pano_bank" / "cameras.json").write_text(json.dumps(cams))

    # One caption per scene (ws_scene.py caption, the panorama's own prompt) when there is one, else Matrix-3D's.
    caption = PANOS.parent / "in" / f"{case}.txt"
    if not caption.exists():
        caption = PANOS / case / "m3d_straight" / "prompt.txt"
    caption = caption.read_text().strip() if caption.exists() else ""
    meta = {"case": case, "pano": pano_name, "scale": scale, "camera_height": CAMERA_HEIGHT, "centre": c.tolist(),
            "walks": walks, "free": dict(zip(map(float, yaws), map(float, free))), "trajs": {}}
    for name, t in trajs.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        start = d.parent / "start_frame.png"
        if name.endswith("traj0"):
            Image.fromarray(crop(full, t["poses"][0])).save(start)
        rgb, holes = render(point_rendering, points, colors, t["poses"])
        if name.endswith("traj0"):
            rgb[0], holes[0] = np.asarray(Image.open(start)), 0
        write_video(d / "render.mp4", rgb)
        write_video(d / "render_mask.mp4", holes)
        (d / "camera.json").write_text(json.dumps({
            "extrinsic": [p.tolist() for p in t["poses"]], "intrinsic": [intrinsic()] * NFRAME,
            "width": W, "height": H, "type": t["kind"], "rotation_deg": 0}))
        cap = d / "traj_caption.json"
        if not cap.exists():
            cap.write_text(json.dumps({"prompt": caption}))
        meta["trajs"][name] = {"yaw": t["yaw"], "kind": t["kind"], "length": t["length"],
                               "holes": [round(float((h > 127).mean()), 3) for h in holes]}
        Image.fromarray(np.concatenate([rgb[0], rgb[NFRAME // 2], rgb[-1]], 1)).save(d.parent / f"{d.name}_render.jpg")
        print(name, t["kind"], f"{t['length']:.1f} m", "holes at end", meta["trajs"][name]["holes"][-1], flush=True)
    (DATA / case / "prep.json").write_text(json.dumps(meta, indent=1))
    prep_sheet(case, full, pts[keep], walks, trajs, yaws, free)


def prep_sheet(case, full, pts, walks, trajs, yaws, free):
    """Panorama with the chosen directions, a top view (obstacles, walks) and each leg's render at start,
    middle and end: what the user approves before the cloud run."""
    from PIL import ImageFont
    out = DATA / case
    font = ImageFont.load_default(size=22)
    pw = 1600
    pano = Image.fromarray(full).resize((pw, pw // 2))
    dr = ImageDraw.Draw(pano)
    cols = [(255, 60, 60), (60, 200, 255), (255, 220, 0), (120, 255, 120)]
    for k, (y, f) in enumerate(walks):
        u = ((0.5 - y / 360) % 1.0) * pw
        dr.line([(u, 0), (u, pw // 2)], fill=cols[k], width=4)
        dr.text((u + 6, pw // 2 - 160), f"walk {k}: {f:.1f} m", fill=cols[k], font=font)
    # Top view: 20 m square, obstacle points grey, walks coloured; x (panorama centre) up, y (left) left.
    s, half = 800, 10.0
    top = Image.new("RGB", (s, s), (20, 20, 20))
    td = ImageDraw.Draw(top)
    hgt = pts[:, 2] + CAMERA_HEIGHT
    ob = pts[(hgt > OBSTACLE_Z[0]) & (hgt < OBSTACLE_Z[1])]
    ob = ob[np.random.default_rng(0).choice(len(ob), min(len(ob), 60000), replace=False)]

    def px(x, y):
        return s / 2 - y / half * s / 2, s / 2 - x / half * s / 2
    for x, y in ob[:, :2]:
        if abs(x) < half and abs(y) < half:
            td.point(px(x, y), fill=(150, 150, 150))
    for yw, f in zip(np.radians(yaws), free):
        td.point(px(f * math.cos(yw), f * math.sin(yw)), fill=(90, 90, 255))
    for name, t in trajs.items():
        k = 0 if name.startswith("wonder") else int(name.split("_")[1].split("/")[0])
        cs = [-p[:3, :3].T @ p[:3, 3] for p in t["poses"]]
        td.line([px(c[0], c[1]) for c in cs], fill=cols[k], width=3 if "wonder" not in name else 1)
    td.text((10, 10), "top view, 20 m; blue = free distance", fill=(255, 255, 255), font=font)
    rows = [pano.resize((1600, 800))]
    head = Image.new("RGB", (1600, 800), (0, 0, 0))
    head.paste(top, (0, 0))
    ImageDraw.Draw(head).text((820, 20), f"{case}: {len(trajs)} legs\n" + "\n".join(
        f"{n}: {t['kind']} {t['length']:.1f} m" for n, t in trajs.items()), fill=(255, 255, 255), font=font)
    rows.append(head)
    for name in trajs:
        im = Image.open(out / "scene" / f"{name}_render.jpg").resize((1600, 308))
        ImageDraw.Draw(im).text((10, 10), name, fill=(255, 255, 0), font=font)
        rows.append(im)
    sheet = Image.new("RGB", (1600, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out / "sheet_prep.jpg", quality=88)
    print(out / "sheet_prep.jpg")


# name: (platform, preset, project); None is SIM2CHIP_NEBIUS_PROJECT (eu-north1). The RTX PRO 6000 (96 GB, 1.80 $/h
# against 4.50 for the H100) is not in eu-north1: eu-south1's default project has the quota (2026-10-21).
GPUS = {"h100": ("gpu-h100-sxm", "1gpu-16vcpu-200gb", None),  # 80 GB: 14B transformer bf16 + fp32 umT5 ~60 GB
        "rtx6000": ("gpu-rtx6000-a", "1gpu-24vcpu-218gb", "project-e07kvxwxma00dk2srnj8vg")}

JOB = r'''#!/bin/bash
exec > /tmp/job.log 2>&1
set -xeo pipefail  # a failed step ends the job instead of burning GPU time on the next one
mkdir -p /tmp/scene /tmp/out
up() { curl -s -X PUT --upload-file /tmp/job.log "$LOG_URL" > /dev/null || true; }
# Whatever legs exist go up every 5 min and at exit: the job timeout kills the container without running traps.
up_out() { tar -C /tmp -cf /tmp/out_up.tar out && curl -s -X PUT --upload-file /tmp/out_up.tar "__OUT_URL__" > /dev/null || true; }
trap 'echo "exit status $?"; up_out; up' EXIT
( set +xe; i=0; while true; do sleep 60; up; i=$((i + 1)); [ $((i % 5)) -eq 0 ] && up_out; done ) &
cd /opt/WorldStereo
nvidia-smi
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
export HF_XET_HIGH_PERFORMANCE=1 TORCHDYNAMO_DISABLE=1  # torch.compile off: minutes of compiling for 8 runs
T0=$(date +%s)
python - <<'PY'
from huggingface_hub import snapshot_download
# The DMD checkpoint holds the whole transformer: the base one's 65 GB are never read (run_ws.py below).
snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                  ignore_patterns=["transformer/diffusion_pytorch_model*"] if "dmd" in "__MODEL__" else None)
snapshot_download("hanshanxue/WorldStereo", allow_patterns=["__MODEL__/config.json", "__MODEL__/model.safetensors"])
snapshot_download("facebook/dinov2-base")
PY
echo "weights seconds $(( $(date +%s) - T0 ))"
curl -sf "__SCENE_URL__" | tar -C /tmp/scene -xf -
ls /tmp/scene/__CASE__
# Resume: earlier legs' results go back where the runner looks; it skips them and loads them into its memory.
if [ -n "__RESUME_URL__" ]; then curl -sf "__RESUME_URL__" | tar -C /tmp/out -xf -; find /tmp/out -name "*_result.mp4"; fi
T0=$(date +%s)
# Upstream keeps the fp32 umT5 (23 GB) on the GPU next to the 17B transformer: 80 GB ran out on the first leg
# (2026-10-21). The wrapper parks the text encoder on the CPU and lends it the GPU only while it encodes.
cat > /tmp/run_ws.py <<'PY'
import runpy, sys, torch
sys.path.insert(0, ".")
from models.worldstereo_wrapper import WorldStereo
load = WorldStereo.from_pretrained.__func__


def from_pretrained(cls, *a, **k):
    ws = load(cls, *a, **k)
    # (the DMD pipeline already returns its own device from its _execution_device() method)
    pipe, dev = ws.pipeline, torch.device(k["device"])
    te = getattr(pipe.text_encoder, "_orig_mod", pipe.text_encoder).to("cpu")
    forward = te.forward

    def lent(*args, **kw):
        te.to(dev)
        try:
            return forward(*args, **kw)
        finally:
            te.to("cpu")
            torch.cuda.empty_cache()
    te.forward = lent
    pipe.text_encoder = te
    torch.cuda.empty_cache()
    print(f"text encoder on cpu; gpu allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return ws


WorldStereo.from_pretrained = classmethod(from_pretrained)
# Upstream builds the transformer from the base model's 65 GB and then overwrites every key with the DMD
# checkpoint (no missing keys, 2026-10-22). Diffusers gets the base config next to the DMD file instead, so only
# that file is read; the ControlNet keys it ignores are loaded by upstream right after, as before.
from pathlib import Path
from unittest import mock
from huggingface_hub import hf_hub_download
import models.worldstereo_wrapper as wsw
load_transformer = WorldStereo._load_transformer


def _load_transformer(cfg, model_type, weights_path, **kw):
    if "dmd" not in model_type:
        return load_transformer(cfg, model_type, weights_path, **kw)
    model = wsw.WorldStereoRefSModel
    upstream_from_pretrained = model.from_pretrained

    def from_dmd(base, subfolder=None, **k):
        conf = Path(base, subfolder, "config.json")
        conf = conf if conf.exists() else hf_hub_download(base, "config.json", subfolder=subfolder)
        d = Path("/tmp/wan_dmd") / subfolder
        d.mkdir(parents=True, exist_ok=True)
        for name, src in (("config.json", conf), ("diffusion_pytorch_model.safetensors", weights_path)):
            (d / name).unlink(missing_ok=True)
            (d / name).symlink_to(src)
        return upstream_from_pretrained(str(d.parent), subfolder=subfolder, **k)
    with mock.patch.object(model, "from_pretrained", from_dmd):
        return load_transformer(cfg, model_type, weights_path, **kw)


WorldStereo._load_transformer = staticmethod(_load_transformer)
# Legs upstream's order does not know (ws_nbv.py's nbv_*, spin_*) run last, after every leg it does know is in
# memory. A leg with chain_from.txt starts on the last frame that leg generated (a 360 in quarter turns).
import glob, os
from PIL import Image
import src.data_utils as du
upstream_sort, upstream_load = du.sort_trajs, du.load_mutli_traj_dataset
def sort_trajs(scene_path):
    known = upstream_sort(scene_path)
    extra = sorted(p for p in glob.glob(f"{scene_path}/*/traj0/render.mp4") if p not in known)
    print("extra legs after upstream's:", [os.path.dirname(p) for p in extra], flush=True)
    return known + extra
def load_mutli_traj_dataset(cfg, input_path, output_path, view_id, traj_id, device, ref_index=None, model_type=None,
                            task_type="panorama"):
    chain = f"{input_path}/{view_id}/chain_from.txt"
    if os.path.exists(chain):
        prev = open(chain).read().strip()
        Image.fromarray(du.get_last_video_frame(f"{output_path}/{prev}/{model_type}_result.mp4")).save(
            f"{input_path}/{view_id}/start_frame.png")
        print(f"{view_id} starts on the last frame of {prev}", flush=True)
    return upstream_load(cfg, input_path, output_path, view_id, traj_id, device, ref_index, model_type, task_type)
du.sort_trajs, du.load_mutli_traj_dataset = sort_trajs, load_mutli_traj_dataset
sys.argv = ["run_multi_traj.py", *sys.argv[1:]]
runpy.run_path("run_multi_traj.py", run_name="__main__")
PY
torchrun --nproc_per_node=1 /tmp/run_ws.py --model_type __MODEL__ --task_type panorama \
  --input_path /tmp/scene/__CASE__ --output_path /tmp/out
# (no --local_files_only: with it transformers cannot load the base model's tokenizer, it looks for a root
# config.json the Diffusers repo lacks)
echo "generation seconds $(( $(date +%s) - T0 ))"
find /tmp/out -type f | head -200
'''


def image_tag() -> str:
    """build.sh's tag: the hash of the Dockerfile without comments."""
    lines = (ROOT / "experiments/worldstereo/Dockerfile").read_text().splitlines(keepends=True)
    return hashlib.sha256("".join(ln for ln in lines if not ln.startswith("#")).encode()).hexdigest()[:12]


def preflight(case: str, image: str):
    """Free checks before paying for a GPU: the walks the job will read and the pushed image."""
    scene = DATA / case / "scene"
    legs = [t for t in scene.glob("*/traj*") if t.is_dir()]
    need = ("render.mp4", "render_mask.mp4", "camera.json", "traj_caption.json")
    missing = [f"{t.relative_to(scene)}/{n}" for t in legs for n in need if not (t / n).exists()]
    missing += [f"{t.parent.name}/start_frame.png" for t in legs if not (t.parent / "start_frame.png").exists()
                and not (t.parent / "chain_from.txt").exists()]
    missing += [] if (scene / "pano_bank/cameras.json").exists() else ["pano_bank/cameras.json"]
    if not legs or missing:
        sys.exit(f"scene not ready ({len(legs)} legs), missing: {missing}")
    if subprocess.run(["docker", "manifest", "inspect", image], capture_output=True).returncode:
        sys.exit(f"{image} is not in the registry: run experiments/worldstereo/build.sh")
    print(f"preflight ok: {len(legs)} legs, image pushed")


def launch(case: str, gpu: str, max_hours: float, dry: bool = False, resume: bool = False):
    """Upload the scene, write job.sh and create the Nebius job; with dry, everything but the job. With resume,
    the legs already generated (out/out/<case>) go along: only the new ones (ws_nbv.py's) are generated."""
    image = f"{os.environ['SIM2CHIP_NEBIUS_REGISTRY']}/eyeforge-worldstereo:{image_tag()}"
    preflight(case, image)
    sys.path.insert(0, str(ROOT))
    from sim2chip import config  # noqa: F401  (loads .env)
    from sim2chip.cloud_sync import S3Store
    store = S3Store()
    exp = max_hours + 2
    tar = DATA / case / "scene.tar"
    with tarfile.open(tar, "w") as t:
        t.add(DATA / case / "scene", arcname=case,
              filter=lambda ti: None if ti.name.endswith("_render.jpg") else ti)
    store.put(BUCKET, f"{PREFIX}/in/{case}.tar", tar.read_bytes())
    resume_url = ""
    if resume:
        done = DATA / case / "resume.tar"
        with tarfile.open(done, "w") as t:
            t.add(DATA / case / "out" / "out" / case, arcname=case)
        store.put(BUCKET, f"{PREFIX}/in/{case}_resume.tar", done.read_bytes())
        resume_url = store.url(BUCKET, f"{PREFIX}/in/{case}_resume.tar", exp)

    def put_url(key):
        return store.s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": f"{PREFIX}/out/{key}"},
                                               ExpiresIn=int(exp * 3600))
    script = DATA / "job.sh"
    script.write_text(JOB.replace("__MODEL__", MODEL).replace("__CASE__", case)
                      .replace("__SCENE_URL__", store.url(BUCKET, f"{PREFIX}/in/{case}.tar", exp))
                      .replace("__OUT_URL__", put_url(f"{case}_out.tar")).replace("__RESUME_URL__", resume_url))
    cli = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
    platform, preset, project = GPUS[gpu]
    args = [cli, "ai", "job", "create", "--name", f"eyeforge-worldstereo-{int(time.time())}", "--image", image,
            "--platform", platform, "--preset", preset, "--on-demand", "--timeout", f"{int(max_hours * 60)}m",
            "--disk-size", "250Gi", "--shm-size", "32Gi",
            "--parent-id", project or os.environ["SIM2CHIP_NEBIUS_PROJECT"], "--inject-file", f"{script}:/opt/ws_job.sh",
            "--container-command", "bash", "--args", "/opt/ws_job.sh",
            "--env", f"LOG_URL={put_url('job.log')}", "--async"]
    if project:  # the registry is in eu-north1: a job elsewhere needs credentials to pull (2026-10-21)
        token = subprocess.run([cli, "iam", "get-access-token"], capture_output=True, text=True, check=True).stdout.strip()
        args += ["--registry-username", "iam", "--registry-password", token]
    if dry:
        print(" ".join("<secret>" if a == args[-1] and project else a for a in args[:args.index("--env") + 1]),
              "LOG_URL=<presigned> ...", f"\njob script: {script}")
        return
    r = subprocess.run(args, capture_output=True, text=True)
    print(f"{case} on {gpu} ({image})", r.stdout[-300:], r.stderr[-500:])


def fetch(case: str):
    sys.path.insert(0, str(ROOT))
    from sim2chip import config  # noqa: F401
    from sim2chip.cloud_sync import S3Store
    out = DATA / case / "out"
    out.mkdir(parents=True, exist_ok=True)
    store = S3Store()
    for k in store.listing(BUCKET, f"{PREFIX}/out/"):
        f = out / k.rsplit("/", 1)[-1]
        f.write_bytes(store.s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        if f.suffix == ".tar":
            with tarfile.open(f) as t:
                try:
                    t.extractall(out, filter="data")
                except TypeError:  # Python before 3.11.4 has no extraction filters
                    t.extractall(out)
        print(f)


def sheet(case: str):
    """Per leg: the point-cloud guide (top) against WorldStereo's frames (bottom) at 0, 1/4, 1/2, 3/4 and the
    end, plus the memory frames it retrieved. Needs a video reader (eyeforge-sam3d env)."""
    import imageio.v3 as iio
    from PIL import ImageFont
    font = ImageFont.load_default(size=22)
    scene, res = DATA / case / "scene", DATA / case / "out" / "out" / case
    cw, chh = 400, 231
    rows = []
    for vid in sorted(res.glob(f"*/traj*/{MODEL}_result.mp4"), key=lambda p: (p.parts[-2] == "traj1", str(p))):
        name = f"{vid.parts[-3]}/{vid.parts[-2]}"
        gen = iio.imread(vid, plugin="pyav")
        ren = iio.imread(scene / name / "render.mp4", plugin="pyav")
        idx = [round(i * (len(gen) - 1) / 4) for i in range(5)]
        row = Image.new("RGB", (cw * 5, chh * 2 + 30), (0, 0, 0))
        for j, i in enumerate(idx):
            row.paste(Image.fromarray(ren[min(i, len(ren) - 1)]).resize((cw, chh)), (j * cw, 30))
            row.paste(Image.fromarray(gen[i]).resize((cw, chh)), (j * cw, 30 + chh))
        ImageDraw.Draw(row).text((8, 4), f"{name}: guide (top), WorldStereo (bottom)", fill=(255, 255, 0), font=font)
        rows.append(row)
    sheet = Image.new("RGB", (cw * 5, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(DATA / case / "sheet_result.jpg", quality=88)
    print(DATA / case / "sheet_result.jpg")


CAPTION = ("Describe this scene in one paragraph of about 90 words for a video model that will walk forward into "
           "it: the ground, what lies ahead and on each side, the materials, the light and the weather. Plain "
           "description only, no camera words, no lists.")


def caption(case: str):
    """Each walk's prompt (upstream's traj_caption.json) from Cosmos Reason 2 on its start frame, through
    sim2chip's local VLM; leg 2 reuses leg 1's."""
    import base64
    import io
    sys.path.insert(0, str(ROOT))
    from sim2chip.local_vlm import DEFAULT_MODEL, MODEL
    scene = DATA / case / "scene"
    for start in sorted(scene.glob("*/start_frame.png")):
        buf = io.BytesIO()
        Image.open(start).convert("RGB").save(buf, "JPEG", quality=90)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                            {"type": "text", "text": CAPTION}]}]
        text = MODEL.chat(DEFAULT_MODEL, msg, 400, 0)["text"].split("</think>")[-1].strip().strip('"')
        text = " ".join(text.split())
        for t in [p for p in start.parent.glob("traj*") if p.is_dir()]:
            (t / "traj_caption.json").write_text(json.dumps({"prompt": text}))
        print(f"{start.parent.name}: {text}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "caption", "launch", "fetch", "sheet", "spin"])
    ap.add_argument("--case", default="photo21")
    ap.add_argument("--pano", default="pano_s1_paste.jpg")
    ap.add_argument("--gpu", default="h100", choices=list(GPUS))
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--dry", action="store_true", help="launch: everything but creating the job")
    ap.add_argument("--resume", action="store_true", help="launch: ship the generated legs, generate only new ones")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.case, a.pano)
    elif a.cmd == "caption":
        caption(a.case)
    elif a.cmd == "launch":
        launch(a.case, a.gpu, a.max_hours, a.dry, a.resume)
    elif a.cmd == "fetch":
        fetch(a.case)
    elif a.cmd == "sheet":
        sheet(a.case)
    elif a.cmd == "spin":
        spin(a.case)
