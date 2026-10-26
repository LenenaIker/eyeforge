"""WorldStereo 2.0 (github.com/FuchengSu/WorldStereo, code Apache 2.0, weights MIT, base Wan2.1-I2V-14B Apache 2.0)
on the route's panorama (pano.py): several walks out of the photosphere along the walkable directions, each one
remembering the frames the earlier walks generated (the model's spatial-stereo memory). Upstream has not released
its panorama preprocessing (HY-World 2.0's WorldNav does it, under a licence that excludes the EU), so prep builds
the scene folder run_multi_traj.py reads:

    pano_bank/images/0000.png + cameras.json   27 perspective crops of the panorama (the initial memory)
    reconstruct_<k>/start_frame.png            the panorama seen from the walk's first camera
    reconstruct_<k>/traj0/                     walk k, leg 1: camera.json, render.mp4, render_mask.mp4,
                                               traj_caption.json (point cloud of the panorama rendered on the path)
    reconstruct_<k>/traj1/                     leg 2, on from the end of leg 1 (starts on leg 1's last frame)
    wonder_0/traj0/                            along walk 0 turning round to face the start (memory check)
    wonder_1/traj0/                            walk 0's first half, climbing while looking down

Runs in SIM2CHIP_SCENE_PYTHON (pytorch3d, MoGe-2)."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import camera
from .camera import CAMERA_HEIGHT, NFRAME, OBSTACLE_Z, PITCH, H, W
from .panorama import PANO_W, contract
from .paths import PANOS, PHOTOS, Case, add_moge, moge_ckpt, tools

CORRIDOR = 0.5  # half width, metres: a walker this wide stops at the first obstacle point
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
GROUND_BELOW = -0.5  # the scale's ground: what lies 30+ degrees below the horizon (direction z under this)...
GROUND_WITHIN = 15.0  # ...within this many (unscaled) metres


# ── depth and walkable directions ───────────────────────────────────────────────────────────────────────────

def pano_distance(pano: np.ndarray, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """MoGe-2 on 20 perspective views of the panorama, merged into one distance map (MoGe's own panorama
    recipe, with the metric v2 model). Returns distance (PANO_W/2 x PANO_W) and its valid mask (False = sky)."""
    if cache.exists():
        z = np.load(cache)
        return z["distance"], z["mask"]
    import cv2
    import torch
    add_moge()
    from moge.model.v2 import MoGeModel
    from moge.utils.panorama import get_panorama_cameras, merge_panorama_depth, split_panorama_image
    try:
        import utils3d_moge as utils3d
    except ImportError:
        import utils3d
    model = MoGeModel.from_pretrained(moge_ckpt()).cuda().eval()
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


def obstacle_xy(points: np.ndarray, within: float) -> np.ndarray:
    """(x, y) of the points between OBSTACLE_Z above the ground, closer than within metres."""
    hgt = points[:, 2] + CAMERA_HEIGHT
    obs = points[(hgt > OBSTACLE_Z[0]) & (hgt < OBSTACLE_Z[1])][:, :2]
    return obs[np.linalg.norm(obs, axis=1) < within]


def walkable(points: np.ndarray, step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Free distance along each azimuth: how far a CORRIDOR-wide walker goes before an obstacle point."""
    obs = obstacle_xy(points, MAX_FREE + MARGIN + 1)
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
    obs = obstacle_xy(points, CENTRE_SEARCH + CENTRE_CLEAR + 1)
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

    def gap(y, ys):
        return min(abs((y - c + 180) % 360 - 180) for c in ys)
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
        e = camera.smooth(t)
        pos = np.asarray(start, float) + (np.asarray(end, float) - np.asarray(start, float)) * t
        out.append(camera.w2c(pos + np.array([0, 0, z[0] + (z[1] - z[0]) * e]), yaw,
                              pitch[0] + (pitch[1] - pitch[0]) * e))
    return out


def straight(start, yaw: float, length: float) -> list[np.ndarray]:
    y = math.radians(yaw)
    d = np.array([math.cos(y), math.sin(y), 0.0])
    return [camera.w2c(np.asarray(start) + d * length * t, yaw, PITCH) for t in np.linspace(0, 1, NFRAME)]


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
    back = [camera.w2c((c + d * lb) * t, yaw + 180 * camera.smooth(t), PITCH) for t in np.linspace(0, 1, NFRAME)]
    trajs["wonder_0/traj0"] = dict(yaw=yaw, poses=back, kind="turn back", length=lb)
    lr = min(LEG, free) * 0.5
    trajs["wonder_1/traj0"] = dict(yaw=yaw, poses=between([0, 0, 0], c + d * lr, yaw, (0.0, RAISE),
                                                          (PITCH, RAISE_PITCH)), kind="raise", length=lr)
    return trajs


# ── rendering ───────────────────────────────────────────────────────────────────────────────────────────────

def renderer():
    """Upstream's point renderer (pytorch3d, radius 0.008, 20 points per pixel, grey where no point lands);
    kornia is only imported for an edge filter the route does not use."""
    sys.modules.setdefault("kornia", types.ModuleType("kornia"))
    sys.path.insert(0, str(tools() / "WorldStereo"))
    from src.pointcloud import point_rendering
    return point_rendering


def render(point_rendering, points, colors, poses) -> tuple[np.ndarray, np.ndarray]:
    """RGB and hole mask (255 where no point lands) of the point cloud at each pose."""
    import torch
    k = torch.tensor([camera.intrinsic()] * len(poses), dtype=torch.float32)
    rgb, holes = [], []
    for i in range(0, len(poses), 4):
        r, m = point_rendering(K=k[i:i + 4], w2cs=torch.tensor(np.stack(poses[i:i + 4]), dtype=torch.float32),
                               points=points, colors=colors, device="cuda", h=H, w=W, render_radius=0.008,
                               points_per_pixel=20, background_color=[0, 0, 0])
        rgb.append(((r.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy())
        holes.append((m[:, 0] * 255).byte().cpu().numpy())
    return np.concatenate(rgb), np.concatenate(holes)


def write_leg(d: Path, poses: list, rgb: np.ndarray, holes: np.ndarray, kind: str):
    """A leg folder of WorldStereo's input: render.mp4, render_mask.mp4 and camera.json."""
    camera.write_video(d / "render.mp4", rgb)
    camera.write_video(d / "render_mask.mp4", holes)
    (d / "camera.json").write_text(json.dumps({
        "extrinsic": [p.tolist() for p in poses], "intrinsic": [camera.intrinsic()] * NFRAME,
        "width": W, "height": H, "type": kind, "rotation_deg": 0}))


def prep(case_name: str, pano_name: str):
    """The panorama's depth and scale, the walks and the scene folder; then the sheet the user approves before
    the cloud run."""
    import cv2
    import torch
    case = Case(case_name)
    out = case.scene
    out.mkdir(parents=True, exist_ok=True)
    full = np.asarray(Image.open(PANOS / case.name / pano_name).convert("RGB"))
    pano = cv2.resize(full, (PANO_W, PANO_W // 2), interpolation=cv2.INTER_AREA)
    dist, valid = pano_distance(pano, case.distance)
    dirs = camera.pano_dirs(*dist.shape)
    keep = valid & ~camera.edge_mask(dist) & np.isfinite(dist) & (dist > 0)

    # Scale: the ground (median height of what lies 30+ degrees below the horizon, within 15 m) sits
    # CAMERA_HEIGHT below the camera.
    pts = dirs * dist[..., None]
    low = keep & (dirs[..., 2] < GROUND_BELOW) & (dist < GROUND_WITHIN)
    scale = CAMERA_HEIGHT / -np.median(pts[low][:, 2])
    pts *= scale
    dist = dist * scale

    c = centre(pts[keep])
    yaws, free = walkable(pts[keep] - c)
    walks = pick_directions(yaws, free)
    print(f"{case.name}: scale {scale:.3f}, centre {c[:2].round(2).tolist()}, walks (yaw, free m): "
          f"{[(round(y), round(f, 1)) for y, f in walks]}")
    if not walks:
        raise SystemExit("no walkable direction")
    trajs = plan(walks, c)

    points = torch.tensor(dirs[keep] * contract(dist[keep])[:, None], dtype=torch.float32)
    colors = torch.tensor(pano[keep] / 127.5 - 1, dtype=torch.float32)
    point_rendering = renderer()

    bank = out / "pano_bank" / "images"
    bank.mkdir(parents=True, exist_ok=True)
    cams = {}
    for i, m in enumerate(camera.bank_cameras()):
        Image.fromarray(camera.crop(full, m)).save(bank / f"{i:04d}.png")
        cams[f"{i:04d}"] = {"extrinsic": m.tolist(), "intrinsic": camera.intrinsic()}
    (out / "pano_bank" / "cameras.json").write_text(json.dumps(cams))

    # One caption per scene (erase.caption: the place alone, no movable objects).
    caption = PHOTOS / f"{case.name}.txt"
    caption = caption.read_text().strip() if caption.exists() else ""
    meta = {"case": case.name, "pano": pano_name, "scale": scale, "camera_height": CAMERA_HEIGHT,
            "centre": c.tolist(), "walks": walks, "free": dict(zip(map(float, yaws), map(float, free))), "trajs": {}}
    for name, t in trajs.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        start = d.parent / "start_frame.png"
        if name.endswith("traj0"):
            Image.fromarray(camera.crop(full, t["poses"][0])).save(start)
        rgb, holes = render(point_rendering, points, colors, t["poses"])
        if name.endswith("traj0"):
            rgb[0], holes[0] = np.asarray(Image.open(start)), 0
        write_leg(d, t["poses"], rgb, holes, t["kind"])
        cap = d / "traj_caption.json"
        if not cap.exists():
            cap.write_text(json.dumps({"prompt": caption}))
        meta["trajs"][name] = {"yaw": t["yaw"], "kind": t["kind"], "length": t["length"],
                               "holes": [round(float((h > 127).mean()), 3) for h in holes]}
        Image.fromarray(np.concatenate([rgb[0], rgb[NFRAME // 2], rgb[-1]], 1)).save(d.parent / f"{d.name}_render.jpg")
        print(name, t["kind"], f"{t['length']:.1f} m", "holes at end", meta["trajs"][name]["holes"][-1], flush=True)
    case.prep_json.write_text(json.dumps(meta, indent=1))
    prep_sheet(case, full, pts[keep], walks, trajs, yaws, free)


def prep_sheet(case: Case, full, pts, walks, trajs, yaws, free):
    """Panorama with the chosen directions, a top view (obstacles, walks) and each leg's render at start,
    middle and end: what the user approves before the cloud run."""
    from PIL import ImageFont
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
    ImageDraw.Draw(head).text((820, 20), f"{case.name}: {len(trajs)} legs\n" + "\n".join(
        f"{n}: {t['kind']} {t['length']:.1f} m" for n, t in trajs.items()), fill=(255, 255, 255), font=font)
    rows.append(head)
    for name in trajs:
        im = Image.open(case.scene / f"{name}_render.jpg").resize((1600, 308))
        ImageDraw.Draw(im).text((10, 10), name, fill=(255, 255, 0), font=font)
        rows.append(im)
    save_rows(rows, case.dir / "sheet_prep.jpg")


def save_rows(rows: list, path: Path, width: int | None = None):
    """Images stacked top to bottom into one JPEG sheet."""
    sheet = Image.new("RGB", (width or max(r.width for r in rows), sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(path, quality=88)
    print(path)


def result_sheet(case_name: str):
    """Per leg: the point-cloud guide (top) against WorldStereo's frames (bottom) at 0, 1/4, 1/2, 3/4 and the
    end. Needs a video reader (SIM2CHIP_SCENE_PYTHON)."""
    import imageio.v3 as iio
    from PIL import ImageFont
    case = Case(case_name)
    font = ImageFont.load_default(size=22)
    cw, chh = 400, 231
    rows = []
    for vid in sorted(case.results.glob("*/traj*/*_result.mp4"), key=lambda p: (p.parts[-2] == "traj1", str(p))):
        name = f"{vid.parts[-3]}/{vid.parts[-2]}"
        gen = iio.imread(vid, plugin="pyav")
        ren = iio.imread(case.scene / name / "render.mp4", plugin="pyav")
        idx = [round(i * (len(gen) - 1) / 4) for i in range(5)]
        row = Image.new("RGB", (cw * 5, chh * 2 + 30), (0, 0, 0))
        for j, i in enumerate(idx):
            row.paste(Image.fromarray(ren[min(i, len(ren) - 1)]).resize((cw, chh)), (j * cw, 30))
            row.paste(Image.fromarray(gen[i]).resize((cw, chh)), (j * cw, 30 + chh))
        ImageDraw.Draw(row).text((8, 4), f"{name}: guide (top), WorldStereo (bottom)", fill=(255, 255, 0), font=font)
        rows.append(row)
    save_rows(rows, case.dir / "sheet_result.jpg", cw * 5)
