"""Next-best-view rounds for a site's GEN3C clips (docs/plan_moverse_photosphere_2026-10-18.md, "Planificación de
recorridos por next-best-view"). Deterministic and without a GPU: the site's current scene (scene.npz and
layers.json, as render_walk.py reads them) is ray cast with Open3D on the CPU.

- audit: what the student's cameras see (a 1 m grid within STUDENT_RADIUS_M, 8 headings, as coverage.py) and which
  of their pixels land on a guess (the backstop, HOLE_LAYERS, nearer than NEAR_M) or on nothing below the horizon.
  The guessed spots are kept as voxels on the backstop, weighted by how many student pixels see them.
- plan: candidate clips that start from a well covered view and move sideways (parallax uncovers, turning does
  not): the perimeter walk (round 1: facing in or out, the score decides) or, later, strafes and arcs around each
  cluster of holes. Each candidate is scored by the student's hole voxels its frames see, and a greedy pick by gain
  (what earlier picks already see does not count again) keeps the best CLIPS_PER_ROUND. Each pick gets its GEN3C
  cache: renders of the current scene along its own path, holes masked, so GEN3C paints only what nobody saw
  (renders of a mesh, not of a point cloud, as Matrix-3D found better).
- launch / fetch: the clips on Nebius with gen3c_probe's runner (model loaded once per job). launch shows the cost
  estimate and needs --confirm (no cloud spend without asking).
- fuse: tsdf_gen3c.py with every round's clips so far, then the new scene's audit.

Run in the SIM2CHIP_SCENE_PYTHON env (Open3D, boto3):

    python experiments/sphere_walk/nbv.py audit  --place <place> [--scene TS2_scene]
    python experiments/sphere_walk/nbv.py plan   --place <place> --round 1 [--scene TS2_scene] [--mode perimeter]
    python experiments/sphere_walk/nbv.py launch --place <place> --round 1 [--jobs 2] [--confirm]
    python experiments/sphere_walk/nbv.py fetch  --place <place> --round 1
    python experiments/sphere_walk/nbv.py fuse   --place <place> --round 1

A round lives in <dir>/walk/N<round>/ (plan.json, plan.png, paths.json, <clip>.npz and, after fetch, the clips);
its fused scene in <dir>/walk/N<round>_scene/, the next round's default scene."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import gen3c_walk as g  # noqa: E402
from gen3c_batch import GPUS, PROMPT, obstacles, place_dir  # noqa: E402
from gen3c_probe import BLOCK, RUNNER, intrinsics  # noqa: E402
from prep import CLEARANCE_M, reachable  # noqa: E402
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

# The student's cameras (coverage.py's grid, at a lower resolution: only shares are counted).
STUDENT_RADIUS_M = 5.0
STUDENT_GRID_M = 1.0
STUDENT_HEADINGS = tuple(range(0, 360, 45))
STUDENT_PITCH_DEG = 15.0
STUDENT_FOV_DEG = 68.0
STUDENT_SIZE = (128, 96)

# What is a hole: a ray that ends on the backstop (our proxy pushed back, right only from the capture point; in a
# B_layers scene its back layer) nearer than NEAR_M, or on nothing below the horizon. The backstop further away is
# far geometry the panorama saw and the fusion does not reach (tsdf_gen3c's MAX_DEPTH_M).
HOLE_LAYERS = ("far", "farback", "back")
NEAR_M = 20.0
VOXEL_M = 0.25        # hole spots matched between the student's rays and a clip's rays
CLUSTER_M = 0.5       # hole voxels this near (from above) are one cluster: what one occluder hides

# Clips: GEN3C's frames, scored at 1/8 of their size on one in SCORE_EVERY.
CLIP_PITCH_DEG = 10.0
SCORE_SIZE = (160, 88)
SCORE_EVERY = 10
# A clip starts from a view we know: GEN3C's first frame is our render, so it must have almost no holes (Taming
# Video Diffusion Prior and ViewCrafter both start from a known view). No frame may be mostly hole: ViewCrafter
# drops poses past 60 % unseen, Taming past 10 %; GEN3C painted 13 % of our first site's views well. A guess.
START_MAX_HOLE = 0.03
FRAME_MAX_HOLE = 0.35
AREA_M = 8.0          # a clip's camera stays this near the capture point
CLIPS_PER_ROUND = 4
MIN_GAIN = 0.001      # a clip must clear this share of the student's pixels
STOP_HOLE = 0.01      # the site is complete under this share of holes
# The perimeter walk: arcs of SEGMENT_M on the movement area's edge (smaller circles where something stands there),
# the camera looking sideways, in or out.
PERIMETER_RADII_M = (STUDENT_RADIUS_M, 4.0, 3.0)
SEGMENT_M = 4.0
# Around a cluster of holes: the camera SPAN_M across, STANDOFFS_M away, from every AZIMUTH_STEP_DEG.
TOP_CLUSTERS = 6
STANDOFFS_M = (3.0, 5.0, 8.0)
AZIMUTH_STEP_DEG = 30
SPAN_M = 3.0

# GEN3C's cache: renders of the current scene at these frames of the clip's own path, plus the first frame turned
# by KEY_TURNS_DEG (what lies beside the path). The first key is GEN3C's input image.
KEY_FRAMES = (0, 30, 60, 90, 120)
KEY_TURNS_DEG = (-60.0, 60.0)
STEPS = 35            # 18 blurred the far buildings on turns (gen3c_probe, 2026-10-19)
PREFIX = "gen3c_nbv"
# Cost estimate per job: setup (pip, weights) and the model's load once, then per clip the cache and 35 steps.
# From gen3c_probe's L40S job (2026-10-19: load 252 s, 12-key cache 376 s, 35 steps 884 s; ~1 h for ~1.8 $).
JOB_SETUP_S = 600
JOB_LOAD_S = 252
CLIP_S = 1100
USD_PER_HOUR = 1.8
L2W = render_plan.LEVEL_TO_WORLD


def heading(dx: float, dy: float) -> float:
    """The heading (degrees, the levelled frame's yaw) looking along world (dx, dy): 0 is +y, 90 is +x."""
    return math.degrees(math.atan2(dx, dy)) % 360


def smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def pack(keys: np.ndarray) -> np.ndarray:
    """Integer voxel coordinates (N, 3) into one int64 each."""
    k = keys.astype(np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def unpack(code: np.ndarray) -> np.ndarray:
    m = (1 << 21) - 1
    return np.stack([code >> 42, (code >> 21) & m, code & m], -1) - (1 << 20)


class Scene:
    """A walk scene (render_walk.py's scene.npz and layers.json) for ray casting: what each ray hits, how far, and,
    for the GEN3C cache, its colour."""

    def __init__(self, scene_dir: Path):
        self.dir = scene_dir
        data = np.load(scene_dir / "scene.npz")
        self.rc = o3d.t.geometry.RaycastingScene()
        self.layers = {}
        for layer in json.loads((scene_dir / "layers.json").read_text()):
            name = layer["name"]
            faces = data[f"{name}_faces"]
            if not len(faces):
                continue
            pts = data[f"{name}_pts"].astype(np.float32)
            gid = self.rc.add_triangles(o3d.core.Tensor(pts), o3d.core.Tensor(faces.astype(np.uint32)))
            entry = {"name": name, "faces": faces, "pts": pts, "hole": name in HOLE_LAYERS}
            if "tex" in layer:
                entry["uv"], entry["tex"] = data[f"{name}_uv"], scene_dir / layer["tex"]
            else:
                entry["col"] = data[f"{name}_col"]
            self.layers[gid] = entry
        self.hole_ids = np.array([k for k, v in self.layers.items() if v["hole"]], np.uint32)
        self._tex = {}

    def cast(self, poses: list, size: tuple[int, int], fov: float) -> dict:
        """Rays through every pixel of every pose (camera-to-world, OpenCV axes, Isaac's world): (N, H, W) arrays."""
        w, h = size
        cam = capture.rays(w, h, fov, 0.0, 0.0).reshape(-1, 3)
        P = np.asarray(poses, np.float64)
        d = np.einsum("nij,kj->nki", P[:, :3, :3], cam)
        o = np.broadcast_to(P[:, None, :3, 3], d.shape)
        rays = np.concatenate([o, d], -1).reshape(-1, 6).astype(np.float32)
        ans = self.rc.cast_rays(o3d.core.Tensor(rays))
        n = len(P)
        t = ans["t_hit"].numpy().reshape(n, h, w)
        gid = ans["geometry_ids"].numpy().reshape(n, h, w)
        miss = ~np.isfinite(t)
        hole = (np.isin(gid, self.hole_ids) & (t < NEAR_M)) | (miss & (d[..., 2].reshape(n, h, w) < 0))
        pts = o.reshape(n, h, w, 3) + np.where(miss, 0, t)[..., None] * d.reshape(n, h, w, 3)
        return {"t": t, "gid": gid, "miss": miss, "hole": hole, "pts": pts, "cam_z": cam[:, 2].reshape(h, w),
                "pid": ans["primitive_ids"].numpy().reshape(n, h, w),
                "bary": ans["primitive_uvs"].numpy().reshape(n, h, w, 2)}

    def hole_voxels(self, hit: dict) -> np.ndarray:
        """The packed voxels of the hole rays that end on a surface (not the misses), one entry per ray."""
        sel = hit["hole"] & ~hit["miss"]
        return pack(np.floor(hit["pts"][sel] / VOXEL_M))

    def colors(self, hit: dict) -> np.ndarray:
        """sRGB uint8 of every ray's hit, black where nothing was hit."""
        out = np.zeros((*hit["t"].shape, 3), np.float32)
        b = hit["bary"]
        wts = np.stack([1 - b[..., 0] - b[..., 1], b[..., 0], b[..., 1]], -1)
        for gid, layer in self.layers.items():
            sel = (hit["gid"] == gid) & ~hit["miss"]
            if not sel.any():
                continue
            pid, w = hit["pid"][sel], wts[sel]
            if "col" in layer:
                c = (layer["col"][layer["faces"][pid]] * w[..., None]).sum(1)
                out[sel] = srgb(c) * 255
            else:
                uv = (layer["uv"].reshape(-1, 3, 2)[pid] * w[..., None]).sum(1)
                tex = self.texture(layer["tex"])
                th, tw = tex.shape[:2]
                x = np.floor((uv[:, 0] % 1.0) * tw).astype(int) % tw
                y = np.clip(np.floor((1 - uv[:, 1]) * th).astype(int), 0, th - 1)
                out[sel] = tex[y, x]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def texture(self, path: Path) -> np.ndarray:
        if path not in self._tex:
            self._tex[path] = np.asarray(Image.open(path).convert("RGB"))
        return self._tex[path]


# --- audit ---------------------------------------------------------------------------------------------------


def student_poses(d: Path, h: float) -> tuple[list, list]:
    obs = obstacles(d)
    r = np.arange(-STUDENT_RADIUS_M, STUDENT_RADIUS_M + 1e-6, STUDENT_GRID_M)
    points = [(float(x), float(y)) for x in r for y in r if math.hypot(x, y) <= STUDENT_RADIUS_M + 1e-6
              and ((x, y) == (0, 0) or reachable(np.array([x, y]), obs))]
    poses, where = [], []
    for x, y in points:
        for hd in STUDENT_HEADINGS:
            v = {"yaw_deg": float(hd), "pitch_deg": STUDENT_PITCH_DEG}
            poses.append(render_plan.camera_pose(v, h, (x, y, 0)))
            where.append((x, y, hd))
    return poses, where


def audit(d: Path, scene: Scene) -> dict:
    """The student's holes: their share of all pixels and, per hole voxel, the share of all pixels that see it."""
    h = json.loads((d / "geometry.json").read_text())["camera_height_m"]
    poses, where = student_poses(d, h)
    codes, shares = [], []
    for k in range(0, len(poses), 64):
        hit = scene.cast(poses[k:k + 64], STUDENT_SIZE, STUDENT_FOV_DEG)
        shares += list(hit["hole"].mean((1, 2)))
        codes.append(scene.hole_voxels(hit))
    codes = np.concatenate(codes)
    total = len(poses) * STUDENT_SIZE[0] * STUDENT_SIZE[1]
    vox, count = np.unique(codes, return_counts=True)
    per_point = {}
    for (x, y, _), s in zip(where, shares):
        per_point[(x, y)] = max(per_point.get((x, y), 0.0), float(s))
    return {"views": len(poses), "hole": float(np.mean(shares)), "voxels": vox, "weight": count / total,
            "worst": [{"x": x, "y": y, "hole": round(s, 4)} for (x, y), s in per_point.items()]}


def clusters(a: dict) -> list[dict]:
    """Hole voxels grouped from above (CLUSTER_M, 8-connected): what one occluder hides. Heaviest first."""
    if not len(a["voxels"]):
        return []
    c = (unpack(a["voxels"]) + 0.5) * VOXEL_M
    ij = np.floor(c[:, :2] / CLUSTER_M).astype(int)
    lo = ij.min(0)
    grid = np.zeros(ij.max(0) - lo + 1, bool)
    grid[tuple((ij - lo).T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    which = lab[tuple((ij - lo).T)]
    out = []
    for k in range(1, n + 1):
        sel = which == k
        w = a["weight"][sel]
        p = (c[sel] * w[:, None]).sum(0) / w.sum()
        out.append({"centre": p.round(2).tolist(), "weight": float(w.sum()), "voxels": int(sel.sum()),
                    "distance_m": round(float(np.hypot(*p[:2])), 1), "heading": round(heading(*p[:2]))})
    return sorted(out, key=lambda q: -q["weight"])


# --- candidates ----------------------------------------------------------------------------------------------


def clip(name: str, kind: str, xy: np.ndarray, yaw: np.ndarray, **info) -> dict:
    return {"name": name, "kind": kind, "xy": xy, "yaw": yaw % 360, **info}


def sectors() -> int:
    return math.ceil(2 * math.pi * PERIMETER_RADII_M[0] / SEGMENT_M)


def perimeter_candidates() -> list[dict]:
    """Arcs on circles around the capture point, the camera looking straight out or in (perpendicular to the walk),
    both ways round. slot: the arc's sector, so the round walks each part of the edge once."""
    n = sectors()
    t = smooth(np.linspace(0, 1, g.FRAMES))
    out = []
    for r in PERIMETER_RADII_M:
        for k in range(n):
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            for way, (s, e) in (("ccw", (a0, a1)), ("cw", (a1, a0))):
                th = s + (e - s) * t
                xy = r * np.stack([np.cos(th), np.sin(th)], -1)
                out_yaw = np.array([heading(math.cos(q), math.sin(q)) for q in th])
                for face, yaw in (("out", out_yaw), ("in", out_yaw + 180)):
                    out.append(clip(f"per_r{r:g}_s{k:02d}_{face}_{way}", "perimeter", xy, yaw, slot=k, radius=r,
                                    facing=face))
    return out


def cluster_candidates(cl: list[dict]) -> list[dict]:
    """Around each heavy cluster: strafes (the camera's heading fixed on the cluster's centre from the middle of the
    walk) and arcs (always looking at it), SPAN_M across, both ways."""
    t = smooth(np.linspace(0, 1, g.FRAMES))
    out = []
    for ci, c in enumerate(cl[:TOP_CLUSTERS]):
        p = np.array(c["centre"][:2])
        for dist in STANDOFFS_M:
            for az in range(0, 360, AZIMUTH_STEP_DEG):
                u = np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])
                mid = p + dist * u
                perp = np.array([-u[1], u[0]])
                half = SPAN_M / 2 / dist  # the arc's half angle for the same span
                for way, sgn in (("a", 1.0), ("b", -1.0)):
                    s = sgn * (t - 0.5) * 2
                    xy = mid + s[:, None] * (SPAN_M / 2) * perp
                    yaw = np.full(g.FRAMES, heading(*(p - mid)))
                    out.append(clip(f"c{ci}_d{dist:g}_az{az:03d}_strafe_{way}", "strafe", xy, yaw, slot=None,
                                    cluster=ci))
                    ang = math.radians(az) + s * half
                    xy = p + dist * np.stack([np.cos(ang), np.sin(ang)], -1)
                    yaw = np.array([heading(*(p - q)) for q in xy])
                    out.append(clip(f"c{ci}_d{dist:g}_az{az:03d}_arc_{way}", "arc", xy, yaw, slot=None, cluster=ci))
    return out


def feasible(c: dict, tree: cKDTree, obs: np.ndarray) -> bool:
    """Inside the area, clear of anything standing by CLEARANCE_M all along, and its start reachable on foot."""
    xy = c["xy"]
    if np.hypot(*xy.T).max() > AREA_M:
        return False
    if len(obs) and tree.query(xy)[0].min() < CLEARANCE_M:
        return False
    return not np.any(xy[0]) or not len(obs) or reachable(xy[0], obs)


def poses_of(c: dict, h: float, frames=None) -> list:
    idx = range(g.FRAMES) if frames is None else frames
    return [render_plan.camera_pose({"yaw_deg": float(c["yaw"][k]), "pitch_deg": CLIP_PITCH_DEG}, h,
                                    (*c["xy"][k], 0.0)) for k in idx]


def score(c: dict, scene: Scene, h: float, voxels: np.ndarray) -> bool:
    """The student's hole voxels the clip's frames see (c["sees"], indices into the audit's voxels) and its frames'
    hole shares; False if it starts or passes through too much unknown."""
    frames = list(range(0, g.FRAMES, SCORE_EVERY)) + ([g.FRAMES - 1] if (g.FRAMES - 1) % SCORE_EVERY else [])
    first = scene.cast(poses_of(c, h, frames[:1]), SCORE_SIZE, g.HFOV)
    c["start_hole"] = float(first["hole"].mean())
    if c["start_hole"] > START_MAX_HOLE:
        return False
    hit = scene.cast(poses_of(c, h, frames[1:]), SCORE_SIZE, g.HFOV)
    c["max_hole"] = float(hit["hole"].mean((1, 2)).max())
    if c["max_hole"] > FRAME_MAX_HOLE:
        return False
    seen = np.unique(np.concatenate([scene.hole_voxels(first), scene.hole_voxels(hit)]))
    i = np.clip(np.searchsorted(voxels, seen), 0, max(len(voxels) - 1, 0))
    c["sees"] = i[voxels[i] == seen] if len(voxels) else np.zeros(0, np.int64)
    return True


def greedy(cands: list[dict], weight: np.ndarray, budget: int) -> list[dict]:
    """Highest gain first, each gain counting only what earlier picks do not already see; one clip per perimeter
    sector. Every clip costs the same (121 frames at 35 steps), so gain per cost is the gain."""
    covered = np.zeros(len(weight), bool)
    picked, slots = [], set()
    while len(picked) < budget:
        best, gain = None, MIN_GAIN
        for c in cands:
            if any(c is q for q in picked) or (c["slot"] is not None and c["slot"] in slots):
                continue
            s = c["sees"]
            gn = float(weight[s[~covered[s]]].sum())
            if gn > gain or (gn == gain and best is not None and c["name"] < best["name"]):
                best, gain = c, gn
        if best is None:
            break
        best["gain"] = gain
        covered[best["sees"]] = True
        picked.append(best)
        if best["slot"] is not None:
            slots.add(best["slot"])
    return picked


# --- GEN3C inputs ---------------------------------------------------------------------------------------------


def level_w2c(pose, h: float) -> np.ndarray:
    """World to camera in the levelled frame (GEN3C's world in gen3c_probe) from a camera-to-world in Isaac's."""
    P = np.asarray(pose, np.float64)
    m = np.eye(4)
    m[:3, :3] = L2W.T @ P[:3, :3]
    m[:3, 3] = L2W.T @ (P[:3, 3] - np.array([0.0, 0.0, h]))
    return np.linalg.inv(m).astype(np.float32)


def cache_npz(c: dict, scene: Scene, h: float, out: Path) -> dict:
    """GEN3C's cache for a clip: our renders at KEY_FRAMES of its path and its first frame turned by KEY_TURNS_DEG,
    holes masked out (depth 0), and the path. The first key is the clip's first frame, GEN3C's input image."""
    path = poses_of(c, h)
    keys = [path[k] for k in KEY_FRAMES]
    y0 = float(c["yaw"][0])
    keys += [render_plan.camera_pose({"yaw_deg": y0 + dt, "pitch_deg": CLIP_PITCH_DEG}, h, (*c["xy"][0], 0.0))
             for dt in KEY_TURNS_DEG]
    imgs, deps = [], []
    for pose in keys:
        hit = scene.cast([pose], (g.W, g.H), g.HFOV)
        imgs.append(scene.colors(hit)[0])
        known = ~hit["hole"][0] & ~hit["miss"][0]
        deps.append(np.where(known, hit["t"][0] * hit["cam_z"], 0).astype(np.float32))
    K = intrinsics()
    dep = np.stack(deps)[:, None]
    np.savez(out / f"{c['name']}.npz",
             images=(np.stack(imgs).transpose(0, 3, 1, 2) / 127.5 - 1).astype(np.float16),
             depth=dep.astype(np.float16), mask=(dep > 0).astype(np.uint8),
             K=np.repeat(K[None], len(keys), 0), w2c=np.stack([level_w2c(p, h) for p in keys]),
             w2cs_all=np.stack([level_w2c(p, h) for p in path]), Ks_all=np.repeat(K[None], len(path), 0))
    Image.fromarray(imgs[0]).save(out / f"{c['name']}_first.jpg", quality=90)
    return {"name": c["name"], "cache": c["name"], "steps": STEPS, "poses": path,
            "view": {"yaw_deg": y0, "pitch_deg": CLIP_PITCH_DEG, "fov_deg": g.HFOV, "size": [g.W, g.H]}}


# --- steps ---------------------------------------------------------------------------------------------------


def round_dir(place: str, k: int) -> Path:
    return place_dir(place) / "walk" / f"N{k:02d}"


def default_scene(place: str, k: int) -> str:
    w = place_dir(place) / "walk"
    for name in [f"N{j:02d}_scene" for j in range(k - 1, 0, -1)] + ["TS_scene", "BL_scene"]:
        if (w / name / "scene.npz").exists():
            return name
    raise SystemExit("no scene to plan from: build B_layers (layers_ours.py) or a TSDF scene first")


def summary(a: dict) -> dict:
    return {"views": a["views"], "hole": round(a["hole"], 4), "hole_voxels": int(len(a["voxels"])),
            "worst_points": sorted(a["worst"], key=lambda q: -q["hole"])[:5]}


def do_audit(place: str, scene_name: str):
    d = place_dir(place)
    t0 = time.time()
    scene = Scene(d / "walk" / scene_name)
    a = audit(d, scene)
    s = {**summary(a), "clusters": clusters(a)[:10], "seconds": round(time.time() - t0)}
    (d / "walk" / scene_name / "audit.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "clusters"}), flush=True)
    return s


def do_plan(place: str, k: int, scene_name: str | None, mode: str | None, budget: int | None):
    d = place_dir(place)
    out = round_dir(place, k)
    out.mkdir(parents=True, exist_ok=True)
    scene_name = scene_name or default_scene(place, k)
    mode = mode or ("perimeter" if k == 1 else "targeted")
    h = json.loads((d / "geometry.json").read_text())["camera_height_m"]
    t0 = time.time()
    scene = Scene(d / "walk" / scene_name)
    print(f"scene {scene_name} loaded in {time.time() - t0:.0f} s", flush=True)
    a = audit(d, scene)
    cl = clusters(a)
    print(f"audit: holes {100 * a['hole']:.2f} % of the student's pixels, {len(a['voxels'])} voxels, "
          f"{len(cl)} clusters ({time.time() - t0:.0f} s)", flush=True)
    plan = {"place": place, "round": k, "scene": scene_name, "mode": mode, "audit": summary(a), "clusters": cl[:10]}
    if a["hole"] < STOP_HOLE:
        plan.update(status="complete", clips=[])
        (out / "plan.json").write_text(json.dumps(plan, indent=1))
        print("complete: holes under", STOP_HOLE, flush=True)
        return
    cands = perimeter_candidates() if mode == "perimeter" else cluster_candidates(cl)
    if budget is None:  # the whole edge once, or a round of CLIPS_PER_ROUND
        budget = sectors() if mode == "perimeter" else CLIPS_PER_ROUND
    obs = obstacles(d)
    tree = cKDTree(obs) if len(obs) else None
    cands = [c for c in cands if feasible(c, tree, obs)]
    scored = [c for c in cands if score(c, scene, h, a["voxels"])]
    print(f"{len(cands)} feasible candidates, {len(scored)} start known ({time.time() - t0:.0f} s)", flush=True)
    picked = greedy(scored, a["weight"], budget)
    clips = []
    for i, c in enumerate(picked):
        c["name"] = f"{place}_N{k:02d}_{i}_{c['name']}"
        clips.append(cache_npz(c, scene, h, out))
        print(f"{c['name']}: gain {100 * c['gain']:.2f} %, start {100 * c['start_hole']:.1f} %, "
              f"max {100 * c['max_hole']:.1f} %", flush=True)
    (out / "paths.json").write_text(json.dumps({"place": place, "paths": clips}))
    gain = sum(c["gain"] for c in picked)
    plan.update(status="planned" if picked else "stuck", candidates=len(cands), scored=len(scored),
                expected_hole=round(a["hole"] - gain, 4),
                clips=[{"name": c["name"], "kind": c["kind"], "gain": round(c["gain"], 4),
                        "start_hole": round(c["start_hole"], 4), "max_hole": round(c["max_hole"], 4),
                        "start_xy": c["xy"][0].round(2).tolist(), "end_xy": c["xy"][-1].round(2).tolist(),
                        **{q: c[q] for q in ("radius", "facing", "cluster") if q in c}} for c in picked],
                estimate=estimate(len(picked), 1), seconds=round(time.time() - t0))
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    draw(a, cl, picked, obs, out / "plan.png")
    print(f"expected holes after the round: {100 * plan['expected_hole']:.2f} % (map {out / 'plan.png'})")


def estimate(clips: int, jobs: int) -> dict:
    per_job = [len(range(j, clips, jobs)) for j in range(jobs)]
    secs = [JOB_SETUP_S + JOB_LOAD_S + n * CLIP_S for n in per_job if n]
    return {"jobs": len(secs), "hours_longest": round(max(secs, default=0) / 3600, 2),
            "usd": round(sum(secs) / 3600 * USD_PER_HOUR, 2)}


def draw(a: dict, cl: list[dict], picked: list[dict], obs: np.ndarray, path: Path):
    """From above: the student's points by their worst heading's holes, the hole voxels, what stands, and each
    picked clip's path with its looking direction every 30 frames."""
    s, r = 40, AREA_M + 4
    size = int(2 * r * s)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    px = lambda x, y: (size / 2 + x * s, size / 2 - y * s)  # noqa: E731
    for x, y in obs[:: max(1, len(obs) // 20000)]:
        if abs(x) < r and abs(y) < r:
            dr.point(px(x, y), fill=(160, 160, 160))
    if len(a["voxels"]):
        c = (unpack(a["voxels"]) + 0.5) * VOXEL_M
        wmax = a["weight"].max()
        for (x, y, _), w in zip(c, a["weight"]):
            q = 1 + 3 * w / wmax
            X, Y = px(x, y)
            dr.ellipse([X - q, Y - q, X + q, Y + q], fill=(220, 40, 40))
    for q in a["worst"]:
        X, Y = px(q["x"], q["y"])
        v = min(1.0, q["hole"] / 0.1)
        dr.ellipse([X - 8, Y - 8, X + 8, Y + 8], fill=(int(255 * v), int(200 * (1 - v)), 60))
    for i, c in enumerate(cl[:TOP_CLUSTERS]):
        X, Y = px(*c["centre"][:2])
        dr.text((X + 6, Y - 6), f"c{i}", fill=(120, 0, 0))
    X0, Y0 = px(0, 0)
    dr.ellipse([X0 - 5, Y0 - 5, X0 + 5, Y0 + 5], outline=(0, 0, 0), width=2)
    for i, c in enumerate(picked):
        col = [(30, 90, 220), (20, 160, 90), (200, 120, 0), (150, 40, 180)][i % 4]
        dr.line([px(*p) for p in c["xy"]], fill=col, width=3)
        for k in range(0, g.FRAMES, 30):
            x, y = c["xy"][k]
            yaw = math.radians(c["yaw"][k])
            dr.line([px(x, y), px(x + 0.8 * math.sin(yaw), y + 0.8 * math.cos(yaw))], fill=col, width=2)
        dr.text(px(*c["xy"][0]), str(i), fill=col)
    dr.text((10, 10), f"holes {100 * a['hole']:.2f} %  (grid {STUDENT_RADIUS_M:g} m, 1 m)", fill=(0, 0, 0))
    img.save(path)


def do_launch(place: str, k: int, jobs: int, max_hours: float, gpu: str, confirm: bool, max_usd: float):
    from sim2chip.cloud_sync import S3Store
    out = round_dir(place, k)
    clips = json.loads((out / "paths.json").read_text())["paths"]
    est = estimate(len(clips), jobs)
    print(f"{len(clips)} clips in {est['jobs']} jobs on {gpu}: ~{est['hours_longest']} h, ~{est['usd']} $ (estimate)")
    if est["usd"] > max_usd:
        raise SystemExit(f"over the cap of {max_usd} $: fewer clips (plan --budget) or a higher --max-usd")
    if not confirm:
        print("nothing launched: run again with --confirm")
        return
    store = S3Store()
    prefix = f"{PREFIX}/{place}/N{k:02d}"
    exp = max_hours + 2
    url = lambda key, op="get_object": store.s3.generate_presigned_url(  # noqa: E731
        op, Params={"Bucket": g.BUCKET, "Key": key}, ExpiresIn=int(exp * 3600))
    cli = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
    platform, preset = GPUS[gpu]
    for j in range(jobs):
        part = clips[j::jobs]
        if not part:
            continue
        fetch = []
        for c in part:
            store.put(g.BUCKET, f"{prefix}/in/{c['cache']}.npz", (out / f"{c['cache']}.npz").read_bytes())
            key = f"{prefix}/in/{c['cache']}.npz"
            fetch.append(f'curl -s -o /tmp/in/{c["cache"]}.npz "{url(key)}"')
        spec = [{"name": c["name"], "npz": f"/tmp/in/{c['cache']}.npz", "steps": c["steps"], "prompt": PROMPT,
                 "out": url(f"{prefix}/out/{c['name']}.mp4", "put_object"),
                 "buf": url(f"{prefix}/out/{c['name']}_buffer.mp4", "put_object")} for c in part]
        store.put(g.BUCKET, f"{prefix}/in/clips{j}.json", json.dumps(spec).encode())
        script = out / f"job{j}.sh"
        script.write_text(g.JOB.replace("__CASES__", BLOCK.format(clips=url(f"{prefix}/in/clips{j}.json"),
                                                                 fetch="\n".join(fetch), runner=RUNNER)))
        assert script.stat().st_size < 60_000, "the script must fit Nebius' 64 KiB inject limit"
        args = [cli, "ai", "job", "create", "--name", f"eyeforge-gen3c-nbv{k}-{j}-{int(time.time())}", "--image",
                g.IMAGE, "--platform", platform, "--preset", preset, "--on-demand", "--timeout",
                f"{int(max_hours * 60)}m", "--parent-id", os.environ["SIM2CHIP_NEBIUS_PROJECT"], "--inject-file",
                f"{script}:/opt/gen3c_job.sh", "--container-command", "bash", "--args", "/opt/gen3c_job.sh",
                "--env", f"LOG_URL={url(f'{prefix}/out/job{j}.log', 'put_object')}", "--async"]
        r = subprocess.run(args, capture_output=True, text=True)
        print(f"job {j}: {len(part)} clips", r.stdout[-300:], r.stderr[-500:])


def do_fetch(place: str, k: int):
    from sim2chip.cloud_sync import S3Store
    out = round_dir(place, k)
    store = S3Store()
    for key in store.listing(g.BUCKET, f"{PREFIX}/{place}/N{k:02d}/out/"):
        dst = out / key.rsplit("/", 1)[-1]
        dst.write_bytes(store.s3.get_object(Bucket=g.BUCKET, Key=key)["Body"].read())
        print(dst)


def do_fuse(place: str, k: int, extra: list[str]):
    """tsdf_gen3c.py over G_gen3c's straight paths (if any), every round up to k and `extra` clips, into
    N<k>_scene, then its audit."""
    adds = list(extra)
    for j in range(1, k + 1):
        rd = round_dir(place, j)
        if (rd / "paths.json").exists():
            adds += [f"{rd.name}/{c['name']}" for c in json.loads((rd / "paths.json").read_text())["paths"]
                     if (rd / f"{c['name']}.mp4").exists()]
    cmd = [sys.executable, str(HERE / "tsdf_gen3c.py"), "--dir", str(place_dir(place)), "--out", f"N{k:02d}_scene"]
    if adds:
        cmd += ["--add", *adds]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    s = do_audit(place, f"N{k:02d}_scene")
    plan = json.loads((round_dir(place, k) / "plan.json").read_text())
    print(f"holes: {100 * plan['audit']['hole']:.2f} % before, {100 * plan.get('expected_hole', 0):.2f} % expected, "
          f"{100 * s['hole']:.2f} % now")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["audit", "plan", "launch", "fetch", "fuse"])
    ap.add_argument("--place", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--scene", help="the scene folder under walk/ (default: the last round's, else TS_scene, "
                    "else BL_scene)")
    ap.add_argument("--mode", choices=["perimeter", "targeted"], help="default: perimeter in round 1")
    ap.add_argument("--budget", type=int, help=f"clips in the round (default: one per perimeter sector, "
                    f"{CLIPS_PER_ROUND} for a targeted round)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=2.5)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--gpu", choices=list(GPUS), default="l40s")
    ap.add_argument("--confirm", action="store_true", help="launch: spend on Nebius (asked first)")
    ap.add_argument("--extra", nargs="*", default=[], help="fuse: more clips, <folder under walk>/<clip>")
    a = ap.parse_args()
    if a.step == "audit":
        do_audit(a.place, a.scene or default_scene(a.place, a.round))
    elif a.step == "plan":
        do_plan(a.place, a.round, a.scene, a.mode, a.budget)
    elif a.step == "launch":
        do_launch(a.place, a.round, a.jobs, a.max_hours, a.gpu, a.confirm, a.max_usd)
    elif a.step == "fetch":
        do_fetch(a.place, a.round)
    else:
        do_fuse(a.place, a.round, a.extra)


if __name__ == "__main__":
    main()
