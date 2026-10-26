"""Next-best-view for a WorldStereo scene: the walks' frames are fitted into a splat, and the next legs look where
no generated frame backs the splat (conflict.py's targets: no data and holes, over every student camera).

- audit: the student cameras render the splat; each target pixel becomes a (voxel, direction bin) key weighted by
  how many student pixels need it, and the keys are grouped from above into clusters (what one car hides).
- plan: candidate legs start on a frame WorldStereo already made (its image is the leg's start frame: sharp and
  consistent) and move sideways, diagonally or back, or arc round a cluster, turning at most MAX_TURN_DEG over the
  leg. A leg must stay in the area, clear of obstacles, and no frame may be mostly unknown. Each is scored by the
  target keys its frames see; a greedy pick by gain keeps the best. The picked legs are written as WorldStereo
  inputs: start_frame.png, camera.json, the splat's render as guide with the targets as holes (render_mask) and
  the panorama's sky, the caption of the leg it starts from.
- stage: the planned legs into the scene folder; then the resume step generates only them, after the walks.

Writes data/worldstereo_test/<case>/nbv/<arm>/: audit.npz, audit.json, sheet_audit.jpg, plan.json, sheet_plan.jpg
and legs/<leg>/traj0/ (SIM2CHIP_GSPLAT_PYTHON)."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import camera, student
from .camera import NFRAME, H, W
from .conflict import Evidence, evidence_views
from .paths import Case
from .config import arm_views

STUDENT_SIZE = (128, 96)
MIN_VIEW_DEPTH_M = 2.0  # a student view whose median depth is nearer faces a hedge from arm's length: left out
VOXEL_M = 0.25
BIN_DEG = 20.0         # target keys: voxel and the student's direction to it, in BIN_DEG azimuth bins
CLUSTER_M = 0.5
# Legs: 21 frames like the walks', from a made frame.
START_EVERY = 2        # every 2nd frame of each walk is a start
START_SKIP = ("nbv_", "spin_")  # legs that are not starts (an earlier plan's, the closed 360 test's)
MOVES_DEG = (90, -90, 45, -45, 135, -135, 180)  # direction of travel against the start's heading
MOVE_M = (2.0, 3.5)
TOP_CLUSTERS = 6
ARC_M = 3.5
ARC_RANGE_M = (2.0, 9.0)
MAX_TURN_DEG = 60.0    # over the leg: about 3 deg per frame, the user found the walks' 9 too fast (2026-10-22)
AREA_MIN_M = 6.0       # legs stay within the student area + 1 m, and never less than this
SCORE_FRAMES = (0, 5, 10, 15, 20)
SCORE_SCALE = 4


def keys_of(X, from_pos):
    """Target keys: voxel of X and the azimuth bin of the direction from from_pos to X, packed in one int64."""
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


def bin_shift(keys: np.ndarray, s: int) -> np.ndarray:
    """The same voxels in the neighbouring direction bin (a view from nearly the same direction may sit there)."""
    n = int(360 // BIN_DEG)
    return keys - keys % 32 + (keys % 32 + s) % n


def clusters(keys: np.ndarray, weight: np.ndarray) -> list[dict]:
    """Target voxels grouped from above (CLUSTER_M, 8-connected), heaviest first."""
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


def nbv_classify(sp, ev: Evidence, w2cs: np.ndarray, K: np.ndarray, w: int, h: int):
    """Next-best-view's targets: code 0 sky, 2 known (confirmed or in conflict), 3 target (no data, or a hole); the
    target keys; RGB; world points (holes take the depth of the nearest surface pixel, so a leg that looks there
    counts as seeing them); and the non-sky mask."""
    import torch
    from scipy import ndimage
    rgb, dep, al = student.render(sp, w2cs, K, w, h)
    surf = al > 0.5
    fill = dep.clone()
    for i in range(len(dep)):
        s = surf[i].cpu().numpy()
        if s.any() and not s.all():
            j = ndimage.distance_transform_edt(~s, return_distances=False, return_indices=True)
            fill[i] = torch.as_tensor(dep[i].cpu().numpy()[j[0], j[1]], device=dep.device)
    X = camera.lift(fill, K, w2cs)
    e = ev.points(X.reshape(-1, 3), colour=False)
    has = ((e["agree"] + e["pano_agree"] + e["contra"] + e["pano_contra"]) > 0).reshape(dep.shape)
    sky = torch.as_tensor(np.stack([ev.sky(m, K, w, h) for m in w2cs]), device=dep.device) & ~surf
    code = torch.full(dep.shape, 2, dtype=torch.uint8, device=dep.device)
    code[~has | ~surf] = 3
    code[sky] = 0
    pos = torch.as_tensor(np.linalg.inv(np.asarray(w2cs))[:, :3, 3], dtype=torch.float32, device=dep.device)
    P = pos[:, None, None].expand_as(X)
    tgt = code == 3
    return code, keys_of(X[tgt], P[tgt]), rgb, X, code != 0


def evidence(case: Case, arm: str, sp) -> Evidence:
    return Evidence(case, sp, evidence_views(arm))


def audit(case_name: str, arm: str, radius: float):
    import torch
    case = Case(case_name)
    t0 = time.time()
    dev = torch.device("cuda")
    sp = student.load_splat(case.splat_ply(arm), dev)
    ev = evidence(case, arm, sp)
    obs = student.obstacles(case, sp)
    poses, where = student.poses(obs, radius, student.seed(case))
    K = camera.intrinsic_fov(*STUDENT_SIZE, student.STUDENT_FOV_DEG)
    _, dep, alpha = student.render(sp, poses, K, *STUDENT_SIZE)
    med = torch.stack([d[a > student.ALPHA_MIN].median() if (a > student.ALPHA_MIN).any() else d.new_tensor(1e6)
                       for d, a in zip(dep, alpha)]).cpu().numpy()
    keep = med >= MIN_VIEW_DEPTH_M
    poses, where = poses[keep], [w for w, k in zip(where, keep) if k]
    shares, keys, codes, rgbs = [], [], [], []
    for s in range(0, len(poses), 32):
        code, k, rgb, _, _ = nbv_classify(sp, ev, poses[s:s + 32], K, *STUDENT_SIZE)
        shares += (code == 3).float().mean((1, 2)).tolist()
        keys.append(k.cpu().numpy())
        codes.append(code.cpu().numpy())
        rgbs.append(rgb.cpu().numpy())
    keys = np.concatenate(keys)
    total = len(poses) * STUDENT_SIZE[0] * STUDENT_SIZE[1]
    uk, cnt = np.unique(keys, return_counts=True)
    codes, rgbs = np.concatenate(codes), np.concatenate(rgbs)
    d = case.nbv(arm)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "audit.npz", keys=uk, weight=cnt / total, poses=poses, where=np.array(where),
                        shares=np.array(shares), obstacles=obs.astype(np.float32))
    cl = clusters(uk, cnt / total)
    info = {"arm": arm, "radius": radius, "views": len(poses), "facing_hedge": int((~keep).sum()),
            "positions": len({(x, y) for x, y, _ in where}),
            "hole": float(np.mean(shares)), "sky": float((codes == 0).mean()), "known": float((codes == 2).mean()),
            "keys": int(len(uk)), "clusters": cl[:10], "seconds": round(time.time() - t0)}
    (d / "audit.json").write_text(json.dumps(info, indent=1))
    audit_sheet(case, d, info, obs, uk, cnt / total, cl, where, shares, codes, rgbs, radius)
    print(json.dumps({k: v for k, v in info.items() if k != "clusters"}), flush=True)


def top_view(obs: np.ndarray, paths: list, radius: float, keys=None, weight=None, cl=None, picked=None,
             size: int = 800, span: float = 20.0):
    """20 m square from above: x (the panorama's centre) up, y (left) left. Obstacles grey, walked cameras blue,
    target weight red, clusters numbered, picked legs coloured."""
    im = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(im)

    def px(x, y):
        return size / 2 - y / span * size, size / 2 - x / span * size
    r = radius / span * size
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


def fitted_legs(case: Case, arm: str) -> set[str]:
    """The generated legs the arm was fitted to (its views file)."""
    return {str(n).rsplit("/", 1)[0] for n in np.load(case.splat / arm_views(arm))["name"]
            if not str(n).startswith(("pano/", "fixer_"))}


def walked_paths(case: Case, arm: str) -> list:
    """Camera paths of the legs the arm was fitted to."""
    out = []
    for leg in sorted(fitted_legs(case, arm)):
        cams = json.loads((case.scene / leg / "camera.json").read_text())["extrinsic"]
        out.append([tuple(camera.pose_of(np.array(m))[0][:2]) for m in cams])
    return out


def overlay(rgb: np.ndarray, code: np.ndarray) -> np.ndarray:
    o = rgb.astype(np.float32)
    o[code == 3] = o[code == 3] * 0.35 + np.array([255, 0, 0]) * 0.65
    return o.astype(np.uint8)


def audit_sheet(case, d, info, obs, keys, weight, cl, where, shares, codes, rgbs, radius):
    from PIL import ImageFont
    font = ImageFont.load_default(size=20)
    tv = top_view(obs, walked_paths(case, info["arm"]), radius, keys, weight, cl)
    order = np.argsort(shares)[::-1]
    pick = list(order[:8]) + list(order[len(order) // 2:len(order) // 2 + 4])
    tw, th = 384, 288
    sh = Image.new("RGB", (tv.width + 3 * tw, max(tv.height, 4 * th) + 40), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    dr.text((8, 8), f"{case.name} audit on {info['arm']}: student targets {info['hole']:.1%} of pixels (known "
            f"{info['known']:.1%}, sky {info['sky']:.1%}); red = no generated frame backs it",
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


def starts(case: Case, arm: str) -> list[dict]:
    """Every START_EVERY-th frame of each walk the arm was fitted to: its pose, image and caption."""
    out = []
    fitted = fitted_legs(case, arm)
    for leg in [g for g in case.legs(START_SKIP) if g in fitted]:
        cams = json.loads((case.scene / leg / "camera.json").read_text())["extrinsic"]
        frames = camera.video_frames(case.leg_video(leg), [k / (len(cams) - 1) for k in range(len(cams))])
        cap = case.scene / leg / "traj_caption.json"
        for f in range(0, len(cams), START_EVERY):
            p, yaw, pitch = camera.pose_of(np.array(cams[f]))
            out.append({"from": f"{leg}/{f:02d}", "pos": p, "yaw": yaw, "pitch": pitch, "image": frames[f],
                        "caption": json.loads(cap.read_text())["prompt"] if cap.exists() else ""})
    return out


def candidates(st: list[dict], cl: list[dict]) -> list[dict]:
    t = camera.smooth(np.linspace(0, 1, NFRAME))
    out = []
    for s in st:
        p0 = s["pos"]
        for mv in MOVES_DEG:
            a = math.radians(s["yaw"] + mv)
            for dist in MOVE_M:
                xy = p0[:2] + t[:, None] * dist * np.array([math.cos(a), math.sin(a)])
                out.append({"name": f"{s['from']}_move{mv:+d}_{dist:g}", "kind": "move", "start": s, "xy": xy,
                            "yaw": np.full(NFRAME, s["yaw"])})
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
                off = camera.wrap(s["yaw"] - look[0])
                yaw = look + (1 - t) * off
                if np.abs(camera.wrap(yaw - s["yaw"])).max() > MAX_TURN_DEG:
                    continue
                out.append({"name": f"{s['from']}_arc{ci}{'+' if sgn > 0 else '-'}", "kind": "arc", "start": s,
                            "xy": xy, "yaw": yaw, "cluster": ci})
    return out


def leg_w2cs(c: dict) -> np.ndarray:
    z = float(c["start"]["pos"][2])
    return np.stack([camera.w2c([x, y, z], yw, c["start"]["pitch"]) for (x, y), yw in zip(c["xy"], c["yaw"])]
                    ).astype(np.float32)


def greedy(cands: list[dict], weight: np.ndarray, budget: int, min_gain: float) -> list[dict]:
    covered = np.zeros(len(weight), bool)
    picked = []
    while len(picked) < budget:
        best, gain = None, min_gain
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


def plan(case_name: str, arm: str, radius: float, legs: int, max_unknown: float, min_gain: float, prefix: str):
    """The legs that see the most target weight; written as legs/<prefix><k>/traj0."""
    import torch
    case = Case(case_name)
    t0 = time.time()
    dev = torch.device("cuda")
    d = case.nbv(arm)
    a = np.load(d / "audit.npz")
    hk, hw, obs = a["keys"], a["weight"], a["obstacles"]
    cl = json.loads((d / "audit.json").read_text())["clusters"]
    sp = student.load_splat(case.splat_ply(arm), dev)
    ev = evidence(case, arm, sp)
    free = student.free_space(obs, student.seed(case))
    cands = candidates(starts(case, arm), cl)
    K = np.array(camera.intrinsic(), np.float32)
    K[:2] /= SCORE_SCALE
    w, h = W // SCORE_SCALE, H // SCORE_SCALE
    area = max(AREA_MIN_M, radius + 1.0)
    ok = []
    for c in cands:
        # The start is a walked frame: already there, so only the rest must keep clear.
        if np.hypot(*c["xy"].T).max() > area or not free(c["xy"][1:]).all():
            continue
        code, _, _, X, surf = nbv_classify(sp, ev, leg_w2cs(c)[list(SCORE_FRAMES)], K, w, h)
        unk = (code == 3).float().mean((1, 2))
        c["unknown"] = unk.tolist()
        if unk.max().item() > max_unknown:
            continue
        # What its frames see counts, target for them or not: the leg looks at it from its own direction.
        P = torch.as_tensor(np.linalg.inv(leg_w2cs(c)[list(SCORE_FRAMES)])[:, :3, 3], device=dev)
        P = P[:, None, None].expand_as(X)
        seen = keys_of(X[surf], P[surf]).unique().cpu().numpy()
        near = np.unique(np.concatenate([bin_shift(seen, s) for s in (-1, 0, 1)]))
        i = np.clip(np.searchsorted(hk, near), 0, len(hk) - 1)
        c["sees"] = i[hk[i] == near]
        ok.append(c)
    print(f"{len(cands)} candidates, {len(ok)} feasible, {time.time() - t0:.0f} s", flush=True)
    picked = greedy(ok, hw, legs, min_gain)
    names = [f"{prefix}{k:02d}" for k in range(len(picked))]
    write_legs(d, sp, ev, picked, names)
    info = {"arm": arm, "candidates": len(cands), "feasible": len(ok), "hole_weight": float(hw.sum()),
            "covered": float(sum(c["gain"] for c in picked)), "seconds": round(time.time() - t0),
            "legs": [{"name": n, "from": c["start"]["from"], "kind": c["kind"], "candidate": c["name"],
                      "gain": round(c["gain"], 5), "max_unknown": round(max(c["unknown"]), 3),
                      "length_m": round(float(np.linalg.norm(np.diff(c["xy"], axis=0), axis=1).sum()), 2)}
                     for n, c in zip(names, picked)]}
    (d / "plan.json").write_text(json.dumps(info, indent=1))
    plan_sheet(case, d, info, obs, hk, hw, cl, picked, radius)
    print(json.dumps({k: v for k, v in info.items() if k != "legs"}), flush=True)
    for leg in info["legs"]:
        print(leg, flush=True)


def write_legs(d: Path, sp, ev: Evidence, picked: list[dict], names: list[str]):
    """WorldStereo inputs per picked leg: the splat's render as guide, the targets as holes, the panorama's sky."""
    out = d / "legs"
    if out.exists():
        shutil.rmtree(out)
    K = np.array(camera.intrinsic(), np.float32)
    for c, name in zip(picked, names):
        t = out / name / "traj0"
        t.mkdir(parents=True)
        m = leg_w2cs(c)
        code, _, rgb, _, _ = nbv_classify(sp, ev, m, K, W, H)
        rgb = rgb.cpu().numpy()
        hole, sky = (code == 3).cpu().numpy(), (code == 0).cpu().numpy()
        for f in range(len(m)):  # sky from the panorama, so the guide leaves only the targets to invent
            rgb[f][sky[f]] = ev.sky_colour(m[f], K, W, H)[sky[f]]
        rgb[hole] = 0
        rgb[0], hole[0] = c["start"]["image"], False
        Image.fromarray(c["start"]["image"]).save(t.parent / "start_frame.png")
        camera.write_video_cv(t / "render.mp4", list(rgb))
        camera.write_video_cv(t / "render_mask.mp4", list(np.repeat((hole * 255).astype(np.uint8)[..., None], 3, -1)))
        (t / "camera.json").write_text(json.dumps({
            "extrinsic": [p.tolist() for p in m.astype(np.float64)], "intrinsic": [camera.intrinsic()] * NFRAME,
            "width": W, "height": H, "type": c["kind"], "rotation_deg": 0}))
        (t / "traj_caption.json").write_text(json.dumps({"prompt": c["start"]["caption"]}))
        c["guide"] = [rgb[f] for f in (0, NFRAME // 2, NFRAME - 1)]
        c["holes"] = float(hole[1:].mean())


def plan_sheet(case, d, info, obs, hk, hw, cl, picked, radius):
    from PIL import ImageFont
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    tv = top_view(obs, walked_paths(case, info["arm"]), radius, hk, hw, cl, picked)
    tw, th = 320, 185
    sh = Image.new("RGB", (tv.width + 3 * tw, max(tv.height, len(picked) * (th + 22)) + 40), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    dr.text((8, 8), f"{case.name} plan on {info['arm']}: {len(picked)} legs cover {info['covered']:.1%} of "
            f"{info['hole_weight']:.1%} target pixels. Guide: start (a walked frame), middle, end; black = to invent",
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


def stage(case_name: str, arm: str, prefix: str):
    """The planned legs into WorldStereo's scene folder (earlier legs of that prefix there are replaced)."""
    case = Case(case_name)
    for old in case.scene.glob(f"{prefix}*"):
        shutil.rmtree(old)
    for leg in sorted((case.nbv(arm) / "legs").glob(f"{prefix}*")):
        shutil.copytree(leg, case.scene / leg.name)
        print(case.scene / leg.name)
