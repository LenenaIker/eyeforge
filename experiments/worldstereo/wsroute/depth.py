"""Per-generation depth with MapAnything (Apache 2.0 weights) and the cross-generation consistency metric.

Each generated leg goes through MapAnything on its own (one forward per leg, never several legs together, or the model
would reconcile two different SUVs into one in between and hide the very conflict we measure), with the cameras and
intrinsics we asked WorldStereo for. Then every leg's points are projected into the other legs' cameras and compared
with their depth:

    agree     |z_other - z| < REL_TOL * z
    conflict  z_other < (1 - REL_TOL) * z: the other leg puts a surface where this one sees through it (free-space
              violation). A point behind (z_other > z) may just be hidden here, so it counts as neither.

consistency = agree / (agree + conflict), weighted by overlap. Drift: MapAnything takes our cameras only as a hint and
returns the ones the video shows; anchored at the first frame, how far (cm) each leg strays from what we asked.
Colour error is the mean absolute difference (0-255) on the agreeing pixels after each frame's mean colour is removed
(WorldStereo shifts the tone between legs). Runs in SIM2CHIP_MAPANYTHING_PYTHON; writes depth_ma/<leg>.npz,
scales.json and consistency.json."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import camera
from .camera import H, W
from .panorama import Panorama
from .paths import Case, tools

CKPT = "facebook/map-anything-apache"
REL_TOL = 0.05         # 5 cm at 1 m, 50 cm at 10 m
MAX_DEPTH = 40.0       # metres, as splat.py
FRAME_STEP = 2         # every 2nd frame of a leg
PIXEL_STEP = 3         # every 3rd pixel of a source frame
ANCHOR_PATH_M = 0.5    # a leg that moves less has no parallax to fix MapAnything's scale (see anchor)
MIN_OVERLAP = 2000     # comparison pixels for a pair to count
# The groups of legs the report compares: phase 1 (the walks) and each set of planned legs.
SETS = {"phase1": ("reconstruct_", "wonder_"), "P2": ("nbv_",), "NA": ("nba_",)}


def group(leg: str) -> str:
    return next((k for k, v in SETS.items() if leg.startswith(v)), "other")


def leg_file(case: Case, leg: str) -> Path:
    return case.depth / f"{leg.replace('/', '__')}.npz"


def camera_file(case: Case, leg: str) -> Path:
    """camera.json of a leg; legs an older test moved out of scene/ live in legsets/."""
    for base in (case.scene, *sorted((case.dir / "legsets").glob("*"))):
        if (base / leg / "camera.json").exists():
            return base / leg / "camera.json"
    raise FileNotFoundError(leg)


def depth(case_name: str):
    """MapAnything on every generated leg that has no depth yet."""
    import imageio.v3 as iio
    import torch
    os.environ.setdefault("HF_HOME", str(tools() / "hf_cache"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    case = Case(case_name)
    t0 = time.time()
    model = MapAnything.from_pretrained(CKPT).cuda().eval()
    case.depth.mkdir(exist_ok=True)
    for leg in case.legs():
        dst = leg_file(case, leg)
        if dst.exists():
            continue
        frames = iio.imread(case.leg_video(leg), plugin="pyav")
        cam = json.loads(camera_file(case, leg).read_text())
        w2c = np.array(cam["extrinsic"], np.float32)
        K = np.array(cam["intrinsic"][0], np.float32)
        c2w = np.linalg.inv(w2c)  # the route's cameras are OpenCV (x right, y down, z forward), as MapAnything's
        views = [{"img": torch.from_numpy(f), "intrinsics": torch.from_numpy(K),
                  "camera_poses": torch.from_numpy(m), "is_metric_scale": torch.tensor([True])}
                 for f, m in zip(frames, c2w)]
        with torch.no_grad():
            pred = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True,
                               amp_dtype="bf16", apply_mask=True, mask_edges=True)
        d = np.stack([p["depth_z"][0, ..., 0].float().cpu().numpy() for p in pred])
        mask = np.stack([p["mask"][0, ..., 0].cpu().numpy() for p in pred]).astype(bool)
        conf = np.stack([p["conf"][0].float().cpu().numpy() for p in pred])
        Ko = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in pred])
        po = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in pred])
        img = np.stack([p["img_no_norm"][0].float().cpu().numpy() for p in pred])
        drift = np.linalg.norm(po[:, :3, 3] - c2w[:, :3, 3], axis=1)
        np.savez_compressed(dst, depth=d.astype(np.float16), mask=mask, conf=conf.astype(np.float16), K=Ko,
                            c2w=c2w, c2w_out=po, img=(np.clip(img, 0, 1) * 255).astype(np.uint8))
        print(f"{leg}: {len(frames)} frames at {d.shape[2]}x{d.shape[1]}, valid {mask.mean():.2f}, "
              f"output poses off the input by {drift.max() * 100:.1f} cm at most, {time.time() - t0:.0f} s", flush=True)


def anchor(case_name: str):
    """scales.json: per leg, the factor that brings MapAnything's depth to the panorama's point cloud (median ratio
    on the pixels that cloud covers, every frame). A walk's metric cameras fix MapAnything's scale (it lands within
    10 % of the panorama), but an in-place spin has no parallax and the scale is the model's guess: the same frame
    came out 1.7x deeper in the spin's forward than in the walk's it chains from. So only legs that move less than
    ANCHOR_PATH_M are anchored; a walk keeps scale 1 (its ratio is printed: a gap there is content, not gauge)."""
    import cv2
    case = Case(case_name)
    pts = Panorama(case).points()[0]
    scales = {}
    for leg in case.legs():
        f = leg_file(case, leg)
        if not f.exists():
            continue
        z = np.load(f)
        c2w, dep, msk = z["c2w"], z["depth"], z["mask"]
        w2c = np.linalg.inv(c2w)
        ratios = []
        for k in range(len(dep)):
            d = dep[k].astype(np.float32)
            ma = cv2.resize(np.where(msk[k], d, 0), (W, H), interpolation=cv2.INTER_NEAREST)
            pr = camera.zbuffer(pts, w2c[k])
            ok = (ma > 0) & (pr > 0) & (pr < MAX_DEPTH)
            if ok.sum() > 500:
                ratios.append(np.median(pr[ok] / ma[ok]))
        ratio = float(np.median(ratios)) if ratios else 1.0
        path = float(np.linalg.norm(c2w[:, :3, 3] - c2w[0, :3, 3], axis=1).max())
        scales[leg] = ratio if path < ANCHOR_PATH_M else 1.0
        print(f"{leg}: panorama / MapAnything {ratio:.2f}, path {path:.1f} m, scale {scales[leg]:.2f}", flush=True)
    (case.depth / "scales.json").write_text(json.dumps(scales, indent=1))


def load(case: Case, leg: str, last: float = 1.0) -> dict:
    """A leg's MapAnything output, its first share last of frames, depth scaled by scales.json."""
    z = dict(np.load(leg_file(case, leg)))
    n = max(1, int(round(len(z["depth"]) * last)))
    for k in ("depth", "mask", "conf", "K", "c2w", "img"):
        z[k] = z[k][:n]
    sf = case.depth / "scales.json"
    z["scale"] = json.loads(sf.read_text()).get(leg, 1.0) if sf.exists() else 1.0
    z["depth"] = z["depth"].astype(np.float32) * z["scale"]
    z["ok"] = z["mask"] & (z["depth"] > 0) & (z["depth"] < MAX_DEPTH)
    return z


def points(z: dict) -> tuple[np.ndarray, np.ndarray]:
    """World points and colours of a leg, every FRAME_STEP-th frame and PIXEL_STEP-th pixel."""
    P, C = [], []
    for f in range(0, len(z["depth"]), FRAME_STEP):
        yy, xx = np.nonzero(z["ok"][f][::PIXEL_STEP, ::PIXEL_STEP])
        yy, xx = yy * PIXEL_STEP, xx * PIXEL_STEP
        d = z["depth"][f][yy, xx]
        K = z["K"][f]
        cam = np.stack([(xx + 0.5 - K[0, 2]) / K[0, 0] * d, (yy + 0.5 - K[1, 2]) / K[1, 1] * d, d], -1)
        m = z["c2w"][f]
        P.append(cam @ m[:3, :3].T + m[:3, 3])
        im = z["img"][f].astype(np.float32)
        C.append(im[yy, xx] - im[z["ok"][f]].mean(0))
    return np.concatenate(P), np.concatenate(C)


def compare(src: tuple, z: dict) -> tuple[int, int, float]:
    """src's points in each FRAME_STEP-th camera of leg z, each against z's depth at its own pixel (a shared cell
    would pick its nearest point and call slanted ground a conflict): (agree, conflict, colour error sum)."""
    P, C = src
    agree = conflict = 0
    cerr = 0.0
    for f in range(0, len(z["depth"]), FRAME_STEP):
        h, w = z["depth"][f].shape
        m = np.linalg.inv(z["c2w"][f])
        p = P @ m[:3, :3].T + m[:3, 3]
        keep = p[:, 2] > 0.1
        p, c = p[keep], C[keep]
        K = z["K"][f]
        u = np.floor(p[:, 0] / p[:, 2] * K[0, 0] + K[0, 2]).astype(int)
        v = np.floor(p[:, 1] / p[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, zs, c = u[ok], v[ok], p[ok, 2], c[ok]
        valid = z["ok"][f][v, u]
        u, v, zs, c = u[valid], v[valid], zs[valid], c[valid]
        zt = z["depth"][f][v, u]
        im = z["img"][f].astype(np.float32)
        ct = im[v, u] - im[z["ok"][f]].mean(0)
        rel = (zs - zt) / np.maximum(zt, 1e-3)
        a = np.abs(rel) < REL_TOL
        agree += int(a.sum())
        conflict += int((rel < -REL_TOL).sum())
        cerr += float(np.abs(c[a] - ct[a]).mean(1).sum())
    return agree, conflict, cerr


def consistency(case_name: str, last: float = 1.0):
    """The metric over every pair of legs from different folders, by group, plus each leg's camera drift."""
    case = Case(case_name)
    t0 = time.time()
    have = {p.stem.replace("__", "/") for p in case.depth.glob("*.npz")}
    legs = [leg for leg in case.legs() if leg in have]
    data = {leg: load(case, leg, last) for leg in legs}
    pts = {leg: points(data[leg]) for leg in legs}
    pairs = {}
    for a in legs:
        for b in legs:
            if a.split("/")[0] == b.split("/")[0]:
                continue  # trajectories of one leg folder share a forward of WorldStereo
            ag, co, ce = compare(pts[a], data[b])
            if ag + co >= MIN_OVERLAP:
                pairs[f"{a}>{b}"] = (ag, co, ce)
    selfc = []
    for leg in legs:  # control: a leg against its own cameras (one forward, should agree)
        ag, co, _ = compare(pts[leg], data[leg])
        selfc.append(ag / max(ag + co, 1))
    print(f"control, each leg against itself: consistency median {np.median(selfc):.3f}, min {min(selfc):.3f}")
    names = [*SETS, "other"]
    tests = [("phase1 vs phase1", lambda ga, gb: ga == gb == "phase1"),
             *[(f"{s} vs phase1", lambda ga, gb, s=s: {ga, gb} == {s, "phase1"}) for s in names if s != "phase1"],
             *[(f"{s} vs {s}", lambda ga, gb, s=s: ga == gb == s) for s in names if s != "phase1"]]
    rows = {}
    for name, test in tests:
        sel = [v for k, v in pairs.items() if test(*(group(x) for x in k.split(">")))]
        if not sel:
            continue
        ag, co, ce = (sum(x[i] for x in sel) for i in range(3))
        rows[name] = {"pairs": len(sel), "consistency": ag / (ag + co), "colour_error": ce / max(ag, 1),
                      "overlap": ag + co}
    for k, v in rows.items():
        print(f"{k:20s} pairs {v['pairs']:3d}  consistency {v['consistency']:.3f}  colour error "
              f"{v['colour_error']:.1f}  overlap {v['overlap']}")
    per_leg = {}
    for k, (ag, co, ce) in pairs.items():
        for leg in k.split(">"):
            s = per_leg.setdefault(leg, [0, 0])
            s[0] += ag
            s[1] += co
    # Obedience: MapAnything takes our cameras as a hint and returns the poses the video shows (relative to its
    # first frame); after anchoring both at that frame, how far a leg's cameras drift from the ones we asked for.
    obey = {}
    for leg in legs:
        ci, co = data[leg]["c2w"], data[leg]["c2w_out"][:len(data[leg]["c2w"])]
        al = ci[0] @ np.linalg.inv(co[0]) @ co
        al[:, :3, 3] = ci[0, :3, 3] + (al[:, :3, 3] - ci[0, :3, 3]) * data[leg]["scale"]
        obey[leg] = float(np.linalg.norm(al[:, :3, 3] - ci[:, :3, 3], axis=1).max() * 100)
    for g in names:
        v = [obey[leg] for leg in legs if group(leg) == g]
        if v:
            worst = max((leg for leg in legs if group(leg) == g), key=obey.get)
            print(f"{g:8s} camera drift from the asked path, max per leg: median {np.median(v):.0f} cm, worst "
                  f"{max(v):.0f} cm ({worst})")
    result = {"last": last, "rel_tol": REL_TOL, "groups": rows, "drift_cm": obey,
              "legs": {leg: s[0] / (s[0] + s[1]) for leg, s in sorted(per_leg.items())},
              "pairs": {k: {"agree": a, "conflict": c} for k, (a, c, _) in pairs.items()}}
    name = "consistency.json" if last == 1.0 else f"consistency_last{last:g}.json"
    (case.depth / name).write_text(json.dumps(result, indent=1))
    print(f"{time.time() - t0:.0f} s -> {case.depth / name}")
