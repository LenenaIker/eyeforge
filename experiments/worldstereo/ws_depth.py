"""Per-generation depth with MapAnything (Apache 2.0 weights) and the cross-generation consistency metric.

Each generated leg goes through MapAnything on its own (one forward per leg, never several legs together, or the model
would reconcile two different SUVs into one in between and hide the very conflict we measure), with the cameras and
intrinsics we asked WorldStereo for. Then every leg's points are projected into the other legs' cameras and compared
with their depth:

    agree     |z_other - z| < REL_TOL * z
    conflict  z_other < (1 - REL_TOL) * z: the other leg puts a surface where this one sees through it (free-space
              violation). A point behind (z_other > z) may just be hidden here, so it counts as neither.

consistency = agree / (agree + conflict), weighted by overlap. Drift: MapAnything takes our cameras only as a hint and
returns the ones the video shows; anchored at the first frame, how far (cm) each leg strays from what we asked. Colour error is the mean absolute difference (0-255)
on the agreeing pixels after each frame's mean colour is removed (WorldStereo shifts the tone between legs).

    <mapanything env>/bin/python experiments/worldstereo/ws_depth.py depth --case photo21
    <mapanything env>/bin/python experiments/worldstereo/ws_depth.py anchor --case photo21    # scale per leg
    <mapanything env>/bin/python experiments/worldstereo/ws_depth.py consensus --case photo21 --views views_s3.npz
    <mapanything env>/bin/python experiments/worldstereo/ws_depth.py consistency --case photo21 [--last 0.8]

Writes data/worldstereo_test/<case>/depth_ma/<leg>.npz and consistency.json. --last keeps only the first share of
each clip (R3, the end of a clip drifts most)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

CKPT = "facebook/map-anything-apache"
REL_TOL = 0.05         # 5 cm at 1 m, 50 cm at 10 m
MAX_DEPTH = 40.0       # metres, as ws_splat
FRAME_STEP = 2         # every 2nd frame of a leg
PIXEL_STEP = 3         # every 3rd pixel of a source frame
ANCHOR_PATH_M = 0.5    # a leg that moves less has no parallax to fix MapAnything's scale (see anchor)
MIN_OVERLAP = 2000     # comparison pixels for a pair to count
# The tests' leg sets (ws_splat arms): phase 1 is in all of them.
SETS = {"phase1": ("reconstruct_", "wonder_", "view", "step_"), "S1": ("spin_",), "S3": ("spg_",), "P2": ("nbv_",), "NA": ("nba_",),
        "K0": ("k0_",), "K1": ("k1_",), "K2": ("k2_",),
        "C0": ("c0_",), "C1": ("c1_",), "C2": ("c2_",), "C3": ("c3_",)}  # ws_cache.py: phase 1 again, seed 1024  # ws_knobs.py variants of spg_0_*


def out_dir(case: str) -> Path:
    return ws.DATA / case / "depth_ma"


def camera_file(case: str, leg: str) -> Path:
    """camera.json of a leg; legs another test moved out of scene/ live in legsets/ or nbv/legs/."""
    root = ws.DATA / case
    for base in (root / "scene", *sorted((root / "legsets").glob("*")), root / "nbv" / "legs"):
        if (base / leg / "camera.json").exists():
            return base / leg / "camera.json"
    raise FileNotFoundError(leg)


def all_legs(case: str) -> list[str]:
    res = ws.DATA / case / "out" / "out" / case
    return sorted(f"{p.parts[-3]}/{p.parts[-2]}" for p in res.glob(f"*/traj*/{ws.MODEL}_result.mp4"))


def depth(case: str):
    import imageio.v3 as iio
    import torch
    os.environ.setdefault("HF_HOME", str(ws.TOOLS / "hf_cache"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    t0 = time.time()
    model = MapAnything.from_pretrained(CKPT).cuda().eval()
    res = ws.DATA / case / "out" / "out" / case
    out = out_dir(case)
    out.mkdir(exist_ok=True)
    for leg in all_legs(case):
        dst = out / f"{leg.replace('/', '__')}.npz"
        if dst.exists():
            continue
        frames = iio.imread(res / leg / f"{ws.MODEL}_result.mp4", plugin="pyav")
        cam = json.loads(camera_file(case, leg).read_text())
        w2c = np.array(cam["extrinsic"], np.float32)
        K = np.array(cam["intrinsic"][0], np.float32)
        c2w = np.linalg.inv(w2c)  # ws_test's cameras are OpenCV (x right, y down, z forward), as MapAnything's
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


def anchor(case: str):
    """scales.json: per leg, the factor that brings MapAnything's depth to the panorama's point cloud (median ratio
    on the pixels that cloud covers, every frame). A walk's metric cameras fix MapAnything's scale (it lands within
    10 % of the panorama), but an in-place spin has no parallax and the scale is the model's guess: the same frame
    came out 1.7x deeper in the spin's forward than in the walk's it chains from. So only legs that move less than
    ANCHOR_PATH_M are anchored; a walk keeps scale 1 (its ratio is printed: a gap there is content, not gauge)."""
    import cv2
    import ws_splat as wsp
    pts = wsp.panorama_points(case)[0]
    scales = {}
    for leg in all_legs(case):
        f = out_dir(case) / f"{leg.replace('/', '__')}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        w2c = np.linalg.inv(z["c2w"])
        ratios = []
        for k in range(len(z["depth"])):
            d = z["depth"][k].astype(np.float32)
            ma = cv2.resize(np.where(z["mask"][k], d, 0), (ws.W, ws.H), interpolation=cv2.INTER_NEAREST)
            pr = wsp.zbuffer(pts, w2c[k])
            ok = (ma > 0) & (pr > 0) & (pr < MAX_DEPTH)
            if ok.sum() > 500:
                ratios.append(np.median(pr[ok] / ma[ok]))
        ratio = float(np.median(ratios)) if ratios else 1.0
        path = float(np.linalg.norm(z["c2w"][:, :3, 3] - z["c2w"][0, :3, 3], axis=1).max())
        scales[leg] = ratio if path < ANCHOR_PATH_M else 1.0
        print(f"{leg}: panorama / MapAnything {ratio:.2f}, path {path:.1f} m, scale {scales[leg]:.2f}", flush=True)
    (out_dir(case) / "scales.json").write_text(json.dumps(scales, indent=1))


def load(case: str, leg: str, last: float) -> dict:
    z = dict(np.load(out_dir(case) / f"{leg.replace('/', '__')}.npz"))
    n = max(1, int(round(len(z["depth"]) * last)))
    for k in ("depth", "mask", "conf", "K", "c2w", "img"):
        z[k] = z[k][:n]
    sf = out_dir(case) / "scales.json"
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


def consistency(case: str, last: float):
    t0 = time.time()
    have = {p.stem.replace("__", "/") for p in out_dir(case).glob("*.npz")}
    legs = [l for l in all_legs(case) if l in have]
    data = {l: load(case, l, last) for l in legs}
    pts = {l: points(data[l]) for l in legs}
    pairs = {}
    for a in legs:
        for b in legs:
            if a.split("/")[0] == b.split("/")[0]:
                continue  # trajectories of one leg folder share a forward of WorldStereo
            ag, co, ce = compare(pts[a], data[b])
            if ag + co >= MIN_OVERLAP:
                pairs[f"{a}>{b}"] = (ag, co, ce)
    selfc = []
    for l in legs:  # control: a leg against its own cameras (one forward, should agree)
        ag, co, _ = compare(pts[l], data[l])
        selfc.append(ag / max(ag + co, 1))
    print(f"control, each leg against itself: consistency median {np.median(selfc):.3f}, min {min(selfc):.3f}")
    group = lambda l: next(k for k, v in SETS.items() if l.startswith(v))
    rows = {}
    for name, test in (("phase1 vs phase1", lambda ga, gb: ga == gb == "phase1"),
                       *[(f"{s} vs phase1", lambda ga, gb, s=s: {ga, gb} == {s, "phase1"}) for s in SETS if s != "phase1"],
                       *[(f"{s} vs {s}", lambda ga, gb, s=s: ga == gb == s) for s in SETS if s != "phase1"]):
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
    for l in legs:
        ci, co = data[l]["c2w"], data[l]["c2w_out"][:len(data[l]["c2w"])]
        al = ci[0] @ np.linalg.inv(co[0]) @ co
        al[:, :3, 3] = ci[0, :3, 3] + (al[:, :3, 3] - ci[0, :3, 3]) * data[l]["scale"]
        obey[l] = float(np.linalg.norm(al[:, :3, 3] - ci[:, :3, 3], axis=1).max() * 100)
    for g in SETS:
        v = [obey[l] for l in legs if l.startswith(SETS[g])]
        if v:
            print(f"{g:8s} camera drift from the asked path, max per leg: median {np.median(v):.0f} cm, worst "
                  f"{max(v):.0f} cm ({max((l for l in legs if l.startswith(SETS[g])), key=obey.get)})")
    result = {"last": last, "rel_tol": REL_TOL, "groups": rows, "drift_cm": obey,
              "legs": {l: s[0] / (s[0] + s[1]) for l, s in sorted(per_leg.items())},
              "pairs": {k: {"agree": a, "conflict": c} for k, (a, c, _) in pairs.items()}}
    name = "consistency.json" if last == 1.0 else f"consistency_last{last:g}.json"
    (out_dir(case) / name).write_text(json.dumps(result, indent=1))
    print(f"{time.time() - t0:.0f} s -> {out_dir(case) / name}")


def consensus(case: str, views: str, rule: str = "majority", tol: float = 0.1):
    """Source per zone (R5): every pixel of every generated frame in a ws_splat views file is projected into the
    other legs' cameras (every FRAME_STEP-th frame) and checked against their depth. A pixel whose point most of the
    legs that see that spot contradict (it sits in front of their surface: they see through it) stops teaching; one
    no other leg sees, or that the majority confirms, keeps teaching. Writes depth_ma/consensus_<views>.npz (name,
    keep at ws_test's frame size), tol is looser than the metric's for MapAnything's own noise.
    rule "closest": a contradiction only counts when it comes from a camera nearer to that spot than this frame's
    (the nearer view rules), so a car a spin saw from 3 m is not voted away by walks that saw it from 10 m.
    rule "first" (A1, invent once): legs in generation order (sorted, as upstream's sort_trajs); a pixel stops
    teaching when any earlier leg saw that spot (its point lands on a valid pixel there and is not hidden behind that
    leg's surface), whatever depth it gave. Each place is taught by the first leg that saw it, so no place gets two
    versions to average; only what is new to a leg teaches."""
    import cv2
    import torch
    dev = torch.device("cuda")
    t0 = time.time()
    vz = np.load(ws.DATA / case / "splat" / views)
    names = [str(n) for n in vz["name"] if not str(n).startswith("pano/")]
    legs = sorted({n.rsplit("/", 1)[0] for n in names})
    data = {l: load(case, l, 1.0) for l in legs}
    gpu = {l: (torch.as_tensor(data[l]["depth"], device=dev), torch.as_tensor(data[l]["ok"], device=dev),
               torch.as_tensor(np.linalg.inv(data[l]["c2w"]), device=dev), torch.as_tensor(data[l]["K"], device=dev))
           for l in legs}
    keep_all, drop_share = [], {}
    for n in names:
        leg, f = n.rsplit("/", 1)
        f = int(f)
        z = data[leg]
        h, w = z["depth"][f].shape
        yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        d = gpu[leg][0][f]
        K = gpu[leg][3][f]
        cam = torch.stack([(xx + 0.5 - K[0, 2]) / K[0, 0] * d, (yy + 0.5 - K[1, 2]) / K[1, 1] * d, d], -1)
        c2w = torch.as_tensor(z["c2w"][f], device=dev)
        P = cam.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]
        agree = torch.zeros(h * w, device=dev)
        conflict = torch.zeros(h * w, device=dev)
        for other in legs:
            if rule == "first" and other >= leg:
                break
            if rule != "first" and other.split("/")[0] == leg.split("/")[0]:
                continue
            od, ook, ow2c, oK = gpu[other]
            for g in range(0, len(od), FRAME_STEP):
                p = P @ ow2c[g][:3, :3].T + ow2c[g][:3, 3]
                zc = p[:, 2].clamp(min=1e-3)
                u = torch.floor(p[:, 0] / zc * oK[g][0, 0] + oK[g][0, 2]).long()
                v = torch.floor(p[:, 1] / zc * oK[g][1, 1] + oK[g][1, 2]).long()
                inside = (p[:, 2] > 0.1) & (u >= 0) & (u < od.shape[2]) & (v >= 0) & (v < od.shape[1])
                u, v = u.clamp(0, od.shape[2] - 1), v.clamp(0, od.shape[1] - 1)
                seen = inside & ook[g][v, u]
                rel = (p[:, 2] - od[g][v, u]) / od[g][v, u].clamp(min=1e-3)
                agree += (seen & (rel.abs() < tol)).float()
                against = seen & (rel < -tol)
                if rule == "closest":
                    against = against & (p[:, 2] < d.reshape(-1))
                conflict += against.float()
        drop = (agree + conflict > 0) if rule == "first" else (conflict > agree)
        keep = ~(z["ok"][f].reshape(-1) & drop.cpu().numpy())
        keep = keep.reshape(h, w).astype(np.uint8)
        drop_share[n] = float(1 - keep[z["ok"][f]].mean()) if z["ok"][f].any() else 0.0
        keep_all.append(cv2.resize(keep, (ws.W, ws.H), interpolation=cv2.INTER_NEAREST))
    out = out_dir(case) / f"consensus_{Path(views).stem}{'' if rule == 'majority' else '_' + rule}.npz"
    np.savez_compressed(out, name=np.array(names), keep=np.stack(keep_all))
    for g in sorted({n.split("_")[0] for n in names}):
        v = [x for k, x in drop_share.items() if k.startswith(g)]
        print(f"{g:12s} share of valid pixels dropped: mean {np.mean(v):.2f}, max {max(v):.2f}")
    print(f"{time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["depth", "anchor", "consistency", "consensus"])
    ap.add_argument("--views", default="views_s3.npz", help="consensus: the ws_splat views file whose frames to mask")
    ap.add_argument("--rule", default="majority", choices=["majority", "closest", "first"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--last", type=float, default=1.0, help="share of each clip kept from its start")
    a = ap.parse_args()
    if a.cmd == "depth":
        depth(a.case)
    elif a.cmd == "consensus":
        consensus(a.case, a.views, a.rule)
    elif a.cmd == "anchor":
        anchor(a.case)
    else:
        consistency(a.case, a.last)
