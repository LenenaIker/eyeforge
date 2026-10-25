"""Walk test W.2, in edge-sdg: per arm, place and distance from the capture point,
- holes: share of magenta pixels left;
- psnr_centre: PSNR of the view at the capture point against the panorama's cut, outside arm A's holes (what was
  seen must stay);
- coherence: mean absolute RGB difference (0-255) between neighbouring views on one heading (0-1, 1-2, 2-3 m and
  3-5 m), each pixel of the nearer view carried to the farther one by its depth and kept where the farther view
  sees the same surface (consecutive distances of views.json); "filled" counts only pixels that were holes in arm A (what the arm invented).
Depth: the arm's own (<arm>_depth/NAME.npy) when it has one, else arm A's mesh with each hole given the farthest
depth around it (a hole is what was behind).

    python experiments/sphere_walk/metrics.py --dir data/sphere_test/<place> --arm A [--arm B_lama ...]

Writes <dir>/walk/metrics_<arm>.json and prints a summary per distance."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MAGENTA = np.array([255, 0, 255])
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
SAME_SURFACE = 0.05  # relative depth difference under which the farther view sees the same point


def dist_of(move):
    return 0 if move == "0" else int(move[1:])


def is_magenta(img):
    return (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)


def filled_depth(dep, hole, cell=8):
    """Holes take the farthest depth around them: max-pooled on cells of `cell` px, grown cell by cell until no
    hole cell is empty, and read back per pixel."""
    H, W = dep.shape
    h, w = -(-H // cell), -(-W // cell)
    pad = np.zeros((h * cell, w * cell), np.float32)
    pad[:H, :W] = dep
    far = pad.reshape(h, cell, w, cell).max((1, 3))
    for _ in range(64):
        empty = far <= 0
        if not empty.any():
            break
        p = np.pad(far, 1)
        grown = np.max([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], 0)
        far = np.where(empty, grown, far)
    up = np.repeat(np.repeat(far, cell, 0), cell, 1)[:H, :W]
    d = dep.copy()
    todo = (d <= 0) & hole
    d[todo] = up[todo]
    return d


def intrinsics(v):
    w, h = v["size"]
    f = w / 2 / math.tan(math.radians(v["fov_deg"]) / 2)
    return f, w / 2, h / 2


def carry(va, da, vb):
    """Pixels of view a with their depth, as pixel coordinates and depth in view b (OpenCV cameras)."""
    f, cx, cy = intrinsics(va)
    h, w = da.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pc = np.stack([(u - cx) / f * da, (v - cy) / f * da, da, np.ones_like(da)], -1)
    Pa, Pb = np.asarray(va["pose"]), np.asarray(vb["pose"])
    pb = pc @ (np.linalg.inv(Pb) @ Pa).T
    z = pb[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ub, vb_ = pb[..., 0] / z * f + cx - 0.5, pb[..., 1] / z * f + cy - 0.5
    return ub, vb_, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", action="append", required=True)
    args = ap.parse_args()
    w = Path(args.dir) / "walk"
    views = {v["name"]: v for v in json.loads((w / "views.json").read_text())["views"]}
    for arm in args.arm:
        rows = []
        cache = {}

        def get(name):
            if name not in cache:
                img = np.asarray(Image.open(w / arm / f"{name}.png").convert("RGB")).astype(np.float32)
                hole_a = np.asarray(Image.open(w / "holes" / f"{name}.png")) > 127
                own = w / f"{arm}_depth" / f"{name}.npy"
                dep = np.load(own if own.exists() else w / "depth" / f"{name}.npy").astype(np.float32)
                if not own.exists():
                    dep = filled_depth(dep, hole_a)
                cache[name] = (img, hole_a, dep)
            return cache[name]

        # Matrix-3D's arm has four headings only; a view a person cannot walk to (prep.reachable) is not scored.
        have = {n for n, v in views.items() if (w / arm / f"{n}.png").exists() and v.get("reachable", True)}
        for n, v in views.items():
            if n not in have:
                continue
            img, hole_a, _ = get(n)
            row = {"name": n, "heading": v["heading"], "dist": dist_of(v["move"]), "holes": float(is_magenta(img).mean())}
            if v["move"] == "0":
                real = np.asarray(Image.open(w / f"real_h{v['heading']:03d}.png").convert("RGB")).astype(np.float32)
                seen = ~hole_a  # what the sphere saw from here: holes are scored by the other measures
                row["psnr_centre"] = float(10 * np.log10(255 ** 2 / max(((img - real)[seen] ** 2).mean(), 1e-6)))
            rows.append(row)
        coh = []
        dists = sorted({dist_of(v["move"]) for v in views.values()})
        pairs = [(f"{k}{a}" if a else "0", f"{k}{b}") for k in "fbrl" for a, b in zip(dists[:-1], dists[1:])]
        for hd in sorted({v["heading"] for v in views.values()}):
            for a, b in pairs:
                na, nb = f"h{hd:03d}_{a}", f"h{hd:03d}_{b}"
                if na not in have or nb not in have:
                    continue
                ia, ha, da = get(na)
                ib, _, db = get(nb)
                ub, vb, z = carry(views[na], da, views[nb])
                H, W = db.shape
                ui, vi = np.round(np.nan_to_num(ub, nan=-1)).astype(int), np.round(np.nan_to_num(vb, nan=-1)).astype(int)
                ok = (da > 0) & (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                ok &= ~is_magenta(ia)
                uc, vc = np.clip(ui, 0, W - 1), np.clip(vi, 0, H - 1)
                ok &= np.abs(db[vc, uc] - z) < SAME_SURFACE * z
                ok &= ~is_magenta(ib)[vc, uc]
                diff = np.abs(ia - ib[vc, uc]).mean(-1)
                coh.append({"pair": f"{na}-{nb}", "dist": dist_of(b), "step_m": dist_of(b) - dist_of(a),
                            "all": float(diff[ok].mean()) if ok.any() else None,
                            "filled": float(diff[ok & ha].mean()) if (ok & ha).sum() > 200 else None,
                            "filled_px": int((ok & ha).sum())})
        (w / f"metrics_{arm}.json").write_text(json.dumps({"views": rows, "coherence": coh}, indent=1))
        print(f"== {arm} ({Path(args.dir).name})")
        cen = [r["psnr_centre"] for r in rows if "psnr_centre" in r]
        print(f"  psnr at the centre {np.mean(cen):.2f} dB")
        for dist in dists:
            hs = [r["holes"] for r in rows if r["dist"] == dist]
            ca = [c["all"] for c in coh if c["dist"] == dist and c["all"] is not None]
            cf = [c["filled"] for c in coh if c["dist"] == dist and c["filled"] is not None]
            print(f"  {dist} m: holes {100 * np.mean(hs):.1f} % (max {100 * np.max(hs):.1f}), coherence all "
                  f"{np.mean(ca) if ca else float('nan'):.1f}, filled {np.mean(cf) if cf else float('nan'):.1f}")


if __name__ == "__main__":
    main()
