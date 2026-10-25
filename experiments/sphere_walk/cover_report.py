"""How a splat's quality at a GEN3C frame it never trained on falls with that frame's distance to its training
cameras (gs_train.py --cover --hold ...): the evidence for the coverage term of nbv.py's score (Mind the Gap, 2026:
error tracks each view's angular distance to its nearest training view).

Per held-out frame, against every training camera (panorama cuts and GEN3C frames):
- t_any: the nearest camera centre, whatever its heading;
- r_any: the smallest angle between optical axes, wherever the camera is;
- t_aligned: the nearest centre among cameras looking within the frame's half field of view of its own axis (a
  camera that could show the same surfaces);
- par_med, par_near: t_aligned over the frame's median and 10th percentile depth, the parallax in radians, free of
  the site's scale (the same move matters less in front of a far facade than over the near ground);
- delta: the pose distance nbv.py's coverage uses, in degrees and free of the site's scale: over the cameras within
  the half field of view, the least axis angle plus the move over the frame's median depth (as an angle).

Per site, psnr_all = floor + (top - floor) * exp(-delta / delta0) is fitted and written to
<site>/walk/cover_fit.json, the kernel of nbv.py's coverage term (measured per site, not set by hand).

    python experiments/sphere_walk/cover_report.py data/sphere_test/<place>/walk/COV_W [more cover folders]

Prints Spearman correlations and tables binned by quantiles of each measure; writes cover_report.json next to the
first folder."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

QUALITY = ("psnr_all", "psnr_hole", "lpips", "sharp", "clear")
BINS = 4


def measures(row, train, half_fov):
    P = np.array(row["c2w"])
    c, axis = P[:3, 3], P[:3, 2]
    T = np.array([t["c2w"] for t in train])
    dt = np.linalg.norm(T[:, :3, 3] - c, axis=1)
    ang = np.degrees(np.arccos(np.clip(T[:, :3, 2] @ axis, -1, 1)))
    ok = ang <= half_fov
    t_al = float(dt[ok].min()) if ok.any() else math.inf
    delta = float((ang + np.degrees(dt / row["depth_med"]))[ok].min()) if ok.any() else math.inf
    return {"t_any": float(dt.min()), "r_any": float(ang.min()), "t_aligned": t_al,
            "par_med": t_al / row["depth_med"], "par_near": t_al / row["depth_near"], "delta": delta}


def kernel(delta, floor, top, delta0):
    return floor + (top - floor) * np.exp(-delta / delta0)


def fit(rows):
    x = np.array([r["delta"] for r in rows])
    y = np.array([r["psnr_all"] for r in rows])
    p0 = (np.percentile(y, 5), np.percentile(y, 95), float(np.median(x)))
    (floor, top, delta0), _ = curve_fit(kernel, x, y, p0=p0, bounds=([0, 0, 0.1], [60, 60, 180]))
    resid = y - kernel(x, floor, top, delta0)
    return {"floor_db": round(float(floor), 2), "top_db": round(float(top), 2), "delta0_deg": round(float(delta0), 2),
            "rmse_db": round(float(np.sqrt((resid ** 2).mean())), 2), "frames": len(rows),
            "delta_max_deg": round(float(x.max()), 1)}


def main():
    rows = []
    for folder in map(Path, sys.argv[1:]):
        cov = json.loads((folder / "cover.json").read_text())
        site = folder.parents[1].name
        fov = {p["name"]: p["view"]["fov_deg"] for f in folder.parent.glob("N*/paths.json")
               for p in json.loads(f.read_text())["paths"]}  # cover.json before it kept each frame's
        for r in cov["held"]:
            half_fov = (r.get("fov_deg") or fov[r["name"].rsplit("_f", 1)[0]]) / 2
            rows.append({**r, **measures(r, cov["train"], half_fov), "run": f"{site[-9:]}/{folder.name}"})
    keys = ("t_any", "r_any", "t_aligned", "par_med", "par_near", "delta")
    finite = [r for r in rows if math.isfinite(r["t_aligned"])]
    print(f"{len(rows)} held-out frames, {len(rows) - len(finite)} with no camera looking their way")
    corr = {}
    for k in keys:
        corr[k] = {}
        for q in QUALITY:
            pairs = [(r[k], r[q]) for r in finite if r[q] is not None]
            rho, p = spearmanr(*zip(*pairs))
            corr[k][q] = {"rho": round(float(rho), 2), "p": float(f"{p:.1e}"), "n": len(pairs)}
    print("Spearman rho (p)   " + "  ".join(f"{q:>14}" for q in QUALITY))
    for k in keys:
        print(f"{k:>18} " + "  ".join(f"{corr[k][q]['rho']:+.2f} ({corr[k][q]['p']:.0e})" for q in QUALITY))
    tables = {}
    for k in keys:
        v = np.array([r[k] for r in finite])
        edges = np.quantile(v, np.linspace(0, 1, BINS + 1))
        tables[k] = []
        print(f"\nbinned by {k}")
        for i in range(BINS):
            sel = [r for r in finite if edges[i] <= r[k] <= edges[i + 1]]
            row = {"from": round(float(edges[i]), 3), "to": round(float(edges[i + 1]), 3), "n": len(sel)}
            for q in QUALITY:
                vals = [r[q] for r in sel if r[q] is not None]
                row[q] = round(float(np.mean(vals)), 3) if vals else None
            tables[k].append(row)
            print(f"  {row['from']:7.3f}-{row['to']:7.3f} n {row['n']:3d}  " +
                  "  ".join(f"{q} {row[q]}" for q in QUALITY))
    runs = sorted({r["run"] for r in rows})
    print("\nper run (spans only)")
    for run in runs:
        sel = [r for r in rows if r["run"] == run and r["span"]]
        if sel:
            print(f"  {run}: n {len(sel)} psnr_all {np.mean([r['psnr_all'] for r in sel]):.2f} sharp "
                  f"{np.mean([r['sharp'] for r in sel]):.2f} t_aligned median "
                  f"{np.median([r['t_aligned'] for r in sel]):.2f}")
    print("\nkernel per site: psnr_all = floor + (top - floor) * exp(-delta / delta0)")
    for folder in {Path(f).parent: None for f in sys.argv[1:]}:
        site = folder.parent.name
        sel = [r for r in finite if r["run"].startswith(site[-9:] + "/")]
        k = {**fit(sel), "runs": sorted({r["run"] for r in sel})}
        (folder / "cover_fit.json").write_text(json.dumps(k, indent=1))
        print(f"  {site}: {k}")
    out = Path(sys.argv[1]) / "cover_report.json"
    out.write_text(json.dumps({"corr": corr, "tables": tables,
                               "rows": [{k: r[k] for k in r if k != "c2w"} for r in rows]}, indent=1, default=float))


if __name__ == "__main__":
    main()
