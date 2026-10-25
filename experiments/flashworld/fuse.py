"""FlashWorld's six per-move splats of one photo (flashworld_test.py run) fused into one scene for Replicator.

Every generation starts at the photo's camera, so their rotations agree, but each picks its own depth scale. Steps:
1. scale: each splat's depth from the photo's camera against the per-pixel median of all six (median ratio where
   both are opaque); the six are brought to that common scale.
2. views: each splat renders itself (RGB, alpha, depth) at VIEWS_PER_MOVE poses along its own path; pixels where
   it is not opaque teach nothing.
3. fuse: the six splats, scaled and thinned to KEEP_PER_MOVE Gaussians each, become one splat that gsplat refits
   to every view at once (L1 colour + depth); where the six disagree the fit averages them, Gaussians no view
   confirms lose opacity and are pruned.
4. export: splat.ply (DC colour only, what 3DGRUT's NuRec exporter reads) and a sheet with renders between paths.

Units stay FlashWorld's (the path length is 1, not metric). Run in FlashWorld's env:

    <flashworld env>/bin/python experiments/flashworld/fuse.py --case photo21

Writes data/flashworld_test/out/<case>/fused/: splat.ply, scales.json, sheet.jpg. USD: gs_usd.py in 3DGRUT's venv,
then scene.usda next to it (written here)."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "flashworld_test"
REPO = Path(os.environ.get("FLASHWORLD_REPO", "/media/lenena-iker/Crucial/tools/FlashWorld"))
sys.path.insert(0, str(REPO))
from utils import quaternion_to_matrix, sample_from_dense_cameras  # noqa: E402

from gsplat import rasterization  # noqa: E402

MOVES = ["orbit", "left", "right", "up", "back", "forward"]
H, W = 480, 704
C0 = 0.28209479177387814
VIEWS_PER_MOVE = 48
KEEP_PER_MOVE = 600_000  # 6 x 4 M Gaussians do not train in 24 GB; the most opaque are kept
STEPS = 6000
ALPHA_MIN = 0.5          # a target pixel teaches only where its own splat is this opaque
DEPTH_WEIGHT = 0.1


def load_ply(path: Path, dev) -> dict:
    """FlashWorld's export (utils.export_gaussians): activations inverted, every SH coefficient as f_dc_*."""
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    sh = np.stack([v[n] for n in sorted((n for n in names if n.startswith("f_dc_")), key=lambda n: int(n[5:]))], 1)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
    return {"means": t(np.stack([v["x"], v["y"], v["z"]], 1)),
            "quats": t(np.stack([v[f"rot_{i}"] for i in range(4)], 1)),
            "log_scales": t(np.stack([v[f"scale_{i}"] for i in range(3)], 1)),
            "logit_opac": t(v["opacity"]),
            "sh": t(sh).reshape(len(sh), -1, 3)}


def cams_of(move: str, case: str, n: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """n camera-to-world matrices (OpenCV axes, as gsplat wants) along a move's path, sampled as FlashWorld's
    video is, and the intrinsics matrix."""
    req = json.loads((DATA / "in" / f"{case}__{move}.json").read_text())
    dense = torch.tensor([c["quaternion"] + c["position"] + [c["fx"] / W, c["fy"] / H, c["cx"] / W, c["cy"] / H]
                          for c in req["cameras"]], dtype=torch.float32, device=dev)
    cams = sample_from_dense_cameras(dense, torch.linspace(0, 1, n, device=dev))
    c2w = torch.eye(4, device=dev).repeat(n, 1, 1)
    c2w[:, :3, :3] = quaternion_to_matrix(cams[:, :4])
    c2w[:, :3, 3] = cams[:, 4:7]
    c2w[:, :3, 1:3] *= -1  # OpenGL to OpenCV, as FlashWorld's gaussian_render
    c = req["cameras"][0]
    K = torch.tensor([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], dtype=torch.float32, device=dev)
    return c2w, K


def render(g: dict, c2w: torch.Tensor, K: torch.Tensor, sh_degree: int | None = None):
    """RGB, alpha and alpha-weighted depth of each camera, one at a time (unpacked, see flashworld_test.py)."""
    deg = int(math.sqrt(g["sh"].shape[1])) - 1 if sh_degree is None else sh_degree
    out = []
    for m in c2w:
        rgbd, alpha, _ = rasterization(g["means"], F.normalize(g["quats"], dim=-1), g["log_scales"].exp(),
                                       torch.sigmoid(g["logit_opac"]), g["sh"], torch.linalg.inv(m)[None], K[None],
                                       W, H, sh_degree=deg, render_mode="RGB+D", packed=False, near_plane=0.01,
                                       far_plane=1000)
        out.append((rgbd[0, ..., :3].clamp(0, 1), alpha[0, ..., 0], rgbd[0, ..., 3]))
    return out


def scaled(g: dict, s: float) -> dict:
    return {**g, "means": g["means"] * s, "log_scales": g["log_scales"] + math.log(s)}


def thin(g: dict, n: int) -> dict:
    keep = torch.topk(g["logit_opac"], min(n, len(g["logit_opac"]))).indices
    return {k: v[keep] for k, v in g.items()}


def fuse(case: str):
    dev = torch.device("cuda")
    out = DATA / "out" / case / "fused"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    splats = {m: load_ply(DATA / "out" / case / m / "gaussians.ply", dev) for m in MOVES}

    # 1. One depth scale: each splat's depth from the photo's camera against the median of all six.
    c2w0, K = cams_of(MOVES[0], case, 1, dev)
    first = {m: render(g, c2w0, K)[0] for m, g in splats.items()}
    depths = torch.stack([torch.where(a > 0.9, d / a.clamp(min=1e-6), torch.nan) for _, a, d in first.values()])
    ref = depths.nanmedian(dim=0).values
    scales = {}
    for m, d in zip(MOVES, depths):
        ok = d.isfinite() & ref.isfinite()
        scales[m] = float((ref[ok] / d[ok]).median())
    print("scales", {m: round(s, 3) for m, s in scales.items()}, flush=True)

    # 2. Each splat's own views, at the common scale.
    views = []
    for m in MOVES:
        c2w, K = cams_of(m, case, VIEWS_PER_MOVE, dev)
        c2w[:, :3, 3] *= scales[m]
        with torch.no_grad():
            for cam, (rgb, a, d) in zip(c2w, render(scaled(splats[m], scales[m]), c2w, K)):
                views.append((m, cam, rgb, a > ALPHA_MIN, torch.where(a > 1e-3, d / a.clamp(min=1e-3), 0)))
    print(f"{len(views)} views, {time.time() - t0:.0f} s", flush=True)

    # 3. One splat refitted to every view: the six thinned and scaled, DC colour only.
    parts = [thin(scaled(splats[m], scales[m]), KEEP_PER_MOVE) for m in MOVES]
    del splats
    torch.cuda.empty_cache()
    params = {k: torch.nn.Parameter(torch.cat([p[k] for p in parts]).contiguous()) for k in
              ("means", "quats", "log_scales", "logit_opac")}
    params["sh"] = torch.nn.Parameter(torch.cat([p["sh"][:, :1] for p in parts]).contiguous())
    lrs = {"means": 1.6e-4, "quats": 1e-3, "log_scales": 5e-3, "logit_opac": 5e-2, "sh": 2.5e-3}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr, "name": k} for k, lr in lrs.items()], eps=1e-15)
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        m, cam, rgb, mask, depth = views[rng.integers(len(views))]
        (pred, alpha, pd), = render(params, cam[None], K, sh_degree=0)
        pd = pd / alpha.clamp(min=1e-3)
        loss = (pred - rgb).abs()[mask].mean() + DEPTH_WEIGHT * ((pd - depth).abs() / depth.clamp(min=1e-2))[mask].mean()
        # Pixels no splat covered keep the fit from inventing opacity there.
        loss = loss + 0.01 * alpha[~mask].mean() if (~mask).any() else loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f} s", flush=True)
    keep = torch.sigmoid(params["logit_opac"].detach()) > 0.02
    g = {k: v.detach()[keep] for k, v in params.items()}
    print(f"kept {int(keep.sum())} of {len(keep)} Gaussians", flush=True)

    # 4. Export and sheet: each move's end, then poses between moves no single generation had.
    write_ply(g, out / "splat.ply")
    (out / "scales.json").write_text(json.dumps(scales, indent=1))
    sheet(g, case, scales, K, dev, out / "sheet.jpg")
    print(f"done {time.time() - t0:.0f} s", flush=True)


def write_ply(g: dict, path: Path):
    n = len(g["means"])
    cols = {"x": g["means"][:, 0], "y": g["means"][:, 1], "z": g["means"][:, 2],
            **{f"f_dc_{i}": g["sh"][:, 0, i] for i in range(3)}, "opacity": g["logit_opac"],
            **{f"scale_{i}": g["log_scales"][:, i] for i in range(3)}, **{f"rot_{i}": g["quats"][:, i] for i in range(4)}}
    arr = np.empty(n, dtype=[(k, "f4") for k in cols])
    for k, v in cols.items():
        arr[k] = v.float().cpu().numpy()
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def sheet(g: dict, case: str, scales: dict, K, dev, path: Path, cell=(352, 240)):
    """Row 1: the fused splat at each move's end. Row 2: between pairs of moves (mean of the two end poses)."""
    ends = {}
    for m in MOVES:
        c2w, _ = cams_of(m, case, 2, dev)
        c2w[:, :3, 3] *= scales[m]
        ends[m] = c2w[1]
    pairs = [("left", "forward"), ("right", "forward"), ("up", "left"), ("back", "right"), ("orbit", "up"),
             ("back", "left")]
    between = []
    for a, b in pairs:
        m = ends[a].clone()
        m[:3, 3] = (ends[a][:3, 3] + ends[b][:3, 3]) / 2
        between.append(m)
    rows = [("fused, end of " + ", ".join(MOVES), torch.stack(list(ends.values()))),
            ("fused, between " + ", ".join(f"{a}+{b}" for a, b in pairs), torch.stack(between))]
    canvas = Image.new("RGB", (cell[0] * 6, (cell[1] + 16) * len(rows)), "white")
    dr = ImageDraw.Draw(canvas)
    with torch.no_grad():
        for r, (label, cams) in enumerate(rows):
            dr.text((4, r * (cell[1] + 16) + 2), label, fill="black")
            for c, (rgb, a, _) in enumerate(render(g, cams, K, sh_degree=0)):
                im = (rgb + (1 - a[..., None])).clamp(0, 1)  # white where empty, as FlashWorld's videos
                im = Image.fromarray((im.cpu().numpy() * 255).astype(np.uint8)).resize(cell)
                canvas.paste(im, (c * cell[0], r * (cell[1] + 16) + 16))
    canvas.save(path, quality=88)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    fuse(ap.parse_args().case)


if __name__ == "__main__":
    main()
