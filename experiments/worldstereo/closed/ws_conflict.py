"""Where do WorldStereo's own frames back the splat a student camera sees, and where not (2026-10-25)?

Each pixel of a student view is lifted to its splat point X and projected into every generated frame the splat was
trained on (MapAnything depth per leg, depth_ma/, the cameras we asked for). A frame that has X in view and a valid
depth d there, with z the depth of X in that frame:
  agrees       |d - z| < TOL z
  contradicts  d > (1 + TOL) z: it looks through where X should be (free space there)
  says nothing d < (1 - TOL) z: X is behind its surface (hidden)
The panorama counts as one more leg (distance from its centre, the origin). Per pixel, the class:
  confirmed  some frame agrees and fewer than half of those with evidence contradict
  conflict   half or more of the frames with evidence contradict
  no data    a surface no frame agrees with or contradicts (why: hidden, masked by MapAnything, or never in view)
  hole       the splat renders nothing (and the panorama says it is not sky)
  sky        nothing rendered where the panorama has sky
MapAnything's depth runs 10 to 30 % short of the splat (scale drift per leg): every frame is calibrated by its
median depth ratio against the splat on its own view, or no data inflates from ~16 to ~55 %.

    <gsplat env>/bin/python experiments/worldstereo/ws_conflict.py grid --case photo21_np --arm G2 [--radius 10]
    <gsplat env>/bin/python experiments/worldstereo/ws_conflict.py sheet --case photo21_np --arm G2 --spots "9,-4;8,-4"
    <gsplat env>/bin/python experiments/worldstereo/ws_conflict.py gaussians --case photo21_np --arm G2
    <gsplat env>/bin/python experiments/worldstereo/ws_conflict.py prune --case photo21_np --arm G2 --rule unbacked

Writes data/worldstereo_test/<case>/conflict/<arm>/: grid.npz, grid.json, sheet_grid.jpg, sheet_spots.jpg,
gaussians.npz. prune writes the splat arm <arm>_<rule> (splat/arms/<arm>_<rule>/splat.ply)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_nbv as nb  # noqa: E402
import ws_test as ws  # noqa: E402

TOL = 0.10
MAX_CONTRA_M = 39.0    # MapAnything's far depths are squashed: no contradiction from beyond this
CALIB_MIN_PX = 500
GRID_SIZE = (320, 240)  # analysis resolution of a student view
CODE_SIZE = (80, 60)    # stored class maps
HEADING_TOL_DEG = 45.0  # a walked frame counts for "distance to walked" when it faces within this
CONFIRMED, CONFLICT, NODATA, HOLE, SKY = range(5)
CLASS_RGB = {CONFIRMED: (40, 160, 60), CONFLICT: (250, 170, 0), NODATA: (220, 0, 0), HOLE: (220, 0, 220),
             SKY: (40, 80, 220)}
DIST_BINS = (0, 0.5, 1, 2, 3, 4, 5, 6, 8, 99)


def out_dir(case: str, arm: str) -> Path:
    d = ws.DATA / case / "conflict" / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


class Evidence:
    """The generated frames (calibrated MapAnything depth, colour) and the panorama, as evidence about points."""

    def __init__(self, case: str, sp, pano: bool = True):
        import torch
        dev = sp["means"].device
        self.dev = dev
        self.case = case
        names = sorted({str(n).rsplit("/", 1)[0] for n in np.load(ws.DATA / case / "splat" / "views.npz")["name"]}
                       - {"pano"})
        scales = json.loads((ws.DATA / case / "depth_ma" / "scales.json").read_text())
        D, Msk, KK, W2C, IMG, LEG = [], [], [], [], [], []
        for li, leg in enumerate(names):
            z = np.load(ws.DATA / case / "depth_ma" / f"{leg.replace('/', '__')}.npz")
            D.append(z["depth"].astype(np.float32) * scales.get(leg, 1.0))
            Msk.append(z["mask"])
            KK.append(z["K"])
            W2C.append(np.linalg.inv(z["c2w"]))
            IMG.append(z["img"].astype(np.float32))
            LEG += [li] * len(z["depth"])
        T = lambda a: torch.as_tensor(np.concatenate(a), device=dev)  # noqa: E731
        self.D, self.Msk, self.KK, self.W2C, self.IMG = T(D), T(Msk), T(KK).float(), T(W2C).float(), T(IMG)
        self.IMG -= self.IMG.mean((1, 2), keepdim=True)  # tone shifts between legs are not conflicts
        self.LEG = torch.as_tensor(LEG, device=dev)
        self.names = names
        self.F, self.fh, self.fw = self.D.shape
        self.ratio = self.calibrate(sp)
        self.D /= self.ratio[:, None, None]
        meta = json.loads((ws.DATA / case / "prep.json").read_text())
        zp = np.load(ws.DATA / case / "distance.npz")
        self.sky_pano = ~zp["mask"]
        self.pano = pano
        self.pdist = torch.as_tensor(zp["distance"] * meta["scale"], device=dev).float()
        self.pvalid = torch.as_tensor(zp["mask"], device=dev)
        self.pratio = self.calibrate_pano(sp) if pano else 1.0
        self.pdist /= self.pratio
        c2w = np.linalg.inv(self.W2C.cpu().numpy())
        self.frame_pos = c2w[:, :3, 3]
        self.frame_yaw = np.degrees(np.arctan2(c2w[:, 1, 2], c2w[:, 0, 2]))
        print(f"{len(names)} legs, {self.F} frames at {self.fw}x{self.fh}; MapAnything/splat depth per frame: median "
              f"{self.ratio.median():.2f} ({self.ratio.min():.2f}-{self.ratio.max():.2f}); panorama/splat "
              f"{self.pratio:.2f}", flush=True)

    def calibrate(self, sp):
        """Per frame: median MapAnything depth / splat depth on its own view."""
        import torch
        out = []
        for f in range(self.F):
            _, dep, al = nb.render(sp, self.W2C[f:f + 1].cpu().numpy(), self.KK[f].cpu().numpy(), self.fw, self.fh)
            m = self.Msk[f] & (al[0] > 0.5) & (dep[0] > 0.3) & (dep[0] < 30)
            out.append((self.D[f][m] / dep[0][m]).median() if m.sum() > CALIB_MIN_PX
                       else torch.tensor(1.0, device=self.dev))
        return torch.stack(out)

    def calibrate_pano(self, sp) -> float:
        """Median panorama distance / splat distance, from four level views at the panorama's centre."""
        import torch
        K = nb.intrinsic(256, 256, 90.0)
        m = np.stack([ws.w2c([0, 0, 0], hd, 0.0) for hd in (0, 90, 180, 270)]).astype(np.float32)
        _, dep, al = nb.render(sp, m, K, 256, 256)
        P = nb.lift(dep, K, m)[al > 0.5]
        rr, pd, ok = self.pano_lookup(P)
        ok &= (rr > 0.5) & (rr < 30)
        return float((pd[ok] / rr[ok]).median()) if ok.sum() > CALIB_MIN_PX else 1.0

    def pano_lookup(self, P):
        import torch
        ph, pw = self.pdist.shape
        rr = torch.linalg.norm(P, dim=1)
        yaw = torch.atan2(P[:, 1], P[:, 0])
        pitch = torch.asin((P[:, 2] / rr.clamp(min=1e-6)).clamp(-1, 1))
        pu = ((0.5 - yaw / (2 * np.pi)) * pw).long().clamp(0, pw - 1)
        pv = ((0.5 - pitch / np.pi) * ph).long().clamp(0, ph - 1)
        return rr, self.pdist[pv, pu], self.pvalid[pv, pu]

    def points(self, P, colour: bool = True) -> dict:
        """P (N, 3) world points -> per point frame counts (agree, contra), legs with evidence, whether any frame had
        it in view or in view with valid depth, the colour spread among agreeing frames, and the panorama's say."""
        import torch
        N = len(P)
        z = lambda dt=torch.float32: torch.zeros(N, dtype=dt, device=self.dev)  # noqa: E731
        ag, co, inview, valid = z(), z(), z(torch.bool), z(torch.bool)
        legs = torch.zeros(N, len(self.names), dtype=torch.bool, device=self.dev)
        csum = torch.zeros(N, 3, device=self.dev) if colour else None
        csq = torch.zeros(N, 3, device=self.dev) if colour else None
        for f in range(self.F):
            cam = P @ self.W2C[f, :3, :3].T + self.W2C[f, :3, 3]
            zc = cam[:, 2]
            u = cam[:, 0] / zc.clamp(min=1e-3) * self.KK[f, 0, 0] + self.KK[f, 0, 2]
            v = cam[:, 1] / zc.clamp(min=1e-3) * self.KK[f, 1, 1] + self.KK[f, 1, 2]
            ok = (zc > 0.2) & (u >= 0) & (u < self.fw - 1) & (v >= 0) & (v < self.fh - 1)
            ui, vi = u.long().clamp(0, self.fw - 1), v.long().clamp(0, self.fh - 1)
            inview |= ok
            ok &= self.Msk[f][vi, ui]
            valid |= ok
            d = self.D[f][vi, ui]
            a = ok & ((d - zc).abs() < TOL * zc)
            c = ok & (d > (1 + TOL) * zc) & (d < MAX_CONTRA_M)
            ag += a
            co += c
            legs[:, self.LEG[f]] |= a | c
            if colour:
                col = self.IMG[f][vi, ui]
                csum += col * a[:, None]
                csq += col * col * a[:, None]
        out = {"agree": ag, "contra": co, "legs": legs.sum(1), "inview": inview, "valid": valid}
        if colour:
            n = ag.clamp(min=1)[:, None]
            out["spread"] = (csq / n - (csum / n) ** 2).clamp(min=0).sqrt().mean(1)
        rr, pd, pv = self.pano_lookup(P)
        pa = pv & ((pd - rr).abs() < TOL * rr) if self.pano else z(torch.bool)
        pc = pv & (pd > (1 + TOL) * rr) & (pd < MAX_CONTRA_M) if self.pano else z(torch.bool)
        out["pano_agree"], out["pano_contra"] = pa, pc
        return out

    def sky(self, m: np.ndarray, K: np.ndarray, w: int, h: int) -> np.ndarray:
        """Pixels whose ray points up into the panorama's sky."""
        ph, pw = self.sky_pano.shape
        j, i = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        ray = np.stack([(j - K[0, 2]) / K[0, 0], (i - K[1, 2]) / K[1, 1], np.ones_like(j)], -1) \
            @ np.linalg.inv(m)[:3, :3].T
        yaw = np.arctan2(ray[..., 1], ray[..., 0])
        pitch = np.arcsin(ray[..., 2] / np.linalg.norm(ray, axis=-1))
        return self.sky_pano[np.clip(((0.5 - pitch / np.pi) * ph).astype(int), 0, ph - 1),
                             np.clip(((0.5 - yaw / (2 * np.pi)) * pw).astype(int), 0, pw - 1)] & (ray[..., 2] > 0)

    def sky_colour(self, m: np.ndarray, K: np.ndarray, w: int, h: int) -> np.ndarray:
        """The panorama's colour along each pixel's ray (sky is at infinity: the same from anywhere)."""
        if not hasattr(self, "pano_rgb"):
            meta = json.loads((ws.DATA / self.case / "prep.json").read_text())
            self.pano_rgb = np.asarray(Image.open(ws.PANOS / self.case / meta["pano"]).convert("RGB"))
        ph, pw = self.pano_rgb.shape[:2]
        j, i = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        ray = np.stack([(j - K[0, 2]) / K[0, 0], (i - K[1, 2]) / K[1, 1], np.ones_like(j)], -1) \
            @ np.linalg.inv(m)[:3, :3].T
        yaw = np.arctan2(ray[..., 1], ray[..., 0])
        pitch = np.arcsin(ray[..., 2] / np.linalg.norm(ray, axis=-1))
        return self.pano_rgb[np.clip(((0.5 - pitch / np.pi) * ph).astype(int), 0, ph - 1),
                             np.clip(((0.5 - yaw / (2 * np.pi)) * pw).astype(int), 0, pw - 1)]

    def walked(self, x: float, y: float, hd: float) -> float:
        """Distance from (x, y) to the nearest generated frame facing within HEADING_TOL_DEG of hd."""
        dy = np.abs(nb.wrap(self.frame_yaw - hd))
        p = self.frame_pos[dy < HEADING_TOL_DEG]
        return float(np.hypot(p[:, 0] - x, p[:, 1] - y).min()) if len(p) else 99.0


def classify(ev: Evidence, sp, m: np.ndarray, K: np.ndarray, w: int, h: int, colour: bool = False) -> dict:
    """Class map of one student view plus the per-pixel evidence it came from."""
    _, dep, al = nb.render(sp, m[None], K, w, h)
    surf = (al[0] > 0.5).cpu().numpy()
    e = ev.points(nb.lift(dep, K, m[None]).reshape(-1, 3), colour)
    e = {k: v.reshape(h, w).cpu().numpy() for k, v in e.items()}
    ag = e["agree"] + e["pano_agree"]
    co = e["contra"] + e["pano_contra"]
    has = (ag + co) > 0
    share = np.where(has, co / np.maximum(ag + co, 1), np.nan)
    sky = ev.sky(m, K, w, h)
    code = np.full((h, w), NODATA, np.uint8)
    code[has & (share < 0.5)] = CONFIRMED
    code[has & (share >= 0.5)] = CONFLICT
    code[~surf] = HOLE
    code[~surf & sky] = SKY
    e.update(code=code, share=share, surf=surf, has=has)
    return e


def nbv_classify(sp, ev: Evidence, w2cs: np.ndarray, K: np.ndarray, w: int, h: int):
    """ws_nbv.classify's contract with this file's targets: code 0 sky, 2 known (confirmed or in conflict), 3 target
    (no data, or a hole); the target keys; RGB; world points (holes take the depth of the nearest surface pixel, so
    a leg that looks there counts as seeing them); and the non-sky mask."""
    import torch
    from scipy import ndimage
    rgb, dep, al = nb.render(sp, w2cs, K, w, h)
    surf = al > 0.5
    fill = dep.clone()
    for i in range(len(dep)):
        s = surf[i].cpu().numpy()
        if s.any() and not s.all():
            j = ndimage.distance_transform_edt(~s, return_distances=False, return_indices=True)
            fill[i] = torch.as_tensor(dep[i].cpu().numpy()[j[0], j[1]], device=dep.device)
    X = nb.lift(fill, K, w2cs)
    e = ev.points(X.reshape(-1, 3), colour=False)
    has = ((e["agree"] + e["pano_agree"] + e["contra"] + e["pano_contra"]) > 0).reshape(dep.shape)
    sky = torch.as_tensor(np.stack([ev.sky(m, K, w, h) for m in w2cs]), device=dep.device) & ~surf
    code = torch.full(dep.shape, 2, dtype=torch.uint8, device=dep.device)
    code[~has | ~surf] = 3
    code[sky] = 0
    pos = torch.as_tensor(np.linalg.inv(np.asarray(w2cs))[:, :3, 3], dtype=torch.float32, device=dep.device)
    P = pos[:, None, None].expand_as(X)
    tgt = code == 3
    return code, nb.keys_of(X[tgt], P[tgt]), rgb, X, code != 0


def stats(e: dict) -> dict:
    code = e["code"]
    ground = code != SKY
    sf = max(int((code[ground] != HOLE).sum()), 1)
    nod = code == NODATA
    nn_ = max(int(nod.sum()), 1)
    return {"hole": float((code == HOLE).sum() / max(ground.sum(), 1)),
            "nodata": float(nod.sum() / sf), "conflict_px": float((code == CONFLICT).sum() / sf),
            "conflict_share": float(np.nanmean(e["share"][e["has"] & e["surf"]])) if (e["has"] & e["surf"]).any() else 0.0,
            "hidden": float((nod & e["valid"]).sum() / nn_),
            "masked": float((nod & e["inview"] & ~e["valid"]).sum() / nn_),
            "unviewed": float((nod & ~e["inview"]).sum() / nn_)}


def grid(case: str, arm: str, radius: float, pano: bool):
    import torch
    t0 = time.time()
    dev = torch.device("cuda")
    sp = nb.load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", dev)
    ev = Evidence(case, sp, pano)
    nb.STUDENT_RADIUS_M = radius
    w2cs, where = nb.student_poses(nb.obstacles(case, sp))
    w, h = GRID_SIZE
    K = nb.intrinsic(w, h, nb.STUDENT_FOV_DEG)
    rows, codes = [], []
    for k, (x, y, hd) in enumerate(where):
        e = classify(ev, sp, w2cs[k], K, w, h)
        s = stats(e)
        s.update(x=x, y=y, heading=hd, walked=ev.walked(x, y, hd))
        rows.append(s)
        codes.append(np.asarray(Image.fromarray(e["code"]).resize(CODE_SIZE, Image.NEAREST)))
        if k % 100 == 0:
            print(f"{k}/{len(where)} views, {time.time() - t0:.0f} s", flush=True)
    d = out_dir(case, arm)
    keys = list(rows[0])
    np.savez_compressed(d / ("grid.npz" if pano else "grid_nopano.npz"), codes=np.stack(codes),
                        **{k: np.array([r[k] for r in rows]) for k in keys})
    summary = summarise(rows, radius)
    summary.update(case=case, arm=arm, pano=pano, views=len(rows), seconds=round(time.time() - t0),
                   ratio_median=float(ev.ratio.median()), pano_ratio=ev.pratio)
    (d / ("grid.json" if pano else "grid_nopano.json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["all"]), flush=True)
    for b in summary["by_walked"]:
        print(b, flush=True)
    if pano:
        grid_sheet(case, arm, ev.frame_pos, rows, summary)


def summarise(rows: list[dict], radius: float) -> dict:
    keys = ("hole", "nodata", "conflict_px", "conflict_share", "hidden", "masked", "unviewed")
    mean = lambda rs: {k: round(float(np.mean([r[k] for r in rs])), 4) for k in keys} | {"views": len(rs)}  # noqa
    out = {"all": mean(rows), "by_walked": [], "by_radius": []}
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        rs = [r for r in rows if lo <= r["walked"] < hi]
        if rs:
            out["by_walked"].append({"walked_m": f"{lo}-{hi}"} | mean(rs))
    for lo in range(0, int(radius), 2):
        rs = [r for r in rows if lo <= math.hypot(r["x"], r["y"]) < lo + 2]
        if rs:
            out["by_radius"].append({"radius_m": f"{lo}-{lo + 2}"} | mean(rs))
    return out


def grid_sheet(case, arm, frame_pos, rows, summary):
    """Top views, one per measure (mean of the 8 headings at each grid spot, green 0, red at the scale's top),
    generated frames in white; plus the bins by distance to walked and by radius."""
    font = ImageFont.load_default(size=18)
    S, R = 520, max(math.hypot(r["x"], r["y"]) for r in rows) + 1.5
    px = lambda x, y: (int(S / 2 - y / R * S / 2), int(S / 2 - x / R * S / 2))  # noqa: E731  x up, y left
    maps = [("holes + no data", lambda r: r["hole"] + r["nodata"], 0.15), ("conflict", lambda r: r["conflict_px"], 0.4)]
    im = Image.new("RGB", (2 * S + 760, S + 60))
    dr = ImageDraw.Draw(im)
    cell = S / 2 / R * nb.STUDENT_GRID_M / 2
    for n, (title, f, top) in enumerate(maps):
        spots = {}
        for r in rows:
            spots.setdefault((r["x"], r["y"]), []).append(f(r))
        for (x, y), v in spots.items():
            t = min(float(np.mean(v)) / top, 1.0)
            cx, cy = px(x, y)
            dr.rectangle([n * S + cx - cell, 40 + cy - cell, n * S + cx + cell, 40 + cy + cell],
                         fill=(int(255 * t), int(200 * (1 - t)), 40))
        for p in frame_pos:
            cx, cy = px(p[0], p[1])
            if 0 <= cx < S and 0 <= cy < S:
                dr.point((n * S + cx, 40 + cy), fill=(255, 255, 255))
        dr.text((n * S + 8, 40), f"{title} (red {top:.0%}+)", fill=(255, 255, 0), font=font)
    dr.text((8, 8), f"{case} {arm}: per grid spot, mean of 8 headings; x up, y left; white = generated frames",
            fill=(255, 255, 255), font=font)
    lines = ["by distance to walked  views  hole  no data  conflict | of no data: hidden masked unviewed"]
    for b in summary["by_walked"]:
        lines.append(f"{b['walked_m']:>8s} m  {b['views']:5d}  {b['hole']:.3f}  {b['nodata']:.3f}  {b['conflict_px']:.3f}"
                     f" | {b['hidden']:.2f} {b['masked']:.2f} {b['unviewed']:.2f}")
    lines += ["", "by radius from the centre"]
    for b in summary["by_radius"]:
        lines.append(f"{b['radius_m']:>8s} m  {b['views']:5d}  {b['hole']:.3f}  {b['nodata']:.3f}  {b['conflict_px']:.3f}"
                     f" | {b['hidden']:.2f} {b['masked']:.2f} {b['unviewed']:.2f}")
    a = summary["all"]
    lines += ["", f"all {a['views']} views: hole {a['hole']:.3f}, no data {a['nodata']:.3f}, conflict {a['conflict_px']:.3f}"]
    dr.multiline_text((2 * S + 10, 40), "\n".join(lines), fill=(255, 255, 255), font=ImageFont.load_default(size=15),
                      spacing=6)
    p = out_dir(case, arm) / "sheet_grid.jpg"
    im.save(p, quality=88)
    print(p, flush=True)


def grid_sheet_only(case: str, arm: str):
    """Redraw sheet_grid.jpg from grid.npz/grid.json (frame positions from depth_ma, no GPU)."""
    d = out_dir(case, arm)
    z = np.load(d / "grid.npz")
    keys = [k for k in z.files if k != "codes"]
    rows = [{k: z[k][i].item() for k in keys} for i in range(len(z["x"]))]
    names = sorted({str(n).rsplit("/", 1)[0] for n in np.load(ws.DATA / case / "splat" / "views.npz")["name"]} - {"pano"})
    pos = np.concatenate([np.load(ws.DATA / case / "depth_ma" / f"{n.replace('/', '__')}.npz")["c2w"][:, :3, 3]
                          for n in names])
    grid_sheet(case, arm, pos, rows, json.loads((d / "grid.json").read_text()))


def sheet(case: str, arm: str, spots: str, radius: float):
    """Every heading at the given spots: render | legs with evidence | conflict share | colour spread | targets in
    red (no data) and magenta (holes) on the render."""
    import cv2
    import torch
    dev = torch.device("cuda")
    sp = nb.load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", dev)
    ev = Evidence(case, sp)
    nb.STUDENT_RADIUS_M = radius
    w2cs, where = nb.student_poses(nb.obstacles(case, sp))
    want = [tuple(float(v) for v in t.split(",")) for t in spots.split(";")]
    ks = [k for s_ in want for k, (x, y, hd) in enumerate(where) if (x, y) == s_]
    W, H = 640, 480
    w, h = GRID_SIZE
    K, Kf = nb.intrinsic(w, h, nb.STUDENT_FOV_DEG), nb.intrinsic(W, H, nb.STUDENT_FOV_DEG)
    font = ImageFont.load_default(size=20)
    tw, th = 480, 360
    heads = ["render", "legs that saw it (black 0, white 4+), sky blue, hole magenta",
             "geometric conflict share (grey = no data)", "colour spread among agreeing frames",
             "targets: no data red, holes magenta"]
    sh = Image.new("RGB", (5 * tw, 70 + len(ks) * (th + 34)))
    dr = ImageDraw.Draw(sh)
    dr.text((6, 6), f"{case} {arm}: every heading at {want}. TOL {TOL:.0%}, panorama counts as a leg",
            fill=(255, 255, 255), font=font)
    for c, t in enumerate(heads):
        dr.text((c * tw + 6, 40), t, fill=(0, 255, 255), font=ImageFont.load_default(size=15))

    def cmap(v, hi, e):
        o = cv2.cvtColor(cv2.applyColorMap((np.clip(np.nan_to_num(v) / hi, 0, 1) * 255).astype(np.uint8),
                                           cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        o[e["code"] == NODATA] = 90
        o[e["code"] == HOLE] = CLASS_RGB[HOLE]
        o[e["code"] == SKY] = CLASS_RGB[SKY]
        return o

    for n, k in enumerate(ks):
        e = classify(ev, sp, w2cs[k], K, w, h, colour=True)
        s = stats(e)
        legs = np.repeat((np.clip(e["legs"] / 4, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
        legs[e["code"] == HOLE] = CLASS_RGB[HOLE]
        legs[e["code"] == SKY] = CLASS_RGB[SKY]
        full = nb.render(sp, w2cs[k][None], Kf, W, H)[0][0].cpu().numpy()
        big = cv2.resize(e["code"], (W, H), interpolation=cv2.INTER_NEAREST)
        red = full.copy()
        for cls in (NODATA, HOLE):
            m = big == cls
            red[m] = (0.4 * red[m] + 0.6 * np.array(CLASS_RGB[cls])).astype(np.uint8)
        y = 70 + n * (th + 34)
        x0, y0, hd = where[k]
        dr.text((6, y + 4), f"({x0:.0f},{y0:.0f}) m heading {hd}: {ev.walked(x0, y0, hd):.1f} m from walked | hole "
                f"{s['hole']:.2f} | no data {s['nodata']:.2f} (hidden {s['hidden']:.2f}, unviewed {s['unviewed']:.2f})"
                f" | conflict {s['conflict_share']:.2f}", fill=(255, 255, 0), font=font)
        for c, im in enumerate([full, legs, cmap(e["share"], 0.6, e), cmap(e["spread"], 40, e), red]):
            sh.paste(Image.fromarray(im).resize((tw, th), Image.NEAREST if c in (1, 2, 3) else Image.BILINEAR),
                     (c * tw, y + 30))
    p = out_dir(case, arm) / "sheet_spots.jpg"
    sh.save(p, quality=88)
    print(p, flush=True)


def gaussians(case: str, arm: str):
    """The same evidence for every Gaussian's centre: what WorldStereo's frames say about where it sits."""
    import torch
    t0 = time.time()
    sp = nb.load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", torch.device("cuda"))
    ev = Evidence(case, sp)
    out = {}
    for i in range(0, len(sp["means"]), 500_000):
        e = ev.points(sp["means"][i:i + 500_000], colour=False)
        for k in ("agree", "contra", "inview", "valid", "pano_agree", "pano_contra"):
            out.setdefault(k, []).append(e[k].cpu().numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    ag = out["agree"] + out["pano_agree"]
    co = out["contra"] + out["pano_contra"]
    n = len(ag)
    print(f"{n} Gaussians: backed {np.mean(ag > co):.3f}, contradicted {np.mean((co >= ag) & (co > 0)):.3f}, "
          f"unbacked {np.mean(ag + co == 0):.3f} (of which never in view {np.mean(~out['inview'][ag + co == 0]):.3f}), "
          f"{time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(out_dir(case, arm) / "gaussians.npz", **out)


def compare(case: str, arms: list[str], spots: str, radius: float):
    """The same student cameras (every heading at the spots) rendered from each arm, side by side."""
    import torch
    dev = torch.device("cuda")
    nb.STUDENT_RADIUS_M = radius
    sps = {a: nb.load_splat(ws.DATA / case / "splat" / "arms" / a / "splat.ply", dev) for a in arms}
    w2cs, where = nb.student_poses(nb.obstacles(case, sps[arms[0]]))
    want = [tuple(float(v) for v in t.split(",")) for t in spots.split(";")]
    ks = [k for s_ in want for k, (x, y, hd) in enumerate(where) if (x, y) == s_]
    W, H = 640, 480
    K = nb.intrinsic(W, H, nb.STUDENT_FOV_DEG)
    imgs = {a: nb.render(sp, w2cs[ks], K, W, H)[0].cpu().numpy() for a, sp in sps.items()}
    font = ImageFont.load_default(size=20)
    tw, th = 480, 360
    sh = Image.new("RGB", (len(arms) * tw, 40 + len(ks) * (th + 28)))
    dr = ImageDraw.Draw(sh)
    dr.text((6, 8), f"{case}: " + " | ".join(f"{a} ({len(sps[a]['means']) / 1e6:.2f}M)" for a in arms),
            fill=(255, 255, 255), font=font)
    for n, k in enumerate(ks):
        y = 40 + n * (th + 28)
        dr.text((6, y + 2), f"({where[k][0]:.0f},{where[k][1]:.0f}) heading {where[k][2]}", fill=(255, 255, 0), font=font)
        for c, a in enumerate(arms):
            sh.paste(Image.fromarray(imgs[a][n]).resize((tw, th)), (c * tw, y + 26))
    p = ws.DATA / case / "conflict" / f"sheet_compare_{'_'.join(arms)}.jpg"
    sh.save(p, quality=86)
    print(p, flush=True)


RULES = {
    # no frame and no panorama agrees with or contradicts where it sits: hidden or never looked at
    "unbacked": lambda ag, co: (ag + co) == 0,
    # also those that more frames look through than agree with: floaters in free space
    "unbacked_contra": lambda ag, co: ((ag + co) == 0) | (co > ag),
}


def prune(case: str, arm: str, rule: str):
    from plyfile import PlyData, PlyElement
    g = np.load(out_dir(case, arm) / "gaussians.npz")
    ag = g["agree"] + g["pano_agree"]
    co = g["contra"] + g["pano_contra"]
    drop = RULES[rule](ag, co)
    src = ws.DATA / case / "splat" / "arms" / arm / "splat.ply"
    v = PlyData.read(src)["vertex"].data
    assert len(v) == len(drop), "gaussians.npz is from another splat"
    dst = ws.DATA / case / "splat" / "arms" / f"{arm}_{rule}"
    dst.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v[~drop], "vertex")]).write(dst / "splat.ply")
    print(f"{rule}: dropped {drop.sum()} of {len(drop)} ({drop.mean():.1%}) -> {dst / 'splat.ply'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "grid_sheet", "sheet", "gaussians", "prune", "compare"])
    ap.add_argument("--arms", default="G2,G2_unbacked,G2_unbacked_contra")
    ap.add_argument("--case", default="photo21_np")
    ap.add_argument("--arm", default="G2")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--spots", default="9,-4;8,-4")
    ap.add_argument("--rule", default="unbacked", choices=list(RULES))
    ap.add_argument("--no-pano", action="store_true", help="leave the panorama out of the evidence")
    a = ap.parse_args()
    {"grid": lambda: grid(a.case, a.arm, a.radius, not a.no_pano),
     "grid_sheet": lambda: grid_sheet_only(a.case, a.arm), "sheet": lambda: sheet(a.case, a.arm, a.spots, a.radius),
     "gaussians": lambda: gaussians(a.case, a.arm), "prune": lambda: prune(a.case, a.arm, a.rule),
     "compare": lambda: compare(a.case, a.arms.split(","), a.spots, a.radius)}[a.cmd]()
