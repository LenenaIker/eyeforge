"""U1: WorldStereo generation laid out like upstream's own panorama example, on an e2e case's panorama (row U1 of
docs/pruebas_reconstruccion_worldstereo.md). The e2e plan (ws_test.plan) walks 3-6.5 m per first leg and 4 m more
per second leg, with 50-90 % of every guide frame empty; upstream's example (examples/panorama, measured
2026-10-24) and the paper do otherwise, and its memory is trained on 30-70 % masks:

    view<i>/traj2, traj0, traj1   first, at the photo's spot, three directions 120 degrees apart: rise and look down
                                  while orbiting right, orbit right, orbit left (51 degrees around a point ORBIT x the
                                  median distance ahead). Upstream's guides there are ~2 % empty; they fill the
                                  memory before anything moves. traj0/traj1 retrieve from traj2 (upstream's rule).
    reconstruct_<k>/traj0         walk k's first STEP metres (upstream: 0.8-2.3 m)
    reconstruct_<k>/traj1         a small look round at traj0's end (rise, look down, sway, back), run last by
                                  upstream as its "iterative" leg; NOT a second walk as in the e2e plan
    wonder_1/traj0                the raised leg of the e2e plan (user, 2026-10-24)
    step_<k>_<j>/traj0            walk k on, STEP metres at a time up to the e2e reach, each starting on the leg
                                  before's last frame (chain_from.txt) after the cache has grown with it

The guide cloud keeps the sky (at the contraction's far limit, upstream's depth squashing), so sky no longer counts
as a hole. The cache grows leg by leg with ws_cache_job.py's aligned placement (cache_align.txt).

    <eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_upstream.py stage --photo photo27   # scene, sheet
    python experiments/worldstereo/ws_upstream.py run                                              # VM (approval)
    python experiments/worldstereo/ws_upstream.py link --photo photo27                             # as case u1_<photo>

U2 (--batch u2): U1 measured worse away from the centre because the chained step legs stall, and better at the centre
(view orbits, aligned cache). U2 keeps the view orbits and the aligned cache, takes the e2e plan's legs, and starts
every one of them at the photo's spot: reconstruct_<k>/traj0 is e2e's first leg, wonder_<2+k>/traj0 goes from the
spot to the end of e2e's second leg in one leg, wonder_0/1 are e2e's. Nothing starts on a generated frame.

`set -a; . ./.env; set +a` before run, like ws_vm.py."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

BATCH = "u1"
STEP = 2.0          # metres per walking leg
FIRST_MAX = 2.5     # the first leg (photo spot -> walk line) stops after this many metres
ORBIT = 0.3         # orbit radius as a share of the median distance ahead (paper, appendix C)
ORBIT_M = (0.8, 2.0)
ORBIT_DEG = 51.0    # upstream's example
AERIAL_RISE = 0.7   # x the orbit radius (upstream: 0.57 for 0.81)
AERIAL_PITCH = -45.0
VIEW_YAWS = (0.0, 120.0, -120.0)
LOOK_RISE, LOOK_PITCH, LOOK_SWAY = 0.4, -20.0, 18.0  # reconstruct traj1, upstream's ~0.3 m, -26 deg, +-18 deg
COLS = [(255, 60, 60), (60, 200, 255), (255, 220, 0), (120, 255, 120), (255, 120, 255)]


def ease(t):
    return t * t * (3 - 2 * t)


def fwd(yaw):
    y = math.radians(yaw)
    return np.array([math.cos(y), math.sin(y), 0.0])


def orbit(yaw0, r, deg, z=(0.0, 0.0), pitch=(ws.PITCH, ws.PITCH), start=(0.0, 0.0, 0.0)):
    """Round a point r metres ahead of start: camera moves sideways (right for deg > 0) and keeps facing it."""
    c = np.asarray(start, float) + r * fwd(yaw0)
    out = []
    for t in np.linspace(0, 1, ws.NFRAME):
        e = ease(t)
        a = yaw0 + deg * e
        out.append(ws.w2c(c - r * fwd(a) + [0, 0, z[0] + (z[1] - z[0]) * e], a, pitch[0] + (pitch[1] - pitch[0]) * e))
    return out


def aerial(yaw0, r):
    """Upstream's up-right-aerial: first half rise and look down, second half orbit right at that height."""
    h = AERIAL_RISE * r
    half = ws.NFRAME // 2
    out = []
    for t in np.linspace(0, 1, half + 1):
        e = ease(t)
        out.append(ws.w2c(-0.3 * r * fwd(yaw0) * e + [0, 0, h * e], yaw0, ws.PITCH + (AERIAL_PITCH - ws.PITCH) * e))
    top = np.linalg.inv(out[-1])[:3, 3]
    rest = orbit(yaw0, r, 60.0, (top[2], top[2]), (AERIAL_PITCH, AERIAL_PITCH), (top[0], top[1], 0.0))
    return out + rest[1:ws.NFRAME - half]


def look(pos, yaw):
    """A small look round at pos, ending where it started (upstream's reconstruct traj1)."""
    out = []
    for t in np.linspace(0, 1, ws.NFRAME):
        b = math.sin(math.pi * t)  # 0 -> 1 -> 0
        out.append(ws.w2c(np.asarray(pos) + [0, 0, LOOK_RISE * b], yaw + LOOK_SWAY * math.sin(2 * math.pi * t),
                          ws.PITCH + (LOOK_PITCH - ws.PITCH) * b))
    return out


def ahead(dist, valid, yaw, half=30.0):
    """Median distance (metres) of the panorama within half degrees of yaw, near the horizon."""
    dirs = ws.pano_dirs(*dist.shape)
    yy = np.degrees(np.arctan2(dirs[..., 1], dirs[..., 0]))
    band = valid & (np.abs((yy - yaw + 180) % 360 - 180) < half) & (np.abs(dirs[..., 2]) < 0.2) & (dist > 0)
    return float(np.median(dist[band])) if band.sum() > 100 else 8.0


def plan(meta, dist, valid):
    """Folder -> {poses, kind, length, group, chain}."""
    c = np.asarray(meta["centre"], float)
    trajs = {}
    for i, yv in enumerate(VIEW_YAWS):
        r = float(np.clip(ORBIT * ahead(dist, valid, yv), *ORBIT_M))
        trajs[f"view{i}/traj2"] = dict(poses=aerial(yv, r), kind="up-right-aerial", length=r, group=4)
        trajs[f"view{i}/traj0"] = dict(poses=orbit(yv, r, ORBIT_DEG), kind="right orbit", length=r, group=4)
        trajs[f"view{i}/traj1"] = dict(poses=orbit(yv, r, -ORBIT_DEG), kind="left orbit", length=r, group=4)
    for k, (yaw, free) in enumerate(meta["walks"]):
        d = fwd(yaw)
        reach = min(2 * ws.LEG, free)  # the e2e plan's reach: two 4 m legs
        goal = c + d * min(STEP, reach)
        n = float(np.linalg.norm(goal))
        end = goal * min(1.0, FIRST_MAX / n) if n > 0 else goal
        trajs[f"reconstruct_{k}/traj0"] = dict(poses=ws.between([0, 0, 0], end, yaw), kind="walk", length=n, group=k)
        trajs[f"reconstruct_{k}/traj1"] = dict(poses=look(end, yaw), kind="look round", length=0.0, group=k)
        prev, at, j = f"reconstruct_{k}/traj0", end, 1
        along = float((end - c) @ d)
        while along < reach - 0.25:
            along = min(along + STEP, reach)
            nxt = c + d * along
            trajs[f"step_{k}_{j}/traj0"] = dict(poses=ws.between(at, nxt, yaw), kind="step", group=k, chain=prev,
                                                length=float(np.linalg.norm(nxt - at)))
            prev, at, j = f"step_{k}_{j}/traj0", nxt, j + 1
    yaw, free = meta["walks"][0]
    lr = min(ws.LEG, free) * 0.5
    trajs["wonder_1/traj0"] = dict(poses=ws.between([0, 0, 0], c + fwd(yaw) * lr, yaw, (0.0, ws.RAISE),
                                                    (ws.PITCH, ws.RAISE_PITCH)), kind="raise", length=lr, group=0)
    return trajs


def plan_u2(meta, dist, valid):
    """U2's legs (module docstring)."""
    trajs = {n: t for n, t in plan(meta, dist, valid).items() if n.startswith("view")}
    c = np.asarray(meta["centre"], float)
    for n, t in ws.plan(meta["walks"], c).items():
        if n.endswith("traj1"):
            end = np.linalg.inv(t["poses"][-1])[:3, 3]
            n = f"wonder_{2 + int(n.split('_')[1].split('/')[0])}/traj0"
            t = dict(poses=ws.between([0, 0, 0], end, t["yaw"]), kind="far walk", length=float(np.linalg.norm(end)))
        trajs[n] = dict(poses=t["poses"], kind=t["kind"], length=t["length"],
                        group=int(n.split("_")[1].split("/")[0]) % 4 if n.startswith("reconstruct") else 0)
    for n, t in trajs.items():
        if n.startswith("wonder_") and int(n.split("_")[1].split("/")[0]) >= 2:
            t["group"] = int(n.split("_")[1].split("/")[0]) - 2
    return trajs


def clouds(case):
    """The guide cloud of the e2e case (ws_test.guide_cloud) and its sky at the contraction's far limit."""
    import cv2
    import torch
    meta = json.loads((ws.DATA / case / "prep.json").read_text())
    full = np.asarray(Image.open(ws.PANOS / case / meta["pano"]).convert("RGB"))
    pano = cv2.resize(full, (ws.PANO_W, ws.PANO_W // 2), interpolation=cv2.INTER_AREA)
    z = np.load(ws.DATA / case / "distance.npz")
    dist, valid = z["distance"] * meta["scale"], z["mask"]
    points, colors, _ = ws.guide_cloud(case)
    keep = valid & ~ws.edge_mask(dist) & np.isfinite(dist) & (dist > 0)
    far = 2 * np.median(dist[keep]) * 8 * 0.999
    dirs = ws.pano_dirs(*dist.shape)
    sky = ~valid & (dirs[..., 2] > -0.05)
    sp = torch.tensor(dirs[sky] * far, dtype=torch.float32)
    sc = torch.tensor(pano[sky] / 127.5 - 1, dtype=torch.float32)
    return meta, full, dist, valid, (points, colors), (torch.cat([points, sp]), torch.cat([colors, sc]))


def stage(photo: str):
    case = f"e2e_{photo}"
    src = ws.DATA / case / "scene"
    out = ws.DATA / BATCH / "scene" / f"{BATCH}_{photo}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    meta, full, dist, valid, (gp, gc), (ap, ac) = clouds(case)
    shutil.copytree(src / "pano_bank", out / "pano_bank")
    caption = (src / "reconstruct_0" / "traj0" / "traj_caption.json").read_text()
    np.savez(out / "cache_init.npz", points=ap.numpy().astype(np.float32), colors=ac.numpy().astype(np.float16))
    (out / "cache_align.txt").write_text("umeyama\n")
    trajs = (plan_u2 if BATCH == "u2" else plan)(meta, dist, valid)
    rnd = ws.renderer()
    info = {"case": case, "photo": photo, "legs": {}}
    for name, t in trajs.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        rgb, holes = ws.render(rnd, ap, ac, t["poses"])
        _, holes_nosky = ws.render(rnd, gp, gc, t["poses"])
        if "chain" in t:
            (d.parent / "chain_from.txt").write_text(t["chain"])
            holes[0] = 0  # its first frame is the leg before's last one
        elif name.endswith("traj1") and name.startswith("reconstruct_"):
            holes[0] = 0  # upstream starts it on traj0's last frame
        else:
            Image.fromarray(ws.crop(full, t["poses"][0])).save(d.parent / "start_frame.png")
            rgb[0], holes[0] = np.asarray(Image.open(d.parent / "start_frame.png")), 0
        ws.write_video(d / "render.mp4", rgb)
        ws.write_video(d / "render_mask.mp4", holes)
        (d / "camera.json").write_text(json.dumps({
            "extrinsic": [p.tolist() for p in t["poses"]], "intrinsic": [ws.intrinsic()] * ws.NFRAME,
            "width": ws.W, "height": ws.H, "type": t["kind"], "rotation_deg": 0}))
        (d / "traj_caption.json").write_text(caption)
        h = (holes[1:] > 127).mean(axis=(1, 2))
        info["legs"][name] = {"kind": t["kind"], "length": round(t["length"], 2), "group": t["group"],
                              "chain": t.get("chain"), "holes_mean": round(float(h.mean()), 3),
                              "holes_max": round(float(h.max()), 3),
                              "holes_nosky_mean": round(float((holes_nosky[1:] > 127).mean()), 3)}
        Image.fromarray(np.concatenate([rgb[0], rgb[ws.NFRAME // 2], rgb[-1]], 1)).save(d.parent / f"{d.name}_render.jpg")
        print(f"{name:22s} {t['kind']:16s} {t['length']:.1f} m  holes {h.mean():.2f} (max {h.max():.2f}, "
              f"without sky in the cloud {info['legs'][name]['holes_nosky_mean']:.2f})", flush=True)
    (ws.DATA / BATCH / f"{BATCH}_{photo}.json").write_text(json.dumps(info, indent=1))
    sheet(photo, meta, trajs, info)


def e2e_holes(case):
    """The e2e plan's legs and their guide holes (prep.json), for the sheet's comparison."""
    meta = json.loads((ws.DATA / case / "prep.json").read_text())
    return {n: float(np.mean(t["holes"][1:])) for n, t in meta["trajs"].items()}


def sheet(photo, meta, trajs, info):
    """Top view of every leg (coloured by walk, views white), the hole table against the e2e plan, and each leg's
    guide at start, middle and end: what the user approves before the cloud run."""
    from PIL import ImageFont
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=17)
    s, half = 900, 11.0
    top = Image.new("RGB", (s, s), (20, 20, 20))
    td = ImageDraw.Draw(top)

    def px(x, y):
        return s / 2 - y / half * s / 2, s / 2 - x / half * s / 2
    for r in range(2, 12, 2):
        a, b = px(r, r), px(-r, -r)
        td.ellipse([a[0], a[1], b[0], b[1]], outline=(55, 55, 55))
    for name, t in trajs.items():
        cs = [np.linalg.inv(p)[:3, 3] for p in t["poses"]]
        col = (230, 230, 230) if t["group"] == 4 else COLS[t["group"]]
        td.line([px(c[0], c[1]) for c in cs], fill=col, width=2 if t["kind"] in ("look round", "raise") else 4)
        td.ellipse([*(np.array(px(cs[-1][0], cs[-1][1])) - 4), *(np.array(px(cs[-1][0], cs[-1][1])) + 4)], fill=col)
    td.text((10, 10), f"{BATCH}_{photo}: top view, rings every 2 m; white = view orbits", fill=(255, 255, 255), font=font)

    e2e = e2e_holes(f"e2e_{photo}")
    legs = info["legs"]
    lines = [f"{BATCH.upper()}: {len(legs)} legs (e2e: {len(e2e)})", "guide holes per frame, mean (max), sky now in the cloud:", ""]
    for n, l in legs.items():
        lines.append(f"{n:20s} {l['kind'][:13]:13s} {l['length']:4.1f} m  {l['holes_mean']:.2f} ({l['holes_max']:.2f})"
                     + ("  *" if l["chain"] else ""))
    lines += ["", "* starts on the leg before's last frame: its guide here is", "  the panorama only; in the run the cache "
              "fills it first", "", "e2e plan (same panorama), mean holes:"]
    lines += [f"{n:20s} {v:.2f}" for n, v in e2e.items()]
    head = Image.new("RGB", (1800, max(s, 30 + 21 * len(lines))), (0, 0, 0))
    head.paste(top, (0, 0))
    ImageDraw.Draw(head).multiline_text((920, 10), "\n".join(lines), fill=(255, 255, 255), font=small, spacing=4)
    rows = [head]
    root = ws.DATA / BATCH / "scene" / f"{BATCH}_{photo}"
    names = list(trajs)
    for i in range(0, len(names), 2):
        row = Image.new("RGB", (1800, 196), (0, 0, 0))
        for j, n in enumerate(names[i:i + 2]):
            im = Image.open(root / f"{n.split('/')[0]}" / f"{n.split('/')[1]}_render.jpg").resize((891, 171))
            row.paste(im, (j * 909, 25))
            l = legs[n]
            ImageDraw.Draw(row).text((j * 909 + 4, 2), f"{n}  {l['kind']}  holes {l['holes_mean']:.2f}",
                                     fill=(255, 255, 0), font=small)
        rows.append(row)
    out = Image.new("RGB", (1800, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    path = ws.DATA / BATCH / f"sheet_prep_{BATCH}_{photo}.jpg"
    out.save(path, quality=85)
    print(path)


def run():
    import ws_cache
    import ws_vm
    ws_vm.MAX_HOURS = 1.5  # ~30 legs plus the cache's MapAnything per leg
    ws_cache.run(BATCH)


def link(photo: str):
    """The run's results as an ordinary case data/worldstereo_test/u1_<photo>/ (scene, prep.json, out/out/<case>),
    so ws_splat.py views/train and ws_depth.py read it like any e2e case."""
    case = f"{BATCH}_{photo}"
    dst = ws.DATA / case
    got = ws.DATA / BATCH / "out" / "out" / BATCH / case
    if not got.exists():
        sys.exit(f"no results at {got}: fetch first")
    for d, s in ((dst / "scene", ws.DATA / BATCH / "scene" / case), (dst / "out" / "out" / case, got)):
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)
    meta = json.loads((ws.DATA / f"e2e_{photo}" / "prep.json").read_text())
    info = json.loads((ws.DATA / BATCH / f"{case}.json").read_text())
    meta["case"], meta["trajs"] = case, {n: {"kind": l["kind"], "length": l["length"], "holes": [l["holes_mean"]]}
                                         for n, l in info["legs"].items()}
    (dst / "prep.json").write_text(json.dumps(meta, indent=1))
    for f in ("distance.npz",):
        shutil.copy(ws.DATA / f"e2e_{photo}" / f, dst / f)
    pano = ws.PANOS / case  # ws_splat and ws_depth read the panorama by case name
    if not pano.exists():
        pano.symlink_to(ws.PANOS / f"e2e_{photo}", target_is_directory=True)
    print(f"linked {case}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "run", "link"])
    ap.add_argument("--photo", default="photo27")
    ap.add_argument("--batch", default=BATCH, choices=["u1", "u2"])
    a = ap.parse_args()
    BATCH = a.batch
    if a.cmd == "run":
        run()
    else:
        {"stage": stage, "link": link}[a.cmd](a.photo)
