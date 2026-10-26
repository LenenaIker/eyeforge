"""NVIDIA Fixer (Difix3D+'s successor, Cosmos-Predict2 0.6B, NVIDIA Open Model License) on a WorldStereo splat
(ws_splat.py): does it clean the splat's smears and ghosts without inventing? Test F0 of
docs/pruebas_reconstruccion_worldstereo.md: the fix only as a post-process of the renders, the splat untouched.

Fixer works at 1024x576. The splat is rendered straight at that size with the frames' scale (focal 1.2 x 240) and a
slightly wider field of view, so nothing is stretched or padded with made-up pixels; afterwards the raw and the
fixed render go through the same resampling back to the 832x480 frame and are scored like ws_splat.py score
(phase 1's held-out frames, non-sky pixels). Novel views are ws_splat.py train's: raised 1.4 m and 2.5 m beside the
walks, never seen by any frame. Fidelity = LPIPS between the raw and the fixed render: high means Fixer changed
what the splat shows, not only cleaned it.

    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py renders --case photo21 --arm G2
    python experiments/worldstereo/ws_fixer.py fix --case photo21 --timestep 250       # Docker, fixer-cosmos-env
    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py score --case photo21

Test F1, Difix3D+'s distillation in one round: the splat is rendered at pseudo views that step from the nearest
training frame towards a student camera (ws_nbv.py audit, the ones with at most MAX_HOLES unknown) at 1/3, 2/3 and
the whole way, Fixer cleans them, and ws_splat.py train --arm F1 fits G2's recipe again with them added (code
PSEUDO, weight PSEUDO_WEIGHT, no depth; pixels the splat left empty teach nothing).

    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py pseudo --case photo21 --arm G2 --tag F1
    python experiments/worldstereo/ws_fixer.py fix --case photo21 --tag F1 --timestep 150
    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py views --case photo21 --tag F1 --timestep 150
    <gsplat env>/bin/python experiments/worldstereo/ws_splat.py train --case photo21 --arm F1
    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py compare --case photo21 --arms G2 F1

Test F3 (2026-10-25), masked and progressive: the student cameras 2 to 4.5 m from any walked frame with the most
targets (ws_conflict.py grid: no data and conflict pixels), spread out; each round steps a third further from the
nearest training frame, renders the latest splat, and only the target pixels of the fixed render teach (the rest is
NONE, so Fixer cannot soften what the frames already back). Rounds accumulate their pseudo views.

    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py pseudo_cf --case u2_photo27 --arm G2 --tag F3a --step 1
    python experiments/worldstereo/ws_fixer.py fix --case u2_photo27 --tag F3a --timestep 200
    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py views --case u2_photo27 --tag F3a --timestep 200
    <gsplat env>/bin/python experiments/worldstereo/ws_splat.py train --case u2_photo27 --arm F3a
    then --arm F3a --tag F3b --step 2 --base views_f3a.npz, and F3c (step 3, base views_f3b.npz)

Writes data/worldstereo_test/<case>/splat/arms/<tag>/: in/*.png (raw renders), meta.npz, t<step>/*.png (fixed),
score.json, sheet.jpg; views makes splat/views_<tag>.npz, compare splat/arms/sheet_<arms>.jpg. Fixer's code and
weights live in data/fixer/ (src/, models/)."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

FW, FH = 1024, 576          # Fixer's working size
SCALE = FH / ws.H           # 1.2: the frames' pixels, bigger
FIXER = ws.ROOT / "data" / "fixer"
IMAGE = "fixer-cosmos-env"
TIMESTEPS = (150, 250, 400)  # upstream's examples use 250, its script defaults to 400
HOLDOUT = 7                 # ws_splat.py's
MAX_HOLES = 0.25            # a student camera with more unknown pixels is not a pseudo view: Fixer cleans, not fills
PSEUDO_TARGETS = 40         # student cameras, spread out (farthest point on position and heading)
PSEUDO_STEPS = (1 / 3, 2 / 3, 1.0)
MIN_ALPHA = 0.5             # a pseudo-view pixel the splat covers less than this teaches nothing
F3_WALKED_M = (2.0, 4.5)    # where Fixer can still help: nearer is already backed, farther needs new content
F3_SPOT_SIDE_M = (0.0, -0.5, 0.5)  # with --spots: each camera also half a metre to either side
F3_TARGET_DILATE = 4        # frame pixels around a target that teach too (Fixer's edits do not stop at the mask)


def big_intrinsic() -> np.ndarray:
    f = ws.F * SCALE
    return np.array([[f, 0, FW / 2], [0, f, FH / 2], [0, 0, 1]], np.float32)


def to_frame(im: np.ndarray) -> np.ndarray:
    """A 1024x576 render back to the 832x480 frame: the same rays, exact scale and offset."""
    import cv2
    ox = (FW - ws.W * SCALE) / 2
    m = np.array([[1 / SCALE, 0, -ox / SCALE + 0.5 / SCALE - 0.5], [0, 1 / SCALE, 0.5 / SCALE - 0.5]], np.float32)
    return cv2.warpAffine(im, m, (ws.W, ws.H), flags=cv2.INTER_LINEAR)


def novel_poses() -> tuple[list[str], np.ndarray]:
    """ws_splat.py train's novel views."""
    labels, mats = [], []
    for yaw in range(0, 360, 45):
        for pos, pitch in (([0, 0, 1.4], -25.0), ([2.5 * math.cos(math.radians(yaw + 20)),
                                                  2.5 * math.sin(math.radians(yaw + 20)), 0], -5.0)):
            labels.append(f"yaw {yaw} {'up 1.4 m' if pos[2] else '2.5 m off, beside'}")
            mats.append(ws.w2c(pos, yaw, pitch).astype(np.float32))
    return labels, np.stack(mats)


def renders(case: str, arm: str, tag: str):
    """The arm's splat at the held-out frames and the novel views, 1024x576, into <tag>/in/."""
    import torch
    from ws_nbv import load_splat, render
    dev = torch.device("cuda")
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    code, w2c, name = z["code"], z["w2c"], z["name"]
    frame = np.array([int(str(s).rsplit("/", 1)[1]) for s in name])
    held = [i for i in range(len(name)) if not str(name[i]).startswith("pano/") and frame[i] % HOLDOUT == HOLDOUT // 2]
    labels, novel = novel_poses()
    out = ws.DATA / case / "splat" / "arms" / tag
    (out / "in").mkdir(parents=True, exist_ok=True)
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", dev)
    rgb, _, alpha = render(sp, np.concatenate([w2c[held], novel]), big_intrinsic(), FW, FH)
    files = [f"held_{k:02d}.png" for k in range(len(held))] + [f"novel_{k:02d}.png" for k in range(len(novel))]
    for f, im in zip(files, rgb.cpu().numpy()):
        Image.fromarray(im).save(out / "in" / f)
    np.savez_compressed(out / "meta.npz", files=np.array(files), held=np.array(held), held_name=name[held],
                        novel_label=np.array(labels), alpha=np.stack([to_frame(a) for a in alpha.cpu().numpy()]),
                        arm=arm)
    print(f"{len(files)} renders of {arm} -> {out / 'in'}")


def look(m: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Position, yaw and pitch (degrees) of a world-to-camera from ws.w2c (no roll)."""
    c2w = np.linalg.inv(m)
    fwd = c2w[:3, 2]
    return c2w[:3, 3], math.degrees(math.atan2(fwd[1], fwd[0])), math.degrees(math.asin(np.clip(fwd[2], -1, 1)))


def pseudo(case: str, arm: str, tag: str):
    """Pseudo views between the training frames and the student cameras, the arm's splat rendered there."""
    import torch
    from ws_nbv import load_splat, render
    au = np.load(ws.DATA / case / "nbv" / "audit.npz")
    ok = np.nonzero(au["shares"] <= MAX_HOLES)[0]
    tgt = [look(au["poses"][i]) for i in ok]
    feat = np.array([[*p[:2], 2 * math.cos(math.radians(y)), 2 * math.sin(math.radians(y))] for p, y, _ in tgt])
    pick = [0]  # farthest point: 1 m of position weighs like ~30 degrees of heading
    for _ in range(min(PSEUDO_TARGETS, len(tgt)) - 1):
        d = np.min(np.linalg.norm(feat[:, None] - feat[pick][None], axis=-1), 1)
        pick.append(int(d.argmax()))
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    name = z["name"]
    frame = np.array([int(str(s).rsplit("/", 1)[1]) for s in name])
    train = [i for i in range(len(name)) if not str(name[i]).startswith("pano/")
             and frame[i] % HOLDOUT != HOLDOUT // 2]
    src = [look(z["w2c"][i]) for i in train]
    sfeat = np.array([[*p[:2], 2 * math.cos(math.radians(y)), 2 * math.sin(math.radians(y))] for p, y, _ in src])
    mats, labels = [], []
    for j in pick:
        p1, y1, t1 = tgt[j]
        s = int(np.linalg.norm(sfeat - feat[j], axis=1).argmin())
        p0, y0, t0 = src[s]
        dy = (y1 - y0 + 180) % 360 - 180
        for f in PSEUDO_STEPS:
            mats.append(ws.w2c(p0 + f * (p1 - p0), y0 + f * dy, t0 + f * (t1 - t0)).astype(np.float32))
            labels.append(f"{name[train[s]]} -> student {ok[j]} at {f:.2f}")
    out = ws.DATA / case / "splat" / "arms" / tag
    (out / "in").mkdir(parents=True, exist_ok=True)
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", torch.device("cuda"))
    rgb, _, alpha = render(sp, np.stack(mats), big_intrinsic(), FW, FH)
    files = [f"pseudo_{k:03d}.png" for k in range(len(mats))]
    for f, im in zip(files, rgb.cpu().numpy()):
        Image.fromarray(im).save(out / "in" / f)
    np.savez_compressed(out / "meta.npz", files=np.array(files), w2c=np.stack(mats), label=np.array(labels),
                        alpha=np.stack([to_frame(a) for a in alpha.cpu().numpy()]), arm=arm)
    print(f"{len(files)} pseudo views of {arm} ({len(pick)} student cameras) -> {out / 'in'}")


def pseudo_cf(case: str, arm: str, tag: str, step: int, spots: str = "", walked=F3_WALKED_M, n: int = PSEUDO_TARGETS):
    """F3's pseudo views: student cameras from ws_conflict.py's grid of G2, stepped step/3 of the way from their
    nearest training frame, rendered from the arm's splat, with the target pixels of the arm's own evidence. With
    spots, every heading at those grid spots instead (each also F3_SPOT_SIDE_M to either side)."""
    import torch
    import ws_conflict as wc
    import ws_nbv
    from scipy import ndimage
    g = np.load(ws.DATA / case / "conflict" / "G2" / "grid.npz")
    score = g["nodata"] + g["conflict_px"]
    ok = np.nonzero((g["walked"] >= walked[0]) & (g["walked"] < walked[1]))[0]
    ok = ok[np.argsort(-score[ok])][:4 * n]  # the worst, then spread out
    feat = np.array([[g["x"][i], g["y"][i], 2 * math.cos(math.radians(g["heading"][i])),
                      2 * math.sin(math.radians(g["heading"][i]))] for i in ok])
    pick = [0]
    for _ in range(min(n, len(ok)) - 1):
        d = np.min(np.linalg.norm(feat[:, None] - feat[pick][None], axis=-1), 1)
        pick.append(int(d.argmax()))
    side = (0.0,)
    if spots:
        want = [tuple(float(v) for v in t.split(",")) for t in spots.split(";")]
        ok = np.array([i for i in range(len(g["x"])) if (g["x"][i], g["y"][i]) in want])
        feat = np.array([[g["x"][i], g["y"][i], 2 * math.cos(math.radians(g["heading"][i])),
                          2 * math.sin(math.radians(g["heading"][i]))] for i in ok])
        pick, side = list(range(len(ok))), F3_SPOT_SIDE_M
    sp = ws_nbv.load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", torch.device("cuda"))
    def student(j, s):
        h = math.radians(g["heading"][ok[j]])
        return look(ws.w2c([g["x"][ok[j]] - s * math.sin(h), g["y"][ok[j]] + s * math.cos(h), 0.0],
                           g["heading"][ok[j]], ws_nbv.STUDENT_PITCH_DEG))
    pairs = [(j, s) for j in pick for s in side]
    tgt = [student(j, s) for j, s in pairs]
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    name = z["name"]
    frame = np.array([int(str(s).rsplit("/", 1)[1]) for s in name])
    train = [i for i in range(len(name)) if not str(name[i]).startswith("pano/")
             and frame[i] % HOLDOUT != HOLDOUT // 2]
    src = [look(z["w2c"][i]) for i in train]
    sfeat = np.array([[*p[:2], 2 * math.cos(math.radians(y)), 2 * math.sin(math.radians(y))] for p, y, _ in src])
    mats, labels = [], []
    f = step / 3
    for (j, sd), (p1, y1, t1) in zip(pairs, tgt):
        s_ = int(np.linalg.norm(sfeat - feat[j], axis=1).argmin())
        p0, y0, t0 = src[s_]
        dy = (y1 - y0 + 180) % 360 - 180
        mats.append(ws.w2c(p0 + f * (p1 - p0), y0 + f * dy, t0 + f * (t1 - t0)).astype(np.float32))
        labels.append(f"{name[train[s_]]} -> student ({g['x'][ok[j]]:.0f},{g['y'][ok[j]]:.0f}) "
                      f"heading {g['heading'][ok[j]]:.0f}{f' {sd:+.1f} m aside' if sd else ''} at {f:.2f}")
    mats = np.stack(mats)
    out = ws.DATA / case / "splat" / "arms" / tag
    (out / "in").mkdir(parents=True, exist_ok=True)
    rgb, _, alpha = ws_nbv.render(sp, mats, big_intrinsic(), FW, FH)
    ev = wc.Evidence(case, sp)
    target = []
    for m in mats:
        e = wc.classify(ev, sp, m, big_intrinsic(), FW, FH)
        t = to_frame(np.isin(e["code"], (wc.NODATA, wc.CONFLICT)).astype(np.uint8) * 255) > 127
        target.append(ndimage.binary_dilation(t, iterations=F3_TARGET_DILATE))
    files = [f"pseudo_{k:03d}.png" for k in range(len(mats))]
    for fn, im in zip(files, rgb.cpu().numpy()):
        Image.fromarray(im).save(out / "in" / fn)
    target = np.stack(target)
    np.savez_compressed(out / "meta.npz", files=np.array(files), w2c=mats, label=np.array(labels),
                        alpha=np.stack([to_frame(a) for a in alpha.cpu().numpy()]), target=target, arm=arm)
    print(f"{len(files)} pseudo views of {arm} at {f:.2f} of the way, targets {target.mean():.2f} of their pixels "
          f"-> {out / 'in'}")


def views(case: str, tag: str, timestep: int, base: str = "views.npz"):
    """splat/views_<tag>.npz: base (views.npz, or an earlier round's) plus the fixed pseudo views (fixer_<tag>/NNN).
    With F3's target masks only those pixels teach."""
    from ws_splat import PSEUDO, NONE
    out = ws.DATA / case / "splat" / "arms" / tag
    meta = np.load(out / "meta.npz")
    z = np.load(ws.DATA / case / "splat" / base)
    ims = np.stack([to_frame(np.asarray(Image.open(out / f"t{timestep}" / f).convert("RGB"))) for f in meta["files"]])
    teach = meta["alpha"] >= MIN_ALPHA
    if "target" in meta.files:
        teach &= meta["target"]
    code = np.where(teach, PSEUDO, NONE).astype(np.uint8)
    n = len(ims)
    np.savez_compressed(ws.DATA / case / "splat" / f"views_{tag.lower()}.npz",
                        img=np.concatenate([z["img"], ims]), code=np.concatenate([z["code"], code]),
                        depth=np.concatenate([z["depth"], np.zeros((n, ws.H, ws.W), np.float16)]),
                        w2c=np.concatenate([z["w2c"], meta["w2c"]]),
                        name=np.concatenate([z["name"], [f"fixer_{tag}/{k:03d}" for k in range(n)]]))
    print(f"views_{tag.lower()}.npz: {len(z['name'])} views + {n} pseudo, {np.mean(code == PSEUDO):.2f} of their"
          " pixels teach")


def compare(case: str, names: list[str]):
    """Arms side by side on ws_splat.py train's novel and held-out views, with their uniform scores."""
    from PIL import ImageFont
    base = ws.DATA / case / "splat" / "arms"
    nov = {a: np.load(base / a / "novel.npz") for a in names}
    sc = {a: json.loads((base / a / "score.json").read_text()) for a in names if (base / a / "score.json").exists()}
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    cw, ch = 560, 323
    first = nov[names[0]]
    rows = [(f"novel: {lab}", [(a, nov[a]["img"][k]) for a in names]) for k, lab in enumerate(first["label"])]
    for k in (0, len(first["held_name"]) // 2, len(first["held_name"]) - 1):
        rows.append((f"held out: {first['held_name'][k]}",
                     [("WorldStereo", first["held_gt"][k])] + [(a, nov[a]["held"][k]) for a in names]))
    cols = max(len(r[1]) for r in rows)
    head = 40 + 30 * len(names)
    sh = Image.new("RGB", (cw * cols, head + len(rows) * (ch + 28)), (0, 0, 0))
    d = ImageDraw.Draw(sh)
    d.text((8, 8), f"{case}: {' vs '.join(names)} (uniform score on phase 1's held-out frames)", fill=(255, 255, 255),
           font=font)
    for j, a in enumerate(names):
        f = sc.get(a, {})
        d.text((8, 40 + 30 * j), f"{a}: " + "  ".join(f"{k} {f[k]:.3f}" for k in ("seen", "hole", "ssim", "lpips")
                                                     if k in f), fill=(255, 255, 0), font=font)
    for r, (label, tiles) in enumerate(rows):
        y = head + r * (ch + 28)
        d.text((8, y + 2), label, fill=(255, 255, 0), font=font)
        for k, (tl, im) in enumerate(tiles):
            tile = Image.fromarray(im).resize((cw, ch))
            ImageDraw.Draw(tile).text((6, 4), tl, fill=(0, 255, 255), font=small)
            sh.paste(tile, (k * cw, y + 28))
    path = base / f"sheet_{'_'.join(names)}.jpg"
    sh.save(path, quality=88)
    print(path)


def fix(case: str, tag: str, timestep: int):
    """Fixer on <tag>/in/ -> <tag>/t<timestep>/, upstream's inference script in its container."""
    out = ws.DATA / case / "splat" / "arms" / tag
    (out / f"t{timestep}").mkdir(exist_ok=True)
    (FIXER / "src" / "models").mkdir(exist_ok=True)
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "HOME=/tmp", "-v", f"{(FIXER / 'src').resolve()}:/work",
           "-v", f"{(FIXER / 'models').resolve()}:/work/models", "-v", f"{out.resolve()}:/data", IMAGE,
           "-c", f"python /work/src/inference_pretrained_model.py --model /work/models/pretrained/pretrained_fixer.pkl"
                 f" --input /data/in --output /data/t{timestep} --timestep {timestep} --resolution 1024"]
    subprocess.run(cmd, check=True)


def score(case: str, tag: str):
    """Raw against fixed: held-out PSNR (seen, invented), SSIM, LPIPS on non-sky pixels; novel-view sharpness;
    fidelity (LPIPS raw vs fixed). score.json and sheet.jpg."""
    import lpips
    import torch
    sys.path.insert(0, str(ws.ROOT / "experiments" / "sphere_walk"))
    from gs_train import ssim_map
    from ws_splat import HOLE, SEEN, SKY, sharpness
    dev = torch.device("cuda")
    out = ws.DATA / case / "splat" / "arms" / tag
    meta = np.load(out / "meta.npz")
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    img, code = z["img"], z["code"]
    files, held, alpha = list(meta["files"]), meta["held"], meta["alpha"]
    runs = ["in"] + [f"t{t}" for t in TIMESTEPS if (out / f"t{t}").is_dir() and any((out / f"t{t}").iterdir())]
    ims = {r: [to_frame(np.asarray(Image.open(out / r / f).convert("RGB"))) for f in files] for r in runs}
    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    t = lambda a: torch.as_tensor(a, device=dev).float().div(255).permute(2, 0, 1)  # noqa: E731
    res = {}
    with torch.no_grad():
        for r in runs:
            s = {"seen": [], "hole": [], "ssim": [], "lpips": [], "sharp": [], "fidelity": []}
            for k, i in enumerate(held):
                pr, g = ims[r][k] / 255, img[i] / 255
                for key, cval in (("seen", SEEN), ("hole", HOLE)):
                    m = code[i] == cval
                    if m.sum() > 1000:
                        s[key].append(float(-10 * np.log10(((pr - g) ** 2)[m].mean())))
                ground = torch.as_tensor(code[i] != SKY, device=dev).float()
                s["ssim"].append(float((ssim_map(t(ims[r][k]), t(img[i])).mean(0) * ground).sum() / ground.sum()))
                d = lp(t(ims[r][k])[None] * ground, t(img[i])[None] * ground, normalize=True)[0, 0]
                s["lpips"].append(float((d * ground).sum() / ground.sum()))
            for k in range(len(held), len(files)):
                a = alpha[k]
                s["sharp"].append(sharpness(ims[r][k], a))
                cov = torch.as_tensor(a > 0.5, device=dev).float()
                d = lp(t(ims[r][k])[None] * cov, t(ims["in"][k])[None] * cov, normalize=True)[0, 0]
                s["fidelity"].append(float((d * cov).sum() / cov.sum()))
            res[r] = {k: round(float(np.nanmean(v)), 4) for k, v in s.items()}
            print(r, res[r], flush=True)
    (out / "score.json").write_text(json.dumps({"arm": str(meta["arm"]), "runs": res}, indent=1))
    sheet(out, case, runs, ims, img, meta, res)


def sheet(out: Path, case: str, runs, ims, img, meta, res):
    from PIL import ImageFont
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    cw, ch = 560, 323
    labels, held = list(meta["novel_label"]), meta["held"]
    names = {"in": f"{meta['arm']} raw"} | {r: f"Fixer {r}" for r in runs[1:]}
    rows = [(f"novel: {lab}", [(names[r], ims[r][len(held) + k]) for r in runs])
            for k, lab in enumerate(labels) if "beside" in lab]
    rows += [(f"novel: {lab}", [(names[r], ims[r][len(held) + k]) for r in runs])
             for k, lab in enumerate(labels) if "up" in lab][::2]
    for k in (0, len(held) // 2, len(held) - 1):
        rows.append((f"held out: {meta['held_name'][k]}",
                     [("WorldStereo", img[held[k]])] + [(names[r], ims[r][k]) for r in runs]))
    cols = max(len(r[1]) for r in rows)
    head = 40 + 30 * len(runs)
    sh = Image.new("RGB", (cw * cols, head + len(rows) * (ch + 28)), (0, 0, 0))
    d = ImageDraw.Draw(sh)
    d.text((8, 8), f"{case}: Fixer on the {meta['arm']} splat's renders (lower LPIPS, higher SSIM/PSNR = closer to"
                   f" WorldStereo; fidelity = LPIPS raw vs fixed, lower = fewer changes)", fill=(255, 255, 255), font=font)
    for j, r in enumerate(runs):
        f = res[r]
        d.text((8, 40 + 30 * j), f"{names[r]}: PSNR seen {f['seen']:.2f} invented {f['hole']:.2f}  SSIM {f['ssim']:.3f}"
               f"  LPIPS {f['lpips']:.3f}  sharpness {f['sharp']:.0f}  fidelity {f['fidelity']:.3f}",
               fill=(255, 255, 0), font=font)
    for r, (label, tiles) in enumerate(rows):
        y = head + r * (ch + 28)
        d.text((8, y + 2), label, fill=(255, 255, 0), font=font)
        for k, (tl, im) in enumerate(tiles):
            tile = Image.fromarray(im).resize((cw, ch))
            ImageDraw.Draw(tile).text((6, 4), tl, fill=(0, 255, 255), font=small)
            sh.paste(tile, (k * cw, y + 28))
    sh.save(out / "sheet.jpg", quality=88)
    print(out / "sheet.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["renders", "fix", "score", "pseudo", "pseudo_cf", "views", "compare"])
    ap.add_argument("--case", default="photo21")
    ap.add_argument("--arm", default="G2", help="renders, pseudo: the splat to render")
    ap.add_argument("--arms", nargs="+", default=["G2", "F1"], help="compare: the arms side by side")
    ap.add_argument("--tag", default="F0")
    ap.add_argument("--timestep", type=int, default=250)
    ap.add_argument("--step", type=int, default=1, help="pseudo_cf: thirds of the way to the student camera")
    ap.add_argument("--spots", default="", help='pseudo_cf: "x,y;x,y", every heading at these grid spots')
    ap.add_argument("--walked", default=",".join(map(str, F3_WALKED_M)), help="pseudo_cf: band of distance to walked, m")
    ap.add_argument("--n", type=int, default=PSEUDO_TARGETS, help="pseudo_cf: student cameras per round")
    ap.add_argument("--base", default="views.npz", help="views: the views file the pseudo views are added to")
    a = ap.parse_args()
    {"renders": lambda: renders(a.case, a.arm, a.tag), "fix": lambda: fix(a.case, a.tag, a.timestep),
     "score": lambda: score(a.case, a.tag), "pseudo": lambda: pseudo(a.case, a.arm, a.tag),
     "pseudo_cf": lambda: pseudo_cf(a.case, a.arm, a.tag, a.step, a.spots,
                                                 tuple(float(v) for v in a.walked.split(",")), a.n),
     "views": lambda: views(a.case, a.tag, a.timestep, a.base), "compare": lambda: compare(a.case, a.arms)}[a.cmd]()
