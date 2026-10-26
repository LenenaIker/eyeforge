"""NVIDIA Fixer (Difix3D+'s successor, Cosmos-Predict2 0.6B, NVIDIA Open Model License) teaching the splat through
pseudo views, recipe F3 (2026-10-25): masked and progressive.

The student cameras of the base arm's conflict grid within the config's band of distance to walked, the worst (most
no data and conflict pixels) first, spread out; round k steps k/3 of the way from each one's nearest training frame
towards it and renders the latest splat there; Fixer cleans the renders; only the target pixels (no data or
conflict under the latest splat's evidence, grown F3_TARGET_DILATE px) teach, the rest is NONE, so Fixer cannot
soften what the frames already back. Rounds accumulate their pseudo views: F3a on views.npz, F3b on views_f3a.npz...

Fixer works at 1024x576. The splat is rendered straight at that size with the frames' scale (focal 1.2 x 240) and a
slightly wider field of view, so nothing is stretched or padded with made-up pixels; the fixed render goes back to
the 832x480 frame by the exact inverse.

    pseudo  SIM2CHIP_GSPLAT_PYTHON  splat/arms/<tag>/in/*.png, meta.npz
    fix     docker                  splat/arms/<tag>/t<timestep>/*.png (upstream's script in fixer-cosmos-env)
    views   SIM2CHIP_GSPLAT_PYTHON  splat/views_<tag>.npz
Fixer's code and weights live in data/fixer/ (src/, models/)."""
from __future__ import annotations

import os
import subprocess

import numpy as np
from PIL import Image

from . import camera, student
from .camera import F, H, W
from .conflict import CONFLICT, NODATA, Evidence, classify, evidence_views
from .paths import FIXER, Case
from .config import arm_views
from .splat import NONE, PSEUDO, frame_index, is_held

FW, FH = 1024, 576          # Fixer's working size
SCALE = FH / H              # 1.2: the frames' pixels, bigger
IMAGE = "fixer-cosmos-env"
MIN_ALPHA = 0.5             # a pseudo-view pixel the splat covers less than this teaches nothing
F3_TARGET_DILATE = 4        # frame pixels around a target that teach too (Fixer's edits do not stop at the mask)
CANDIDATES_PER_VIEW = 4     # the worst 4n student cameras, then n spread out among them


def big_intrinsic() -> np.ndarray:
    f = F * SCALE
    return np.array([[f, 0, FW / 2], [0, f, FH / 2], [0, 0, 1]], np.float32)


def to_frame(im: np.ndarray) -> np.ndarray:
    """A 1024x576 render back to the 832x480 frame: the same rays, exact scale and offset."""
    import cv2
    ox = (FW - W * SCALE) / 2
    m = np.array([[1 / SCALE, 0, -ox / SCALE + 0.5 / SCALE - 0.5], [0, 1 / SCALE, 0.5 / SCALE - 0.5]], np.float32)
    return cv2.warpAffine(im, m, (W, H), flags=cv2.INTER_LINEAR)


def pseudo(case_name: str, arm: str, tag: str, step: int, rounds: int, walked: tuple, n: int, grid_arm: str = "G2"):
    """Round step's pseudo views (step/rounds of the way), rendered from the arm's splat, with the target pixels of
    the arm's own evidence; the student cameras come from grid_arm's conflict grid."""
    import torch
    from scipy import ndimage
    case = Case(case_name)
    g = np.load(case.conflict(grid_arm) / "grid.npz")
    gx, gy, gh = g["x"], g["y"], g["heading"]
    score = g["nodata"] + g["conflict_px"]
    walked_m = g["walked"]
    ok = np.nonzero((walked_m >= walked[0]) & (walked_m < walked[1]))[0]
    ok = ok[np.argsort(-score[ok])][:CANDIDATES_PER_VIEW * n]  # the worst, then spread out
    feat = camera.look_features([(gx[i], gy[i]) for i in ok], [gh[i] for i in ok])
    pick = camera.farthest(feat, n)
    sp = student.load_splat(case.splat_ply(arm), torch.device("cuda"))
    tgt = [camera.pose_of(camera.w2c([gx[ok[j]], gy[ok[j]], 0.0], gh[ok[j]], student.STUDENT_PITCH_DEG), True)
           for j in pick]
    z = np.load(case.splat / arm_views(grid_arm))
    name, w2c = z["name"], z["w2c"]
    frame = frame_index(name)
    train = [i for i in range(len(name))
             if not str(name[i]).startswith("pano/") and not is_held(str(name[i]), frame[i])]
    src = [camera.pose_of(w2c[i], True) for i in train]
    sfeat = camera.look_features([p[:2] for p, _, _ in src], [y for _, y, _ in src])
    mats, labels = [], []
    f = step / rounds
    for j, (p1, y1, t1) in zip(pick, tgt):
        s_ = int(np.linalg.norm(sfeat - feat[j], axis=1).argmin())
        p0, y0, t0 = src[s_]
        dy = (y1 - y0 + 180) % 360 - 180
        mats.append(camera.w2c(p0 + f * (p1 - p0), y0 + f * dy, t0 + f * (t1 - t0)).astype(np.float32))
        labels.append(f"{name[train[s_]]} -> student ({gx[ok[j]]:.0f},{gy[ok[j]]:.0f}) heading {gh[ok[j]]:.0f} "
                      f"at {f:.2f}")
    mats = np.stack(mats)
    out = case.arm(tag)
    (out / "in").mkdir(parents=True, exist_ok=True)
    rgb, _, alpha = student.render(sp, mats, big_intrinsic(), FW, FH)
    ev = Evidence(case, sp, evidence_views(arm))
    target = []
    for m in mats:
        e = classify(ev, sp, m, big_intrinsic(), FW, FH)
        t = to_frame(np.isin(e["code"], (NODATA, CONFLICT)).astype(np.uint8) * 255) > 127
        target.append(ndimage.binary_dilation(t, iterations=F3_TARGET_DILATE))
    files = [f"pseudo_{k:03d}.png" for k in range(len(mats))]
    for fn, im in zip(files, rgb.cpu().numpy()):
        Image.fromarray(im).save(out / "in" / fn)
    target = np.stack(target)
    np.savez_compressed(out / "meta.npz", files=np.array(files), w2c=mats, label=np.array(labels),
                        alpha=np.stack([to_frame(a) for a in alpha.cpu().numpy()]), target=target, arm=arm)
    print(f"{len(files)} pseudo views of {arm} at {f:.2f} of the way, targets {target.mean():.2f} of their pixels "
          f"-> {out / 'in'}")


def fix(case_name: str, tag: str, timestep: int):
    """Fixer on <tag>/in/ -> <tag>/t<timestep>/, upstream's inference script in its container."""
    out = Case(case_name).arm(tag)
    (out / f"t{timestep}").mkdir(exist_ok=True)
    (FIXER / "src" / "models").mkdir(exist_ok=True)
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "HOME=/tmp", "-v", f"{(FIXER / 'src').resolve()}:/work",
           "-v", f"{(FIXER / 'models').resolve()}:/work/models", "-v", f"{out.resolve()}:/data", IMAGE,
           "-c", f"python /work/src/inference_pretrained_model.py --model /work/models/pretrained/pretrained_fixer.pkl"
                 f" --input /data/in --output /data/t{timestep} --timestep {timestep} --resolution 1024"]
    subprocess.run(cmd, check=True)


def views(case_name: str, tag: str, timestep: int, base: str):
    """splat/views_<tag>.npz: base plus the fixed pseudo views (fixer_<tag>/NNN); only their target pixels teach."""
    case = Case(case_name)
    out = case.arm(tag)
    meta = np.load(out / "meta.npz")
    files, alpha, target, mats = meta["files"], meta["alpha"], meta["target"], meta["w2c"]
    ims = np.stack([to_frame(np.asarray(Image.open(out / f"t{timestep}" / f).convert("RGB"))) for f in files])
    code = np.where((alpha >= MIN_ALPHA) & target, PSEUDO, NONE).astype(np.uint8)
    n = len(ims)
    z = np.load(case.splat / base)
    np.savez_compressed(case.splat / f"views_{tag.lower()}.npz",
                        img=np.concatenate([z["img"], ims]), code=np.concatenate([z["code"], code]),
                        depth=np.concatenate([z["depth"], np.zeros((n, H, W), np.float16)]),
                        w2c=np.concatenate([z["w2c"], mats]),
                        name=np.concatenate([z["name"], [f"fixer_{tag}/{k:03d}" for k in range(n)]]))
    print(f"views_{tag.lower()}.npz: {n} pseudo views added, {np.mean(code == PSEUDO):.2f} of their pixels teach")
