"""GenFusion's repair model (CVPR 2025, MIT code, DynamiCrafter fine-tune) on a WorldStereo splat: test GF1, F1
(ws_fixer.py) with the corrector swapped. Fixer cleans one view at a time and smoothed the texture (F0, F1, FX);
GenFusion repairs a 16-frame video of the splat's renders, colour and disparity together, so the frames of a clip
must agree with each other and with the splat's depth.

Each clip walks from the nearest training frame to a student camera (ws_nbv.py audit, the ones with at most
MAX_HOLES unknown) in 16 steps, so it starts on a view the splat knows well. The splat is rendered at GenFusion's
960x512 with the frames' scale (focal 512/480 x 240) and a slightly wider field of view; the reference frame for its
image embedding is that training frame. Frames KEEP (1/3, 2/3 and the whole way, as F1) of each repaired clip go
back to the 832x480 frame and are added like F1's: code PSEUDO, weight PSEUDO_WEIGHT, no depth.

Only GenFusion's Generation/ (lvdm, MIT) is imported; its Reconstruction/ is the Inria 2DGS code (non-commercial)
and is not used. The repair call follows upstream's Extrapolator.repair and OutpaintBase's pre-processing.

    <gsplat env>/bin/python experiments/worldstereo/ws_genfusion.py pseudo --case e2e_photo27 --arm G2
    <genfusion env>/bin/python experiments/worldstereo/ws_genfusion.py fix --case e2e_photo27
    <gsplat env>/bin/python experiments/worldstereo/ws_genfusion.py views --case e2e_photo27
    <gsplat env>/bin/python experiments/worldstereo/ws_splat.py train --case e2e_photo27 --arm GF1
    <gsplat env>/bin/python experiments/worldstereo/ws_fixer.py compare --case e2e_photo27 --arms G2 GF1

Writes data/worldstereo_test/<case>/splat/arms/GF1/: clips.npz (renders), meta.npz, fixed/clip_NN.png strips,
sheet_clips.jpg; views makes splat/views_gf1.npz."""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402
from ws_fixer import MAX_HOLES, MIN_ALPHA, PSEUDO_TARGETS, look  # noqa: E402

GW, GH = 960, 512                # GenFusion's working size
SCALE = GH / ws.H                # 1.067: the frames' pixels, bigger
FRAMES = 16                      # the model's clip length
KEEP = (5, 10, 15)               # frames of each clip that become pseudo views
GUIDANCE = 3.2                   # upstream's masked-reconstruction setting
GENFUSION = Path("/media/lenena-iker/Crucial/tools/genfusion")
CKPT = GENFUSION / "ckpt" / "epoch=59-step=34000.ckpt"
CONFIG = GENFUSION / "src" / "Reconstruction" / "generation_infer.yaml"
HOLDOUT = 7                      # ws_splat.py's


def big_intrinsic() -> np.ndarray:
    f = ws.F * SCALE
    return np.array([[f, 0, GW / 2], [0, f, GH / 2], [0, 0, 1]], np.float32)


def to_frame(im: np.ndarray) -> np.ndarray:
    """A 960x512 render back to the 832x480 frame: the same rays, exact scale and offset."""
    import cv2
    ox = (GW - ws.W * SCALE) / 2
    m = np.array([[1 / SCALE, 0, -ox / SCALE + 0.5 / SCALE - 0.5], [0, 1 / SCALE, 0.5 / SCALE - 0.5]], np.float32)
    return cv2.warpAffine(im, m, (ws.W, ws.H), flags=cv2.INTER_LINEAR)


def out_dir(case: str, tag: str) -> Path:
    return ws.DATA / case / "splat" / "arms" / tag


def pseudo(case: str, arm: str, tag: str):
    """One 16-frame clip per picked student camera, from the nearest training frame, rendered by the arm's splat."""
    import torch
    from ws_nbv import load_splat, render
    au = np.load(ws.DATA / case / "nbv" / "audit.npz")
    ok = np.nonzero(au["shares"] <= MAX_HOLES)[0]
    tgt = [look(au["poses"][i]) for i in ok]
    feat = np.array([[*p[:2], 2 * math.cos(math.radians(y)), 2 * math.sin(math.radians(y))] for p, y, _ in tgt])
    pick = [0]  # farthest point: 1 m of position weighs like ~30 degrees of heading (ws_fixer.py's)
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
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", torch.device("cuda"))
    rgbs, deps, alphas, mats, refs, labels = [], [], [], [], [], []
    for j in pick:
        p1, y1, t1 = tgt[j]
        s = int(np.linalg.norm(sfeat - feat[j], axis=1).argmin())
        p0, y0, t0 = src[s]
        dy = (y1 - y0 + 180) % 360 - 180
        clip = np.stack([ws.w2c(p0 + f * (p1 - p0), y0 + f * dy, t0 + f * (t1 - t0)).astype(np.float32)
                         for f in np.linspace(0, 1, FRAMES)])
        rgb, dep, alpha = render(sp, clip, big_intrinsic(), GW, GH)
        rgbs.append(rgb.cpu().numpy())
        deps.append(dep.cpu().numpy().astype(np.float16))
        alphas.append(alpha.cpu().numpy().astype(np.float16))
        mats.append(clip)
        refs.append(z["img"][train[s]])
        labels.append(f"{name[train[s]]} -> student {ok[j]}")
    out = out_dir(case, tag)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "clips.npz", rgb=np.stack(rgbs), depth=np.stack(deps), alpha=np.stack(alphas),
             w2c=np.stack(mats), ref=np.stack(refs), label=np.array(labels), arm=arm)
    print(f"{len(pick)} clips of {FRAMES} frames from {arm} -> {out / 'clips.npz'}")


def load_model():
    """Upstream's Extrapolator.create_model, without the OpenCLIP download (the checkpoint holds those weights)."""
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(GENFUSION / "src" / "Generation"))
    from utils.utils import instantiate_from_config
    cfg = OmegaConf.load(CONFIG).pop("model")
    cfg["params"]["unet_config"]["params"]["use_checkpoint"] = False
    cfg["params"]["cond_stage_config"]["params"]["version"] = None
    cfg["params"]["img_cond_stage_config"]["params"]["version"] = None
    model = instantiate_from_config(cfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd), strict=True)
    del sd
    model.eval()
    model.perframe_ae = True
    return model.half().cuda()


def to_video(x, size, mode):
    """[T,H,W,C] array -> [C,T,h,w] tensor at GenFusion's size."""
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(x).float().permute(3, 0, 1, 2)[None]
    kw = {"align_corners": False} if mode == "trilinear" else {}
    return F.interpolate(t, size=(t.shape[2], *size), mode=mode, **kw)[0]


def disparity(depth: np.ndarray):
    """OutpaintBase.depth_preprocess: disparity over the clip normalised to [-1, 1], empty pixels farthest."""
    import torch
    d = torch.as_tensor(depth.astype(np.float32))
    ok = d > 0
    disp = torch.zeros_like(d)
    disp[ok] = 1.0 / (d[ok] + 1e-10)
    if ok.any():
        lo, hi = disp[ok].min(), disp[ok].max()
        disp[ok] = (disp[ok] - lo) / (hi - lo + 1e-10)
    return to_video((disp * 2 - 1)[..., None].numpy(), (GH, GW), "nearest")


def fix(case: str, tag: str, limit: int | None = None):
    """Repair every clip (Extrapolator.repair: 25 DDIM steps, eta 1, guidance rescale 0.7, fs 30)."""
    import torch
    import torch.utils.checkpoint  # noqa: F401 (lvdm.common reads it without importing it)
    from einops import rearrange
    sys.path.insert(0, str(GENFUSION / "src" / "Generation"))
    from lvdm.models.samplers.ddim import DDIMSampler
    model = load_model()
    out = out_dir(case, tag)
    z = np.load(out / "clips.npz")
    (out / "fixed").mkdir(exist_ok=True)
    fixed = []
    for c in range(min(limit or len(z["rgb"]), len(z["rgb"]))):
        t0 = time.time()
        rgb = to_video(z["rgb"][c] / 255.0 * 2 - 1, (GH, GW), "trilinear")
        dep = disparity(z["depth"][c])
        ref = to_video(z["ref"][c][None] / 255.0 * 2 - 1, (GH, GW), "trilinear")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            rgb, dep, ref = (v[None].cuda().half() for v in (rgb, dep, ref))
            refs = rearrange(ref, "b c l h w -> (b l) c h w")
            emb = model.image_proj_model(model.embedder(refs))
            emb = rearrange(emb, "(b t) l c -> b (t l) c", b=1)
            cat = model.encode_first_stage(torch.cat([rgb, dep], 1)).half()
            cond = {"c_crossattn": [emb], "c_concat": [cat]}
            uemb = model.image_proj_model(model.embedder(torch.zeros_like(refs)))
            uc = {"c_crossattn": [rearrange(uemb, "(b t) l c -> b (t l) c", b=1)], "c_concat": [cat]}
            fs = torch.tensor([30], dtype=torch.long, device="cuda")
            samples, _ = DDIMSampler(model).sample(
                S=25, conditioning=cond, batch_size=1, shape=[4, FRAMES, GH // 8, GW // 8], verbose=False,
                unconditional_guidance_scale=GUIDANCE, unconditional_conditioning=uc, eta=1.0, cfg_img=None,
                mask=None, x0=None, x_T=None, fs=fs, timestep_spacing="uniform", precision=16,
                guidance_rescale=0.7, unconditional_conditioning_img_nonetext=None)
            res = model.decode_first_stage(samples, return_depth=True)[0]  # [4, T, H, W]
        im = ((res[:3].clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        fixed.append(im)
        Image.fromarray(np.concatenate([np.concatenate([z["rgb"][c][k], im[k]], 0) for k in (0, *KEEP)], 1)) \
            .resize((GW * 2, GH)).save(out / "fixed" / f"clip_{c:02d}.jpg", quality=85)
        print(f"clip {c}: {time.time() - t0:.0f} s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    np.savez(out / "fixed.npz", rgb=np.stack(fixed))


def views(case: str, tag: str):
    """splat/views_<tag>.npz: views.npz plus frames KEEP of every repaired clip (named genfusion_<tag>/NNN)."""
    from ws_splat import NONE, PSEUDO
    out = out_dir(case, tag)
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    c = np.load(out / "clips.npz")
    fx = np.load(out / "fixed.npz")["rgb"]
    ims = np.stack([to_frame(fx[i, k]) for i in range(len(fx)) for k in KEEP])
    alpha = np.stack([to_frame(c["alpha"][i, k].astype(np.float32)) for i in range(len(fx)) for k in KEEP])
    w2c = np.stack([c["w2c"][i, k] for i in range(len(fx)) for k in KEEP])
    code = np.where(alpha >= MIN_ALPHA, PSEUDO, NONE).astype(np.uint8)
    n = len(ims)
    np.savez_compressed(ws.DATA / case / "splat" / f"views_{tag.lower()}.npz",
                        img=np.concatenate([z["img"], ims]), code=np.concatenate([z["code"], code]),
                        depth=np.concatenate([z["depth"], np.zeros((n, ws.H, ws.W), np.float16)]),
                        w2c=np.concatenate([z["w2c"], w2c]),
                        name=np.concatenate([z["name"], [f"genfusion_{tag}/{k:03d}" for k in range(n)]]))
    print(f"views_{tag.lower()}.npz: {len(z['name'])} views + {n} pseudo, {np.mean(code == PSEUDO):.2f} of their"
          " pixels teach")
    sheet(case, tag, c, fx)


def sheet(case: str, tag: str, c, fx):
    """A few clips: the splat's render over GenFusion's repair at the start and the KEEP frames."""
    from PIL import ImageFont
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    cw, ch = 480, 256
    clips = np.linspace(0, len(fx) - 1, min(8, len(fx))).astype(int)
    cols = (0, *KEEP)
    sh = Image.new("RGB", (cw * len(cols), 36 + len(clips) * 2 * ch), (0, 0, 0))
    ImageDraw.Draw(sh).text((8, 6), f"{case}: {c['arm']} splat render (upper) vs GenFusion repair (lower), "
                                     f"frames {cols} of 16", fill=(255, 255, 255), font=font)
    for r, i in enumerate(clips):
        for k, f in enumerate(cols):
            for s, (name, im) in enumerate((("splat", c["rgb"][i, f]), ("GenFusion", fx[i, f]))):
                t = Image.fromarray(im).resize((cw, ch))
                ImageDraw.Draw(t).text((6, 4), f"{name}: clip {i} frame {f}", fill=(255, 255, 0), font=small)
                sh.paste(t, (k * cw, 36 + (2 * r + s) * ch))
    sh.save(out_dir(case, tag) / "sheet_clips.jpg", quality=85)
    print(out_dir(case, tag) / "sheet_clips.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pseudo", "fix", "views"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--arm", default="G2", help="pseudo: the splat to render")
    ap.add_argument("--tag", default="GF1")
    ap.add_argument("--limit", type=int, help="fix: only the first N clips (a smoke test)")
    a = ap.parse_args()
    {"pseudo": lambda: pseudo(a.case, a.arm, a.tag), "fix": lambda: fix(a.case, a.tag, a.limit),
     "views": lambda: views(a.case, a.tag)}[a.cmd]()
