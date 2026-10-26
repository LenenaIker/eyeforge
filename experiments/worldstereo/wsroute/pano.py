"""Step 2: the erased photo outpainted to a 360 panorama by Qwen-Image-Edit-2511 (Apache 2.0) with mickmumpitz's
QWEN-EDIT_360 LoRA (MIT, trained on Poly Haven CC0 HDRIs; licence checked on its HF page 2026-10-26, repo sha
f7626f2) and lightx2v's Lightning 4-step LoRA (Apache 2.0). The recipe is the author's ComfyUI workflow: LoRA at
0.9, 4 steps, cfg 1, a 2048x1024 output, then a second pass (Lightning only) that rolls the wrap to the middle,
whitens a bar over it and asks the model to remove it. It replaced WorldGen's LoRA on FLUX.1-Fill-dev
(non-commercial) on 2026-10-26: docs/plan_sustituir_flux_2026-10-26.md, tests in experiments/panogen/qwen360_test.py.

Two steps, two envs:

    pano        SIM2CHIP_ERASE_PYTHON (the eraser's diffusers, no OpenCV): the photo placed on the 2:1 canvas by
                its EXIF field of view (the horizon, longitude 0 at the centre), grey elsewhere, as the edit's
                input; writes out/<case>/pano_s<seed>.jpg and pano_s<seed>.json
    pano_paste  SIM2CHIP_SCENE_PYTHON (OpenCV, SciPy): Qwen redraws the photo too, at its own field of view (about
                1.5 times smaller than EXIF's), so the photo goes back at PASTE_W where the pano drew it: fx scale,
                yaw, pitch, roll and one radial term fitted to the pano; writes pano_s<seed>_paste.jpg, the route's
                panorama"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .paths import PANOS, PHOTOS, env, tools

GEN_W, GEN_H = 2048, 1024   # the workflow's output size
GREY = 127                  # the canvas around the placed photo
PASTE_W = 4096              # the pasted copy: wide enough to keep the photo's own detail
PHOTO_LONG_EDGE = 2048
LORA = ("mickmumpitz/QWEN-EDIT_360", "251018_MICKMUMPITZ_QWEN-EDIT_360_03.safetensors")
LORA_WEIGHT = 0.9
LIGHTNING = ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
MODEL = "Qwen-Image-Edit-2511 nf4 + mickmumpitz/QWEN-EDIT_360 (0.9) + Lightning 4 steps"
PROMPT = ("Create a 360 panoramic image of {scene}. An HDRI image captured on a 360 degree camera, insta 360, "
          "equirectangular projection. Keep the style the same.")
SEAM_PROMPT = ("Remove the visible seam in the middle of the image. Remove the white bar. Keep the rest of the "
               "image exactly the same.")
SEAM_BAR, SEAM_BLUR = 200, 20  # the workflow's mask size and blur, px at GEN_W
MIN_FIT = 0.8  # correlation under which the fit did not find the photo (the tests' lowest was 0.86): no paste


def exif_fx(im: Image.Image, src: Path) -> float:
    """Focal length in pixels of the oriented full image, from EXIF's 35 mm equivalent (43.27 mm diagonal)."""
    f35 = im.getexif().get_ifd(0x8769).get(0xA405)
    if not f35:
        raise SystemExit(f"{src}: no 35 mm equivalent focal length in EXIF")
    w, h = ImageOps.exif_transpose(im).size
    return f35 / 43.27 * math.hypot(w, h)


def ingest(src: Path, name: str):
    """A phone photo as the route's input: oriented, long edge PHOTO_LONG_EDGE, and its focal length in that
    image's pixels (data/panogen_test/in/<name>.png and .json)."""
    raw = Image.open(src)
    fx = exif_fx(raw, src)
    im = ImageOps.exif_transpose(raw).convert("RGB")
    s = PHOTO_LONG_EDGE / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    PHOTOS.mkdir(parents=True, exist_ok=True)
    im.save(PHOTOS / f"{name}.png")
    f = fx * s
    hfov, vfov = (math.degrees(2 * math.atan(n / 2 / f)) for n in im.size)
    (PHOTOS / f"{name}.json").write_text(json.dumps({"fx": f, "hfov": hfov, "vfov": vfov}))
    print(f"{name}: {im.width}x{im.height} fx {f:.0f} px, fov {hfov:.0f} x {vfov:.0f} deg")


def photo_coords(pw: int, ph: int, fx: float, w: int, h: int, yaw: float = 0.0, pitch: float = 0.0,
                 roll: float = 0.0, k1: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pixel of a w x h equirectangular canvas (longitude 0 at the centre, Matrix-3D's i2p_init), where it
    falls in a pw x ph photo seen with focal fx, turned by yaw, pitch and roll (degrees), with one radial
    distortion term k1 (per squared half-width of the photo): u, v (float32, -1 outside) and the inside mask."""
    lon = np.radians((np.arange(w) + 0.5) / w * 360 - 180)[None, :]
    lat = np.radians(90 - (np.arange(h) + 0.5) / h * 180)[:, None]
    x, y, z = np.cos(lat) * np.sin(lon), np.sin(lat) * np.ones_like(lon), np.cos(lat) * np.cos(lon)
    a, p, r = map(math.radians, (yaw, pitch, roll))
    x, z = x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)
    y, z = y * math.cos(p) - z * math.sin(p), y * math.sin(p) + z * math.cos(p)
    x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = x / z, y / z
        d = 1 + k1 * (xn * xn + yn * yn) * (fx / (pw / 2)) ** 2
        u, v = fx * xn * d + pw / 2, ph / 2 - fx * yn * d
    inside = (z > 0) & (u >= 0) & (u < pw - 1) & (v >= 0) & (v < ph - 1)
    return np.where(inside, u, -1).astype(np.float32), np.where(inside, v, -1).astype(np.float32), inside


def place(photo, fx: float, w: int, h: int, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
          k1: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The photo (PIL or HxWx3 uint8 array) warped onto an equirectangular canvas (photo_coords) and its coverage
    mask (1 where the photo lands). The pano step's env has no OpenCV, so this is cv2.remap's INTER_LINEAR in
    numpy, bit for bit: coordinates rounded to 1/32 of a pixel, integer weights. The 4-step edit turns even a
    one-level change of its input into another picture (2026-10-27), and the tested panoramas came from cv2."""
    a = np.asarray(photo, np.uint8).astype(np.int64)
    u, v, inside = photo_coords(a.shape[1], a.shape[0], fx, w, h, yaw, pitch, roll, k1)
    X, Y = np.rint(np.where(inside, u, 0) * 32).astype(np.int64), np.rint(np.where(inside, v, 0) * 32).astype(np.int64)
    x0, y0, ax, ay = X >> 5, Y >> 5, (X & 31)[..., None], (Y & 31)[..., None]
    x1, y1 = np.minimum(x0 + 1, a.shape[1] - 1), np.minimum(y0 + 1, a.shape[0] - 1)
    s = ((a[y0, x0] * (32 - ax) + a[y0, x1] * ax) * (32 - ay) + (a[y1, x0] * (32 - ax) + a[y1, x1] * ax) * ay)
    out = ((s + 512) >> 10).clip(0, 255).astype(np.uint8)
    out[~inside] = 0
    return out, inside.astype(np.uint8)


def short_scene(case: str) -> str:
    """The workflow asks for "a very short description": the first sentence of the caption step's <case>.txt."""
    first = (PHOTOS / f"{case}.txt").read_text().strip().split(". ")[0].rstrip(".")
    return first[0].lower() + first[1:]


def load():
    """The eraser's Qwen-Image-Edit-2511 in 4 bits (erase_qwen.base_pipe) with the 360 and Lightning LoRAs."""
    from huggingface_hub import hf_hub_download
    from .erase_qwen import base_pipe
    pipe = base_pipe(4)
    pipe.load_lora_weights(hf_hub_download(*LORA), adapter_name="pano360")
    pipe.load_lora_weights(hf_hub_download(*LIGHTNING), adapter_name="lightning")
    pipe.enable_model_cpu_offload()
    return pipe


def edit(pipe, image: Image.Image, prompt: str, seed: int, adapters: dict) -> Image.Image:
    import torch
    pipe.set_adapters(list(adapters), adapter_weights=list(adapters.values()))
    return pipe(image=[image], prompt=prompt, negative_prompt=" ", true_cfg_scale=1.0, num_inference_steps=4,
                height=GEN_H, width=GEN_W, generator=torch.Generator("cpu").manual_seed(seed)).images[0]


def seam_mask() -> Image.Image:
    from PIL import ImageFilter
    m = Image.new("L", (GEN_W, GEN_H), 0)
    m.paste(255, ((GEN_W - SEAM_BAR) // 2, 0, (GEN_W + SEAM_BAR) // 2, GEN_H))
    return m.filter(ImageFilter.GaussianBlur(SEAM_BLUR / 2))


def fix_seam(pipe, img: Image.Image, seed: int) -> Image.Image:
    """The workflow's second pass: the wrap rolled to the middle, a white bar over it (screen blend), an edit that
    removes it; only the bar is kept from that edit, then the pano is rolled back."""
    rolled = Image.fromarray(np.roll(np.asarray(img), GEN_W // 2, axis=1))
    m = seam_mask()
    a, w = np.asarray(rolled, np.float32) / 255, np.asarray(m, np.float32)[..., None] / 255
    barred = Image.fromarray(((1 - (1 - a) * (1 - w)) * 255).round().astype(np.uint8))
    fixed = edit(pipe, barred, SEAM_PROMPT, seed + 1, {"lightning": 1.0}).resize(rolled.size, Image.LANCZOS)
    out = Image.composite(fixed, rolled, m)
    return Image.fromarray(np.roll(np.asarray(out), -GEN_W // 2, axis=1))


def record(case: str, seed: int) -> Path:
    return PANOS / case / f"pano_s{seed}.json"


def pano(case: str, seed: int):
    """Placement by EXIF field of view, the 360 edit and the seam pass: out/<case>/pano_s<seed>.jpg and its
    record pano_s<seed>.json (model, prompt, seed, times)."""
    import time
    import torch
    os.environ.setdefault("HF_HUB_CACHE", env("HF_HUB_CACHE") or str(tools() / "hf_cache"))
    if env("SIM2CHIP_QWEN_NF4"):  # the 4-bit pipeline saved by erase_qwen.py --save-nf4
        os.environ.setdefault("SIM2CHIP_QWEN_NF4", env("SIM2CHIP_QWEN_NF4"))
    out = PANOS / case
    out.mkdir(parents=True, exist_ok=True)
    photo = Image.open(PHOTOS / f"{case}.png").convert("RGB")
    cam = json.loads((PHOTOS / f"{case}.json").read_text())
    canvas, inside = place(photo, cam["fx"], GEN_W, GEN_H)
    canvas[inside == 0] = GREY
    Image.fromarray(canvas).save(out / "placed.jpg", quality=92)
    prompt = PROMPT.format(scene=short_scene(case))
    t0 = time.time()
    pipe = load()
    t1 = time.time()
    raw = edit(pipe, Image.fromarray(canvas), prompt, seed, {"pano360": LORA_WEIGHT, "lightning": 1.0})
    t2 = time.time()
    img = fix_seam(pipe, raw, seed)
    img.save(out / f"pano_s{seed}.jpg", quality=95)
    info = {"model": MODEL, "lora": "/".join(LORA), "prompt": prompt, "seed": seed, "size": [GEN_W, GEN_H],
            "load_s": round(t1 - t0, 1), "pano_s": round(t2 - t1, 1), "seam_s": round(time.time() - t2, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    record(case, seed).write_text(json.dumps(info, indent=1))
    print(f"{case} seed {seed}: {json.dumps(info)}", flush=True)


def ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def fit_photo(img: Image.Image, photo: Image.Image, fx: float) -> tuple[dict, float, float]:
    """Where the pano drew the photo: the scale of fx, yaw, pitch, roll (degrees) and k1 that best match the photo
    to the generated pano (normalised cross-correlation of blurred grey levels at 1024 wide, Nelder-Mead from a
    few scales). Returns the parameters, the correlation at the EXIF placement and at the fitted one."""
    import cv2
    from scipy.optimize import minimize
    w, h = 1024, 512
    g = cv2.GaussianBlur(np.asarray(img.convert("L").resize((w, h), Image.BICUBIC), np.float32), (0, 0), 1)
    ph = cv2.GaussianBlur(np.asarray(photo.convert("L"), np.float32), (0, 0), 2)
    k = np.ones((5, 5), np.uint8)

    def score(q):
        u, v, ins = photo_coords(ph.shape[1], ph.shape[0], fx * q[0], w, h, *q[1:])
        c = cv2.remap(ph, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        m = cv2.erode(ins.astype(np.uint8), k).astype(bool)
        return ncc(c, g, m) if m.sum() > 500 else -1.0
    best = None
    for s0 in (0.7, 1.0, 1.5):
        q0 = np.array([s0, 0, 0, 0, 0], float)
        simplex = np.vstack([q0] + [q0 + d for d in np.diag([0.05, 2.0, 2.0, 1.0, 0.05])])
        r = minimize(lambda q: -score(q), q0, method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-5, initial_simplex=simplex, maxiter=1500))
        if best is None or r.fun < best.fun:
            best = r
    q = [round(float(v), 4) for v in best.x]
    return (dict(zip(("fx_scale", "yaw", "pitch", "roll", "k1"), q)), round(score([1, 0, 0, 0, 0]), 3),
            round(-best.fun, 3))


def paste(pano_img: Image.Image, photo: Image.Image, fx: float, q: dict | None = None) -> Image.Image:
    """The generated pano upscaled to PASTE_W with the photo warped back over it at full detail where the pano drew
    it (fit_photo's parameters; None: at its EXIF place), feathered over the outer 2 % of the photo so the seam
    does not show."""
    import cv2
    q = q or {"fx_scale": 1.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "k1": 0.0}
    w, h = PASTE_W, PASTE_W // 2
    canvas, inside = place(photo, fx * q["fx_scale"], w, h, q["yaw"], q["pitch"], q["roll"], q["k1"])
    soft = cv2.GaussianBlur(cv2.erode(inside.astype(np.float32), np.ones((3, 3)), iterations=w // 100), (0, 0), w / 400)
    up = np.asarray(pano_img.resize((w, h), Image.BICUBIC), dtype=np.float32)
    return Image.fromarray((up * (1 - soft[..., None]) + canvas * soft[..., None]).clip(0, 255).astype(np.uint8))


def pano_paste(case: str, seed: int):
    """The photo pasted where the pano drew it: out/<case>/pano_s<seed>_paste.jpg, the fit added to the record.
    A fit under MIN_FIT did not find the photo: the pano goes on alone, upscaled (the user found it good unpasted
    too, 2026-10-26)."""
    out = PANOS / case
    img = Image.open(out / f"pano_s{seed}.jpg").convert("RGB")
    photo = Image.open(PHOTOS / f"{case}.png").convert("RGB")
    fx = json.loads((PHOTOS / f"{case}.json").read_text())["fx"]
    q, n_exif, n_fit = fit_photo(img, photo, fx)
    pasted = bool(n_fit >= MIN_FIT)
    final = paste(img, photo, fx, q) if pasted else img.resize((PASTE_W, PASTE_W // 2), Image.BICUBIC)
    final.save(out / f"pano_s{seed}_paste.jpg", quality=92)
    f = record(case, seed)
    info = json.loads(f.read_text()) if f.exists() else {}
    info["paste"] = {"pasted": pasted, "fit": q, "ncc_exif": n_exif, "ncc_fit": n_fit, "min_fit": MIN_FIT}
    f.write_text(json.dumps(info, indent=1))
    print(f"{case} seed {seed}: photo {'pasted' if pasted else 'not found, no paste'} "
          f"(fit {n_fit}, EXIF {n_exif}, fx x{q['fx_scale']:.2f})", flush=True)
