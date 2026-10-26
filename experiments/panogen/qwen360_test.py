"""Test 1 of docs/plan_sustituir_flux_2026-10-26.md: the F3c route's pano step with the QWEN-EDIT_360 LoRA
(mickmumpitz, MIT, trained on Poly Haven CC0 HDRIs; licence checked on its HF page 2026-10-26, repo sha f7626f2)
on Qwen-Image-Edit instead of WorldGen's LoRA on FLUX.1-Fill-dev (non-commercial). wsroute/pano.py is untouched.

The recipe is the author's ComfyUI workflow (251018_MICKMUMPITZ_QWEN-IMAGE-EDIT-360_1-1.json): LoRA at 0.9 with
the Lightning 4-step LoRA, cfg 1, a 2048x1024 output, then a second pass (Lightning only) that rolls the seam to
the middle, whitens a 200 px bar over it and asks the model to remove it. Its prompt asks for "a very short
description": the first sentence of the case's <case>.txt.

Arms, same seed as FLUX (0):
- A: the photo placed on the 2:1 canvas by its EXIF field of view (pano.place), grey elsewhere, as the input.
- B: the photo as the LoRA expects it: scaled to 1 MP, a 400 px grey margin, letterboxed into 1024x512.
Qwen redraws the photo too, at its own field of view (A drew it about 1.5 times smaller than EXIF's), so the
photo is pasted back at 4096 px where the pano drew it: fx scale, yaw, pitch, roll and one radial term fitted to
the generated pano (fit_photo), with pano.paste's feathering. The EXIF paste is kept as _paste_exif.jpg.

Three steps, two envs (the eraser's env has no OpenCV, which place and paste need):

    $SIM2CHIP_SCENE_PYTHON experiments/panogen/qwen360_test.py prep    # placed canvases and the B input
    $SIM2CHIP_ERASE_PYTHON experiments/panogen/qwen360_test.py gen     # Qwen, both arms (HF_HUB_CACHE set)
    $SIM2CHIP_SCENE_PYTHON experiments/panogen/qwen360_test.py post    # paste, measures, one sheet per case

Writes data/panogen_test/out/<case>/qwen360/ (<arm>_s<seed>.jpg, <arm>_s<seed>_paste.jpg, measures.json), the
sheet <case>/qwen360/sheet.jpg (FLUX, A, B: the whole pano and four 90 degree views) and paste_check.jpg (the
front view generated, pasted at EXIF's place and pasted where it was drawn).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PANOGEN = ROOT / "data" / "panogen_test"
CASES = ["f3c_photo21", "f3c_photo47", "f3c_photo66"]
GEN_W, GEN_H = 2048, 1024
GREY = 127
LORA = ("mickmumpitz/QWEN-EDIT_360", "251018_MICKMUMPITZ_QWEN-EDIT_360_03.safetensors")
LIGHTNING = ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
PROMPT = ("Create a 360 panoramic image of {scene}. An HDRI image captured on a 360 degree camera, insta 360, "
          "equirectangular projection. Keep the style the same.")
SEAM_PROMPT = ("Remove the visible seam in the middle of the image. Remove the white bar. Keep the rest of the "
               "image exactly the same.")
SEAM_BAR, SEAM_BLUR = 200, 20  # the workflow's mask size and blur, px at GEN_W
PAD_B = 400                    # the workflow's margin around the 1 MP photo


def out_dir(case):
    return PANOGEN / "out" / case / "qwen360"


def short_scene(case):
    first = (PANOGEN / "in" / f"{case}.txt").read_text().strip().split(". ")[0].rstrip(".")
    return first[0].lower() + first[1:]


def prep(args):
    sys.path.insert(0, str(ROOT / "experiments" / "worldstereo"))
    from wsroute.pano import place
    for case in args.cases:
        d = out_dir(case)
        d.mkdir(parents=True, exist_ok=True)
        photo = Image.open(PANOGEN / "in" / f"{case}.png").convert("RGB")
        fx = json.loads((PANOGEN / "in" / f"{case}.json").read_text())["fx"]
        canvas, inside = place(photo, fx, GEN_W, GEN_H)
        canvas[inside == 0] = GREY
        Image.fromarray(canvas).save(d / "in_A.png")
        s = math.sqrt(1024 * 1024 / (photo.width * photo.height))
        small = photo.resize((round(photo.width * s), round(photo.height * s)), Image.LANCZOS)
        padded = Image.new("RGB", (small.width + 2 * PAD_B, small.height + 2 * PAD_B), (GREY,) * 3)
        padded.paste(small, (PAD_B, PAD_B))
        k = min(1024 / padded.width, 512 / padded.height)
        padded = padded.resize((round(padded.width * k), round(padded.height * k)), Image.LANCZOS)
        box = Image.new("RGB", (1024, 512), (GREY,) * 3)
        box.paste(padded, ((1024 - padded.width) // 2, (512 - padded.height) // 2))
        box.save(d / "in_B.png")
        print(case, "prepared", flush=True)


def load():
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from huggingface_hub import hf_hub_download
    # The eraser's 2511 pipeline already in 4 bits, before its LoRAs (erase_qwen.py --save-nf4).
    pipe = QwenImageEditPlusPipeline.from_pretrained(os.environ["SIM2CHIP_QWEN_NF4"], torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(hf_hub_download(*LORA), adapter_name="pano360")
    pipe.load_lora_weights(hf_hub_download(*LIGHTNING), adapter_name="lightning")
    pipe.enable_model_cpu_offload()
    return pipe


def edit(pipe, image, prompt, seed, adapters):
    import torch
    pipe.set_adapters(list(adapters), adapter_weights=list(adapters.values()))
    return pipe(image=[image], prompt=prompt, negative_prompt=" ", true_cfg_scale=1.0, num_inference_steps=4,
                height=GEN_H, width=GEN_W, generator=torch.Generator("cpu").manual_seed(seed)).images[0]


def seam_mask():
    from PIL import ImageFilter
    m = Image.new("L", (GEN_W, GEN_H), 0)
    m.paste(255, ((GEN_W - SEAM_BAR) // 2, 0, (GEN_W + SEAM_BAR) // 2, GEN_H))
    return m.filter(ImageFilter.GaussianBlur(SEAM_BLUR / 2))


def fix_seam(pipe, pano, seed):
    """The workflow's second pass: the wrap rolled to the middle, a white bar over it (screen blend), an edit that
    removes it; only the bar is kept from that edit, then the pano is rolled back."""
    rolled = Image.fromarray(np.roll(np.asarray(pano), GEN_W // 2, axis=1))
    m = seam_mask()
    a, w = np.asarray(rolled, np.float32) / 255, np.asarray(m, np.float32)[..., None] / 255
    barred = Image.fromarray(((1 - (1 - a) * (1 - w)) * 255).round().astype(np.uint8))
    fixed = edit(pipe, barred, SEAM_PROMPT, seed + 1, {"lightning": 1.0}).resize(rolled.size, Image.LANCZOS)
    out = Image.composite(fixed, rolled, m)
    return Image.fromarray(np.roll(np.asarray(out), -GEN_W // 2, axis=1))


def gen(args):
    import torch
    t = time.time()
    pipe = load()
    load_s = round(time.time() - t, 1)
    for case in args.cases:
        d = out_dir(case)
        prompt = PROMPT.format(scene=short_scene(case))
        info = json.loads((d / "gen.json").read_text()) if (d / "gen.json").exists() else {}
        for arm in args.arms:
            t = time.time()
            raw = edit(pipe, Image.open(d / f"in_{arm}.png").convert("RGB"), prompt, args.seed,
                       {"pano360": args.lora_weight, "lightning": 1.0})
            raw.save(d / f"{arm}_s{args.seed}_noseam.jpg", quality=95)
            t1 = time.time() - t
            fixed = fix_seam(pipe, raw, args.seed)
            fixed.save(d / f"{arm}_s{args.seed}.jpg", quality=95)
            info[arm] = {"prompt": prompt, "seed": args.seed, "lora_weight": args.lora_weight,
                         "seconds_pano": round(t1, 1), "seconds_total": round(time.time() - t, 1),
                         "load_seconds": load_s, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps({"case": case, "arm": arm, **info[arm]}), flush=True)
            (d / "gen.json").write_text(json.dumps(info, indent=1))


def seam_ratio(img):
    """Mean step across the wrap (last column to first) over the mean step between neighbouring columns: about 1
    when the seam is closed, well above when it shows."""
    a = np.asarray(img, np.float32)
    wrap = np.abs(a[:, 0] - a[:, -1]).mean()
    inner = np.abs(np.diff(a, axis=1)).mean()
    return round(float(wrap / inner), 2)


def place_fit(photo, fx, w, h, yaw=0.0, pitch=0.0, roll=0.0, k1=0.0):
    """pano.place with yaw, roll and one radial distortion term: the photo warped onto an equirectangular canvas
    (longitude 0 at the centre) and its coverage mask."""
    import cv2
    lon = np.radians((np.arange(w) + 0.5) / w * 360 - 180)[None, :]
    lat = np.radians(90 - (np.arange(h) + 0.5) / h * 180)[:, None]
    x, y, z = np.cos(lat) * np.sin(lon), np.sin(lat) * np.ones_like(lon), np.cos(lat) * np.cos(lon)
    a, p, r = map(math.radians, (yaw, pitch, roll))
    x, z = x * math.cos(a) - z * math.sin(a), x * math.sin(a) + z * math.cos(a)
    y, z = y * math.cos(p) - z * math.sin(p), y * math.sin(p) + z * math.cos(p)
    x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
    ph, pw = photo.shape[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = x / z, y / z
        d = 1 + k1 * (xn * xn + yn * yn) * (fx / (pw / 2)) ** 2  # k1 per squared half-width of the photo
        u, v = fx * xn * d + pw / 2, ph / 2 - fx * yn * d
    inside = (z > 0) & (u >= 0) & (u < pw - 1) & (v >= 0) & (v < ph - 1)
    u, v = np.where(inside, u, -1).astype(np.float32), np.where(inside, v, -1).astype(np.float32)
    return cv2.remap(photo, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), inside.astype(np.uint8)


def ncc(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def fit_photo(img, photo, fx):
    """Where the pano drew the photo: the scale of fx, yaw, pitch, roll (degrees) and k1 that best match the photo
    to the generated pano (normalised cross-correlation of blurred grey levels at 1024 wide, Nelder-Mead from a
    few scales). Qwen redraws the photo too, at its own field of view, so the EXIF placement does not hold.
    Returns the parameters, the correlation with the EXIF placement and with the fitted one."""
    import cv2
    from scipy.optimize import minimize
    w, h = 1024, 512
    g = cv2.GaussianBlur(np.asarray(img.convert("L").resize((w, h), Image.BICUBIC), np.float32), (0, 0), 1)
    ph = cv2.GaussianBlur(np.asarray(photo.convert("L"), np.float32), (0, 0), 2)
    k = np.ones((5, 5), np.uint8)

    def score(q):
        c, ins = place_fit(ph, fx * q[0], w, h, *q[1:])
        m = cv2.erode(ins, k).astype(bool)
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
    return dict(zip(("fx_scale", "yaw", "pitch", "roll", "k1"), q)), round(score([1, 0, 0, 0, 0]), 3), \
        round(-best.fun, 3)


def paste_fit(pano_img, photo, fx, q):
    """pano.paste with the photo where the pano drew it (fit_photo's parameters) instead of at its EXIF place."""
    import cv2
    w, h = 4096, 2048
    canvas, inside = place_fit(np.asarray(photo), fx * q["fx_scale"], w, h, q["yaw"], q["pitch"], q["roll"], q["k1"])
    soft = cv2.GaussianBlur(cv2.erode(inside.astype(np.float32), np.ones((3, 3)), iterations=w // 100), (0, 0), w / 400)
    up = np.asarray(pano_img.resize((w, h), Image.BICUBIC), dtype=np.float32)
    return Image.fromarray((up * (1 - soft[..., None]) + canvas * soft[..., None]).clip(0, 255).astype(np.uint8))


def views(pano):
    """Front, left, back and right at 90 degrees horizontal (4:3), pitch 0, with wsroute.camera.crop."""
    sys.path.insert(0, str(ROOT / "experiments" / "worldstereo"))
    from wsroute import camera
    camera.W, camera.H, camera.F = 640, 480, 320.0
    a = np.asarray(pano)
    return [Image.fromarray(camera.crop(a, camera.w2c([0, 0, 0], yaw, 0))) for yaw in (0, 90, 180, -90)]


def sheet(case, rows):
    from PIL import ImageDraw, ImageFont
    pw, vw = 1600, 400
    font = ImageFont.load_default(size=30)
    blocks = []
    for title, pano in rows:
        b = Image.new("RGB", (pw, 50 + pw // 2 + 30 + vw * 3 // 4), "white")
        dr = ImageDraw.Draw(b)
        dr.text((10, 8), title, fill="black", font=font)
        b.paste(pano.resize((pw, pw // 2), Image.LANCZOS), (0, 50))
        y = 50 + pw // 2
        for i, (v, name) in enumerate(zip(views(pano), ("front", "left", "back", "right"))):
            dr.text((i * vw + 6, y), name, fill="black", font=ImageFont.load_default(size=22))
            b.paste(v.resize((vw, vw * 3 // 4), Image.LANCZOS), (i * vw, y + 30))
        blocks.append(b)
    out = Image.new("RGB", (pw, sum(b.height for b in blocks) + 10 * len(blocks)), (200, 200, 200))
    y = 0
    for b in blocks:
        out.paste(b, (0, y))
        y += b.height + 10
    path = out_dir(case) / "sheet.jpg"
    out.save(path, quality=85)
    return path


def paste_check(case, rows):
    """The front 90 degree view of each pano: the generated one, its EXIF paste and its fitted paste."""
    from PIL import ImageDraw, ImageFont
    vw, vh = 640, 480
    font = ImageFont.load_default(size=26)
    out = Image.new("RGB", (vw * 3, (vh + 40) * len(rows)), "white")
    dr = ImageDraw.Draw(out)
    for i, (arm, panos) in enumerate(rows):
        for j, (name, pano) in enumerate(panos):
            dr.text((j * vw + 8, i * (vh + 40) + 6), f"{arm}: {name}", fill="black", font=font)
            out.paste(views(pano)[0], (j * vw, i * (vh + 40) + 40))
    path = out_dir(case) / "paste_check.jpg"
    out.save(path, quality=85)
    return path


def post(args):
    sys.path.insert(0, str(ROOT / "experiments" / "worldstereo"))
    from wsroute.pano import paste
    for case in args.cases:
        d = out_dir(case)
        photo = Image.open(PANOGEN / "in" / f"{case}.png").convert("RGB")
        fx = json.loads((PANOGEN / "in" / f"{case}.json").read_text())["fx"]
        info = json.loads((d / "gen.json").read_text())
        flux = Image.open(PANOGEN / "out" / case / f"pano_s{args.seed}.jpg").convert("RGB")
        q, n_exif, n_fit = fit_photo(flux, photo, fx)
        measures = {"FLUX": {"seam_ratio": seam_ratio(flux), "ncc_exif": n_exif, "ncc_fit": n_fit, "fit": q}}
        rows = [(f"FLUX.1-Fill + WorldGen LoRA (now)  seam {measures['FLUX']['seam_ratio']}  photo fit {n_exif}",
                 Image.open(PANOGEN / "out" / case / f"pano_s{args.seed}_paste.jpg"))]
        checks = []
        for arm in args.arms:
            raw = Image.open(d / f"{arm}_s{args.seed}.jpg").convert("RGB")
            exif = paste(raw, photo, fx)
            exif.save(d / f"{arm}_s{args.seed}_paste_exif.jpg", quality=92)
            q, n_exif, n_fit = fit_photo(raw, photo, fx)
            fitted = paste_fit(raw, photo, fx, q)
            fitted.save(d / f"{arm}_s{args.seed}_paste.jpg", quality=92)
            m = {"seam_ratio": seam_ratio(raw),
                 "seam_ratio_noseam_pass": seam_ratio(Image.open(d / f"{arm}_s{args.seed}_noseam.jpg")),
                 "ncc_exif": n_exif, "ncc_fit": n_fit, "fit": q, "seconds": info[arm]["seconds_total"]}
            measures[arm] = m
            rows.append((f"Qwen 360 arm {arm}  seam {m['seam_ratio']}  fx x{q['fx_scale']:.2f} k1 {q['k1']:+.2f}  "
                         f"photo fit {n_exif} -> {n_fit}  {m['seconds']:.0f} s", fitted))
            checks.append((arm, [("generated", raw), ("EXIF paste", exif), ("fitted paste", fitted)]))
        (d / "measures.json").write_text(json.dumps(measures, indent=1))
        print(case, json.dumps(measures), sheet(case, rows), paste_check(case, checks), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "gen", "post"])
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--arms", nargs="+", default=["A", "B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-weight", type=float, default=0.9)
    args = ap.parse_args()
    {"prep": prep, "gen": gen, "post": post}[args.step](args)


if __name__ == "__main__":
    main()
