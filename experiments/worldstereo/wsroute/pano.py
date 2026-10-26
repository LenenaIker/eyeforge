"""Step 2: the erased photo outpainted to a 360 panorama. WorldGen (github ZiYang-xie/WorldGen, code Apache, LoRA
Apache) fills it with its img2scene LoRA on FLUX.1-Fill-dev, after Matrix-3D's placement (the photo at the horizon,
field of view from EXIF). FLUX.1 [dev] is non-commercial: this step never goes in the product as is.

FLUX.1-Fill-dev runs as nunchaku's int4 transformer (WorldGen's own low-VRAM path; bf16 does not fit in 24 GB); its
text encoders and VAE come from the gated black-forest-labs/FLUX.1-Fill-dev repo (licence accepted on HF). Runs in
SIM2CHIP_PANOGEN_PYTHON (torch 2.7.0, WorldGen, nunchaku). The test of every arm (Matrix-3D's walk, WorldGen as
shipped) is experiments/panogen/panogen_test.py."""
from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .paths import PANOS, PHOTOS, env, tools

PANO_W, PANO_H = 1600, 800  # WorldGen's generation size
PASTE_W = 4096              # the pasted copy: wide enough to keep the photo's own detail
PHOTO_LONG_EDGE = 2048


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


def place(photo: Image.Image, fx: float, w: int, h: int, pitch_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The photo warped onto an equirectangular canvas (longitude 0 at the centre, Matrix-3D's i2p_init with a
    pitch) and its coverage mask (1 where the photo lands). Bilinear, where Matrix-3D used nearest."""
    import cv2
    lon = np.radians((np.arange(w) + 0.5) / w * 360 - 180)[None, :]
    lat = np.radians(90 - (np.arange(h) + 0.5) / h * 180)[:, None]
    x, y, z = np.cos(lat) * np.sin(lon), np.sin(lat) * np.ones_like(lon), np.cos(lat) * np.cos(lon)
    p = math.radians(pitch_deg)
    yc, zc = y * math.cos(p) - z * math.sin(p), y * math.sin(p) + z * math.cos(p)
    pw, ph = photo.size
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * x / zc + pw / 2
        v = ph / 2 - fx * yc / zc
    inside = (zc > 0) & (u >= 0) & (u < pw - 1) & (v >= 0) & (v < ph - 1)
    u, v = np.where(inside, u, -1).astype(np.float32), np.where(inside, v, -1).astype(np.float32)
    canvas = cv2.remap(np.asarray(photo), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return canvas, inside.astype(np.uint8)


def stub_pytorch3d():
    """WorldGen imports pytorch3d for one function (matrix_to_quaternion, splats only); building pytorch3d needs
    nvcc for nothing the fill calls, so a few lines of torch stand in. Its DA-2 depth model (a git submodule) is
    unused by the fill too: a placeholder keeps an env without the submodule working."""
    import torch

    def matrix_to_quaternion(m):
        w = torch.sqrt(torch.clamp(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], min=1e-8)) / 2
        return torch.stack([w, (m[..., 2, 1] - m[..., 1, 2]) / (4 * w), (m[..., 0, 2] - m[..., 2, 0]) / (4 * w),
                            (m[..., 1, 0] - m[..., 0, 1]) / (4 * w)], -1)
    mod = types.ModuleType("pytorch3d")
    mod.transforms = types.ModuleType("pytorch3d.transforms")
    mod.transforms.matrix_to_quaternion = matrix_to_quaternion
    sys.modules.setdefault("pytorch3d", mod)
    sys.modules.setdefault("pytorch3d.transforms", mod.transforms)
    for name in ("da2", "da2.model", "da2.model.spherevit"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["da2.model.spherevit"].SphereViT = getattr(sys.modules["da2.model.spherevit"], "SphereViT", None)


def worldgen() -> Path:
    return Path(env("WORLDGEN_REPO") or tools() / "WorldGen")


def fill_model():
    """WorldGen's build_pano_fill_model(low_vram=True): nunchaku's int4 FLUX.1-Fill-dev with the img2scene LoRA."""
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision
    sys.path.insert(0, str(worldgen() / "src"))
    stub_pytorch3d()
    from worldgen.models.flux_pano_fill_pipeline import FluxFillPipeline
    from worldgen.utils.lora_utils import load_and_fix_lora
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{get_precision()}-flux.1-fill-dev",
                                                                 offload=True)
    # Everything but the bf16 transformer (23 GB, replaced by the int4 one above).
    base = snapshot_download("black-forest-labs/FLUX.1-Fill-dev",
                             ignore_patterns=["transformer/*", "flux1-fill-dev.safetensors", "ae.safetensors",
                                              "*.png", "*.jpg"])
    pipe = FluxFillPipeline.from_pretrained(base, transformer=transformer, torch_dtype=torch.bfloat16)
    lora = hf_hub_download("LeoXie/WorldGen", "models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors")
    state, _ = load_and_fix_lora(lora)
    transformer.update_lora_params(state)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    return pipe


def paste(pano_img: Image.Image, photo: Image.Image, fx: float) -> Image.Image:
    """The generated pano upscaled to PASTE_W with the photo warped back over it at full detail, feathered over
    the outer 2 % of the photo so the seam does not show."""
    import cv2
    w, h = PASTE_W, PASTE_W // 2
    canvas, inside = place(photo, fx, w, h)
    soft = cv2.GaussianBlur(cv2.erode(inside.astype(np.float32), np.ones((3, 3)), iterations=w // 100), (0, 0), w / 400)
    up = np.asarray(pano_img.resize((w, h), Image.BICUBIC), dtype=np.float32)
    return Image.fromarray((up * (1 - soft[..., None]) + canvas * soft[..., None]).clip(0, 255).astype(np.uint8))


def pano(case: str, seed: int):
    """Placement by EXIF field of view, WorldGen LoRA fill (WorldGen's gen_pano_fill_image: 50 steps, guidance 30,
    seam blending), then the photo pasted back at PASTE_W: out/<case>/pano_s<seed>.jpg and pano_s<seed>_paste.jpg."""
    import time
    import cv2
    os.environ.setdefault("HF_HUB_CACHE", env("HF_HUB_CACHE") or str(tools() / "hf_cache"))
    pipe = fill_model()
    from worldgen.pano_gen import gen_pano_fill_image
    out = PANOS / case
    out.mkdir(parents=True, exist_ok=True)
    photo = Image.open(PHOTOS / f"{case}.png").convert("RGB")
    cam = json.loads((PHOTOS / f"{case}.json").read_text())
    prompt = (PHOTOS / f"{case}.txt").read_text().strip()
    canvas, inside = place(photo, cam["fx"], PANO_W, PANO_H)
    # Matrix-3D: the photo's mask shrunk 3 px and inverted, so the fill also redraws the warped edge.
    keep = cv2.erode(inside * 255, np.ones((3, 3), np.uint8), iterations=3)
    Image.fromarray(canvas).save(out / "placed.jpg", quality=92)
    t0 = time.time()
    img = gen_pano_fill_image(pipe, image=Image.fromarray(canvas), mask=Image.fromarray(255 - keep), prompt=prompt,
                              seed=seed, height=PANO_H, width=PANO_W)
    img.save(out / f"pano_s{seed}.jpg", quality=95)
    paste(img, photo, cam["fx"]).save(out / f"pano_s{seed}_paste.jpg", quality=92)
    print(f"{case} seed {seed}: {time.time() - t0:.1f} s", flush=True)
