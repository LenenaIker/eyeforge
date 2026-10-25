"""One phone photo to a full 360 photosphere, as Replicator backgrounds for the sphere route
(docs/roadmap_photosphere.md): instead of one 3D scene per photo, many varied panoramas around the place where the
camera will spawn. Two published systems, test only:

- WorldGen (github ZiYang-xie/WorldGen, code Apache, LoRA Apache) outpaints the photo to a 360 panorama with its
  img2scene LoRA on FLUX.1-Fill-dev. FLUX.1 [dev] is non-commercial: this step never goes in the product as is.
- Matrix-3D (github SkyworkAI/Matrix-3D, code and weights MIT) uses that same WorldGen LoRA for its image-to-pano
  step (code/pano_init vendors an older WorldGen), but places the photo by its field of view first; then its
  panoramic video model (a LoRA on Wan2.2-TI2V-5B, Apache) walks the camera through the panorama, so every frame
  is a new photosphere a step further along. That second step is commercial-safe whatever made the first pano.

Arms:
- pano: Matrix-3D's placement (photo at the horizon, field of view from EXIF instead of MoGe's guess) and the
  WorldGen LoRA fill, SEEDS per case, plus a copy with the photo pasted back at full resolution.
- stock: WorldGen's own image-to-scene placement as shipped (since its 2026-04-11 switch to DA-2 it reads a
  perspective photo as if it were already 360 wide; this arm checks that on PHOTO_STOCK cases).
- video: Matrix-3D's 5B panoramic video, straight forward, from the seed 0 pano of VIDEO_CASES and from the
  user's own full photospheres (REAL_PANOS, data/sphere_test): every frame is a new photosphere along the walk.

The prompt per photo is Matrix-3D's own instruction, answered by Cosmos Reason 2 (local, already in use here)
instead of VideoLLaMA3-7B (16 GB more download for the same job).

FLUX.1-Fill-dev runs as nunchaku's int4 transformer (WorldGen's own low-VRAM path; bf16 does not fit in 24 GB);
its text encoders and VAE come from the gated black-forest-labs/FLUX.1-Fill-dev repo (licence accepted on HF).

    python experiments/panogen/panogen_test.py prep                      # edge-sdg
    python experiments/panogen/panogen_test.py caption                   # teacher-sam3 (Cosmos Reason 2)
    <panogen env>/bin/python experiments/panogen/panogen_test.py pano    # also: stock, video, sheet

The env is /media/lenena-iker/Crucial/tools/envs/panogen (torch 2.7.0, WorldGen, Matrix-3D's DiffSynth-Studio,
nvdiffrast) and the repos /media/lenena-iker/Crucial/tools/{WorldGen,Matrix-3D}. Outputs go to
data/panogen_test/out/<case>/."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "panogen_test"
TOOLS = Path("/media/lenena-iker/Crucial/tools")
WORLDGEN = Path(os.environ.get("WORLDGEN_REPO", TOOLS / "WorldGen"))
MATRIX3D = Path(os.environ.get("MATRIX3D_REPO", TOOLS / "Matrix-3D"))
PANO_W, PANO_H = 1600, 800  # WorldGen's and Matrix-3D's generation size
PASTE_W = 4096              # the pasted copy: wide enough to keep the photo's own detail
SEEDS = [0, 1]
PHOTO_STOCK = ["photo21", "photo31"]
VIDEO_CASES = ["photo21", "photo12", "photo68"]
# Full 360 photospheres the user took (2026-10-16/18), already stitched and filled by the sphere route.
REAL_PANOS = {"plaza": "data/sphere_test/PXL_20261018_130625084/pano.jpg",
              "street": "data/sphere_test/PXL_20261016_172048699/pano.jpg"}
# Matrix-3D's captioning instruction (code/pano_init/prompt/prompt.py), word for word.
CAPTION = ("Generate a high-quality prompt starting with 'a' for Stable Diffusion. "
           "Use only concise, non-repeating, comma-separated phrases. "
           "Follow this structure: a [style+color+material keywords] [main subject], [details], (quality tags) .")


def exif_fx(im: Image.Image, src: Path) -> float:
    """Focal length in pixels of the oriented full image, from EXIF's 35 mm equivalent (43.27 mm diagonal)."""
    f35 = im.getexif().get_ifd(0x8769).get(0xA405)
    if not f35:
        raise SystemExit(f"{src}: no 35 mm equivalent focal length in EXIF")
    w, h = ImageOps.exif_transpose(im).size
    return f35 / 43.27 * math.hypot(w, h)


def prep():
    """Per case the oriented photo (long edge 2048) and its focal length in that image's pixels."""
    sys.path.insert(0, str(ROOT / "experiments" / "lyra1"))
    from lyra1_test import CASES
    d = DATA / "in"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in CASES.items():
        raw = Image.open(ROOT / src)
        fx = exif_fx(raw, ROOT / src)
        im = ImageOps.exif_transpose(raw).convert("RGB")
        s = 2048 / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(d / f"{name}.png")
        f = fx * s
        hfov, vfov = (math.degrees(2 * math.atan(n / 2 / f)) for n in im.size)
        (d / f"{name}.json").write_text(json.dumps({"fx": f, "hfov": hfov, "vfov": vfov}))
        print(f"{name}: {im.width}x{im.height} fx {f:.0f} px, fov {hfov:.0f} x {vfov:.0f} deg")
    for name, src in REAL_PANOS.items():
        Image.open(ROOT / src).convert("RGB").resize((2048, 1024), Image.LANCZOS).save(d / f"{name}.pano.jpg", quality=95)


def caption():
    """Matrix-3D's prompt for each photo, from Cosmos Reason 2 through sim2chip's local VLM (no server needed)."""
    import base64
    import io
    sys.path.insert(0, str(ROOT))
    from sim2chip.local_vlm import DEFAULT_MODEL, MODEL
    for png in sorted([*(DATA / "in").glob("*.png"), *(DATA / "in").glob("*.pano.jpg")]):
        out = png.parent / f"{png.name.split('.')[0]}.txt"
        if out.exists():
            continue
        buf = io.BytesIO()
        im = Image.open(png).convert("RGB")
        im.thumbnail((1024, 1024))
        im.save(buf, "JPEG", quality=90)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                            {"type": "text", "text": CAPTION}]}]
        text = MODEL.chat(DEFAULT_MODEL, msg, 300, 0)["text"]
        text = text.split("</think>")[-1].strip().strip('"').splitlines()[0].strip()
        out.write_text(text)
        print(f"{png.stem}: {text}", flush=True)


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
    nvcc for nothing this test calls, so a few lines of torch stand in."""
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
    # WorldGen's package import also pulls its DA-2 depth model (a git submodule, unused by the pano fill): a
    # placeholder keeps an env without the submodule working (rebuilt 2026-10-23).
    for name in ("da2", "da2.model", "da2.model.spherevit"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["da2.model.spherevit"].SphereViT = getattr(sys.modules["da2.model.spherevit"], "SphereViT", None)


def fill_model():
    """WorldGen's build_pano_fill_model(low_vram=True): nunchaku's int4 FLUX.1-Fill-dev with the img2scene LoRA."""
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.utils import get_precision
    sys.path.insert(0, str(WORLDGEN / "src"))
    stub_pytorch3d()
    from worldgen.models.flux_pano_fill_pipeline import FluxFillPipeline
    from worldgen.utils.lora_utils import load_and_fix_lora
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{get_precision()}-flux.1-fill-dev",
                                                                 offload=True)
    # Everything but the bf16 transformer (23 GB, replaced by the int4 one above).
    base = snapshot_download("black-forest-labs/FLUX.1-Fill-dev",
                             ignore_patterns=["transformer/*", "flux1-fill-dev.safetensors", "ae.safetensors", "*.png", "*.jpg"])
    pipe = FluxFillPipeline.from_pretrained(base, transformer=transformer, torch_dtype=torch.bfloat16)
    lora = hf_hub_download("LeoXie/WorldGen", "models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors")
    state, _ = load_and_fix_lora(lora)
    transformer.update_lora_params(state)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    return pipe


def cases() -> list[str]:
    """The photos; the sheet also takes the real photospheres."""
    return sorted(p.stem for p in (DATA / "in").glob("*.png"))


def pano(only: list[str] | None):
    """The main arm: placement by EXIF field of view, WorldGen LoRA fill (WorldGen's gen_pano_fill_image: 50
    steps, guidance 30, seam blending), one pano per seed; then the photo pasted back at PASTE_W."""
    import cv2
    import torch
    sys.path.insert(0, str(WORLDGEN / "src"))
    pipe = fill_model()
    from worldgen.pano_gen import gen_pano_fill_image
    times = {}
    for case in only or cases():
        out = DATA / "out" / case
        out.mkdir(parents=True, exist_ok=True)
        photo = Image.open(DATA / "in" / f"{case}.png").convert("RGB")
        cam = json.loads((DATA / "in" / f"{case}.json").read_text())
        prompt = (DATA / "in" / f"{case}.txt").read_text().strip()
        canvas, inside = place(photo, cam["fx"], PANO_W, PANO_H)
        # Matrix-3D: the photo's mask shrunk 3 px and inverted, so the fill also redraws the warped edge.
        keep = cv2.erode(inside * 255, np.ones((3, 3), np.uint8), iterations=3)
        Image.fromarray(canvas).save(out / "placed.jpg", quality=92)
        mask = Image.fromarray(255 - keep)
        for seed in SEEDS:
            dst = out / f"pano_s{seed}.jpg"
            if dst.exists():
                continue
            t0 = time.time()
            img = gen_pano_fill_image(pipe, image=Image.fromarray(canvas), mask=mask, prompt=prompt, seed=seed,
                                      height=PANO_H, width=PANO_W)
            times[f"{case}/s{seed}"] = round(time.time() - t0, 1)
            img.save(dst, quality=95)
            paste(img, photo, cam["fx"]).save(out / f"pano_s{seed}_paste.jpg", quality=92)
            print(f"{case} seed {seed}: {times[f'{case}/s{seed}']} s", flush=True)
            torch.cuda.empty_cache()
    merge_times(times)


def paste(pano_img: Image.Image, photo: Image.Image, fx: float) -> Image.Image:
    """The generated pano upscaled to PASTE_W with the photo warped back over it at full detail, feathered over
    the outer 2 % of the photo so the seam does not show."""
    import cv2
    w, h = PASTE_W, PASTE_W // 2
    canvas, inside = place(photo, fx, w, h)
    soft = cv2.GaussianBlur(cv2.erode(inside.astype(np.float32), np.ones((3, 3)), iterations=w // 100), (0, 0), w / 400)
    up = np.asarray(pano_img.resize((w, h), Image.BICUBIC), dtype=np.float32)
    return Image.fromarray((up * (1 - soft[..., None]) + canvas * soft[..., None]).clip(0, 255).astype(np.uint8))


def stock():
    """WorldGen's image-to-scene as shipped: generate_pano with its DA-2 placement (WorldGen's class, low VRAM)."""
    import torch
    sys.path.insert(0, str(WORLDGEN / "src"))
    stub_pytorch3d()
    import worldgen.pano_gen as pg
    pg.build_pano_fill_model = lambda **k: fill_model()  # the main arm's loader (int4)
    import worldgen.worldgen as wgm
    wgm.build_pano_fill_model = pg.build_pano_fill_model
    wg = wgm.WorldGen(mode="i2s", device=torch.device("cuda"), low_vram=True, resolution=PANO_W)
    times = {}
    for case in PHOTO_STOCK:
        dst = DATA / "out" / case / "stock_s0.jpg"
        if dst.exists():
            continue
        t0 = time.time()
        img = wg.generate_pano((DATA / "in" / f"{case}.txt").read_text().strip(),
                               Image.open(DATA / "in" / f"{case}.png").convert("RGB"))
        times[f"{case}/stock"] = round(time.time() - t0, 1)
        img.save(dst, quality=95)
        print(f"{case} stock: {times[f'{case}/stock']} s", flush=True)
    merge_times(times)


def video(only: list[str] | None, mode: str):
    """Matrix-3D's step 2 with the 5B model in low-VRAM mode: its own script under torchrun, from the seed 0 pano.
    It needs Matrix-3D's checkpoints/ layout (Wan2.2-TI2V-5B, the 5B LoRA under Wan-AI/wan-lora, MoGe v1)."""
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OPENCV_IO_ENABLE_OPENEXR": "1"}
    times = {}
    for case in only or [*REAL_PANOS, *VIDEO_CASES]:
        d = DATA / "out" / case / f"m3d_{mode}"
        if (d / "pano_video.mp4").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        src = DATA / "in" / f"{case}.pano.jpg" if case in REAL_PANOS else DATA / "out" / case / "pano_s0.jpg"
        shutil.copy(src, d / "pano_img.jpg")
        shutil.copy(DATA / "in" / f"{case}.txt", d / "prompt.txt")
        t0 = time.time()
        with open(d / "run.log", "w") as log:
            subprocess.run([str(Path(sys.executable).parent / "torchrun"), "--nproc_per_node", "1",
                            str(Path(__file__).parent / "m3d_video.py"), f"--inout_dir={d}", "--resolution=720",
                            "--use_5b_model", "--enable_vram_management", f"--movement_mode={mode}"],
                           cwd=MATRIX3D, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        times[f"{case}/video_{mode}"] = round(time.time() - t0, 1)
        print(f"{case} video {mode}: {times[f'{case}/video_{mode}']} s", flush=True)
    merge_times(times)


def merge_times(new: dict):
    path = DATA / "times.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **new}, indent=1))


def views(pano_img: Image.Image, dirs: list[tuple[float, float]], fov=(64, 50), hw=(240, 320)) -> list[Image.Image]:
    """Pinhole views out of an equirectangular image, as a Replicator camera at the pano's centre would see:
    (yaw, pitch) in degrees, yaw 0 the photo's direction, positive pitch up."""
    import py360convert
    e = np.asarray(pano_img.convert("RGB"))
    return [Image.fromarray(py360convert.e2p(e, fov, u, v, hw).astype(np.uint8)) for u, v in dirs]


def video_frames(path: Path, idx: list[int]) -> list[Image.Image]:
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")
    return [Image.fromarray(frames[min(i, len(frames) - 1)]) for i in idx]


def sheet(only: list[str] | None):
    """Per case: the inputs and panoramas, then what a camera at the centre would see in five directions, then
    (if made) Matrix-3D's walk through the pano seen straight ahead and behind."""
    cell = (320, 240)
    dirs = [(0, 0), (90, 0), (180, 0), (-90, 0), (0, -35)]
    dir_names = ["ahead (photo)", "right", "behind", "left", "ahead, 35 deg down"]
    for case in only or [*cases(), *REAL_PANOS]:
        out = DATA / "out" / case
        rows = []
        if case in REAL_PANOS:
            top = [("real photosphere", Image.open(DATA / "in" / f"{case}.pano.jpg"))]
        elif (out / "pano_s0.jpg").exists():
            top = [("photo", Image.open(DATA / "in" / f"{case}.png")), ("placed", Image.open(out / "placed.jpg"))]
        else:
            continue
        top += [(f"pano seed {s}", Image.open(out / f"pano_s{s}.jpg")) for s in SEEDS if (out / f"pano_s{s}.jpg").exists()]
        if (out / "stock_s0.jpg").exists():
            top.append(("WorldGen stock", Image.open(out / "stock_s0.jpg")))
        rows.append(("inputs and panoramas", top))
        for s in SEEDS:
            p = out / f"pano_s{s}_paste.jpg"
            if p.exists():
                rows.append((f"seed {s}, camera at the centre", list(zip(dir_names, views(Image.open(p), dirs)))))
        for vd in sorted(out.glob("m3d_*/pano_video.mp4")):
            f0, f1, f2 = video_frames(vd, [0, 40, 80])
            ahead = views(f1, [(0, 0)]) + views(f2, [(0, 0)])
            behind = views(f1, [(180, 0)]) + views(f2, [(180, 0)])
            rows.append((f"Matrix-3D walk ({vd.parent.name})",
                         [("pano frame 80", f2), ("ahead, frame 40", ahead[0]), ("ahead, frame 80", ahead[1]),
                          ("behind, frame 40", behind[0]), ("behind, frame 80", behind[1])]))
        ncol = max(len(r[1]) for r in rows)
        canvas = Image.new("RGB", (cell[0] * ncol, (cell[1] + 32) * len(rows)), "white")
        dr = ImageDraw.Draw(canvas)
        for r, (label, items) in enumerate(rows):
            y = r * (cell[1] + 32)
            dr.text((4, y + 2), f"{case}: {label}", fill="black")
            for c, (name, im) in enumerate(items):
                im = im.convert("RGB")
                im.thumbnail(cell)
                canvas.paste(im, (c * cell[0], y + 30))
                dr.text((c * cell[0] + 4, y + 16), name, fill="gray")
        dst = DATA / f"sheet_{case}.jpg"
        canvas.save(dst, quality=88)
        print(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "caption", "pano", "stock", "video", "sheet"])
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--mode", default="straight", help="Matrix-3D movement: straight, s_curve, l_curve, r_curve")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_CACHE", str(TOOLS / "hf_cache"))
    if a.step == "prep":
        prep()
    elif a.step == "caption":
        caption()
    elif a.step == "pano":
        pano(a.case)
    elif a.step == "stock":
        stock()
    elif a.step == "video":
        video(a.case, a.mode)
    else:
        sheet(a.case)


if __name__ == "__main__":
    main()
