"""FlashWorld against photo2scene and Lyra 1 (test only: FlashWorld's weights are CC BY-NC-SA 4.0, never in the
product). FlashWorld (github imlixinyang/FlashWorld) turns one photo plus a camera path into 3D Gaussians in one
feed-forward pass on Wan2.2-TI2V-5B. The test feeds it the same phone photos as experiments/lyra1/lyra1_test.py
and Lyra's six camera moves (orbit, left, right, up, back, forward), one generation per move, so the sheets put
photo2scene, Lyra and FlashWorld side by side. Runs locally on the 3090:

    python experiments/flashworld/flashworld_test.py prep                       # edge-sdg
    <flashworld env>/bin/python experiments/flashworld/flashworld_test.py run  # FlashWorld's own env
    python experiments/flashworld/flashworld_test.py sheet                      # edge-sdg

The env (torch 2.4.1 cu124, gsplat 1.5.3 wheel, FlashWorld's diffusers commit; no spz, no SageAttention) is
/media/lenena-iker/Crucial/tools/envs/flashworld and the repo /media/lenena-iker/Crucial/tools/FlashWorld.
Outputs go to data/flashworld_test/out/<case>/<move>/ (video.mp4, frames, gaussians.ply)."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "flashworld_test"
REPO = Path(os.environ.get("FLASHWORLD_REPO", "/media/lenena-iker/Crucial/tools/FlashWorld"))
N_FRAME, H, W = 24, 480, 704  # FlashWorld's trained resolution and frame count
# Lyra's view order (lyra_static.yaml), so FlashWorld's rows line up with Lyra's on the sheet.
MOVES = ["orbit", "left", "right", "up", "back", "forward"]
ORBIT_DEG = 30.0  # orbit around a point ORBIT_R units ahead; FlashWorld normalises translation, so only shape counts
ORBIT_R = 2.0


def look_quat(R: np.ndarray) -> list[float]:
    """Rotation matrix to FlashWorld's quaternion, real part first."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def yaw(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def path(move: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Camera-to-world (rotation, position) per frame, OpenGL axes as FlashWorld's create_rays: x right, y up,
    looking down -z. Frame 0 is the photo's camera."""
    out = []
    for t in np.linspace(0, 1, N_FRAME):
        if move == "orbit":
            a = ORBIT_DEG * t  # positive yaw turns left; the camera swings right, keeping the centre in view
            R = yaw(a)
            centre = np.array([0, 0, -ORBIT_R])
            out.append((R, centre - R @ np.array([0, 0, -ORBIT_R])))
            continue
        step = {"left": [-1, 0, 0], "right": [1, 0, 0], "up": [0, 1, 0], "back": [0, 0, 1], "forward": [0, 0, -1]}[move]
        out.append((np.eye(3), t * np.array(step, float)))
    return out


def exif_fx(im: Image.Image, src: Path) -> float:
    """Focal length in pixels of the oriented full image, from EXIF's 35 mm equivalent (43.27 mm diagonal)."""
    f35 = im.getexif().get_ifd(0x8769).get(0xA405)
    if not f35:
        raise SystemExit(f"{src}: no 35 mm equivalent focal length in EXIF")
    w, h = ImageOps.exif_transpose(im).size
    return f35 / 43.27 * math.hypot(w, h)


def prep():
    """Per case and move a FlashWorld request JSON: the photo cut to the centre 704:480 band, the camera path and
    the intrinsics scaled to that cut."""
    sys.path.insert(0, str(ROOT / "experiments" / "lyra1"))
    from lyra1_test import CASES
    d = DATA / "in"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in CASES.items():
        raw = Image.open(ROOT / src)
        fx = exif_fx(raw, ROOT / src)
        im = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = im.size
        s = max(W / w, H / h)
        cw, ch = round(W / s), round(H / s)
        im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)).resize((W, H), Image.LANCZOS)
        im.save(d / f"{name}.png")
        f = fx * s
        print(f"{name}: {w}x{h} fx {fx:.0f} px -> {W}x{H} fx {f:.1f} (hfov {math.degrees(2 * math.atan(W / 2 / f)):.0f} deg)")
        for move in MOVES:
            cams = [{"quaternion": look_quat(R), "position": p.tolist(), "fx": f, "fy": f, "cx": W / 2, "cy": H / 2}
                    for R, p in path(move)]
            req = {"image": str(d / f"{name}.png"), "text_prompt": "", "cameras": cams, "image_index": 0}
            (d / f"{name}__{move}.json").write_text(json.dumps(req))


def run(offload_vae: bool, only: list[str] | None):
    """Loads FlashWorld once and generates every prepared request; the trajectory video is rendered from the
    Gaussians and three frames (start, half, end) are cut for the sheet."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # a run of 48 fragments 24 GB
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import imageio.v3 as iio
    import torch
    from huggingface_hub import hf_hub_download
    import app
    from utils import export_gaussians

    # FlashWorld loads T5 in fp32 on the CPU next to the fp32 transformer before its fp8 step: 62 GB of RAM, over
    # the workstation's 61. T5 in bf16 saves ~11 GB, and its one embedding (the prompt is always "") is cached.
    t5_load = app.UMT5EncoderModel.from_pretrained

    class T5:
        @staticmethod
        def from_pretrained(*a, **k):
            return t5_load(*a, **{**k, "torch_dtype": torch.bfloat16})
    app.UMT5EncoderModel = T5
    # FlashWorld pins gsplat 1.5.2, which wants the RGB+D background with its depth channel; 1.5.3 (the wheel
    # that needs no nvcc) appends that channel itself, so only RGB is passed on; and its packed mode checks the
    # background without the camera axis, so it renders unpacked (one camera per call, little extra memory).
    import models.render as render
    rasterize = render.rasterization

    def rasterize_rgb_bg(*a, backgrounds=None, render_mode="RGB", **k):
        if backgrounds is not None and render_mode == "RGB+D" and backgrounds.shape[-1] == 4:
            backgrounds = backgrounds[..., :3]
        return rasterize(*a, backgrounds=backgrounds, render_mode=render_mode, packed=False, **k)
    render.rasterization = rasterize_rgb_bg
    system = app.GenerationSystem(ckpt_path=hf_hub_download("imlixinyang/FlashWorld", "model.ckpt"),
                                  device=torch.device("cuda"), offload_t5=True, offload_vae=offload_vae,
                                  offload_transformer_during_vae=True)
    text_cache, encode = {}, system.encode_text

    def encode_cached(texts):
        if tuple(texts) not in text_cache:
            text_cache[tuple(texts)] = encode(texts)
        return text_cache[tuple(texts)]
    system.encode_text = encode_cached
    times = {}
    for req_path in sorted((DATA / "in").glob("*__*.json")):
        name, move = req_path.stem.split("__")
        if only and name not in only:
            continue
        out = DATA / "out" / name / move
        if (out / "c.jpg").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        req = json.loads(req_path.read_text())
        image = torch.from_numpy(np.array(Image.open(req["image"]).convert("RGB"))).float().permute(2, 0, 1) / 255 * 2 - 1
        cams = torch.tensor([c["quaternion"] + c["position"] + [c["fx"] / W, c["fy"] / H, c["cx"] / W, c["cy"] / H]
                             for c in req["cameras"]], dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        params, _, t_norm = system.generate(cams, N_FRAME, image, req["text_prompt"], req["image_index"], H, W,
                                            video_path=str(out / "video.mp4"))
        times[f"{name}/{move}"] = dt = time.time() - t0
        export_gaussians(params.detach().cpu(), opacity_threshold=0.0, T_norm=t_norm, ply_path=str(out / "gaussians.ply"))
        fr = iio.imread(out / "video.mp4", plugin="pyav")
        for tag, k in (("a", 0), ("b", len(fr) // 2), ("c", len(fr) - 1)):
            Image.fromarray(fr[k]).save(out / f"{tag}.jpg", quality=92)
        print(f"{name} {move}: {dt:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    (DATA / "out" / "times.json").write_text(json.dumps(times, indent=1))


def sheet(cell: int = 360):
    """Per case one JPG: row 1 the photo and photo2scene's scene from the photo's camera; row 2 photo2scene's first
    new views; then Lyra's six moves (if fetched) and FlashWorld's, half way and at the end. Writes
    data/flashworld_test/sheet_<case>.jpg."""
    sys.path.insert(0, str(ROOT / "experiments" / "lyra1"))
    from lyra1_test import CASES
    base, lyra = ROOT / "data" / "lyra1_test" / "baseline", ROOT / "data" / "lyra1_test" / "out" / "lyra_out"
    for name in CASES:
        b, fw = base / name, DATA / "out" / name
        if not fw.exists():
            print(f"{name}: no FlashWorld renders yet")
            continue
        rows = [("photo / photo2scene 10-15",
                 [Image.open(b / "photo.jpg")] + ([Image.open(b / "scene.jpg")] if (b / "scene.jpg").exists() else [])),
                ("photo2scene new views", [Image.open(p) for p in sorted(b.glob("view_*.jpg"))[:6]])]
        lf = next(lyra.rglob(f"frames/{name}*"), None) if lyra.exists() else None
        for tag, when in (("b", "half way"), ("c", "end")):
            if lf is not None:
                rows.append((f"Lyra 1 moves, {when}", [Image.open(p) for p in sorted(lf.glob(f"v*_{tag}.jpg"))]))
        for tag, when in (("b", "half way"), ("c", "end")):
            rows.append((f"FlashWorld moves ({', '.join(MOVES)}), {when}",
                         [Image.open(fw / m / f"{tag}.jpg") for m in MOVES if (fw / m / f"{tag}.jpg").exists()]))
        th = int(cell * 9 / 16)
        canvas = Image.new("RGB", (cell * 6, (th + 18) * len(rows)), "white")
        dr = ImageDraw.Draw(canvas)
        for r, (label, row) in enumerate(rows):
            dr.text((4, r * (th + 18) + 2), label, fill="black")
            for c, im in enumerate(row[:6]):
                canvas.paste(ImageOps.fit(im.convert("RGB"), (cell, th)), (c * cell, r * (th + 18) + 18))
        p = DATA / f"sheet_{name}.jpg"
        canvas.save(p, quality=88)
        print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "run", "sheet"])
    ap.add_argument("--case", action="append", help="run: only these cases (default: all)")
    ap.add_argument("--offload-vae", action="store_true", help="under 10 GB of VRAM but ~10 min per generation")
    a = ap.parse_args()
    if a.step == "prep":
        prep()
    elif a.step == "run":
        run(a.offload_vae, a.case)
    else:
        sheet()


if __name__ == "__main__":
    main()
