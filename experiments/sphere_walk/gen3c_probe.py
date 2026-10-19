"""GEN3C probe (2026-10-19): what makes a site's batch shorter, measured in one Nebius job.

- One process for every clip: last night each clip ran its own script, and of its ~21 min only ~14 were the
  generation itself; the rest reloaded the 7B model and T5-11B (and the same prompt was embedded every time).
- Fewer diffusion steps (STEPS_FAST against GEN3C's default 35).
- A multi-view cache: the cache holds KEY_YAWS x KEY_PITCHES cuts of the panorama with our own depth, not one 80 deg
  view, so a path can turn and look back and GEN3C paints only what no cut saw. GEN3C's own multi-view selector keeps
  the buffers with the most overlap over a whole chunk; ours picks them per frame (a turning path sees different
  cuts along the way).

Clips: single_fast (last night's p000 path, one cut, STEPS_FAST), multi (a path walking TURN_M out along heading 0
while turning to look back, all the cuts, 35 steps) and multi_fast (the same at STEPS_FAST). Run in edge-sdg:

    python experiments/sphere_walk/gen3c_probe.py plan   --place <place>
    python experiments/sphere_walk/gen3c_probe.py launch --place <place> [--max-hours 2]
    python experiments/sphere_walk/gen3c_probe.py fetch  --place <place>

Writes data/sphere_test/<place>/walk/P_probe/: <cache>.npz (the cache's cuts and the path), paths.json (per clip:
its frames' camera-to-world in Isaac's world, as gen3c_batch's), and after fetch the clips and job.log."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen3c_walk as g  # noqa: E402
from gen3c_batch import GPUS, PROMPT, place_dir  # noqa: E402
from sim2chip.cloud_sync import S3Store  # noqa: E402
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

PREFIX = "gen3c_probe"
STEPS_FAST = 18
KEY_YAWS = tuple(range(0, 360, 60))  # 80 deg views every 60 deg: each overlaps the next
KEY_PITCHES = (0.0, 45.0)            # level views reach 25 deg down; the pitched ones bring the near ground
TURN_M = 4.0

RUNNER = r'''
import json, subprocess, sys, time
import numpy as np, torch
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Base, Cache3D_BufferSelector
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
from cosmos_predict1.utils import misc
from cosmos_predict1.utils.io import save_video
import argparse
torch.enable_grad(False)
FRAMES_PER_RENDER = 11


class PerFrameSelector(Cache3D_BufferSelector):
    """The cache's buffers with the most overlap for each target frame (GEN3C's selector picks them per chunk)."""
    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        out = target_w2cs.device
        w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        ks = target_intrinsics.to(self.weight_dtype).to(self.device)
        px_all, mk_all = [], []
        for f in range(0, w2cs.shape[1], FRAMES_PER_RENDER):
            px, mk = Cache3D_Base.render_cache(self, w2cs[:, f:f + FRAMES_PER_RENDER], ks[:, f:f + FRAMES_PER_RENDER],
                                               render_depth, start_frame_idx)
            k = min(self.frame_buffer_max, px.shape[2])
            idx = mk.sum(dim=(3, 4, 5)).topk(k, dim=2).indices  # B, F, k
            pick = lambda t: torch.gather(t, 2, idx[..., None, None, None].expand(-1, -1, -1, *t.shape[3:]))
            px_all.append(pick(px).cpu())
            mk_all.append(pick(mk).cpu())
            del px, mk
            torch.cuda.empty_cache()
        return torch.cat(px_all, 1).to(out), torch.cat(mk_all, 1).to(out)


p = argparse.ArgumentParser()
add_common_arguments(p)
a = p.parse_args(["--checkpoint_dir", "checkpoints"])
clips = json.load(open(sys.argv[1]))
misc.set_random_seed(a.seed)
t0 = time.time()
pipe = Gen3cPipeline(inference_type="video2world", checkpoint_dir="checkpoints", checkpoint_name="Gen3C-Cosmos-7B",
                     prompt_upsampler_dir="Pixtral-12B", enable_prompt_upsampler=False, offload_network=False,
                     offload_tokenizer=False, offload_text_encoder_model=True, offload_prompt_upsampler=False,
                     offload_guardrail_models=False, disable_guardrail=True, disable_prompt_encoder=False,
                     guidance=1, num_steps=35, height=704, width=1280, fps=24, num_video_frames=121, seed=a.seed)
print("timing load", round(time.time() - t0), flush=True)
embed = pipe._run_text_embedding_on_prompt_with_offload
memo = {}
def embed_once(prompts, **kw):
    key = tuple(prompts)
    if key not in memo:
        memo[key] = embed(prompts, **kw)
    return memo[key]
pipe._run_text_embedding_on_prompt_with_offload = embed_once
dev = torch.device("cuda")
for c in clips:
    t0 = time.time()
    z = np.load(c["npz"])
    f32 = lambda k: torch.tensor(z[k].astype(np.float32), device=dev)
    images = f32("images")[None]
    cache = PerFrameSelector(frame_buffer_max=pipe.model.frame_buffer_max, input_image=images,
                             input_depth=f32("depth")[None], input_mask=f32("mask")[None], input_w2c=f32("w2c")[None],
                             input_intrinsics=f32("K")[None], filter_points_threshold=0.05,
                             input_format=["B", "N", "C", "H", "W"], foreground_masking=True)
    w2cs, ks = f32("w2cs_all")[None], f32("Ks_all")[None]
    warp, wmask = cache.render_cache(w2cs, ks)
    t1 = time.time()
    pipe.num_steps = c["steps"]
    video, _ = pipe.generate(prompt=c["prompt"], image_path=images[:, 0].unsqueeze(2), negative_prompt=a.negative_prompt,
                             rendered_warp_images=warp, rendered_warp_masks=wmask)
    t2 = time.time()
    path = f"/tmp/out/{c['name']}.mp4"
    save_video(video=video, fps=24, H=704, W=1280, video_save_quality=5, video_save_path=path)
    buf = ((warp[0, :, 0].permute(0, 2, 3, 1).float().cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    save_video(video=buf, fps=24, H=704, W=1280, video_save_quality=5, video_save_path=f"/tmp/out/{c['name']}_buffer.mp4")
    print("timing", c["name"], "steps", c["steps"], "cache", round(t1 - t0), "generate", round(t2 - t1), flush=True)
    subprocess.run(["curl", "-s", "-X", "PUT", "--upload-file", path, c["out"]])
    subprocess.run(["curl", "-s", "-X", "PUT", "--upload-file", f"/tmp/out/{c['name']}_buffer.mp4", c["buf"]])
'''

BLOCK = '''
curl -s -o /tmp/in/clips.json "{clips}"
{fetch}
cat > run_clips.py <<'PYEOF'
{runner}
PYEOF
CUDA_HOME=/usr/local/cuda PYTHONPATH=$(pwd) python run_clips.py /tmp/in/clips.json
up
'''


def key_view(place: str, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """One cut of the erased panorama (B_layers' front texture) and its z-depth from our proxy, 0 where unknown."""
    d = place_dir(place)
    R = np.array(json.loads((d / "geometry.json").read_text())["levelling"])
    v = {"yaw_deg": yaw, "pitch_deg": pitch, "fov_deg": g.HFOV, "size": [g.W, g.H]}
    rays_l = capture.view_rays(v)
    rays_p = render_plan.pano_rays(rays_l, R)
    tex = np.asarray(Image.open(d / "walk" / "BL_scene" / "tex_front.png").convert("RGB"))
    img = np.clip(capture.sample(tex, rays_p) + 0.5, 0, 255).astype(np.uint8)
    pd = np.load(d / "proxy_distance.npz")
    dist = np.where(pd["valid"], pd["distance"], 0).astype(np.float32)
    x, y = capture.to_pixel(rays_p, dist.shape[1], dist.shape[0])
    dd = dist[np.clip(np.round(y).astype(int), 0, dist.shape[0] - 1), np.round(x).astype(int) % dist.shape[1]]
    return img, (dd * (rays_l @ capture.rotation(yaw, pitch)[:, 2])).astype(np.float32)


def w2c(yaw: float, pitch: float, pos_level) -> np.ndarray:
    """World to camera in the levelled frame (OpenCV axes, the capture point at the origin): GEN3C's world."""
    m = np.eye(4)
    m[:3, :3] = capture.rotation(yaw, pitch)
    m[:3, 3] = pos_level
    return np.linalg.inv(m).astype(np.float32)


def intrinsics() -> np.ndarray:
    f = capture.focal(g.W, g.HFOV)
    return np.array([[f, 0, g.W / 2 - 0.5], [0, f, g.H / 2 - 0.5], [0, 0, 1]], np.float32)


def cache_npz(place: str, out: Path, name: str, keys: list[tuple[float, float]], path: list[tuple[float, float]]):
    """keys: (yaw, pitch) of each cut, the first one the clip's first frame; path: (yaw, metres along heading 0)."""
    imgs, deps = zip(*[key_view(place, y, p) for y, p in keys])
    dep = np.stack(deps)[:, None]
    fwd = capture.rotation(0, 0)[:, 2]
    np.savez(out / f"{name}.npz",
             images=(np.stack(imgs).transpose(0, 3, 1, 2) / 127.5 - 1).astype(np.float16),
             depth=dep.astype(np.float16), mask=(dep > 0).astype(np.uint8),
             K=np.repeat(intrinsics()[None], len(keys), 0), w2c=np.stack([w2c(y, p, (0, 0, 0)) for y, p in keys]),
             w2cs_all=np.stack([w2c(y, 0, s * fwd) for y, s in path]), Ks_all=np.repeat(intrinsics()[None], len(path), 0))


def plan(place: str) -> list[dict]:
    d = place_dir(place)
    out = d / "walk" / "P_probe"
    out.mkdir(parents=True, exist_ok=True)
    h = json.loads((d / "geometry.json").read_text())["camera_height_m"]
    t = np.linspace(0, 1, g.FRAMES)
    straight = [(0.0, 5.0 * x) for x in t]                              # last night's p000
    turn = [(180.0 * x * x * (3 - 2 * x), TURN_M * x) for x in t]       # out along 0, ending looking back
    cache_npz(place, out, "single", [(0.0, 0.0)], straight)
    cache_npz(place, out, "multi", [(y, p) for p in KEY_PITCHES for y in KEY_YAWS], turn)
    fwd = render_plan.to_world(capture.rotation(0, 0)[:, 2], 0)
    poses = lambda path: [render_plan.camera_pose(g.view(y), h, s * fwd) for y, s in path]  # noqa: E731
    clips = [{"name": f"{place}_single_fast", "cache": "single", "steps": STEPS_FAST, "poses": poses(straight)},
             {"name": f"{place}_multi", "cache": "multi", "steps": 35, "poses": poses(turn)},
             {"name": f"{place}_multi_fast", "cache": "multi", "steps": STEPS_FAST, "poses": poses(turn)}]
    (out / "paths.json").write_text(json.dumps({"place": place, "view": g.view(0), "paths": clips}))
    for c in clips:
        print(c["name"], c["cache"], c["steps"])
    return clips


def launch(place: str, max_hours: float, gpu: str):
    out = place_dir(place) / "walk" / "P_probe"
    clips = json.loads((out / "paths.json").read_text())["paths"]
    store = S3Store()
    exp = max_hours + 2
    url = lambda key, op="get_object": store.s3.generate_presigned_url(  # noqa: E731
        op, Params={"Bucket": g.BUCKET, "Key": key}, ExpiresIn=int(exp * 3600))
    fetch = []
    for name in sorted({c["cache"] for c in clips}):
        store.put(g.BUCKET, f"{PREFIX}/in/{name}.npz", (out / f"{name}.npz").read_bytes())
        fetch.append(f'curl -s -o /tmp/in/{name}.npz "{url(f"{PREFIX}/in/{name}.npz")}"')
    spec = [{"name": c["name"], "npz": f"/tmp/in/{c['cache']}.npz", "steps": c["steps"], "prompt": PROMPT,
             "out": url(f"{PREFIX}/out/{c['name']}.mp4", "put_object"),
             "buf": url(f"{PREFIX}/out/{c['name']}_buffer.mp4", "put_object")} for c in clips]
    store.put(g.BUCKET, f"{PREFIX}/in/clips.json", json.dumps(spec).encode())
    script = out / "job.sh"
    script.write_text(g.JOB.replace("__CASES__", BLOCK.format(clips=url(f"{PREFIX}/in/clips.json"),
                                                             fetch="\n".join(fetch), runner=RUNNER)))
    assert script.stat().st_size < 60_000, "the script must fit Nebius' 64 KiB inject limit"
    platform, preset = GPUS[gpu]
    cli = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
    args = [cli, "ai", "job", "create", "--name", f"eyeforge-gen3c-probe-{int(time.time())}", "--image", g.IMAGE,
            "--platform", platform, "--preset", preset, "--on-demand", "--timeout", f"{int(max_hours * 60)}m",
            "--parent-id", os.environ["SIM2CHIP_NEBIUS_PROJECT"], "--inject-file", f"{script}:/opt/gen3c_job.sh",
            "--container-command", "bash", "--args", "/opt/gen3c_job.sh", "--env",
            f"LOG_URL={url(f'{PREFIX}/out/job.log', 'put_object')}", "--async"]
    r = subprocess.run(args, capture_output=True, text=True)
    print(r.stdout[-400:], r.stderr[-600:])


def fetch(place: str):
    out = place_dir(place) / "walk" / "P_probe"
    store = S3Store()
    for k in store.listing(g.BUCKET, f"{PREFIX}/out/"):
        (out / k.rsplit("/", 1)[-1]).write_bytes(store.s3.get_object(Bucket=g.BUCKET, Key=k)["Body"].read())
        print(out / k.rsplit("/", 1)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "launch", "fetch"])
    ap.add_argument("--place", required=True)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--gpu", choices=list(GPUS), default="l40s")
    a = ap.parse_args()
    {"plan": lambda: plan(a.place), "launch": lambda: launch(a.place, a.max_hours, a.gpu),
     "fetch": lambda: fetch(a.place)}[a.step]()


if __name__ == "__main__":
    main()
