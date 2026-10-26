# WorldStereo's run_multi_traj.py with three patches, run by job.sh inside the image (torchrun): the text encoder
# waits on the CPU, the DMD checkpoint loads without the base transformer, and legs upstream's order does not
# know (next-best-view legs) run after its own, a leg with chain_from.txt starting on another leg's last frame.
import runpy, sys, torch
sys.path.insert(0, ".")
from models.worldstereo_wrapper import WorldStereo
load = WorldStereo.from_pretrained.__func__


def from_pretrained(cls, *a, **k):
    ws = load(cls, *a, **k)
    # (the DMD pipeline already returns its own device from its _execution_device() method)
    pipe, dev = ws.pipeline, torch.device(k["device"])
    te = getattr(pipe.text_encoder, "_orig_mod", pipe.text_encoder).to("cpu")
    forward = te.forward

    def lent(*args, **kw):
        te.to(dev)
        try:
            return forward(*args, **kw)
        finally:
            te.to("cpu")
            torch.cuda.empty_cache()
    te.forward = lent
    pipe.text_encoder = te
    torch.cuda.empty_cache()
    print(f"text encoder on cpu; gpu allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return ws


WorldStereo.from_pretrained = classmethod(from_pretrained)
# Upstream builds the transformer from the base model's 65 GB and then overwrites every key with the DMD
# checkpoint (no missing keys, 2026-10-22). Diffusers gets the base config next to the DMD file instead, so only
# that file is read; the ControlNet keys it ignores are loaded by upstream right after, as before.
from pathlib import Path
from unittest import mock
from huggingface_hub import hf_hub_download
import models.worldstereo_wrapper as wsw
load_transformer = WorldStereo._load_transformer


def _load_transformer(cfg, model_type, weights_path, **kw):
    if "dmd" not in model_type:
        return load_transformer(cfg, model_type, weights_path, **kw)
    model = wsw.WorldStereoRefSModel
    upstream_from_pretrained = model.from_pretrained

    def from_dmd(base, subfolder=None, **k):
        conf = Path(base, subfolder, "config.json")
        conf = conf if conf.exists() else hf_hub_download(base, "config.json", subfolder=subfolder)
        d = Path("/tmp/wan_dmd") / subfolder
        d.mkdir(parents=True, exist_ok=True)
        for name, src in (("config.json", conf), ("diffusion_pytorch_model.safetensors", weights_path)):
            (d / name).unlink(missing_ok=True)
            (d / name).symlink_to(src)
        return upstream_from_pretrained(str(d.parent), subfolder=subfolder, **k)
    with mock.patch.object(model, "from_pretrained", from_dmd):
        return load_transformer(cfg, model_type, weights_path, **kw)


WorldStereo._load_transformer = staticmethod(_load_transformer)
# Legs upstream's order does not know (ws_nbv.py's nbv_*, spin_*) run last, after every leg it does know is in
# memory. A leg with chain_from.txt starts on the last frame that leg generated (a 360 in quarter turns).
import glob, os
from PIL import Image
import src.data_utils as du
upstream_sort, upstream_load = du.sort_trajs, du.load_mutli_traj_dataset
def sort_trajs(scene_path):
    known = upstream_sort(scene_path)
    extra = sorted(p for p in glob.glob(f"{scene_path}/*/traj0/render.mp4") if p not in known)
    print("extra legs after upstream's:", [os.path.dirname(p) for p in extra], flush=True)
    return known + extra
def load_mutli_traj_dataset(cfg, input_path, output_path, view_id, traj_id, device, ref_index=None, model_type=None,
                            task_type="panorama"):
    chain = f"{input_path}/{view_id}/chain_from.txt"
    if os.path.exists(chain):
        prev = open(chain).read().strip()
        Image.fromarray(du.get_last_video_frame(f"{output_path}/{prev}/{model_type}_result.mp4")).save(
            f"{input_path}/{view_id}/start_frame.png")
        print(f"{view_id} starts on the last frame of {prev}", flush=True)
    return upstream_load(cfg, input_path, output_path, view_id, traj_id, device, ref_index, model_type, task_type)
du.sort_trajs, du.load_mutli_traj_dataset = sort_trajs, load_mutli_traj_dataset
sys.argv = ["run_multi_traj.py", *sys.argv[1:]]
runpy.run_path("run_multi_traj.py", run_name="__main__")
