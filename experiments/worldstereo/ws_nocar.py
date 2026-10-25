"""Tests N1/N2 of docs/pruebas_reconstruccion_worldstereo.md: the movable objects (cars, people) taken out before
WorldStereo has to invent them from new angles, so the scene it walks is only background (the cars would come back
as 3D objects).

N1 erased them from the panorama (sphere route: capture.prepare, steps/instances.py, steps/erase.py). N2 erases
them from the photo, before WorldGen, so the eraser works on a plain photo and WorldGen's prompt no longer asks for
the SUV. This script prepares the photo for steps/erase_qwen.py (run in SIM2CHIP_SAM3_PYTHON):

    python experiments/worldstereo/ws_nocar.py masks --case photo21
    <SIM2CHIP_ERASE_PYTHON> sim2chip/photo2scene/steps/erase_qwen.py --dir data/worldstereo_test/nocar_photo/photo21

Writes data/worldstereo_test/nocar_photo/<case>/: photo.png, instances.npz (masks, phrases), prepare.json and
erased.png (the union of the masks grown by GROW px, for the contact shadows), the inputs erase_qwen.py reads."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
PHRASES = ("car", "van", "truck", "person")
THRESHOLD = 0.5
GROW = 15


def masks(case: str):
    import torch
    from transformers import Sam3Model, Sam3Processor
    out = ROOT / "data" / "worldstereo_test" / "nocar_photo" / case
    out.mkdir(parents=True, exist_ok=True)
    photo = Image.open(ROOT / "data" / "panogen_test" / "in" / f"{case}.png").convert("RGB")
    photo.save(out / "photo.png")
    model = Sam3Model.from_pretrained("facebook/sam3").cuda().eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    found, phrases = [], []
    for phrase in PHRASES:
        inp = proc(images=photo, text=phrase, return_tensors="pt").to("cuda")
        with torch.no_grad():
            res = model(**inp)
        r = proc.post_process_instance_segmentation(res, threshold=THRESHOLD, mask_threshold=0.5,
                                                    target_sizes=inp.get("original_sizes").tolist())[0]
        for s, m in zip(r["scores"].tolist(), r["masks"].cpu().numpy().astype(bool)):
            found.append(m)
            phrases.append(phrase)
            print(phrase, round(s, 2), f"{m.mean():.3f} of the photo", flush=True)
    union = np.any(found, 0) if found else np.zeros(photo.size[::-1], bool)
    grown = Image.fromarray(union.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * GROW + 1))
    grown.save(out / "erased.png")
    np.savez_compressed(out / "instances.npz", masks=np.stack(found) if found else np.zeros((0, *union.shape), bool),
                        phrases=np.array(phrases))
    (out / "prepare.json").write_text(json.dumps({"instances": phrases}))
    print(f"{len(found)} instances, {union.mean():.3f} of the photo -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["masks"])
    ap.add_argument("--case", default="photo21")
    a = ap.parse_args()
    masks(a.case)
