"""Image to 3D: turn a photo of an object into a textured, light GLB (runs in the `trellis` runtime).

An open-vocabulary detector (OWLv2, Apache 2.0) finds the object named by a text query and crops it,
BiRefNet (MIT) cuts it out and TRELLIS v1 (MIT) generates the mesh. Feasibility test and timings in
spike/NOTES.md ("Imagen a 3D con TRELLIS v1").

License note: TRELLIS bakes the texture with nvdiffrast and a Gaussian rasterizer (mip-splatting), both
non-commercial. Replace that export before any commercial use (TODO.md, 3.E).

Environment: SIM2CHIP_TRELLIS_DIR is the TRELLIS checkout (environment or .env); model weights come from the
Hugging Face cache.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DETECTOR = "google/owlv2-base-patch16-ensemble"
CUTOUT = "ZhengPeng7/BiRefNet"
IMAGE_TO_3D = "microsoft/TRELLIS-image-large"
MIN_SCORE = 0.1  # detections below this are not the object


@dataclass
class Detection:
    score: float
    box: list[float] | None  # x0, y0, x1, y1 in image pixels


class ImageTo3D:
    """Loads the three models once; each stays on the GPU only while it runs (~10 GB peak)."""

    def __init__(self):
        import sys

        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation, Owlv2ForObjectDetection, Owlv2Processor

        from .config import env
        trellis = env("SIM2CHIP_TRELLIS_DIR")
        if not trellis:
            raise RuntimeError("set SIM2CHIP_TRELLIS_DIR to the TRELLIS checkout (environment or .env)")
        sys.path.insert(0, trellis)
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils

        self._post = postprocessing_utils
        self.processor = Owlv2Processor.from_pretrained(DETECTOR)
        self.detector = Owlv2ForObjectDetection.from_pretrained(DETECTOR).eval()
        self.cutout = AutoModelForImageSegmentation.from_pretrained(CUTOUT, trust_remote_code=True).float().eval()
        self.cutout_tf = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(IMAGE_TO_3D)
        self.pipeline.cuda()

    def detect(self, image, query: str) -> Detection:
        """Best box for `query`."""
        import torch

        self.detector.cuda()
        inputs = self.processor(text=[[query]], images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.detector(**inputs)
        self.detector.cpu()
        side = max(image.size)  # OWLv2 pads to a square: boxes are relative to the padded side
        res = self.processor.post_process_grounded_object_detection(
            outputs, threshold=0.05, target_sizes=torch.tensor([[side, side]], device="cuda"))[0]
        if len(res["scores"]) == 0:
            return Detection(0.0, None)
        i = int(res["scores"].argmax())
        return Detection(float(res["scores"][i]), res["boxes"][i].tolist())

    def cut_out(self, image):
        """RGBA image with the background removed."""
        import torch
        from torchvision import transforms

        self.cutout.cuda()
        with torch.no_grad():
            pred = self.cutout(self.cutout_tf(image).unsqueeze(0).cuda())[-1].sigmoid().cpu()[0, 0]
        self.cutout.cpu()
        out = image.copy()
        out.putalpha(transforms.ToPILImage()(pred).resize(image.size))
        return out

    def generate(self, image, det: Detection, seed: int, glb_path: str, margin: float = 0.15):
        """Crop, cut out and build the mesh. Returns the image TRELLIS saw (for human review)."""
        import torch

        crop = image
        if det.box:
            x0, y0, x1, y1 = det.box
            cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * (1 + 2 * margin) / 2
            crop = image.crop([int(max(0, cx - half)), int(max(0, cy - half)),
                               int(min(image.width, cx + half)), int(min(image.height, cy + half))])
        prep = self.pipeline.preprocess_image(self.cut_out(crop))
        torch.cuda.empty_cache()
        outputs = self.pipeline.run(prep, seed=seed, formats=["mesh", "gaussian"], preprocess_image=False)
        glb = self._post.to_glb(outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024,
                                verbose=False)
        glb.export(glb_path)
        del outputs, glb
        torch.cuda.empty_cache()
        return prep
