"""SigLIP 2 embeddings of photos and sentences, in the Python that has it (torch + transformers). The label stage
runs it for classification (teacher.py).

    python -m sim2chip.siglip_worker < request.json

Request: {"model", "items": [{"path", "out"}], "texts": [sentence, ...], "text_out": path}.
Each photo's unit-length embedding goes to its `out` file: {"embedding": [...]}, or {"error": ...} when the
photo is broken. The sentences' embeddings go to `text_out` as {"embeddings": [[...], ...]} in their order.
One progress line per photo on stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def features(out):
    """transformers 5 returns an output object where 4 returned the tensor."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def main() -> int:
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoModel, AutoProcessor

    req = json.load(sys.stdin)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(req["model"]).to(device).eval()
    proc = AutoProcessor.from_pretrained(req["model"])
    unit = lambda e: torch.nn.functional.normalize(e.float(), dim=-1)  # noqa: E731
    print(f"SigLIP 2 on {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}", flush=True)
    if req.get("texts"):
        x = proc(text=req["texts"], padding="max_length", max_length=64, truncation=True,
                 return_tensors="pt").to(device)
        with torch.no_grad():
            t = unit(features(model.get_text_features(**x))).cpu()
        Path(req["text_out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(req["text_out"]).write_text(json.dumps({"embeddings": [[round(v, 6) for v in r] for r in t.tolist()]}))
    for i, item in enumerate(req["items"]):
        out = Path(item["out"])
        try:
            with Image.open(item["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            x = proc(images=im, return_tensors="pt").to(device)
            with torch.no_grad():
                e = unit(features(model.get_image_features(**x))).cpu()[0]
            answer = {"embedding": [round(v, 6) for v in e.tolist()]}
        except Exception as e:  # a broken photo costs that photo only
            answer = {"error": f"{type(e).__name__}: {e}"[:300]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(answer))
        print(f"{i + 1}/{len(req['items'])} {Path(item['path']).name}: {answer.get('error') or 'ok'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
