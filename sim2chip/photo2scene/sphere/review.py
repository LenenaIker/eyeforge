"""S.6 of the sphere route, review: a vision model looks at each extrapolated view with its label boxes and rates it
as training data, as bg_review.py does for one photo's scenes (on HOPE, unreviewed scenes made the student worse
than real photos alone). Per view, not per scene: one bad placement should cost that view only. The model runs on
this machine by default (llm.LOCAL_VLM, Cosmos Reason 2), no cloud spend.

Views without labelled targets are kept unrated: they are negative examples, and the model scored every one of
them 0 for having no car (2026-10-17). Writes <dir>/render/review.json: {"views": {name: {"score", "worst",
"keep"}}, "model"}.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

PROMPT = """We generate synthetic training images for a small detector of {label}s. Each image is a real street \
photo with 3D {label}s rendered into it; red boxes are the labels the detector will learn from.

Synthetic {label}s never look fully real: slightly flat lighting or plain textures cost little. Judge only what \
would teach the detector wrong things: a {label} floating in the air, sunk into the ground, inside or through a \
building or wall, standing where no {label} could be, far too big or too small for its place, broken or \
see-through shapes, a toy or cartoon look, a red box around something that is not a {label}, or a visible {label} \
without a box.

Rate the image as training data from 0 (useless, would hurt the detector) to 10 (as useful as a real photo). Answer \
with one JSON object, nothing else:
{{"score": <0-10>, "worst": ["<the defects that cost most, short phrases>", ...]}}"""
MIN_SCORE = 3   # as bg_review: under it a view is left out of training


def boxed(img_path: Path, boxes: list[dict], out: Path) -> Path:
    im = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for b in boxes:
        dr.rectangle(b["box"], outline=(255, 0, 0), width=3)
    im.save(out, quality=90)
    return out


def review_views(d: Path, label: str, model: str | None = None, min_score: float = MIN_SCORE) -> dict:
    """Rate every view of <d>/render that has labels (render/labels.json) and a final frame."""
    from ...llm import LOCAL_VLM, Client, image_part, parse_json
    d = Path(d)
    model = model or LOCAL_VLM
    labels = json.loads((d / "render" / "labels.json").read_text())
    client = Client()
    tmp = d / "render" / "review"
    tmp.mkdir(exist_ok=True)
    out = {}
    for name, boxes in sorted(labels.items()):
        frame = d / "render" / f"final_{name}.png"
        if not frame.exists():
            continue
        if not boxes:  # a view without targets is a negative example: nothing in it to judge as a target
            out[name] = {"score": None, "worst": [], "keep": True, "note": "no targets"}
            continue
        path = boxed(frame, boxes, tmp / f"{name}.jpg")
        msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(label=label)}, image_part(path)]}]
        answer = None
        for _ in range(2):  # one retry on a bad or cut answer
            try:
                v = parse_json(client.chat(msg, model, purpose="sphere_review", temperature=0.0, max_tokens=4096))
            except ValueError:
                continue
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
                answer = v
                break
        out[name] = {"score": None, "worst": [], "keep": True, "note": "no usable answer"} if answer is None else \
            {"score": answer["score"], "worst": [str(x) for x in answer.get("worst") or []],
             "keep": answer["score"] >= min_score}
    (d / "render" / "review.json").write_text(json.dumps({"views": out, "model": model}, indent=1))
    return out
