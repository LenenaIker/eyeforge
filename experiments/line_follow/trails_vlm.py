"""Vision-language models of Token Factory as path-direction teachers (IDSIA forest trails, or road frames).

    python experiments/line_follow/trails_vlm.py FRAMES OUT [--models a,b] [--workers 8] [--scene trail|road]

Each model gets each frame and answers left / straight / right (where the hiker must turn to keep on the trail),
the same three classes as the dataset (-1 TL, 0 GS, +1 TR). Answers are cached per model in OUT/<model>.json;
the summary has accuracy, accuracy per class, "straight or not", unanswered frames, seconds and dollars.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, image_part, parse_json  # noqa: E402

MODELS = ["moonshotai/Kimi-K2.6", "zai-org/GLM-5.3-Flash", "google/gemma-3-27b-it", "openbmb/MiniCPM-V-4_5",
          "Qwen/Qwen3.8-27B", "MiniMaxAI/MiniMax-M3"]
PROMPTS = {
    "trail": """This photo comes from a camera worn on the head of a hiker who wants to keep walking along a trail.
Where is the trail, compared with the direction the camera looks?
- "left": the trail heads towards the left of the picture (or is off to the left), so the hiker must turn left
- "straight": the trail goes ahead through the middle of the picture
- "right": the trail heads towards the right of the picture (or is off to the right), so the hiker must turn right
Reply with one JSON object: {"direction": "left" | "straight" | "right"}""",
    "road": """This photo comes from a camera on the dashboard of a car that keeps driving along its road.
Where does the road ahead go, so where must the driver steer now?
- "left": the road bends to the left, so the driver steers left
- "straight": the road goes straight ahead, so the driver keeps the wheel straight
- "right": the road bends to the right, so the driver steers right
Reply with one JSON object: {"direction": "left" | "straight" | "right"}""",
}
VALUE = {"left": -1, "straight": 0, "right": 1}


class Counting(Client):
    """Client that adds up tokens and seconds instead of writing a usage log."""

    def __init__(self):
        super().__init__()
        self.tokens = [0, 0]
        self.seconds = 0.0

    def _log(self, model, purpose, usage, latency):
        self.tokens[0] += usage.get("prompt_tokens") or 0
        self.tokens[1] += usage.get("completion_tokens") or 0
        self.seconds += latency


def ask(client: Client, model: str, frame: Path, prompt: str) -> int | None:
    for _ in range(2):
        try:
            text = client.chat([{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(frame)]}],
                               model, purpose="trail_teacher", temperature=0.0, max_tokens=2048, timeout=90,
                               retries=2)
            v = parse_json(text)
            d = str(v.get("direction", "")).lower().strip() if isinstance(v, dict) else ""
            if d in VALUE:
                return VALUE[d]
        except Exception:  # an error or a broken answer: try once more, then leave the frame unanswered
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scene", choices=sorted(PROMPTS), default="trail")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    labels = json.loads((a.frames / "labels.json").read_text())
    names = sorted(labels)
    prices = Client().prices()
    summary_path = a.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for model in a.models.split(","):
        cache = a.out / (model.replace("/", "__") + ".json")
        answers = json.loads(cache.read_text()) if cache.exists() else {}
        todo = [n for n in names if n not in answers]
        client = Counting()
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            for n, v in zip(todo, ex.map(lambda n: ask(client, model, a.frames / n, PROMPTS[a.scene]), todo)):
                answers[n] = v
        cache.write_text(json.dumps(answers))
        p = prices.get(model, {"input": 0, "output": 0})
        y = [labels[n] for n in names]
        pred = [answers[n] for n in names]
        ok = [q == t for q, t in zip(pred, y)]
        row = {"acc": round(sum(ok) / len(ok), 3),
               "per_class": {c: round(sum(q == c for q, t in zip(pred, y) if t == c) / y.count(c), 3)
                             for c in (-1, 0, 1)},
               "gs_vs_not": round(sum((q == 0) == (t == 0) for q, t in zip(pred, y)) / len(y), 3),
               "unanswered": sum(q is None for q in pred)}
        if todo:
            row |= {"usd_per_1000": round((client.tokens[0] * p["input"] + client.tokens[1] * p["output"])
                                          / len(todo) * 1000, 3),
                    "s_per_frame": round(client.seconds / len(todo), 2), "wall_min": round((time.time() - t0) / 60, 1)}
        summary[model] = {**summary.get(model, {}), **row}
        print(model, summary[model], flush=True)
        summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
