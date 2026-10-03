"""Cheap Token Factory VLMs as a path or no-path gate before Molmo2: one photo, answer 1 (a path or road to follow)
or 0, at temperature 0. Compared with SigLIP 2 zero-shot (path_gate.py) on the same photos.

    python experiments/line_follow/path_gate_vlm.py PHOTOS OUT.json [--models a,b]
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

PROMPT = """Is there a path, street, road, sidewalk, bike lane or trail in this photo that the camera could follow
ahead? Answer 1 if yes, 0 if there is nothing to follow (only grass, bushes, a wall, an open field).
Reply with one JSON object: {"path": 1} or {"path": 0}"""
MODELS = ["zai-org/GLM-5.3-Flash", "openbmb/MiniCPM-V-4_5"]


def ask(client, model, photo):
    for _ in range(2):
        try:
            t = time.time()
            v = parse_json(client.chat([{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                                     image_part(photo)]}],
                                       model, purpose="path_gate", temperature=0.0, max_tokens=1024, timeout=90,
                                       retries=2))
            if isinstance(v, dict) and v.get("path") in (0, 1, "0", "1", True, False):
                return int(v["path"]), round(time.time() - t, 2)
        except Exception:
            pass
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    photos = sorted(a.photos.glob("*.jpg"))
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    client = Client()
    for m in a.models.split(","):
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(lambda p: ask(client, m, p), photos))
        res[m] = {p.name: g[0] for p, g in zip(photos, got)}
        secs = [g[1] for g in got if g[1] is not None]
        print(f"{m}: path {sum(v == 1 for v in res[m].values())}, no path {sum(v == 0 for v in res[m].values())}, "
              f"unanswered {sum(v is None for v in res[m].values())}, {sum(secs) / max(1, len(secs)):.1f} s a photo")
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
