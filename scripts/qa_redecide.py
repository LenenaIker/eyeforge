"""Rerun only the Nemotron decision of describe_then_decide on saved descriptions (to tune the prompt cheaply).

    python scripts/qa_redecide.py <job dir> <verdicts with descriptions.json> <out.json>
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sim2chip import spec as spec_mod
from sim2chip.llm import Client, parse_json
from sim2chip.stages.qa_assets import DECIDE, PROBLEMS


def main(job: Path, src: Path, out: Path) -> None:
    spec = spec_mod.load(job / "spec.json")
    assets = {a["uid"]: a for a in json.loads((job / "convert_assets" / "assets.json").read_text())}
    old = json.loads(src.read_text())
    client = Client(usage_log=job / "logs" / "llm_usage_redecide.jsonl")
    model = client.model_for("text")
    classes = "\n".join(f"- {c.id}: {c.description or c.id}" for c in spec.classes)
    ids = ", ".join(c.id for c in spec.classes)

    def one(uid):
        v = old.get(uid)
        if not v or "description" not in v:
            return uid, None
        a = assets[uid]
        text = DECIDE.format(classes=classes, category=a.get("category", ""), name=a.get("name", ""), cls=a["class"],
                             ids=ids, problems=PROBLEMS, description=json.dumps(v["description"]))
        for _ in range(2):
            try:
                r = parse_json(client.chat([{"role": "user", "content": text}], model, purpose="qa_decide",
                                           temperature=0.0, max_tokens=4096))
                if r.get("decision") in ("keep", "discard"):
                    return uid, {**r, "description": v["description"]}
            except ValueError:
                pass
        return uid, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        out.write_text(json.dumps(dict(pool.map(one, assets)), indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
