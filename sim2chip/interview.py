"""Interview: a conversation with Nemotron (Token Factory) that turns the user's needs into a valid spec.

Every model turn is one JSON action:
  {"say": "..."}                         ask the user something (or explain)
  {"search": ["cow", "cattle"]}          look up 3D model categories (Objaverse-LVIS) and their sizes
  {"spec": {...}}                        propose the full spec
A proposed spec is validated; errors go back to the model. A valid spec is summarized and saved only
after the user confirms it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from . import hardware, spec as spec_mod
from .scenes import TEMPLATES
from .scenes.skies import SKIES

SYSTEM = """You are Sim2Chip's setup assistant. Sim2Chip trains a tiny image classifier for low-power hardware
(for example an ESP32-CAM) from synthetic images rendered with 3D models plus a few real photos from the user.
Your job: interview the user and write the job spec. Be brief and friendly, ask one or two questions at a time,
in the user's language.

Find out:
- What the camera must recognize (2 to 5 classes work best on a microcontroller) and what counts as "nothing".
  Suggest merging classes the hardware cannot tell apart at low resolution.
- Where the camera is: environment, mounting height, typical distance, lighting and weather.
- Target hardware (known profiles: {hardware}).
- Whether they have photos: labeled (one folder per class) or raw (unlabeled). Ask for the folder path.
  A few real photos improve accuracy a lot compared with synthetic images only; say so.
- Real-world size of each class (you propose typical sizes, the user confirms).

3D models come from Objaverse-LVIS categories. Before writing classes, use "search" to find categories and
check they have enough models (at least ~10). Only use category names returned by search.
Scene templates available: {templates}. Weather options: {weather}.

Spec format (JSON, defaults shown; omit a field to keep its default):
{schema}
Notes: size_m.measure is "length" (long side from above), "width" (short side from above) or "height";
choose the one that varies least across the class. assets.count is the number of models per class.
user_data.labeled / user_data.raw are folder paths; class_map maps each class id to the subfolder names of
the labeled folder. Keep qa.mode "auto".

Reply ONLY with one JSON object per turn: {{"say": "..."}} or {{"search": ["..."]}} or {{"spec": {{...}}}}."""

EXAMPLE = {"name": "example", "classes": [{"id": "cow", "description": "cows seen from the side or front",
                                           "size_m": {"measure": "length", "range": [2.0, 2.5]},
                                           "assets": {"source": "objaverse_lvis", "categories": ["cow"],
                                                      "count": 20}}],
           "qa": {"mode": "auto"}}


def system_prompt() -> str:
    schema = json.dumps(spec_mod.to_dict(spec_mod.from_dict(EXAMPLE)), indent=1)
    return SYSTEM.format(hardware=", ".join(hardware.PROFILES), templates=", ".join(TEMPLATES),
                         weather=", ".join(sorted({s.weather for s in SKIES})), schema=schema)


class CategoryIndex:
    """Objaverse-LVIS category names with their model counts, searched by substring."""

    def __init__(self, counts: dict[str, int] | None = None, cache_dir: Path | None = None):
        if counts is None:
            import objaverse
            if cache_dir:
                objaverse.BASE_PATH = str(cache_dir)
                objaverse._VERSIONED_PATH = str(cache_dir / "hf-objaverse-v1")
            counts = {k: len(v) for k, v in objaverse.load_lvis_annotations().items()}
        self.counts = counts

    def search(self, queries: list[str], limit: int = 12) -> dict[str, int]:
        hits: dict[str, int] = {}
        for q in queries:
            q = q.lower().strip().replace(" ", "_")
            # Plural forms too: "cows" -> "cow", "buses" -> "bus".
            variants = {q, q[:-1] if q.endswith("s") else q, q[:-2] if q.endswith("es") else q} - {""}
            for name, n in self.counts.items():
                base = name.split("_(")[0]
                if any(v in name or (len(base) >= 4 and base in v) for v in variants):
                    hits[name] = n
        return dict(sorted(hits.items(), key=lambda kv: -kv[1])[:limit])


def summary(s: spec_mod.Spec) -> str:
    lines = [f"Spec '{s.name}' for {s.hardware}:",
             f"  classes: {', '.join(s.labels)} (background: {s.background_class})"]
    for c in s.classes:
        size = f"{c.size_m.measure} {c.size_m.range[0]}-{c.size_m.range[1]} m" if c.size_m else "no 3D size"
        cats = ", ".join(c.assets.categories) if c.assets else "no 3D models"
        lines.append(f"  - {c.id}: {c.description} | {size} | models: {cats}")
    lines.append(f"  camera: {s.camera.height_m[0]}-{s.camera.height_m[1]} m high, scene {s.scene.template}, "
                 f"weather {', '.join(s.scene.weather)}")
    ud = s.user_data
    photos = [f"labeled {ud.labeled}" if ud.labeled else "", f"raw {ud.raw}" if ud.raw else ""]
    lines.append(f"  photos: {', '.join(p for p in photos if p) or 'none (synthetic only)'}")
    return "\n".join(lines)


def run(client, model: str, categories: CategoryIndex, ask: Callable[[str], str], tell: Callable[[str], None],
        max_turns: int = 60) -> spec_mod.Spec | None:
    from .llm import parse_json

    messages = [{"role": "system", "content": system_prompt()}]
    first = ask("Describe what your camera should see: ")
    messages.append({"role": "user", "content": first})
    for _ in range(max_turns):
        text = client.chat(messages, model, purpose="interview", temperature=0.3, max_tokens=4096)
        messages.append({"role": "assistant", "content": text})
        try:
            action = parse_json(text)
        except ValueError:
            messages.append({"role": "user", "content": "Reply with exactly one JSON object as instructed."})
            continue
        if "say" in action:
            messages.append({"role": "user", "content": ask(action["say"] + "\n> ")})
        elif "search" in action:
            found = categories.search(action["search"] if isinstance(action["search"], list) else [action["search"]])
            messages.append({"role": "user", "content": "search results (category: number of models): "
                             + json.dumps(found or "nothing found, try other words")})
        elif "spec" in action:
            try:
                s = spec_mod.from_dict(action["spec"])
                unknown = [cat for c in s.classes if c.assets for cat in c.assets.categories
                           if cat not in categories.counts]
                if unknown:
                    raise spec_mod.SpecError(f"categories not found in Objaverse-LVIS: {unknown}, use search")
            except spec_mod.SpecError as e:
                messages.append({"role": "user", "content": f"The spec is invalid: {e}. Fix it and send it again."})
                continue
            tell(summary(s))
            answer = ask("Is this right? (yes, or tell me what to change) > ")
            if answer.strip().lower() in ("y", "yes", "s", "si", "sí", "ok"):
                return s
            messages.append({"role": "user", "content": f"The user wants changes: {answer}"})
        else:
            messages.append({"role": "user", "content": "Unknown action. Use say, search or spec."})
    tell("The interview did not converge; nothing was saved.")
    return None
