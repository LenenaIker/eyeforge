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
from dataclasses import dataclass
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


YES = {"y", "yes", "yeah", "yep", "ok", "okay", "sure", "correct", "right", "s", "si", "sí", "vale", "claro",
       "correcto", "perfecto", "adelante", "bien", "de acuerdo"}
NO = {"no", "not", "but", "pero", "cambia", "change", "falta", "missing"}


def is_yes(answer: str) -> bool:
    """A short affirmative answer ("Sí, adelante.", "ok!", "yes please") without a change request."""
    import re
    words = re.findall(r"[a-záéíóúñ]+", answer.lower())
    if not words or len(words) > 6 or any(w in NO for w in words):
        return False
    text = " ".join(words)
    return words[0] in YES or text in YES


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


FIRST_QUESTION = "Describe what your camera should see / Describe qué debe ver tu cámara:"
CONFIRM = "Is this right? / ¿Es correcto? (yes/sí, or tell me what to change / o dime qué cambiar)"


@dataclass
class Turn:
    kind: str                       # "say" (a question), "confirm" (spec proposed), "done" or "failed"
    text: str
    spec: spec_mod.Spec | None = None


class Session:
    """One interview, driven one user message at a time (terminal, web or any other front end)."""

    def __init__(self, client, model: str, categories: CategoryIndex, max_turns: int = 60):
        self.client, self.model, self.categories, self.max_turns = client, model, categories, max_turns
        self.messages = [{"role": "system", "content": system_prompt()}]
        self.proposed: spec_mod.Spec | None = None
        self.finished = False

    def reply(self, user_text: str) -> Turn:
        if self.finished:
            raise RuntimeError("the interview is over")
        if self.proposed is not None:
            if is_yes(user_text):
                self.finished = True
                return Turn("done", "", self.proposed)
            self.proposed = None
            user_text = f"The user wants changes: {user_text}"
        self.messages.append({"role": "user", "content": user_text})
        return self._advance()

    def _advance(self) -> Turn:
        from .llm import parse_json

        for _ in range(self.max_turns):
            text = self.client.chat(self.messages, self.model, purpose="interview", temperature=0.3, max_tokens=4096)
            self.messages.append({"role": "assistant", "content": text})
            try:
                action = parse_json(text)
            except ValueError:
                self._note("Reply with exactly one JSON object as instructed.")
                continue
            if not isinstance(action, dict):
                self._note("Reply with exactly one JSON object as instructed.")
                continue
            if "say" in action:
                if not str(action["say"]).strip():
                    self._note("Your say message was empty. Ask the user something or propose the spec.")
                    continue
                return Turn("say", str(action["say"]))
            if "search" in action:
                q = action["search"] if isinstance(action["search"], list) else [action["search"]]
                found = self.categories.search(q)
                self._note("search results (category: number of models): "
                           + json.dumps(found or "nothing found, try other words"))
            elif "spec" in action:
                try:
                    s = spec_mod.from_dict(action["spec"])
                    unknown = [cat for c in s.classes if c.assets for cat in c.assets.categories
                               if cat not in self.categories.counts]
                    if unknown:
                        raise spec_mod.SpecError(f"categories not found in Objaverse-LVIS: {unknown}, use search")
                except spec_mod.SpecError as e:
                    self._note(f"The spec is invalid: {e}. Fix it and send it again.")
                    continue
                self.proposed = s
                return Turn("confirm", summary(s), s)
            else:
                self._note("Unknown action. Use say, search or spec.")
        self.finished = True
        return Turn("failed", "The interview did not converge; nothing was saved.")

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


def run(client, model: str, categories: CategoryIndex, ask: Callable[[str], str], tell: Callable[[str], None],
        max_turns: int = 60) -> spec_mod.Spec | None:
    """Terminal front end: ask() gets the user's answer to a prompt, tell() shows text."""
    session = Session(client, model, categories, max_turns)
    turn = session.reply(ask(FIRST_QUESTION + "\n> "))
    while True:
        if turn.kind == "say":
            turn = session.reply(ask(turn.text + "\n> "))
        elif turn.kind == "confirm":
            tell(turn.text)
            turn = session.reply(ask(CONFIRM + " > "))
        elif turn.kind == "done":
            return turn.spec
        else:
            tell(turn.text)
            return None
