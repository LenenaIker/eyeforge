"""Scene templates. Each template builds an environment in Isaac Sim and places targets, camera and
distractors. The spec picks one by name (scene.template)."""
from __future__ import annotations

TEMPLATES = {"road": "sim2chip.scenes.road:RoadScene", "generic": "sim2chip.scenes.generic:GenericScene",
             "places": "sim2chip.scenes.places:PlacesScene"}


def template_for(scene) -> str:
    """The template a spec's scene section needs: contexts (balconies, rooms...) are built by "places", on top of
    the generic scene."""
    return "places" if scene.contexts else scene.template


def load(name: str):
    """Import lazily: templates need Isaac Sim modules, which exist only inside a running app."""
    import importlib

    try:
        mod, cls = TEMPLATES[name].split(":")
    except KeyError:
        raise KeyError(f"unknown scene template '{name}', known: {sorted(TEMPLATES)}") from None
    return getattr(importlib.import_module(mod), cls)
