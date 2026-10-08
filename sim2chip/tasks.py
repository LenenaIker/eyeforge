"""What changes with the task (classify, count) and with the recipe that made a seed's model, in one place.

Stages ask these instead of each deciding from spec.task, the hardware's counter or the keys of a result:
which metric heads the report, which teacher labels raw photos, whether a detector trains, which file a seed
ships. Stdlib only (it runs in every runtime). The training and scoring code itself stays in the stages; when a
third task arrives (following a line), the parts it shares with these two show which of them become methods here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from . import teacher as teachers
from .hardware import Hardware


@dataclass(frozen=True)
class Task:
    name: str
    metric: str        # headline on the real test, as evaluate and the report name it
    val_key: str       # key of a seed's result (SeedResult) that export picks the shipped seed by
    teacher: str       # who labels the user's raw photos (teacher.py)
    points: bool       # answers are points per object (count), not one class per photo

    def detector(self, hw: Hardware) -> str | None:
        """The pretrained detector this task trains on this hardware, or None for our own model."""
        return hw.counter if self.points and hw.counter != "grid" else None


TASKS = {
    "roi_classification": Task("roi_classification", "accuracy", "val_acc", f"Google {teachers.SIGLIP_NAME}",
                               points=False),
    "count": Task("count", "f1", "val_f1", f"Meta {teachers.NAME}", points=True),
}


def task_of(spec) -> Task:
    return TASKS[spec.task]


@dataclass(frozen=True)
class Recipe:
    """How a seed's model was made, which says how it ships and how it is read."""
    name: str
    boxes: bool = False                   # a box detector (detector.py): read as boxes, float16 weights
    model_file: str = "model_int8.tflite"  # the seed's file that export ships and evaluate scores

    @property
    def quantization(self) -> str | None:
        """float16 for a detector, else None: the hardware profile's own."""
        return "float16" if self.boxes else None


RECIPES = {r.name: r for r in (
    Recipe("fine_tune"),  # the whole network trained (classify, or count with the grid head)
    Recipe("head"),       # few photos: only the last layers learn, scored by cross-validation
    Recipe("picodet_s", boxes=True, model_file="model.tflite"),
)}


def recipe_of(result: dict) -> Recipe:
    """The recipe a seed recorded in train/summary.json (every seed records one)."""
    return RECIPES[result.get("recipe", "fine_tune")]


class SeedResult(TypedDict, total=False):
    """One seed in train/summary.json."""
    recipe: str            # a key of RECIPES
    kb: float              # size of the shipped model
    val_acc: float         # classify: validation (or cross-validated) accuracy
    val_f1: float          # count: points F1 at `threshold`
    val_count_mae: float   # count: mean count error per image
    threshold: float       # count: score from which a cell or box is an object
    cv_curve: dict         # count with few photos: cross-validated F1 by threshold
    counter: str           # detector: the hardware counter it stands for
    input: int             # detector: its input side when it is not the profile's
