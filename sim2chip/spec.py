"""Dataset and model spec: the single source of truth for a job.

Everything that depends on the user's problem (classes, sizes, scene, camera, hardware, real data)
lives here. Stages read the spec and never hardcode a class or an environment.
The spec is written by hand for now; the Nemotron interview will produce it later.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

MEASURES = ("length", "width", "height")  # length and width: long and short side seen from above


class SpecError(ValueError):
    pass


@dataclass
class SizeSpec:
    """Real-world size used to scale 3D assets: `measure` is drawn uniformly from `range` (meters)."""
    measure: str
    range: tuple[float, float]


@dataclass
class AssetQuery:
    """Where to find 3D models for a class. `count` models are split evenly across `categories`."""
    source: str = "objaverse_lvis"
    categories: list[str] = field(default_factory=list)
    count: int = 12


@dataclass
class ClassSpec:
    id: str
    description: str = ""
    size_m: SizeSpec | None = None
    assets: AssetQuery | None = None


@dataclass
class CameraSpec:
    height_m: tuple[float, float] = (2.0, 6.0)
    hfov_deg: float = 60.0
    resolution: tuple[int, int] = (320, 240)
    # Fraction of the image width covered by the target's largest dimension. The camera distance is
    # derived from it, so nobody has to guess a distance per class.
    target_fill: tuple[float, float] = (0.15, 0.55)
    distance_m: tuple[float, float] = (4.0, 80.0)  # hard limits on the derived distance


@dataclass
class SceneSpec:
    template: str = "road"
    weather: list[str] = field(default_factory=lambda: ["clear", "cloudy", "evening", "storm"])
    distractors: list[str] = field(default_factory=lambda: ["tree", "person"])


@dataclass
class UserData:
    """Photos from the user. `labeled` is a folder with one subfolder per source class and `class_map`
    maps each spec class to the source subfolders that belong to it. `raw` is a folder of unlabeled
    photos that the teacher model labels; answers below `label_min_confidence` are set aside."""
    labeled: str | None = None
    raw: str | None = None
    label_min_confidence: float = 0.6
    class_map: dict[str, list[str]] = field(default_factory=dict)
    test_per_class: int = 400
    train_per_class: int = 50
    min_side_px: int = 32


@dataclass
class QASpec:
    """Asset quality control: `none` keeps every asset, `manual` applies a human review, `auto` asks a
    vision model (manual lists, if given, are applied on top)."""
    mode: str = "none"
    exclude: str | None = None          # text file, one uid per line
    class_override: str | None = None   # JSON {uid: {"class": new_class}}
    no_color: str | None = None         # JSON list of uids whose model has no color (always repainted)


@dataclass
class GenerationSpec:
    frames: int = 2000
    empty_ratio: float = 0.2
    seed: int = 0


@dataclass
class TrainingSpec:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    epochs: int = 25
    real_weight: int = 4  # real photos are repeated to weigh ~1/real_weight of the synthetic set


@dataclass
class Spec:
    name: str
    classes: list[ClassSpec]
    hardware: str = "esp32cam"
    task: str = "roi_classification"
    background_class: str = "empty"
    camera: CameraSpec = field(default_factory=CameraSpec)
    scene: SceneSpec = field(default_factory=SceneSpec)
    user_data: UserData = field(default_factory=UserData)
    qa: QASpec = field(default_factory=QASpec)
    generation: GenerationSpec = field(default_factory=GenerationSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)

    @property
    def labels(self) -> list[str]:
        """Model output order: background first, then the spec classes."""
        return [self.background_class] + [c.id for c in self.classes]

    def cls(self, class_id: str) -> ClassSpec:
        for c in self.classes:
            if c.id == class_id:
                return c
        raise KeyError(class_id)

    def section(self, name: str) -> Any:
        return to_dict(getattr(self, name))

    def to_json(self) -> str:
        return json.dumps(to_dict(self), indent=2)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def fingerprint(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _build(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise SpecError(f"{where}: expected an object")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for k, v in data.items():
        t = known[k].type
        if t in ("tuple[float, float]", "tuple[int, int]"):
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise SpecError(f"{where}.{k}: expected [min, max]")
            v = tuple(v)
        kwargs[k] = v
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise SpecError(f"{where}: {e}") from None


def from_dict(data: dict) -> Spec:
    data = dict(data)
    if "classes" not in data or not data["classes"]:
        raise SpecError("spec: 'classes' is required and cannot be empty")
    classes = []
    for i, c in enumerate(data.pop("classes")):
        c = dict(c)
        where = f"classes[{i}]"
        size = _build(SizeSpec, c.pop("size_m"), f"{where}.size_m") if c.get("size_m") else c.pop("size_m", None)
        assets = _build(AssetQuery, c.pop("assets"), f"{where}.assets") if c.get("assets") else c.pop("assets", None)
        classes.append(_build(ClassSpec, {**c, "size_m": size, "assets": assets}, where))
    sections = {"camera": CameraSpec, "scene": SceneSpec, "user_data": UserData, "qa": QASpec,
                "generation": GenerationSpec, "training": TrainingSpec}
    for key, cls in sections.items():
        if key in data:
            data[key] = _build(cls, data[key], key)
    spec = _build(Spec, {**data, "classes": classes}, "spec")
    validate(spec)
    return spec


def validate(spec: Spec) -> None:
    ids = [c.id for c in spec.classes]
    if len(set(ids)) != len(ids):
        raise SpecError("classes: duplicated ids")
    if spec.background_class in ids:
        raise SpecError("background_class cannot also be a class")
    for c in spec.classes:
        if c.size_m:
            if c.size_m.measure not in MEASURES:
                raise SpecError(f"class {c.id}: size_m.measure must be one of {MEASURES}")
            lo, hi = c.size_m.range
            if not 0 < lo <= hi:
                raise SpecError(f"class {c.id}: size_m.range must be 0 < min <= max")
        if c.assets and not c.size_m:
            raise SpecError(f"class {c.id}: 3D assets need size_m to be scaled to real size")
    for k in spec.user_data.class_map:
        if k not in spec.labels:
            raise SpecError(f"user_data.class_map: '{k}' is not a class of the spec")
    if spec.qa.mode not in ("none", "manual", "auto"):
        raise SpecError("qa.mode must be 'none', 'manual' or 'auto'")
    lo, hi = spec.camera.target_fill
    if not 0 < lo <= hi <= 1:
        raise SpecError("camera.target_fill must be within (0, 1]")
    if not 0 <= spec.generation.empty_ratio < 1:
        raise SpecError("generation.empty_ratio must be in [0, 1)")
    if not spec.training.seeds:
        raise SpecError("training.seeds cannot be empty")


def load(path: str | Path) -> Spec:
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise SpecError(f"{path}: invalid JSON ({e})") from None
    return from_dict(data)


def camera_distance(size_m: float, fill: float, hfov_deg: float, limits: tuple[float, float]) -> float:
    """Distance at which an object of `size_m` covers `fill` of the image width (pinhole camera)."""
    d = size_m / (2 * math.tan(math.radians(hfov_deg) / 2) * fill)
    return min(max(d, limits[0]), limits[1])


__all__ = ["Spec", "SpecError", "load", "from_dict", "fingerprint", "camera_distance"]
