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

ASSET_SOURCES = ("objaverse_lvis", "search", "photos")
DISTRACTORS = ("tree", "person")  # what the scene templates can place
WEATHER = ("clear", "cloudy", "evening", "storm", "night")
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
    """Where to find 3D models for a class.

    source "objaverse_lvis": curated Objaverse-LVIS `categories`, `count` split evenly across them.
    source "search": live text search (`queries`, e.g. ["wild boar", "boar"]) over Sketchfab, keeping
    models with an open license that the Objaverse mirror can serve; asset QA then filters them.
    source "photos": only models generated from photos (below).
    `photos` (any source): a folder of photos of this object, or "user_data" for the user's own photos of the
    class, turned into 3D models with an image-to-3D model; up to `generate` models (default 4), best photos first.
    Generated models go through the same asset QA as the others.
    """
    source: str = "objaverse_lvis"
    categories: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    count: int = 12
    photos: str | None = None
    generate: int | None = None  # None keeps fingerprints of specs written before this field existed

    @property
    def n_generate(self) -> int:
        return 4 if self.generate is None else self.generate


@dataclass
class ClassSpec:
    id: str
    description: str = ""
    size_m: SizeSpec | None = None
    assets: AssetQuery | None = None


@dataclass
class PropSpec:
    """Background objects placed around the targets so scenes are not empty (cones and pallets on a
    building site, rocks and logs in a forest). Found by live search like a class, checked by asset QA
    (which also drops models that look like a target class) and never labeled: the camera must ignore them.
    `count` models are kept; each frame shows between 0 and `max_per_frame` instances of this prop."""
    id: str
    queries: list[str]
    size_m: SizeSpec
    description: str = ""
    count: int = 4
    max_per_frame: int = 4


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
    """Where the targets are shown. "road" is a hand-built template; "generic" is built from data: a ground
    texture chosen by `ground_tags` and sky maps chosen by `sky_tags` (both Poly Haven tags, CC0) plus
    `weather`. Setting sky_tags also makes "road" draw its skies from Poly Haven."""
    template: str = "road"
    weather: list[str] = field(default_factory=lambda: ["clear", "cloudy", "evening", "storm"])
    distractors: list[str] = field(default_factory=lambda: ["tree", "person"])
    sky_tags: list[str] = field(default_factory=list)
    ground_tags: list[str] = field(default_factory=list)


@dataclass
class UserData:
    """Photos from the user. `labeled` is a folder with one subfolder per source class and `class_map`
    maps each spec class to the source subfolders that belong to it. `raw` is a folder of unlabeled
    photos that the teacher model labels; answers below `label_min_confidence` are set aside. `test` is an
    optional labeled folder (same layout and class_map as `labeled`) used only for the real test."""
    labeled: str | None = None
    raw: str | None = None
    test: str | None = None
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
    strategy: str = "vision"            # auto mode: "vision" or "describe_then_decide"
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
    props: list[PropSpec] = field(default_factory=list)
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

    def size_of(self, asset_class: str) -> SizeSpec:
        """Real-world size of a class or a prop."""
        for p in self.props:
            if p.id == asset_class:
                return p.size_m
        return self.cls(asset_class).size_m

    def section(self, name: str) -> Any:
        """A spec section as it enters stage fingerprints: empty values (None, [], {}) are left out, so
        adding an optional field later does not invalidate existing jobs."""
        return _prune(to_dict(getattr(self, name)))

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


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v not in (None, [], {})}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


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
    props = []
    for i, p in enumerate(data.pop("props", None) or []):
        p = dict(p)
        where = f"props[{i}]"
        if not isinstance(p.get("size_m"), dict):
            raise SpecError(f"{where}.size_m: required")
        props.append(_build(PropSpec, {**p, "size_m": _build(SizeSpec, p["size_m"], f"{where}.size_m")}, where))
    data["props"] = props
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
        if c.assets:
            if c.assets.source not in ASSET_SOURCES:
                raise SpecError(f"class {c.id}: assets.source must be one of {ASSET_SOURCES}")
            if c.assets.source == "objaverse_lvis" and not c.assets.categories:
                raise SpecError(f"class {c.id}: objaverse_lvis assets need categories")
            if c.assets.source == "search" and not c.assets.queries:
                raise SpecError(f"class {c.id}: search assets need queries")
            if c.assets.count < 1:
                raise SpecError(f"class {c.id}: assets.count must be at least 1")
            if c.assets.source == "photos" and not c.assets.photos:
                raise SpecError(f"class {c.id}: photos assets need a photos folder (or \"user_data\")")
            if c.assets.photos and c.assets.n_generate < 1:
                raise SpecError(f"class {c.id}: assets.generate must be at least 1")
            if c.assets.photos and not (c.assets.queries or c.assets.categories):
                raise SpecError(f"class {c.id}: photo-made models need assets.queries, an English name of the "
                                "object for the detector (e.g. [\"flower pot\"])")
    pids = [p.id for p in spec.props]
    if len(set(pids)) != len(pids) or set(pids) & set(spec.labels):
        raise SpecError("props: ids must be unique and differ from the class ids and background_class")
    for p in spec.props:
        if p.size_m.measure not in MEASURES or not 0 < p.size_m.range[0] <= p.size_m.range[1]:
            raise SpecError(f"prop {p.id}: size_m needs a measure in {MEASURES} and 0 < min <= max")
        if not p.queries:
            raise SpecError(f"prop {p.id}: queries cannot be empty")
        if p.count < 1 or p.max_per_frame < 1:
            raise SpecError(f"prop {p.id}: count and max_per_frame must be at least 1")
    from .scenes import TEMPLATES
    if spec.scene.template not in TEMPLATES:
        raise SpecError(f"scene.template must be one of {sorted(TEMPLATES)}")
    bad = [w for w in spec.scene.weather if w not in WEATHER]
    if bad or not spec.scene.weather:
        raise SpecError(f"scene.weather must be a non-empty subset of {WEATHER}")
    for k in spec.user_data.class_map:
        if k not in spec.labels:
            raise SpecError(f"user_data.class_map: '{k}' is not a class of the spec")
    for k, dirs in spec.user_data.class_map.items():
        if not dirs or any(not str(d).strip() for d in dirs):
            raise SpecError(f"user_data.class_map: '{k}' needs subfolder names; unsorted photos go in user_data.raw")
    bad = [d for d in spec.scene.distractors if d not in DISTRACTORS]
    if bad:
        raise SpecError(f"scene.distractors {bad} are not available, only {list(DISTRACTORS)}")
    if spec.qa.mode not in ("none", "manual", "auto"):
        raise SpecError("qa.mode must be 'none', 'manual' or 'auto'")
    if spec.qa.strategy not in ("vision", "describe_then_decide"):
        raise SpecError("qa.strategy must be 'vision' or 'describe_then_decide'")
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
