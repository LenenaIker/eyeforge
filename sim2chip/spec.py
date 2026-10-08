"""Dataset and model spec: the single source of truth for a job.

Everything that depends on the user's problem (classes, sizes, scene, camera, hardware, real data)
lives here. Stages read the spec and never hardcode a class or an environment.
The Nemotron interview writes it (interview.py); it can also be written by hand.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ASSET_SOURCES = ("objaverse_lvis", "search", "photos", "library")
# roi_classification: one class per image (the object filling most of it, or none).
# count: how many objects of each class are in the image and where (a point per object).
TASKS = ("roi_classification", "count")
ERRORS = ("find_more", "balanced", "fewer_false")  # which mistake hurts the user more (Spec.errors)
# Which rotations and mirrors keep a photo's answer (TrainingSpec.orientation): "upright" scenes have an up
# (sky, horizon, standing objects); "any" when up does not exist (a camera looking straight down, a microscope,
# loose objects on a table); "fixed" when also left and right matter (text, arrows, one-sided parts).
ORIENTATIONS = ("upright", "any", "fixed")
DISTRACTORS = ("tree", "person")  # what the scene templates can place
WEATHER = ("clear", "cloudy", "evening", "storm", "night")
# Places the generic scene can build (scene.contexts): open ground, a building front seen from the street, a
# balcony seen from inside, a room. Where objects can stand in them (ContextSpec.supports):
CONTEXTS = {"ground": ("ground",),
            "facade": ("balcony_floor", "railing", "window_sill", "ground"),
            "balcony": ("balcony_floor", "railing"),
            "indoor": ("floor", "furniture", "window_sill")}
WALLS = ("brick", "plaster", "stone", "concrete", "wood")  # wall materials, as Poly Haven texture tags
RAILINGS = ("bars", "solid")
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
    source "library": only the user's own models from their library (`models`).
    `models` (any source): GLB files of the user's library in the job's input folder (the server puts the
    objects a kit or the chat chose there, with their name and license in objects.json next to them).
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
    models: list[str] = field(default_factory=list)

    @property
    def n_generate(self) -> int:
        return 4 if self.generate is None else self.generate


@dataclass
class ClassSpec:
    id: str
    description: str = ""
    size_m: SizeSpec | None = None
    assets: AssetQuery | None = None
    # Short English noun phrases the counting teacher (SAM 3, English only) looks for in the user's photos.
    # None: the 3D model search names (assets.queries or categories), else the id put in English.
    names: list[str] | None = None
    # Solid paint over the model's own colors, for classes whose real instances come in many colors (cars):
    # the chance per frame (None: never; models without color are always painted) and the metallic range.
    repaint: float | None = None
    metallic: tuple[float, float] | None = None

    @property
    def repaint_p(self) -> float:
        return self.repaint or 0.0

    @property
    def metallic_range(self) -> tuple[float, float]:
        return self.metallic or (0.0, 0.15)


@dataclass
class PropSpec:
    """Background objects placed around the targets so scenes are not empty (cones and pallets on a
    building site, rocks and logs in a forest). Found by live search like a class, checked by asset QA
    (which also drops models that look like a target class) and never labeled: the camera must ignore them.
    `count` models are kept; each frame shows between 0 and `max_per_frame` instances of this prop.
    `models`: GLB files of the user's library (as AssetQuery.models), used as they are; `queries` may then
    be empty."""
    id: str
    queries: list[str]
    size_m: SizeSpec
    description: str = ""
    count: int = 4
    max_per_frame: int = 4
    models: list[str] = field(default_factory=list)


@dataclass
class CameraSpec:
    height_m: tuple[float, float] = (2.0, 6.0)
    hfov_deg: float = 60.0
    resolution: tuple[int, int] = (320, 240)
    # Fraction of the image width covered by the target's largest dimension. The camera distance is
    # derived from it, so nobody has to guess a distance per class.
    target_fill: tuple[float, float] = (0.15, 0.55)
    distance_m: tuple[float, float] = (4.0, 80.0)  # hard limits on the derived distance
    # Degrees below the horizon the camera looks at the target, for cameras looking down at the ground (a phone
    # over a lawn: 60-89). Open-ground frames then place the camera by height and pitch, not by target_fill.
    pitch_deg: tuple[float, float] | None = None


@dataclass
class ContextSpec:
    """One kind of place in the synthetic scenes and its share of the frames. `distance_m` is the camera
    distance to the objects there (facade only, default 6-25 m); `height_m` the camera height above the
    floor it stands on (default a person holding a phone, 1.3-1.8 m; higher for a wall-mounted camera);
    `supports` where the objects stand (default: every support of the kind)."""
    kind: str
    share: float = 1.0
    distance_m: tuple[float, float] | None = None
    supports: list[str] = field(default_factory=list)
    height_m: tuple[float, float] | None = None

    @property
    def eye_height(self) -> tuple[float, float]:
        return self.height_m or (1.3, 1.8)


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
    # Generic template only. Empty contexts: objects on open ground, as always. Usually proposed from the
    # user's photos (scene_profile.py). wall_tags and railings style the buildings of facade/balcony/indoor.
    contexts: list[ContextSpec] = field(default_factory=list)
    wall_tags: list[str] = field(default_factory=list)
    railings: list[str] = field(default_factory=list)
    # Mixed ground (ground_mix.py), generic template: patches of other materials on the ground_tags base (bare
    # spots, fallen leaves; one Poly Haven tag per material), the share of ground they cover and their size, and
    # a hard-edged strip (a path, a pavement; edge_tags) in edge_share of the layouts. Defaults in ground_mix.
    ground_patches: list[str] = field(default_factory=list)
    patch_cover: tuple[float, float] | None = None
    patch_size_m: tuple[float, float] | None = None
    edge_tags: list[str] = field(default_factory=list)
    edge_share: float | None = None


@dataclass
class UserData:
    """Photos from the user. `labeled` is a folder with one subfolder per source class and `class_map`
    maps each spec class to the source subfolders that belong to it. `raw` is a folder of unlabeled
    photos that the teacher model labels. `test` is an optional labeled folder (same layout and class_map as
    `labeled`) used only for the real test. `raw` may list several sources (a library dataset and the photos
    attached in the chat): the web server puts them in one folder of the job before it runs."""
    labeled: str | None = None
    raw: str | list[str] | None = None
    test: str | None = None
    # No longer used, kept so older specs load: the teacher always commits to a class, there is no "unsure" pile
    # (user, 2026-09-30).
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
    max_targets: int | None = None  # count task: up to this many targets per frame (default 6)

    @property
    def n_targets(self) -> int:
        return 6 if self.max_targets is None else self.max_targets


@dataclass
class LookSpec:
    """How the user's camera sees: color (Lab mean and spread), edge sharpness and noise of their photos at
    the model's input size (look.py), measured by `sim2chip match`. Synthetic images are moved towards it
    before training (points, crops)."""
    lab_mean: list[float]
    lab_std: list[float]
    sharpness: float
    noise: float
    n: int = 0  # photos measured


@dataclass
class TrainingSpec:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    epochs: int = 25
    real_weight: int = 4  # real photos are repeated to weigh ~1/real_weight of the synthetic set
    orientation: str | None = None  # one of ORIENTATIONS; None is "upright". Chooses the training augmentation


@dataclass
class Spec:
    name: str
    classes: list[ClassSpec]
    props: list[PropSpec] = field(default_factory=list)
    hardware: str = "esp32cam"
    task: str = "roi_classification"
    # Which mistake hurts the user more: "find_more" (missing an object is worse), "balanced" (None: default) or
    # "fewer_false" (a false alarm is worse). Sets the teacher's threshold (teacher.py).
    errors: str | None = None
    # The class for "none of the objects": right when the classes are objects that may be absent. None when every
    # photo is one of the classes (surfaces, scenes, states: lawn, border, no grass), so no extra class is invented.
    background_class: str | None = "empty"
    camera: CameraSpec = field(default_factory=CameraSpec)
    scene: SceneSpec = field(default_factory=SceneSpec)
    user_data: UserData = field(default_factory=UserData)
    qa: QASpec = field(default_factory=QASpec)
    generation: GenerationSpec = field(default_factory=GenerationSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    look: LookSpec | None = None

    @property
    def labels(self) -> list[str]:
        """Model output order: background first (when there is one), then the spec classes."""
        return ([self.background_class] if self.background_class else []) + [c.id for c in self.classes]

    @property
    def synthetic(self) -> bool:
        """Whether the job renders images from 3D models: some class has models and frames are asked for.
        generation.frames 0 means the user's photos only (the 3D stages would run for nothing, 2026-09-30)."""
        return self.generation.frames > 0 and any(c.assets for c in self.classes)

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
        adding an optional field later does not invalidate existing jobs. The default task counts as empty
        for the same reason: stages that learned about tasks later keep their fingerprints."""
        if name == "task" and self.task == "roi_classification":
            return None
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


_NUMBER = (int, float)
_SCALARS = {"int": (int,), "float": _NUMBER, "str": (str,)}


def _check_type(t: str, v: Any, where: str) -> None:
    """Scalar and list types; nested sections are built separately. Specs come from an LLM, so a wrong
    type must be a SpecError here, not a TypeError deep inside a stage (maybe after paying for a GPU)."""
    optional = t.endswith(" | None")
    base = t.removesuffix(" | None")
    if v is None:
        if not optional and base in _SCALARS:
            raise SpecError(f"{where}: cannot be null")
        return
    kind = _SCALARS.get(base)
    if kind and (isinstance(v, bool) or not isinstance(v, kind)):
        raise SpecError(f"{where}: expected {base}, got {v!r}")
    if base.startswith("tuple[") and not all(isinstance(x, _NUMBER) and not isinstance(x, bool) for x in v):
        raise SpecError(f"{where}: expected numbers, got {list(v)!r}")
    if base in ("list[str]", "list[int]"):
        item = (str,) if base == "list[str]" else (int,)
        if not isinstance(v, list) or not all(isinstance(x, item) and not isinstance(x, bool) for x in v):
            raise SpecError(f"{where}: expected {base}, got {v!r}")


def _build(cls, data: dict, where: str):
    if not isinstance(data, dict):
        raise SpecError(f"{where}: expected an object")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        # Name the valid keys: without them the interview model guessed ("name", "type") until it gave up.
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}, valid keys are {sorted(known)}")
    kwargs = {}
    for k, v in data.items():
        t = known[k].type
        if t == "tuple[float, float] | None" and v is None:
            pass
        elif t in ("tuple[float, float]", "tuple[int, int]", "tuple[float, float] | None"):
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise SpecError(f"{where}.{k}: expected [min, max]")
            v = tuple(v)
        _check_type(t, v, f"{where}.{k}")
        kwargs[k] = v
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise SpecError(f"{where}: {e}") from None


def from_dict(data: dict) -> Spec:
    """Every problem in the data is a SpecError, whatever its shape."""
    if not isinstance(data, dict):
        raise SpecError("spec: expected an object")
    try:
        return _from_dict(data)
    except SpecError:
        raise
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        raise SpecError(f"spec: {e}") from None


def _from_dict(data: dict) -> Spec:
    data = dict(data)
    if "classes" not in data or not data["classes"]:
        raise SpecError("spec: 'classes' is required and cannot be empty")
    classes = []
    for i, c in enumerate(data.pop("classes")):
        where = f"classes[{i}]"
        if not isinstance(c, dict):
            raise SpecError(f"{where}: expected an object with 'id', 'size_m', ..., got {c!r}")
        c = dict(c)
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
    if isinstance(data.get("scene"), dict) and data["scene"].get("contexts"):
        sc = dict(data["scene"])
        sc["contexts"] = [_build(ContextSpec, c, f"scene.contexts[{i}]") for i, c in enumerate(sc["contexts"])]
        data["scene"] = sc
    sections = {"camera": CameraSpec, "scene": SceneSpec, "user_data": UserData, "qa": QASpec,
                "generation": GenerationSpec, "training": TrainingSpec, "look": LookSpec}
    if data.get("look") is None:
        data.pop("look", None)
    for key, cls in sections.items():
        if key in data:
            data[key] = _build(cls, data[key], key)
    spec = _build(Spec, {**data, "classes": classes}, "spec")
    validate(spec)
    return spec


def validate(spec: Spec) -> None:
    """Every rule a spec must pass, section by section; the first broken one is the SpecError."""
    for check in (_validate_labels, _validate_classes, _validate_props, _validate_scene, _validate_user_data,
                  _validate_places, _validate_task, _validate_rest):
        check(spec)


def _validate_labels(spec: Spec) -> None:
    from .hardware import PROFILES
    if spec.hardware not in PROFILES:
        raise SpecError(f"hardware: unknown '{spec.hardware}', known: {sorted(PROFILES)}")
    ids = [c.id for c in spec.classes]
    if len(set(ids)) != len(ids):
        raise SpecError("classes: duplicated ids")
    if spec.background_class is not None and spec.background_class in ids:
        raise SpecError("background_class cannot also be a class")
    if spec.background_class is None and spec.task == "count":
        raise SpecError("background_class: counting needs one (the places with no object)")
    if spec.background_class is None and len(ids) < 2:
        raise SpecError("background_class: with no background class, list at least two classes")


def _validate_classes(spec: Spec) -> None:
    for c in spec.classes:
        if c.names is not None and (not isinstance(c.names, list) or not c.names
                                    or not all(isinstance(n, str) and n.strip() for n in c.names)):
            raise SpecError(f"class {c.id}: names must be a list of short English noun phrases")
        if c.repaint is not None and not 0 <= c.repaint <= 1:
            raise SpecError(f"class {c.id}: repaint must be between 0 and 1")
        if c.metallic is not None and not 0 <= c.metallic[0] <= c.metallic[1] <= 1:
            raise SpecError(f"class {c.id}: metallic must be [min, max] within 0..1")
        if c.size_m:
            if c.size_m.measure not in MEASURES:
                raise SpecError(f"class {c.id}: size_m.measure must be one of {MEASURES}")
            lo, hi = c.size_m.range
            if not 0 < lo <= hi:
                raise SpecError(f"class {c.id}: size_m.range must be 0 < min <= max")
        if c.assets and not c.size_m:
            raise SpecError(f"class {c.id}: 3D assets need size_m to be scaled to real size")
        if c.assets:
            _validate_assets(c)


def _validate_assets(c: ClassSpec) -> None:
    if c.assets.source not in ASSET_SOURCES:
        raise SpecError(f"class {c.id}: assets.source must be one of {ASSET_SOURCES}")
    if c.assets.source == "objaverse_lvis" and not c.assets.categories:
        raise SpecError(f"class {c.id}: objaverse_lvis assets need categories")
    if c.assets.source == "library" and not c.assets.models:
        raise SpecError(f"class {c.id}: library assets need models")
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


def _validate_props(spec: Spec) -> None:
    pids = [p.id for p in spec.props]
    if len(set(pids)) != len(pids) or set(pids) & set(spec.labels):
        raise SpecError("props: ids must be unique and differ from the class ids and background_class")
    for p in spec.props:
        if p.size_m.measure not in MEASURES or not 0 < p.size_m.range[0] <= p.size_m.range[1]:
            raise SpecError(f"prop {p.id}: size_m needs a measure in {MEASURES} and 0 < min <= max")
        if not p.queries and not p.models:
            raise SpecError(f"prop {p.id}: needs queries to search for, or models from the library")
        if p.count < 1 or p.max_per_frame < 1:
            raise SpecError(f"prop {p.id}: count and max_per_frame must be at least 1")


def _validate_scene(spec: Spec) -> None:
    from .scenes import TEMPLATES
    named = sorted(set(TEMPLATES) - {"places"})  # places is chosen by scene.contexts, not by name
    if spec.scene.template not in named:
        raise SpecError(f"scene.template must be one of {named}")
    bad = [w for w in spec.scene.weather if w not in WEATHER]
    if bad or not spec.scene.weather:
        raise SpecError(f"scene.weather must be a non-empty subset of {WEATHER}")
    if spec.scene.template == "road" and not spec.scene.sky_tags:
        from .scenes.skies import SKIES
        if not {s.weather for s in SKIES} & set(spec.scene.weather):
            raise SpecError(f"scene.weather {spec.scene.weather}: the road template's own skies have none of "
                            "these; add scene.sky_tags to take skies from Poly Haven")


def _validate_user_data(spec: Spec) -> None:
    raw = spec.user_data.raw
    if isinstance(raw, list):
        if not raw or not all(isinstance(r, str) and r for r in raw):
            raise SpecError("user_data.raw: a folder, or a list of folders and library datasets")
        if len(raw) == 1:
            spec.user_data.raw = raw[0]
    elif raw is not None and not isinstance(raw, str):
        raise SpecError("user_data.raw: a folder, or a list of folders and library datasets")
    for k in spec.user_data.class_map:
        if k not in spec.labels:
            raise SpecError(f"user_data.class_map: '{k}' is not a class of the spec")
    for k, dirs in spec.user_data.class_map.items():
        if not dirs or any(not str(d).strip() for d in dirs):
            raise SpecError(f"user_data.class_map: '{k}' needs subfolder names; unsorted photos go in user_data.raw")


def _validate_places(spec: Spec) -> None:
    """Contexts, mixed ground and distractors of the scene."""
    _validate_contexts(spec.scene)
    _validate_ground_mix(spec.scene)
    bad = [d for d in spec.scene.distractors if d not in DISTRACTORS]
    if bad:
        raise SpecError(f"scene.distractors {bad} are not available, only {list(DISTRACTORS)}")


def _validate_task(spec: Spec) -> None:
    if spec.errors is not None and spec.errors not in ERRORS:
        raise SpecError(f"errors must be one of {ERRORS}")
    if spec.training.orientation is not None and spec.training.orientation not in ORIENTATIONS:
        raise SpecError(f"training.orientation must be one of {ORIENTATIONS}")
    if spec.task not in TASKS:
        raise SpecError(f"task must be one of {TASKS}")
    if spec.task == "count":
        if spec.user_data.labeled or spec.user_data.test:
            raise SpecError("task count: put the photos in user_data.raw; the teacher marks every object and "
                            "you fix its marks in review (class folders cannot say where the objects are)")
        if spec.generation.n_targets < 1:
            raise SpecError("generation.max_targets must be at least 1")


def _validate_rest(spec: Spec) -> None:
    """Asset QA, camera, generation, look and training."""
    if spec.qa.mode not in ("none", "manual", "auto"):
        raise SpecError("qa.mode must be 'none', 'manual' or 'auto'")
    if spec.qa.strategy not in ("vision", "describe_then_decide"):
        raise SpecError("qa.strategy must be 'vision' or 'describe_then_decide'")
    if spec.camera.pitch_deg and not 0 < spec.camera.pitch_deg[0] <= spec.camera.pitch_deg[1] <= 90:
        raise SpecError("camera.pitch_deg must be [min, max] within (0, 90]")
    lo, hi = spec.camera.target_fill
    if not 0 < lo <= hi <= 1:
        raise SpecError("camera.target_fill must be within (0, 1]")
    if not 0 <= spec.generation.empty_ratio < 1:
        raise SpecError("generation.empty_ratio must be in [0, 1)")
    if spec.look and not (len(spec.look.lab_mean) == len(spec.look.lab_std) == 3):
        raise SpecError("look: lab_mean and lab_std need 3 values (L, a, b)")
    if not spec.training.seeds:
        raise SpecError("training.seeds cannot be empty")


def _validate_ground_mix(sc: SceneSpec) -> None:
    if not (sc.ground_patches or sc.edge_tags):
        return
    if sc.template != "generic" or not sc.ground_tags:
        raise SpecError("scene.ground_patches and edge_tags need the generic template and ground_tags (the base)")
    if sc.patch_cover and not 0 <= sc.patch_cover[0] <= sc.patch_cover[1] <= 0.9:
        raise SpecError("scene.patch_cover must be [min, max] within 0..0.9")
    if sc.patch_size_m and not 0 < sc.patch_size_m[0] <= sc.patch_size_m[1]:
        raise SpecError("scene.patch_size_m must be 0 < min <= max")
    if sc.edge_share is not None and not 0 <= sc.edge_share <= 1:
        raise SpecError("scene.edge_share must be within 0..1")


def _validate_contexts(sc: SceneSpec) -> None:
    if sc.contexts and sc.template != "generic":
        raise SpecError("scene.contexts need scene.template \"generic\"")
    kinds = [c.kind for c in sc.contexts]
    if len(set(kinds)) != len(kinds):
        raise SpecError("scene.contexts: each kind at most once")
    for c in sc.contexts:
        if c.kind not in CONTEXTS:
            raise SpecError(f"scene.contexts: kind must be one of {list(CONTEXTS)}")
        if c.share <= 0:
            raise SpecError(f"scene.contexts {c.kind}: share must be positive")
        if c.distance_m and not 0 < c.distance_m[0] <= c.distance_m[1]:
            raise SpecError(f"scene.contexts {c.kind}: distance_m must be 0 < min <= max")
        bad = [x for x in c.supports if x not in CONTEXTS[c.kind]]
        if bad:
            raise SpecError(f"scene.contexts {c.kind}: supports {bad} not possible there, only {list(CONTEXTS[c.kind])}")
    bad = [w for w in sc.wall_tags if w not in WALLS] + [r for r in sc.railings if r not in RAILINGS]
    if bad:
        raise SpecError(f"scene: unknown {bad}; wall_tags from {WALLS}, railings from {RAILINGS}")


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
