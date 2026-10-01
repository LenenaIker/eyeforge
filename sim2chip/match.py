"""Match the synthetic scenes to the user's photos, before the full render.

Two parts:
1. The camera's look (look.py): color, sharpness and noise measured on the photos, stored in spec.look and
   given to every synthetic image by the points and crops stages. No model involved.
2. The scene, in a few rounds. A small trial render (TRIAL_FRAMES) is set beside the photos; a vision model
   lists how they differ (what is in the scene, the camera, the ground, the objects); a reasoning model turns
   those differences into changes of the spec, only through LEVERS (fields with allowed values or ranges);
   the changed spec is rendered again and kept only if the renders come closer to the photos (KID on
   ImageNet features, lower is closer; the two halves of the user's photos give the floor). Differences no
   lever can fix are reported: they say what the scene builder is missing.

Only the user's training photos are used (never the test side). Trial jobs live in <job>/match/it<k>/ with
copies of the parent job's asset stages, so assets are not fetched or made again.
Output: <job>/match/report.json, <job>/match/spec.json (the proposed spec), mosaics per round.

    sim2chip match specs/macetas.json --job maceta_counter_11
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from . import look
from .pipeline import StageFailed
from .spec import CONTEXTS, DISTRACTORS, RAILINGS, WALLS, WEATHER, LookSpec, Spec, from_dict, to_dict
from .workspace import IMAGE_EXT, Cache, Job, content_key

TRIAL_FRAMES = 48   # per round: enough for KID to move less than the changes we look for (sd ~0.02 at 48)
ROUNDS = 3
MIN_GAIN = 0.02     # KID drop that counts as closer (about one sd of the trial noise)
REF_PHOTOS = 32     # user photos used as the reference
MOSAIC = 6          # images per row in the comparison mosaic
FEATURE_SIZE = 224
LLM_TIMEOUT = 600   # s per model call: vision models may think for minutes on a comparison

# What the reasoning model may change: kind, allowed values or range, and a hint for it.
LEVERS = {
    "scene.weather": ("subset", WEATHER, "sky and sun of the frames"),
    "scene.sky_tags": ("tags", 3, "Poly Haven HDRI tags for the sky maps; a map must carry EVERY tag, so use "
                                  "one or two; empty: chosen from weather"),
    "scene.ground_tags": ("tags", 3, "Poly Haven texture tags for the ground; a texture must carry EVERY tag, "
                                     "so use one or two (e.g. [grass] or [tiles, floor])"),
    "scene.ground_patches": ("tag_each", 3, "materials of patches on the ground, one Poly Haven texture tag each "
                                            "(e.g. dirt, leaves, gravel, moss, mud): bare spots, fallen leaves; "
                                            "needs scene.ground_tags as the base"),
    "scene.patch_cover": ("pair", (0.0, 0.9), "share of the ground the patches cover, [min, max] per layout"),
    "scene.patch_size_m": ("pair", (0.05, 5.0), "typical size of one patch, meters"),
    "scene.edge_tags": ("tag_each", 2, "material of a straight hard-edged strip crossing some ground layouts "
                                       "(a path, a pavement), one texture tag each"),
    "scene.edge_share": ("range", (0.0, 1.0), "share of ground layouts with that strip"),
    "scene.distractors": ("subset", DISTRACTORS, "big generic things placed in open-ground frames"),
    "scene.wall_tags": ("subset", WALLS, "wall materials of buildings (facade, balcony, indoor)"),
    "scene.railings": ("subset", RAILINGS, "balcony railing styles"),
    "scene.contexts.<kind>.share": ("range", (0.0, 1.0), f"share of frames of each place kind {list(CONTEXTS)}; "
                                                          "0 removes it (needs scene.template generic)"),
    "scene.contexts.<kind>.distance_m": ("pair", (0.3, 80.0), "camera distance to the objects in that place"),
    "scene.contexts.<kind>.height_m": ("pair", (0.1, 30.0), "camera height above the floor it stands on"),
    "camera.height_m": ("pair", (0.1, 30.0), "camera height above the ground (open-ground frames)"),
    "camera.target_fill": ("pair", (0.02, 1.0), "share of the image width the object covers"),
    "camera.hfov_deg": ("range", (30.0, 120.0), "horizontal field of view of the user's camera"),
    "camera.pitch_deg": ("pair", (5.0, 89.0), "degrees the camera looks down (89: straight down at the ground, "
                                              "no horizon); open-ground frames only; null: from target_fill"),
    # Empty frames teach the model not to see targets everywhere: kept even when every photo has one.
    "generation.empty_ratio": ("range", (0.1, 0.6), "share of frames without any target"),
    "generation.max_targets": ("int", (1, 20), "count task: most targets in one frame"),
    "props.add": ("prop", 3, "add background objects seen near the targets in the photos: "
                             "{id, queries: [English search words], size_m: {measure: length|width|height, "
                             "range: [min, max] meters}, max_per_frame}; up to 3 per round"),
    "props.remove": ("id", None, "remove a background object by id"),
}
MAX_PROPS = 8


# ---------------------------------------------------------------- photos and images

def reference_photos(spec: Spec, job: Job, limit: int = REF_PHOTOS, seed: int = 0) -> list[Path]:
    """The user's photos on the training side of the job's split (all of them before the first ingest)."""
    from .stages.ingest import read_split
    ud = spec.user_data
    files = []
    for rel in (ud.raw, ud.labeled):
        if rel:
            files += [f for f in sorted(job.resolve_input(rel).rglob("*")) if f.suffix.lower() in IMAGE_EXT]
    sides = read_split(job.stage_dir("ingest"))
    if sides:
        files = [f for f in files if sides.get(content_key(f)) == "train"]
    random.Random(seed).shuffle(files)
    return files[:limit]


def _open(path: Path, size: tuple[int, int] | None = None, fit: bool = False):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    if size:
        im = ImageOps.fit(im, size) if fit else im.resize(size, Image.BILINEAR)
    return im


def small(paths: list[Path], size: int = FEATURE_SIZE) -> list:
    """Square model-input images, squashed like ingest and points do."""
    return [_open(p, (size, size)) for p in paths]


def mosaic(real: list[Path], synth: list, out: Path, cell=(384, 288)) -> Path:
    """Top row the user's photos, bottom row renders (paths or images)."""
    from PIL import Image, ImageDraw
    n = min(MOSAIC, len(real), len(synth))
    w, h = cell
    m = Image.new("RGB", (w * n, h * 2 + 40), "white")
    d = ImageDraw.Draw(m)
    d.text((6, 4), "TOP ROW: REAL PHOTOS FROM THE USER'S CAMERA", fill="black")
    d.text((6, h + 24), "BOTTOM ROW: OUR SYNTHETIC RENDERS", fill="black")
    for i in range(n):
        m.paste(_open(real[i], cell, fit=True), (i * w, 18))
        s = synth[i] if not isinstance(synth[i], Path) else _open(synth[i])
        m.paste(s.resize(cell), (i * w, h + 40))
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(out, quality=88)
    return out


# ---------------------------------------------------------------- closeness

class Features:
    """ImageNet features (MobileNetV2, pooled) for KID. TensorFlow is loaded once, on first use."""

    def __init__(self):
        self._model = None

    def __call__(self, imgs: list):
        import numpy as np
        import tensorflow as tf
        if self._model is None:
            for g in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(g, True)
            self._model = tf.keras.applications.MobileNetV2(input_shape=(FEATURE_SIZE, FEATURE_SIZE, 3),
                                                            include_top=False, weights="imagenet", pooling="avg")
        x = np.stack([np.asarray(i.resize((FEATURE_SIZE, FEATURE_SIZE)), dtype=np.float32) for i in imgs])
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return self._model.predict(x, verbose=0, batch_size=32).astype(np.float64)


def kid(a, b) -> float:
    """Kernel Inception Distance (unbiased, cubic kernel) between two feature sets: fit for small sets."""
    d = a.shape[1]
    k = lambda x, y: (x @ y.T / d + 1) ** 3  # noqa: E731
    m, n = len(a), len(b)
    kaa, kbb = k(a, a), k(b, b)
    v = (kaa.sum() - kaa.trace()) / (m * (m - 1)) + (kbb.sum() - kbb.trace()) / (n * (n - 1)) - 2 * k(a, b).mean()
    return round(float(v), 4)


# ---------------------------------------------------------------- levers

class LeverError(ValueError):
    pass


def lever_values(spec: Spec) -> dict:
    """Current value of every lever that applies to this spec."""
    d = to_dict(spec)
    out = {k: d["scene"][k.split(".")[1]] for k in LEVERS if k.startswith("scene.") and "<kind>" not in k}
    for c in d["scene"]["contexts"]:
        out[f"scene.contexts.{c['kind']}.share"] = c["share"]
        out[f"scene.contexts.{c['kind']}.distance_m"] = c["distance_m"]
        out[f"scene.contexts.{c['kind']}.height_m"] = c["height_m"]
    out.update({f"camera.{k}": d["camera"][k] for k in ("height_m", "target_fill", "hfov_deg", "pitch_deg")})
    out["generation.empty_ratio"] = d["generation"]["empty_ratio"]
    if spec.task == "count":
        out["generation.max_targets"] = spec.generation.n_targets
    out["props"] = [{"id": p["id"], "queries": p["queries"], "size_m": p["size_m"],
                     "max_per_frame": p["max_per_frame"]} for p in d["props"]]
    return out


def _lever_of(name: str) -> tuple[str, tuple]:
    parts = name.split(".")
    if len(parts) == 4 and parts[:2] == ["scene", "contexts"]:
        if parts[2] not in CONTEXTS:
            raise LeverError(f"unknown place kind {parts[2]}")
        key = f"scene.contexts.<kind>.{parts[3]}"
    else:
        key = name
    if key not in LEVERS:
        raise LeverError(f"{name} is not a lever")
    return key, LEVERS[key]


def _check(name: str, value, spec: Spec):
    """The value, checked and clipped to the lever's range; LeverError when it cannot be used."""
    key, (kind, allowed, _) = _lever_of(name)
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    if kind == "subset":
        if not isinstance(value, list) or not all(v in allowed for v in value):
            raise LeverError(f"{name}: expected a list from {list(allowed)}")
        return list(dict.fromkeys(value))
    if kind == "tag_each":
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise LeverError(f"{name}: expected a list of words")
        return list(dict.fromkeys(v.strip().lower() for v in value))[:allowed]
    if kind == "tags":
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            raise LeverError(f"{name}: expected a list of words")
        return [v.strip().lower() for v in value][:allowed]
    if kind in ("range", "int"):
        if not num(value):
            raise LeverError(f"{name}: expected a number")
        v = min(max(value, allowed[0]), allowed[1])
        return int(round(v)) if kind == "int" else round(float(v), 3)
    if kind == "pair":
        if value is None and (key.startswith("scene.contexts") or key == "camera.pitch_deg"):
            return None
        if not (isinstance(value, list) and len(value) == 2 and all(num(v) for v in value)):
            raise LeverError(f"{name}: expected [min, max]")
        lo, hi = (min(max(float(v), allowed[0]), allowed[1]) for v in sorted(value))
        return [round(lo, 3), round(hi, 3)]
    if kind == "prop":
        if not isinstance(value, dict) or not {"id", "queries", "size_m"} <= set(value):
            raise LeverError(f"{name}: expected {{id, queries, size_m, max_per_frame}}")
        v = {"id": str(value["id"]).strip().lower().replace(" ", "_"), "queries": value["queries"],
             "size_m": value["size_m"], "max_per_frame": int(value.get("max_per_frame") or 3)}
        v["max_per_frame"] = min(max(v["max_per_frame"], 1), 6)
        return v
    if kind == "id":
        if value not in [p.id for p in spec.props]:
            raise LeverError(f"{name}: no prop {value!r}")
        return value
    raise LeverError(f"{name}: unknown lever kind")


def _flat(changes: list) -> list[dict]:
    """Changes as single values: models send props.add with a list of props."""
    out = []
    for ch in changes:
        if isinstance(ch, dict) and ch.get("lever") == "props.add" and isinstance(ch.get("value"), list):
            out += [{**ch, "value": v} for v in ch["value"]]
        elif isinstance(ch, dict):
            out.append(ch)
    return out


def patch(spec: Spec, changes: list[dict], tags_ok=None, locked=()) -> tuple[Spec, list[dict], list[dict]]:
    """Apply changes one by one; a change that breaks a rule is skipped. Returns (new spec, applied, rejected).
    tags_ok(lever, tags) says whether Poly Haven has anything with those tags (None: not checked); `locked`
    levers are facts the user gave (their camera height) and are never changed."""
    cur, applied, rejected, added = spec, [], [], 0
    for ch in _flat(changes):
        name, value = ch.get("lever"), ch.get("value")
        try:
            if not isinstance(name, str):
                raise LeverError("missing lever")
            if name in locked:
                raise LeverError(f"{name} is fixed by the user")
            v = _check(name, value, cur)
            if tags_ok and name in ("scene.sky_tags", "scene.ground_tags") and v and not tags_ok(name, v):
                raise LeverError(f"no Poly Haven asset carries all of {v}")
            if tags_ok and LEVERS[_lever_of(name)[0]][0] == "tag_each":
                bad = [t for t in v if not tags_ok("scene.ground_tags", [t])]
                if bad:
                    raise LeverError(f"no Poly Haven texture has the tag {bad}")
            d = to_dict(cur)
            parts = name.split(".")
            if name == "props.add":
                if added >= LEVERS[name][1] or len(d["props"]) >= MAX_PROPS:
                    raise LeverError("too many props")
                if v["id"] in [p["id"] for p in d["props"]]:
                    raise LeverError(f"prop {v['id']} exists")
                d["props"].append({**v, "count": 4})
                added += 1
            elif name == "props.remove":
                d["props"] = [p for p in d["props"] if p["id"] != v]
            elif parts[:2] == ["scene", "contexts"]:
                if d["scene"]["template"] != "generic":
                    raise LeverError("places need the generic scene")
                ctxs = {c["kind"]: c for c in d["scene"]["contexts"]}
                c = ctxs.setdefault(parts[2], {"kind": parts[2], "share": 0.0, "distance_m": None,
                                               "supports": [], "height_m": None})
                c[parts[3]] = v
                d["scene"]["contexts"] = [c for c in ctxs.values() if c["share"] > 0]
                if not d["scene"]["contexts"]:
                    raise LeverError("at least one place kind must stay")
            elif parts[0] == "generation" and parts[1] == "max_targets" and cur.task != "count":
                raise LeverError("max_targets is for the count task")
            else:
                d[parts[0]][parts[1]] = v
            cur = from_dict(d)
            applied.append({**ch, "value": v})
        except (LeverError, ValueError) as e:
            rejected.append({**ch, "error": str(e)})
    return cur, applied, rejected


# ---------------------------------------------------------------- models

COMPARE = """The top row shows real photos from a user's camera. The bottom row shows synthetic 3D renders we made
to train a small camera model that must {task} {classes}. The renders must look like they were taken by the
same camera in the same kind of places.

List the most important ways the renders differ from the real photos, the ones that would make a model trained
on renders fail on the real photos. Look at: what the scene contains and how cluttered it is, the kind of place,
the ground and its texture (patches, wear, grass, dirt), the objects themselves (shape, color, material, what
is in them or on them), their size in the image and how many there are, the camera height, angle and distance,
the light and shadows, and anything that looks impossible or fake. Be concrete and visual. Do not mention color
tone, contrast, sharpness or noise: those are handled separately.

Answer only with JSON:
{{"differences": [{{"aspect": "<scene|place|ground|objects|size|count|camera|light|fake|other>",
   "real": "<what the photos show>", "synthetic": "<what the renders show>", "importance": <1 low .. 3 high>}}],
 "summary": "<one sentence: the biggest gap>"}}"""

PROPOSE = """We make synthetic training images for a small camera model that must {task} {classes}.
A vision model compared our renders with the user's real photos and found these differences:
{differences}

You can only change the scene through these levers (current values below):
{levers}

Current values:
{values}

Tags that exist on Poly Haven (most used first):
{vocab}

Earlier rounds (changes tried and whether the renders came closer to the photos):
{history}

Choose the changes most likely to bring the renders closer to the photos. Change few things at a time (at most
{max_changes}), the most important first, and do not repeat a change that did not help. If a difference cannot
be fixed with these levers, say what lever would be needed. Color tone, contrast, sharpness and noise are
handled separately: ignore them.

Answer only with JSON:
{{"changes": [{{"lever": "<lever name, e.g. scene.contexts.balcony.share>", "value": <new value>,
   "why": "<which difference it fixes>"}}],
 "cannot_fix": [{{"difference": "<short>", "needs": "<the missing lever or feature>"}}]}}"""

MAX_CHANGES = 2  # few at a time: with four, one bad change hid a good one (grass, 2026-09-29)


def _task_words(spec: Spec) -> str:
    return "count and locate" if spec.task == "count" else "recognize"


def _classes(spec: Spec) -> str:
    return ", ".join(c.description or c.id for c in spec.classes)


def compare(client, model: str, spec: Spec, mosaic_paths: list[Path]) -> list[dict]:
    """Differences found by the vision model, one call per mosaic (at the same time: each takes minutes), pooled."""
    from concurrent.futures import ThreadPoolExecutor

    from .llm import ask_json, image_part

    def one(m):
        msg = [{"role": "user", "content": [
            {"type": "text", "text": COMPARE.format(task=_task_words(spec), classes=_classes(spec))},
            image_part(m)]}]
        v = ask_json(client, msg, model, "match_compare", lambda v: isinstance(v.get("differences"), list),
                     temperature=0.0, max_tokens=8192)
        return [d for d in v["differences"] if isinstance(d, dict)] if v else []

    with ThreadPoolExecutor(len(mosaic_paths)) as pool:
        found = [d for ds in pool.map(one, mosaic_paths) for d in ds]
    return sorted(found, key=lambda d: -(d.get("importance") or 0))


def propose(client, model: str, spec: Spec, differences: list[dict], history: list[dict],
            vocab: dict[str, list[str]] | None = None, locked=()) -> dict:
    from .llm import ask_json
    levers = "\n".join(f"- {k}: {kind} {list(a) if isinstance(a, tuple) else a}; {hint}"
                       for k, (kind, a, hint) in LEVERS.items()
                       if (spec.task == "count" or k != "generation.max_targets") and k not in locked)
    if locked:
        values = lever_values(spec)
        levers += "\n\nFacts the user gave, true of the real photos (do not contradict them; explain differences " \
                  "another way): " + ", ".join(f"{k} = {json.dumps(values.get(k))}" for k in locked)
    past = "\n".join(f"- round {h['round']}: {json.dumps(_brief(h['applied']))} -> {h['result']}"
                     + (f"; refused: {json.dumps(_brief(h['rejected'], 'error'))}" if h["rejected"] else "")
                     for h in history) or "none"
    words = "\n".join(f"- {k}: {', '.join(v)}" for k, v in (vocab or {}).items()) or "(not available)"
    text = PROPOSE.format(task=_task_words(spec), classes=_classes(spec),
                          differences=json.dumps(differences, indent=1), levers=levers,
                          values=json.dumps(lever_values(spec), indent=1), history=past, max_changes=MAX_CHANGES, vocab=words)
    v = ask_json(client, [{"role": "user", "content": text}], model, "match_propose",
                 lambda v: isinstance(v.get("changes"), list), temperature=0.2, max_tokens=8192)
    return v or {"changes": [], "cannot_fix": []}


def _brief(changes: list[dict], extra: str | None = None) -> list[dict]:
    return [{"lever": c.get("lever"), "value": c.get("value"), **({extra: c.get(extra)} if extra else {})}
            for c in changes]


def poly_haven_tags(n: int = 60) -> tuple[dict[str, list[str]], object]:
    """The most used Poly Haven tags for skies and ground textures, and a check that some asset carries all
    of a list of tags (patch's tags_ok). Empty vocabulary and no check if Poly Haven cannot be reached."""
    from collections import Counter

    from . import polyhaven
    try:
        cats = {"scene.sky_tags": polyhaven.catalog("hdris"), "scene.ground_tags": polyhaven.catalog("textures")}
    except OSError:
        return {}, None
    vocab = {k: [t for t, _ in Counter(x for a in cat.values() for x in polyhaven._labels(a)).most_common(n)]
             for k, cat in cats.items()}
    return vocab, lambda lever, tags: bool(polyhaven.match(cats[lever], tags))


# ---------------------------------------------------------------- trial renders

def trial_spec(spec: Spec, frames: int = TRIAL_FRAMES) -> Spec:
    d = to_dict(spec)
    d["generation"]["frames"] = frames
    return from_dict(d)


def trial_job(parent: Job, source: Job, k: int) -> Job:
    """<parent>/match/it<k>, with copies of the stage folders `source` has up to the render (copies, not links:
    a stage that reruns there clears its own folder)."""
    from .pipeline import _deps
    from .stages import STAGES
    job = Job(parent.root / "match" / f"it{k}")
    shutil.rmtree(job.root, ignore_errors=True)
    job.root.mkdir(parents=True)
    job.logs.mkdir()
    job.input.symlink_to(parent.input.resolve(), target_is_directory=True)
    for name in _deps(STAGES, "render"):
        src = source.stage_dir(name)
        if src.is_dir():
            shutil.copytree(src, job.stage_dir(name), symlinks=True)
    return job


def render_trial(spec: Spec, made_with: Spec, job: Job, cache: Cache, backend) -> list[Path]:
    """Render `spec` in the trial job. The copied asset stages were made with `made_with`: those the change
    does not reach are taken as they are (their fingerprint is renewed, so code changes since then do not
    rerun the teacher or the image-to-3D model for a trial); the others rerun."""
    from .pipeline import Pipeline, _deps
    from .stages import STAGES
    keep = _deps(STAGES, "render") | {"render"}
    pool = [s for s in STAGES if s.name in keep]
    pipe = Pipeline(spec, job, cache, pool)
    old, new = Pipeline(made_with, job, cache, pool).fingerprints(), pipe.fingerprints()
    for name in keep - {"render"}:
        done = job.done(name)
        if done and old.get(name) == new.get(name):
            job.mark_done(name, new[name], {k: v for k, v in done.items() if k != "fingerprint"})
    backend.run(pipe)
    out = job.stage_dir("render")
    return [out / f"rgb_{lab['frame']:04d}.png" for lab in json.loads((out / "labels.json").read_text())]


# ---------------------------------------------------------------- the loop

def with_look(spec: Spec, user_look: dict) -> Spec:
    return from_dict({**to_dict(spec), "look": to_dict(LookSpec(**{
        k: user_look[k] for k in ("lab_mean", "lab_std", "sharpness", "noise", "n")}))})


def run(spec: Spec, job: Job, cache: Cache, backend, client, rounds: int = ROUNDS, frames: int = TRIAL_FRAMES,
        log=print, locked: tuple[str, ...] = ()) -> dict:
    """Returns the report; writes <job>/match/{report.json, spec.json}."""
    root = job.root / "match"
    root.mkdir(parents=True, exist_ok=True)
    compare_model, reason_model = client.model_for("compare"), client.model_for("reason")
    photos = reference_photos(spec, job)
    if len(photos) < 4:
        raise ValueError("match needs at least 4 of the user's training photos")
    real = small(photos)
    user_look = look.measure(real)
    spec = with_look(spec, user_look)
    feats = Features()
    real_f = feats(real)
    half = len(real_f) // 2
    floor = kid(real_f[:half], real_f[half:])
    log(f"{len(photos)} photos of the user; their look: {user_look}")
    log(f"floor (one half of the photos against the other): KID {floor}")

    def score(renders: list[Path], tag: str) -> dict:
        imgs = small(renders)
        t = look.fit(user_look, look.measure(imgs))
        rng = random.Random(0)
        looked = [look.apply(i, t, rng) for i in imgs]
        rnd = random.Random(1)
        pick = rnd.sample(range(len(renders)), min(MOSAIC * 2, len(renders)))
        sample = rnd.sample(photos, min(MOSAIC * 2, len(photos)))
        mos = [mosaic(sample[:MOSAIC], [looked[i] for i in pick[:MOSAIC]], root / f"{tag}_a.jpg"),
               mosaic(sample[MOSAIC:] or sample, [looked[i] for i in pick[MOSAIC:]] or looked, root / f"{tag}_b.jpg")]
        return {"kid": kid(real_f, feats(looked)), "kid_raw": kid(real_f, feats(imgs)),
                "look_gap": look.distance(user_look, look.measure(looked)), "mosaics": [str(m) for m in mos]}

    vocab, tags_ok = poly_haven_tags()
    best_job = trial_job(job, job, 0)
    best_spec = spec
    log(f"round 0: trial render of {frames} frames")
    best = score(render_trial(trial_spec(spec, frames), trial_spec(spec, frames), best_job, cache, backend),
                 "round0")
    log(f"round 0: KID {best['kid']} (without the camera look {best['kid_raw']})")
    start = dict(best)
    history, missing = [], []

    def save(end: dict, end_spec: Spec) -> dict:
        report = {"photos": len(photos), "user_look": user_look, "floor": floor, "start": start, "end": end,
                  "rounds": history, "cannot_fix": missing, "frames": frames,
                  "models": {"compare": compare_model, "reason": reason_model}}
        (root / "report.json").write_text(json.dumps(report, indent=1))
        (root / "spec.json").write_text(end_spec.to_json() + "\n")
        return report

    for k in range(1, rounds + 1):
        diffs = compare(client, compare_model, best_spec, [Path(m) for m in best["mosaics"]])
        log(f"round {k}: {len(diffs)} differences, e.g. " + "; ".join(
            f"{d.get('synthetic')} (real: {d.get('real')})" for d in diffs[:3]))
        prop = propose(client, reason_model, best_spec, diffs, history, vocab, locked)
        missing += [m for m in prop.get("cannot_fix", []) if isinstance(m, dict)]
        cand, applied, rejected = patch(best_spec, prop.get("changes", []), tags_ok, locked)
        for r in rejected:
            log(f"  rejected {r.get('lever')}: {r['error']}")
        if not applied:
            log(f"round {k}: nothing to change, stop")
            history.append({"round": k, "differences": diffs, "applied": [], "rejected": rejected, "kept": False,
                            "result": "nothing changed", "kid": best["kid"]})
            break
        for a in applied:
            log(f"  {a['lever']} = {json.dumps(a['value'])}  ({a.get('why', '')})")
        tj = trial_job(job, best_job, k)
        try:
            got = score(render_trial(trial_spec(cand, frames), trial_spec(best_spec, frames), tj, cache, backend),
                        f"round{k}")
        except (StageFailed, ValueError, OSError) as e:  # e.g. no model found for a new prop: undo, go on
            log(f"round {k}: the trial render failed ({e}), undone")
            history.append({"round": k, "differences": diffs, "applied": applied, "rejected": rejected,
                            "kept": False, "result": f"undone, the render failed: {e}", "kid": None})
            save(best, best_spec)
            continue
        kept = got["kid"] < best["kid"] - MIN_GAIN
        log(f"round {k}: KID {got['kid']} vs {best['kid']}: {'kept' if kept else 'undone'}")
        history.append({"round": k, "differences": diffs, "applied": applied, "rejected": rejected, "kept": kept,
                        "result": f"{'kept, closer' if kept else 'undone, not closer'} (KID {got['kid']} "
                                  f"vs {best['kid']})", **got})
        if kept:
            best, best_spec, best_job = got, cand, tj
        save(best, best_spec)
    report = save(best, best_spec)
    log(f"KID {start['kid_raw']} (renders as they were) -> {start['kid']} (camera look) -> {best['kid']} "
        f"(scene rounds); floor {floor}")
    return report

