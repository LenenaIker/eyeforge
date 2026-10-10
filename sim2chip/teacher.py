"""The teachers that label the user's raw photos. Which one depends on the task (docs/profesor_modelos.md):

- Count: SAM 3 (Meta), open-vocabulary segmentation from a short noun phrase per class. It scores every object
  it finds, so the threshold follows the user's error preference (spec.errors): chosen on the photos the user
  corrected, or a default when there are none.
- Classification: SigLIP 2 (Google), image and text embeddings. A photo gets the class whose centre is nearest:
  the class description in English, plus the average of the photos the user put in that class. It learns the
  user's own criterion from a few corrections (lawn test 2026-09-30: 0.87 right with 6 corrected photos, SAM 3
  0.60, GLM 0.41), which SAM 3 cannot: it looks for objects, not for what the user means by a class.

Both need a GPU and their own Python (torch, transformers >= 5): SIM2CHIP_SAM3_PYTHON (environment or .env),
or this Python when it has it. SAM 3's weights are gated on Hugging Face: accept the license there and give
HF_TOKEN (or log in with `hf auth login`); HF_HUB_CACHE says where they go. In a Nebius job there is no token:
the weights come from the cache bucket (weights()). This module is stdlib only (boto3 only for that download).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

MODEL = "facebook/sam3"
NAME = "SAM 3"
SIDE = 1024  # pixels of the longest side: small, distant objects need them
FLOOR = 0.2  # lowest score kept, so the threshold can move without running SAM 3 again
# F-beta per preference: find more objects (recall), balanced, or fewer false marks (precision).
BETA = {"find_more": 2.0, "balanced": 1.0, "fewer_false": 0.5}
# Threshold without corrections to learn from (0.5 looked best to the user in the benchmark).
DEFAULT = {"find_more": 0.4, "balanced": 0.5, "fewer_false": 0.6}
THRESHOLDS = [round(0.2 + 0.05 * i, 2) for i in range(15)]  # 0.2 .. 0.9
MIN_CORRECTED = 2  # fewer corrected photos than this say too little to move the threshold
# Nor do fewer marked objects than this: 1 bag on 2 photos moved it to 0.85 and left 2 of 6 photos empty
# (waste_dataset_classes, 2026-10-10).
MIN_MARKS = 5

SIGLIP_MODEL = "google/siglip2-large-patch16-384"  # base: 2 points worse; so400m: no better (2026-09-30)
SIGLIP_NAME = "SigLIP 2"
TEXT_WEIGHT = 1.0  # the description beside the examples: steadier with few of them (worst split 0.67 -> 0.79)


class TeacherError(RuntimeError):
    pass


def python() -> list[str] | None:
    """The interpreter that can run SAM 3 here, or None."""
    from .config import env
    exe = env("SIM2CHIP_SAM3_PYTHON")
    if exe:
        return [exe] if Path(exe).exists() else None
    if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
        return [sys.executable]
    return None


def available() -> bool:
    return python() is not None


def names(c) -> list[str]:
    """The class's short English names: its own (class.names, from the interview), else those of its 3D model
    search (queries or categories), else none."""
    a = c.assets
    return [n.replace("_", " ").strip() for n in (c.names or ((a.queries or a.categories) if a else []))[:3]]


def phrases(spec) -> dict[str, list[str]]:
    """Short English noun phrases per class, as SAM 3 wants them: the 3D model search queries or categories
    the interview wrote (already short English names), else the class id. A job without 3D models has no such
    names and its ids are often in the user's language: stages.label puts those in English (english_phrases)."""
    return {c.id: names(c) or [c.id.replace("_", " ")] for c in spec.classes}


PHRASE_ASK = """SAM 3, an object detector that understands only English, looks for each class below by short
noun phrases. The classes come with their descriptions, often in another language. For each class write 1 to 3
short English noun phrases (1 to 3 words, singular, the most common name first), translating every word and
never copying the original language. Example: {"maceta": "macetas vistas desde cualquier ángulo"} ->
{"maceta": ["flower pot", "plant pot"]}. Answer only JSON {"<class id>": ["<phrase>", ...]}.
Classes:
"""


def english_phrases(meaning: dict[str, str], client=None) -> dict[str, list[str]] | None:
    """{class id: description} -> {class id: English noun phrases} from the text model, or None without an answer
    for every class. Without it SAM 3 gets "maceta" and finds nothing (pots job, 2026-09-30: 0 in 25 photos)."""
    from .llm import parse_json
    try:
        from .llm import Client
        client = client or Client()
        out = parse_json(client.chat([{"role": "user", "content": PHRASE_ASK + json.dumps(meaning, ensure_ascii=False)}],
                                     client.model_for("text"), purpose="teacher_phrases", temperature=0.0,
                                     max_tokens=512))
    except Exception as e:  # the caller warns that the classes stay untranslated
        print(f"English phrases for {NAME} failed: {type(e).__name__}: {e}", flush=True)
        return None
    if not isinstance(out, dict):
        return None
    found = {k: [p.strip() for p in out[k] if isinstance(p, str) and p.strip()][:3]
             for k in meaning if isinstance(out.get(k), list)}
    return found if all(found.get(k) for k in meaning) else None


def _served(module: str, name: str, req: dict, show) -> bool:
    """The request to the teachers kept loaded on the warm GPU (teacher_server.py), when there is one. False when
    it cannot be reached or does not know the request: the caller runs the worker itself."""
    import socket
    from .teacher_server import DONE, VERSION, WORKERS
    path = os.environ.get("SIM2CHIP_TEACHER_SOCKET")
    teacher = next((k for k, m in WORKERS.items() if m == module), None)
    if not path or teacher is None or not Path(path).exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.connect(path)
            s.sendall((json.dumps({"v": VERSION, "teacher": teacher, "req": req}) + "\n").encode())
            for line in s.makefile("r", encoding="utf-8"):
                if line.startswith(DONE):
                    code, _, why = line[len(DONE):].strip().partition(" ")
                    if code != "0":
                        raise TeacherError(f"{name} failed on the warm GPU: {why}")
                    return True
                show(line)
    except OSError:
        return False
    return False  # the server went away mid-answer: run it here


def _worker(module: str, name: str, req: dict, log=print) -> None:
    """Runs a teacher worker in the teacher's Python, the request on stdin, its progress lines to `log`. On the warm
    GPU the teachers stay loaded in teacher_server.py, which answers instead."""

    def show(line: str) -> None:
        if line.startswith((f"{name} on", "  ")) or "/" in line.split(" ", 1)[0]:
            log(f"  {line.rstrip()}")

    if _served(module, name, req, show):
        return
    exe = python()
    if exe is None:
        raise TeacherError(f"the teacher ({name}) needs a GPU Python with torch and transformers: set "
                           "SIM2CHIP_SAM3_PYTHON, or run with --backend nebius")
    from .config import env
    from .subproc import python_env, run_module
    extra = {k: v for k in ("HF_HUB_CACHE", "HF_TOKEN", "TORCH_HOME") if (v := env(k))}
    code, tail = run_module(exe, module, request=req, on_line=show, env=python_env(extra))
    if code != 0:
        raise TeacherError(f"{name} failed (exit {code}):\n" + "\n".join(tail[-10:]))


WEIGHTS_PREFIX = "weights"  # in the cache bucket: weights/<repo id>/<file>, uploaded by scripts/upload_weights.py


def weights(repo: str, log=print, timeline=None) -> str:
    """What the worker loads for a Hugging Face model: the repo id (Hugging Face and its cache), or, in a cloud
    job (SIM2CHIP_WEIGHTS_DIR and S2C_CACHE_BUCKET set), a local folder pulled from the cache bucket the first
    time: no token for gated weights, and the bucket is in the job's region (user's choice, 2026-10-04)."""
    root = os.environ.get("SIM2CHIP_WEIGHTS_DIR")
    bucket = os.environ.get("S2C_CACHE_BUCKET")
    if not (root and bucket):
        return repo
    dst = Path(root) / repo
    if not (dst / "config.json").exists():
        import time
        from .cloud_sync import S3Store
        t0 = time.time()
        store = S3Store()
        keys = list(store.listing(bucket, f"{WEIGHTS_PREFIX}/{repo}/"))
        if not keys:
            raise TeacherError(f"no weights for {repo} in s3://{bucket}/{WEIGHTS_PREFIX}/: run "
                               "scripts/upload_weights.py")
        store.pull(bucket, keys, dst, strip=f"{WEIGHTS_PREFIX}/{repo}")
        seconds = time.time() - t0
        log(f"  {repo}: weights from the cache bucket in {seconds:.0f} s")
        if timeline:
            timeline(f"cloud: pull {repo.split('/')[-1]} weights", seconds)
    return str(dst)


def detect(items: list[tuple[Path, dict[str, Path]]], log=print, timeline=None) -> None:
    """Runs SAM 3 on (photo, {noun phrase: answer file}) pairs; each answer file gets what that phrase found on the
    photo (sam3_worker)."""
    _worker("sim2chip.sam3_worker", NAME, {"model": weights(MODEL, log, timeline), "side": SIDE, "floor": FLOOR,
                                           "items": [{"path": str(p), "outs": {k: str(o) for k, o in outs.items()}}
                                                     for p, outs in items]}, log)


def nms(dets: list[dict], iou: float = 0.5) -> list[dict]:
    """Same-class boxes from several phrases of one class are one object: keep the most confident."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        x0, y0, x1, y1 = d["box"]
        for k in keep:
            if k["class"] != d["class"]:
                continue
            a0, b0, a1, b1 = k["box"]
            inter = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
            union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
            if union > 0 and inter / union > iou:
                break
        else:
            keep.append(d)
    return keep


def embed(items: list[tuple[Path, Path]], texts: list[str], text_out: Path, log=print, timeline=None) -> None:
    """Runs SigLIP 2 on (photo, answer file) pairs and on the sentences (their embeddings to text_out; with no
    sentences nothing is written there)."""
    _worker("sim2chip.siglip_worker", SIGLIP_NAME,
            {"model": weights(SIGLIP_MODEL, log, timeline), "texts": texts, "text_out": str(text_out),
             "items": [{"path": str(p), "out": str(o)} for p, o in items]}, log)


PROMPT_ASK = """Each class below is one kind of photo a small camera model must recognize, with its description,
often in another language. For each class write one short sentence IN ENGLISH (translate every word, never
copy the original language) that describes what such a photo shows, starting with "a photo of". Keep every
detail of the description and add nothing. Example: {"hole": "hueco en la valla"} -> {"hole": "a photo of a
gap in a fence"}. Answer only JSON {"<class id>": "<English sentence>"}.
Classes:
"""


def prompts(spec, client=None) -> dict[str, str]:
    """An English sentence per label for SigLIP: written by the text model from the class descriptions, or the
    class id and description as they are when no model answers."""
    from .llm import parse_json
    meaning = {c.id: c.description or c.id.replace("_", " ") for c in spec.classes}
    if spec.background_class:
        meaning = {spec.background_class: "none of these is in the photo: " + ", ".join(meaning.values()),
                   **meaning}
    plain = {k: f"a photo of {k.replace('_', ' ')}: {v}" for k, v in meaning.items()}
    try:
        from .llm import Client
        client = client or Client()
        out = parse_json(client.chat([{"role": "user", "content": PROMPT_ASK + json.dumps(meaning, ensure_ascii=False)}],
                                     client.model_for("text"), purpose="teacher_prompts", temperature=0.0,
                                     max_tokens=1024))
    except Exception:  # no key or no answer: the words as they are still work, a little worse
        return plain
    return {k: out[k] if isinstance(out, dict) and isinstance(out.get(k), str) and out[k].strip() else plain[k]
            for k in meaning}


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def centres(text: dict[str, list[float]], examples: dict[str, list[list[float]]]) -> dict[str, list[float]]:
    """Each label's centre: its description, plus the mean of the user's photos in it. Photo to photo scores
    are far higher than photo to text ones, so examples are used only when every label has some: a label with
    only its description would never win."""
    if not all(examples.get(k) for k in text):
        return dict(text)
    out = {}
    for k, t in text.items():
        mean = _unit([sum(col) / len(examples[k]) for col in zip(*examples[k])])
        out[k] = _unit([m + TEXT_WEIGHT * x for m, x in zip(mean, t)])
    return out


def nearest(embedding: list[float], centre: dict[str, list[float]]) -> tuple[str, float]:
    """The nearest label and how clearly it wins (softmax share, for review only)."""
    scores = {k: _dot(embedding, c) for k, c in centre.items()}
    top = max(scores.values())
    exp = {k: math.exp(50 * (s - top)) for k, s in scores.items()}
    best = max(scores, key=scores.get)
    return best, round(exp[best] / sum(exp.values()), 3)


def spread(embeddings: dict[str, list[float]], keep: list[str], n: int) -> list[str]:
    """n more keys as different as possible from each other and from `keep` (farthest point first): photos to
    ask the user about that cover the folder, instead of five of the same corner."""
    rest = [k for k in sorted(embeddings) if k not in keep]
    # Each photo's highest similarity to the chosen ones, kept up to date as photos are chosen: ordering a few
    # hundred photos for the next calibration rounds stays a matter of seconds.
    near = {r: max((_dot(embeddings[r], embeddings[c]) for c in keep), default=None) for r in rest}
    picked = []
    while rest and len(picked) < n:
        if near[rest[0]] is not None:
            k = min(rest, key=lambda r: near[r])
        else:  # the most typical photo first
            mean = [sum(col) for col in zip(*embeddings.values())]
            k = max(rest, key=lambda r: _dot(embeddings[r], mean))
        picked.append(k)
        rest.remove(k)
        for r in rest:
            s = _dot(embeddings[r], embeddings[k])
            near[r] = s if near[r] is None else max(near[r], s)
    return picked


def explain_examples(examples: int, used: bool) -> str:
    """One line for review: what the classification teacher learned from the corrections."""
    if used:
        return (f"Sorts each photo by how close it looks to the {examples} photos you corrected and to the class "
                "descriptions. Every correction teaches it your criterion.")
    return ("Sorts each photo by the class descriptions only. Correct at least one photo of each class and it "
            "learns your criterion from them.")


def objects(dets: list[dict], threshold: float) -> list[dict]:
    """Count task: the detections above the threshold as objects (centre point, box kept for review)."""
    return [{"x": round((d["box"][0] + d["box"][2]) / 2, 4), "y": round((d["box"][1] + d["box"][3]) / 2, 4),
             "class": d["class"], "box": d["box"], "score": d["score"]}
            for d in dets if d["score"] >= threshold]


OVERLAP = 0.7  # boxes of two classes overlapping this much (IoU) are one object found under both names


def iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_objects(dets: list[dict]) -> list[list[int]]:
    """Detections that are one object found under several classes: SAM 3 looks for each phrase on its own, so
    classes that are the same thing told apart by an attribute (a blue and a yellow recycling bin) get the same
    box once per class. Groups of indices into `dets`, only those holding two classes or more."""
    parent = list(range(len(dets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if dets[i]["class"] != dets[j]["class"] and iou(dets[i]["box"], dets[j]["box"]) >= OVERLAP:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(dets)):
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len({dets[i]["class"] for i in g}) > 1]


def crop_box(box: list[float], margin: float = 0.1) -> list[float]:
    """The box with a margin around it, inside the photo: SigLIP sees the object and a little of its place."""
    w, h = box[2] - box[0], box[3] - box[1]
    return [max(0.0, box[0] - margin * w), max(0.0, box[1] - margin * h),
            min(1.0, box[2] + margin * w), min(1.0, box[3] + margin * h)]


def match(boxes: list[dict], points: list) -> int:
    """True positives: user points inside a teacher box of the same class, one to one, nearest centre first."""
    used, tp = set(), 0
    for x, y, cls in points:
        inside = [(abs((b["box"][0] + b["box"][2]) / 2 - x) + abs((b["box"][1] + b["box"][3]) / 2 - y), i)
                  for i, b in enumerate(boxes) if i not in used and b["class"] == cls
                  and b["box"][0] <= x <= b["box"][2] and b["box"][1] <= y <= b["box"][3]]
        if inside:
            used.add(min(inside)[1])
            tp += 1
    return tp


def fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return (1 + beta ** 2) * p * r / max(1e-9, beta ** 2 * p + r)


def choose(preference: str, dets: dict[str, list[dict]], truth: dict) -> dict:
    """Count: the threshold for the user's preference, the best F-beta over the corrected photos (truth: key ->
    points), or the default when there are too few of them."""
    beta, default = BETA[preference], DEFAULT[preference]
    keys = [k for k in truth if k in dets]
    if len(keys) < MIN_CORRECTED or sum(len(truth[k]) for k in keys) < MIN_MARKS:
        return {"threshold": default, "preference": preference, "photos": len(keys)}

    def score(t):
        tp = fp = fn = 0
        for k in keys:
            boxes = [d for d in dets[k] if d["score"] >= t]
            pts = [p for p in truth[k] if len(p) > 2]
            a = match(boxes, [tuple(p[:3]) for p in pts])
            tp, fp, fn = tp + a, fp + len(boxes) - a, fn + len(pts) - a
        return fbeta(tp, fp, fn, beta)

    # Ties go to the threshold nearest the default: a flat score says nothing about where to move.
    best = max(THRESHOLDS, key=lambda t: (round(score(t), 4), -abs(t - default)))
    return {"threshold": best, "preference": preference, "photos": len(keys), "fbeta": round(score(best), 3)}


def explain(chosen: dict) -> str:
    """One line for review: what the teacher learned from the corrections."""
    what = {"find_more": "find every object, even with some false marks",
            "balanced": "balance missed objects and false marks",
            "fewer_false": "avoid false marks, even if it misses some"}[chosen["preference"]]
    if "fbeta" not in chosen:  # too few corrections to learn from
        return (f"Marks what it is at least {chosen['threshold']:.0%} sure of, to {what}. Correct a few photos "
                "and it tunes this to them.")
    return (f"Marks what it is at least {chosen['threshold']:.0%} sure of: the setting that best matches your "
            f"{chosen['photos']} corrected photos, to {what}.")
