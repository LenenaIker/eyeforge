"""Teacher: label the user's raw photos with a vision model (Nemotron on Token Factory).

Classification: each photo gets one class of the spec (background included) or "unsure". Photos below the
confidence threshold are set aside for human review instead of being used.
Count task: the teacher boxes every object in the photo; its centre becomes the object's point. Photos
without a valid answer go to "unsure"; photos with no objects are kept (they teach what "none" looks like).
Teacher answers are kept in the shared cache by photo content, model and prompt: the same photo gets the
same answer in every job (so test scores compare across jobs) and is never paid for twice. The user's
corrections from review (input/corrections.json) override the teacher.
The teacher also learns from them: each corrected photo where it disagreed with the user gives a lesson (the
vision model sees its own marks next to the user's and writes the general rule it missed), and the lessons,
merged into a few rules, go into the prompt. The rules change the prompt, so the next run relabels every
photo with them. Lessons are kept per photo (label/lessons.json): a photo the teacher now gets right still
teaches, so the rules do not flip back and forth between runs.
Output: label/<class>/<photo> (copies) or, for count, label/images/<photo>; label/unsure/, label/verdicts.json,
label/lessons.json
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from .. import review
from ..workspace import IMAGE_EXT, content_key, copy_name
from .base import Context, Stage

PROMPT = """You label photos for training a small camera classifier. Classes:
{classes}

Look at the photo and pick the class of the main object in it. Use "{bg}" when none of the other classes is
present. Use "unsure" if the photo is ambiguous, too dark or blurry to tell, or shows several different classes.

Answer only with JSON: {{"class": "<one of: {ids}, unsure>", "confidence": <0 to 1>, "reason": "<short>"}}"""

COUNT_PROMPT = """You mark objects in photos to train a small camera model that counts them. Objects to find:
{classes}

Find every one of them in the photo, including small, distant and partly hidden ones. Mark each object
separately, even when several stand side by side. A box holds the whole visible object as described above,
with everything that belongs to it (a pot and the plant growing in it), not just its main part.

Answer only with JSON: {{"objects": [{{"class": "<one of: {ids}>", "box": [x0, y0, x1, y1]}}]}} with each box
in pixel coordinates of this image, which is {w} x {h} pixels. Use an empty list when there are none."""

RULES = """

Rules learned from the user's corrections of earlier answers (they win over everything above):
{rules}"""

LESSON_COUNT = """A teacher model marked objects in this photo for a dataset that teaches a small camera model to
count them, and the user corrected it. Objects to find:
{classes}

Red boxes: the {n_teacher} objects the teacher marked. Green dots: the {n_user} objects the user says are
really there, one dot per object, on the object itself. Compare them: which objects did the teacher box wrongly
(a part of the object, or the thing next to it, instead of the object), miss, or add?

Write the general rule the teacher should follow from now on, in one sentence (which part of the scene is one
object, what to include or leave out), not a description of this photo.
Answer only with JSON: {{"lesson": "<rule>"}}"""

LESSON_CLASS = """A teacher model labeled this photo for a dataset that teaches a small camera classifier, and the
user corrected it. Classes:
{classes}

The teacher said "{teacher}" ({reason}). The user says "{user}". Write the general rule the teacher should
follow from now on to tell these classes apart, in one sentence, not a description of this photo.
Answer only with JSON: {{"lesson": "<rule>"}}"""

MERGE = """A teacher model labels photos for a small camera model. The user corrected some of its answers, and
each correction gave a lesson. Classes:
{classes}

Lessons:
{lessons}

Merge them into at most 5 short, general rules for the teacher: keep what the user wants, drop duplicates and
anything that only fits one photo.
Answer only with JSON: {{"rules": ["<rule>"]}}"""


class Label(Stage):
    name = "label"
    runtime = "cpu"
    needs_llm = True
    reads_input = True
    rereads = ("verdicts.json", "lessons.json")  # the user's corrections, earlier answers and lessons
    uses_cache = True
    uses = ("user_data", "classes", "background_class", "task")

    def enabled(self, spec) -> bool:
        return bool(spec.user_data.raw)

    def run(self, ctx: Context) -> dict:
        from PIL import Image, ImageOps

        from ..llm import WORKERS, Client, LLMError, ask_json, image_part

        ud, spec = ctx.spec.user_data, ctx.spec
        src = ctx.job.resolve_input(ud.raw)
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise FileNotFoundError(f"user_data.raw: no photos in {src}")
        out = ctx.job.stage_dir(self.name)
        client = Client(usage_log=ctx.job.logs / "llm_usage.jsonl")
        model = client.model_for("vision")
        descr = {c.id: c.description or c.id for c in spec.classes}
        counting = spec.task == "count"
        if counting:
            ids = [c.id for c in spec.classes]
            prompt = COUNT_PROMPT.format(classes="\n".join(f"- {k}: {v}" for k, v in descr.items()),
                                         ids=", ".join(ids), w="{w}", h="{h}")  # size filled per photo
        else:
            classes = "\n".join([f"- {spec.background_class}: none of the classes below, only background"] +
                                [f"- {k}: {v}" for k, v in descr.items()])
            prompt = PROMPT.format(classes=classes, bg=spec.background_class, ids=", ".join(spec.labels))
        previous = json.loads((out / "verdicts.json").read_text()) if (out / "verdicts.json").exists() else {}
        # Corrected photos keep the teacher's own verdict next to the user's: restore it before reusing.
        teacher = {k: (({**v, "objects": v["teacher_objects"]} if "teacher_objects" in v else
                        {**v, "class": v["teacher"], "confidence": v["teacher_confidence"]}) | {"judge": "teacher"}
                       if v.get("judge") == "user" else v) for k, v in previous.items()}
        corrections = review.load_corrections(ctx.job)
        fixes, fix_points = corrections["labels"], corrections.get("points", {})
        learned = learn(ctx, client, src, teacher, fixes, fix_points, descr, counting)
        if learned["rules"]:
            prompt += RULES.format(rules="\n".join(f"- {r}" for r in learned["rules"]))
            ctx.log("teacher rules from your corrections:\n" + "\n".join(f"  - {r}" for r in learned["rules"]))
        prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        reuse = {k: v for k, v in teacher.items() if v.get("model") == model and v.get("prompt") == prompt_id
                 and v.get("reason") != "no valid answer"}
        shared = ctx.cache.root / "labels" / prompt_id / re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        shared.mkdir(parents=True, exist_ok=True)
        keys = {f: content_key(f) for f in files}
        for f in files:
            k, hit = str(f.relative_to(src)), shared / f"{keys[f]}.json"
            if hit.exists():  # the shared answer wins over this job's older one: one truth per photo
                reuse[k] = json.loads(hit.read_text())
        shutil.rmtree(out, ignore_errors=True)
        small = out / "_small"
        small.mkdir(parents=True)
        todo = [f for f in files if str(f.relative_to(src)) not in reuse]
        ctx.log(f"labeling {len(todo)} photos with {model} ({len(files) - len(todo)} already labeled)")

        def ask(i, f):
            # Downscale before sending: the model does not need more pixels than a cheap camera has.
            tmp = small / f"{i:06d}.jpg"
            # Counting needs the small, distant objects too, so it sends more pixels.
            side = 1024 if counting else 512
            with Image.open(f) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((side, side))
                im.save(tmp, quality=90)
                w, h = im.size
            text = prompt.replace("{w}", str(w)).replace("{h}", str(h))
            msg = [{"role": "user", "content": [{"type": "text", "text": text}, image_part(tmp)]}]
            if counting:
                v = ask_json(client, msg, model, "label", lambda v: objects_from(v, ids, w, h) is not None,
                             temperature=0.0, max_tokens=8192)
                return {"objects": objects_from(v, ids, w, h)} if v else None
            return ask_json(client, msg, model, "label", lambda v: v.get("class") in spec.labels + ["unsure"],
                            temperature=0.0, max_tokens=2048)

        def one(item):
            i, f = item
            try:
                v = ask(i, f)
            except (OSError, LLMError) as e:  # a broken photo or a failed call costs that photo only
                ctx.log(f"  {f.name}: {e}")
                v, failed = None, True
            else:
                failed = False
            if v is not None:
                # Cached as soon as it arrives: a later failure must not throw away answers already paid.
                (shared / f"{keys[f]}.json").write_text(json.dumps({**v, "model": model, "prompt": prompt_id}))
                return f, v, False
            if counting:
                return f, {"objects": None, "reason": "no valid answer"}, failed
            return f, {"class": "unsure", "confidence": 0, "reason": "no valid answer"}, failed

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            answers = list(pool.map(one, enumerate(todo)))
        if todo and all(failed for _, _, failed in answers):
            raise RuntimeError(f"the teacher failed on all {len(todo)} photos, see the lines above")
        fresh = {f: v for f, v, _ in answers}
        results = []
        for f in files:
            key = str(f.relative_to(src))
            v = fresh.get(f) or reuse[key]
            v = {**v, "model": model, "prompt": prompt_id}
            if counting and key in fix_points:  # the user's marks win over the teacher's
                v = {**v, "teacher_objects": v["objects"], "judge": "user",
                     "objects": [{"x": x, "y": y, "class": c} for x, y, c in fix_points[key] if c in ids]}
            elif not counting and key in fixes:
                v = {**v, "teacher": v["class"], "teacher_confidence": v.get("confidence", 0),
                     "class": fixes[key], "confidence": 1.0, "judge": "user"}
            results.append((f, v))
        shutil.rmtree(small, ignore_errors=True)
        counts: dict[str, int] = {}
        verdicts = {}
        for f, v in results:
            if counting:
                cls = "unsure" if v["objects"] is None else "images"
            else:
                cls = v["class"] if float(v.get("confidence", 0)) >= ud.label_min_confidence else "unsure"
            d = out / cls
            d.mkdir(exist_ok=True)
            shutil.copy2(f, d / copy_name(str(f.relative_to(src))))
            counts[cls] = counts.get(cls, 0) + 1
            verdicts[str(f.relative_to(src))] = {**v, "assigned": cls}
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        (out / "lessons.json").write_text(json.dumps(learned, indent=1))
        if counting:
            n = sum(len(v["objects"]) for v in verdicts.values() if v["objects"] is not None)
            ctx.log(f"{counts.get('images', 0)} photos with {n} objects marked, {counts.get('unsure', 0)} unsure")
            return {"labels": counts, "objects": n}
        ctx.log(f"labels {counts}")
        return {"labels": counts}


def objects_from(answer: dict, ids: list[str], w: int, h: int) -> list[dict]:
    """The teacher's boxes (pixels of the image it saw) as points in 0..1, keeping the box for review."""
    out = []
    for o in answer["objects"]:
        cls = o.get("class", ids[0] if len(ids) == 1 else None)
        x0, y0, x1, y1 = (float(v) for v in o["box"])
        x0, x1 = sorted((min(max(x0, 0), w), min(max(x1, 0), w)))
        y0, y1 = sorted((min(max(y0, 0), h), min(max(y1, 0), h)))
        if cls not in ids or x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append({"x": round((x0 + x1) / 2 / w, 4), "y": round((y0 + y1) / 2 / h, 4), "class": cls,
                    "box": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]})
    return out


def disagree(teacher: list[dict] | None, user: list, near: float = 0.06) -> bool:
    """The teacher's objects differ from the user's marks: another count, or a mark with no teacher point
    near it (the teacher boxed something else, like the plant above the pot)."""
    if teacher is None or len(teacher) != len(user):
        return True
    left = [(o["x"], o["y"]) for o in teacher]
    for x, y, *_ in user:
        if not left:
            return True
        i = min(range(len(left)), key=lambda j: (left[j][0] - x) ** 2 + (left[j][1] - y) ** 2)
        if (left[i][0] - x) ** 2 + (left[i][1] - y) ** 2 > near ** 2:
            return True
        left.pop(i)
    return False


def learn(ctx: Context, client, src, teacher: dict, fixes: dict, fix_points: dict, descr: dict,
          counting: bool) -> dict:
    """Lessons from the corrected photos where the teacher was wrong, and the rules merged from them.
    A lesson is asked once per photo and correction; the merge only when the lessons change."""
    from PIL import Image, ImageDraw, ImageOps

    from ..llm import LLMError, ask_json, image_part

    out = ctx.job.stage_dir("label")
    path = out / "lessons.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    lessons = {}
    classes = "\n".join(f"- {k}: {v}" for k, v in descr.items())
    model = None
    for key, marks in sorted((fix_points if counting else fixes).items()):
        mark_id = hashlib.sha256(json.dumps(marks).encode()).hexdigest()[:12]
        if old.get("lessons", {}).get(key, {}).get("marks") == mark_id:
            lessons[key] = old["lessons"][key]
            continue
        t = teacher.get(key)
        f = src / key
        if t is None or not f.exists():
            continue
        if counting and not disagree(t.get("objects"), marks) or not counting and t.get("class") == marks:
            continue
        model = model or client.model_for("vision")
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "_lesson.jpg"
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1024, 1024))
            w, h = im.size
            if counting:  # the teacher's boxes in red, the user's points in green
                d, r = ImageDraw.Draw(im), max(4, w // 90)
                for o in t.get("objects") or []:
                    if o.get("box"):
                        x0, y0, x1, y1 = o["box"]
                        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(255, 0, 0), width=max(2, w // 300))
                    else:
                        d.ellipse([o["x"] * w - r, o["y"] * h - r, o["x"] * w + r, o["y"] * h + r], outline=(255, 0, 0))
                for x, y, *_ in marks:
                    d.ellipse([x * w - r, y * h - r, x * w + r, y * h + r], fill=(0, 255, 0), outline=(0, 0, 0))
            im.save(tmp, quality=90)
        if counting:
            text = LESSON_COUNT.format(classes=classes, n_teacher=len(t.get("objects") or []), n_user=len(marks))
        else:
            text = LESSON_CLASS.format(classes=classes, teacher=t.get("class"), user=marks,
                                       reason=t.get("reason", "no reason given"))
        msg = [{"role": "user", "content": [{"type": "text", "text": text}, image_part(tmp)]}]
        try:
            v = ask_json(client, msg, model, "label_lesson", lambda v: isinstance(v.get("lesson"), str),
                         temperature=0.0, max_tokens=2048)
        except LLMError as e:  # no lesson this time: asked again on the next run
            ctx.log(f"  lesson from {key}: {e}")
            v = None
        tmp.unlink(missing_ok=True)
        if v:
            lessons[key] = {"marks": mark_id, "lesson": " ".join(v["lesson"].split())[:300]}
    texts = sorted({x["lesson"] for x in lessons.values()})
    of = hashlib.sha256(json.dumps(texts).encode()).hexdigest()[:12]
    rules = old.get("rules", []) if old.get("rules_of") == of else texts[:1]
    if len(texts) > 1 and old.get("rules_of") != of:
        try:
            v = ask_json(client, [{"role": "user", "content": MERGE.format(
                classes=classes, lessons="\n".join(f"- {x}" for x in texts))}], client.model_for("text"),
                "label_rules", lambda v: isinstance(v.get("rules"), list) and v["rules"], temperature=0.0,
                max_tokens=4096)
        except LLMError as e:
            ctx.log(f"  merging lessons: {e}")
            v = None
        rules = [" ".join(str(r).split())[:300] for r in v["rules"][:5]] if v else texts[:5]
    return {"lessons": lessons, "rules": rules, "rules_of": of}
