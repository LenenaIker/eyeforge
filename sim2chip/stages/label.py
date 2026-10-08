"""Teacher: label the user's raw photos (see teacher.py for why each model).

Classification (SigLIP 2): every photo is embedded once; a photo gets the label whose centre is nearest, the
centre being the label's description in English plus the photos the user put in it (input/corrections.json).
The teacher always commits to a label, never "unsure" (user, 2026-09-30). Calibration (`sample`) asks about
photos as different from each other as possible, so a few corrections cover the folder.
Count (SAM 3): SAM 3 finds every object of each class (one short noun phrase per class) and scores it; every
object above the threshold is marked and its box centre becomes the object's point. Photos with no objects are
kept (they teach what "none" looks like). The threshold follows the user's error preference (spec.errors) and
their corrected photos, which also win over the teacher. Its calibration rounds also show photos as different as
possible (SigLIP 2 embeddings, picked by name when they cannot be had).
Answers are cached by photo content and model (and phrases for SAM 3): the same photo gets the same answer in
every job, and new corrections need no GPU. prefetch() fills SAM 3's answers for the whole folder while the user
corrects the first photos, and gpu_needed() tells when a round can run without a GPU. Photos with no answer (the teacher failed) are left out, in
label/no_answer/ (label/unsure/ for count), and asked again on the next run.
Output: label/<class>/<photo> (copies) or, for count, label/images/<photo>; label/verdicts.json,
label/teacher.json (what the teacher used and a line for review)
"""
from __future__ import annotations

import json
import os
import shutil

from .. import counting, review, teacher
from ..tasks import task_of
from ..workspace import IMAGE_EXT, cache_key, content_key, copy_name
from .base import Context, Stage

NO_ANSWER = "no_answer"  # the teacher failed on the photo: left out, asked again on the next run


class Label(Stage):
    name = "label"
    version = "3"  # 2: SAM 3 instead of a vision model on Token Factory; 3: SigLIP 2 for classification
    runtime = "cpu"  # the teachers run in their own Python (teacher.py); this process plans and writes
    gpu = "any"
    reads_input = True
    rereads = ("verdicts.json",)  # the teacher's earlier answers and the user's corrections
    uses_cache = True
    uses = ("user_data", "classes", "background_class", "task", "errors")
    sample: int | None = None  # calibration: only the photos already answered and this many new ones

    def enabled(self, spec) -> bool:
        return bool(spec.user_data.raw)

    def powered_by(self, spec, backend: str, model: str | None = None) -> str:
        return f"{task_of(spec).teacher} | {self.where(backend)}"

    @staticmethod
    def _timeline(ctx: Context):
        """Where the teacher's own waits go (its weights pulled in a cloud job), next to the stages."""
        return lambda step, seconds: ctx.job.timeline(step, seconds, os.environ.get("SIM2CHIP_RUNS_ON", "local"))

    def run(self, ctx: Context) -> dict:
        ud, spec = ctx.spec.user_data, ctx.spec
        src = ctx.job.resolve_input(ud.raw)
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise FileNotFoundError(f"user_data.raw: no photos in {src}")
        out = ctx.job.stage_dir(self.name)
        previous = json.loads((out / "verdicts.json").read_text()) if (out / "verdicts.json").exists() else {}
        sample = self._sample()
        keys = {f: str(f.relative_to(src)) for f in files}
        corrections = review.load_corrections(ctx.job)
        if task_of(spec).points:
            verdicts, learned, info = self._count(ctx, files, keys, previous, sample, corrections)
        else:
            verdicts, learned, info = self._classify(ctx, files, keys, previous, sample, corrections["labels"])

        counts: dict[str, int] = {}
        for v in verdicts.values():
            counts[v["assigned"]] = counts.get(v["assigned"], 0) + 1
        out.mkdir(parents=True, exist_ok=True)
        for d in out.iterdir():  # copies of an earlier run (verdicts.json was read above)
            shutil.rmtree(d) if d.is_dir() else d.unlink()
        for key, v in verdicts.items():
            d = out / v["assigned"]
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / key, d / copy_name(key))
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        (out / "teacher.json").write_text(json.dumps(learned, indent=1))
        if task_of(spec).points:
            n = sum(len(v["objects"]) for v in verdicts.values() if v["objects"] is not None)
            ctx.log(f"{counts.get('images', 0)} photos with {n} objects marked, {counts.get('unsure', 0)} unsure")
            return {"labels": counts, "objects": n, **info}
        ctx.log(f"labels {counts}")
        return {"labels": counts, **info}

    @staticmethod
    def _prompts(ctx: Context) -> dict[str, str]:
        """teacher.prompts, cached by the labels and their descriptions: written once, the same in every job."""
        spec = ctx.spec
        what = [spec.background_class] + [[c.id, c.description] for c in spec.classes]
        f = ctx.cache.root / "labels" / "prompts" / f"{cache_key(what, sort_keys=False)}.json"
        if f.exists():
            return json.loads(f.read_text())
        found = teacher.prompts(spec)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(found, indent=1))
        return found

    @staticmethod
    def _phrases(ctx: Context) -> dict[str, list[str]]:
        """teacher.phrases, with the classes that have no English names from a 3D model search put in English
        by the text model. Cached by the classes and their descriptions, so every run asks SAM 3 the same."""
        spec = ctx.spec
        found = teacher.phrases(spec)
        missing = {c.id: c.description or c.id.replace("_", " ") for c in spec.classes if not teacher.names(c)}
        if not missing:
            return found
        f = ctx.cache.root / "labels" / "phrases" / \
            f"{cache_key(missing)}.json"
        if f.exists():
            return {**found, **json.loads(f.read_text())}
        english = teacher.english_phrases(missing)
        if english is None:
            ctx.log(f"WARNING: could not put {', '.join(missing)} in English; {teacher.NAME} understands only "
                    "English and may find nothing")
            return found
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(english, indent=1))
        return {**found, **english}

    def _classify(self, ctx, files, keys, previous, sample, fixes):
        prompts = self._prompts(ctx)
        model = teacher.SIGLIP_MODEL.split("/")[-1]
        shared = ctx.cache.root / "labels" / "siglip" / model
        text_id = cache_key(prompts)
        text_file = ctx.cache.root / "labels" / "siglip_text" / model / f"{text_id}.json"
        # Every photo, also when calibrating: embedding is cheap and picking varied photos needs them all.
        answer_file = {f: shared / f"{content_key(f)}.json" for f in files}
        todo = [f for f in files if not answer_file[f].exists()]
        ctx.log(f"embedding {len(todo)} photos with {teacher.SIGLIP_NAME} ({len(files) - len(todo)} already done); "
                + "; ".join(f"{k}: {v}" for k, v in prompts.items()))
        if todo or not text_file.exists():
            teacher.embed([(f, answer_file[f]) for f in todo], list(prompts.values()), text_file, log=ctx.log,
                          timeline=self._timeline(ctx))
        emb = {}
        for f in files:
            a = json.loads(answer_file[f].read_text()) if answer_file[f].exists() else {"error": "no answer"}
            if "error" in a:
                ctx.log(f"  {keys[f]}: {a['error']}")
                answer_file[f].unlink(missing_ok=True)  # asked again on the next run
            else:
                emb[keys[f]] = a["embedding"]
        if not emb:
            raise RuntimeError(f"the teacher failed on all {len(files)} photos, see the lines above")
        if sample:
            keep = [k for k in previous if k in emb]
            chosen = set(keep + teacher.spread(emb, keep, sample))
            files = [f for f in files if keys[f] in chosen or keys[f] in previous]

        text = dict(zip(prompts, json.loads(text_file.read_text())["embeddings"]))
        examples = {lab: [emb[k] for k, c in fixes.items() if c == lab and k in emb] for lab in text}
        centre = teacher.centres(text, examples)
        used = all(examples.values())
        n = sum(len(v) for v in examples.values())
        ctx.log(f"learning from {n} corrected photos" if used else
                f"using the descriptions only ({n} corrected photos, at least one per label needed)")
        verdicts = {}
        for f in files:
            key = keys[f]
            if key not in emb:
                v = {"class": NO_ANSWER, "confidence": 0, "reason": "no valid answer"}
            else:
                cls, conf = teacher.nearest(emb[key], centre)
                v = {"class": cls, "confidence": conf,
                     "reason": f"{teacher.SIGLIP_NAME}: closest to {cls}" + (f", from your {n} photos" if used else "")}
            v["model"] = teacher.SIGLIP_MODEL
            if key in fixes:  # the user's label wins over the teacher's
                v = {**v, "teacher": v["class"], "teacher_confidence": v["confidence"],
                     "class": fixes[key], "confidence": 1.0, "judge": "user"}
            verdicts[key] = {**v, "assigned": v["class"]}
        learned = {"model": teacher.SIGLIP_MODEL, "prompts": prompts, "examples": n, "used_examples": used,
                   "note": teacher.explain_examples(n, used)}
        return verdicts, learned, {"examples": n}

    def _files(self, ctx):
        """The user's raw photos and their keys (path in the raw folder)."""
        src = ctx.job.resolve_input(ctx.spec.user_data.raw)
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        return files, {f: str(f.relative_to(src)) for f in files}

    def _sample(self) -> int | None:
        return self.sample or int(os.environ.get("SIM2CHIP_LABEL_SAMPLE", "0")) or None

    @staticmethod
    def _previous(ctx) -> dict:
        f = ctx.job.stage_dir("label") / "verdicts.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def answers_dir(self, ctx):
        """Where SAM 3's answers for this spec's phrases live, shared by every job: photo content x model x phrases."""
        answers_id = cache_key([teacher.MODEL, teacher.SIDE, teacher.FLOOR, self._phrases(ctx)])
        return ctx.cache.root / "labels" / answers_id / "sam3"

    @staticmethod
    def _embedding_file(ctx, f):
        return ctx.cache.root / "labels" / "siglip" / teacher.SIGLIP_MODEL.split("/")[-1] / f"{content_key(f)}.json"

    def _embeddings(self, ctx, files, keys, compute: bool = True) -> dict[str, list[float]]:
        """SigLIP 2 embeddings of the photos (the same files classification uses), computed for those missing when
        `compute`. Count uses them only to pick varied photos for calibration: a teacher error here is logged and
        the photos are picked by name instead."""
        out = {f: self._embedding_file(ctx, f) for f in files}
        todo = [f for f in files if not out[f].exists()]
        if todo and compute:
            try:
                teacher.embed([(f, out[f]) for f in todo], [], ctx.cache.root / "labels" / "siglip_text" / "_none.json",
                              log=ctx.log, timeline=self._timeline(ctx))
            except teacher.TeacherError as e:
                ctx.log(f"WARNING: could not compare the photos with {teacher.SIGLIP_NAME}, picking them by name: {e}")
        emb = {}
        for f in files:
            a = json.loads(out[f].read_text()) if out[f].exists() else {}
            if "embedding" in a:
                emb[keys[f]] = a["embedding"]
            else:
                out[f].unlink(missing_ok=True)
        return emb

    def _pick(self, ctx, files, keys, previous, sample, compute: bool = True) -> list:
        """A count calibration round: the photos already shown and `sample` new ones, as different from each other
        and from those as possible (SigLIP 2), or spread over the folder by name when the photos could not be
        compared (names often follow places or days)."""
        rest = [f for f in files if keys[f] not in previous]
        emb = self._embeddings(ctx, files, keys, compute)
        new = set(teacher.spread(emb, [k for k in previous if k in emb], sample)) if emb else set()
        picked = [f for f in rest if keys[f] in new]
        if len(picked) < sample:  # photos SigLIP 2 could not read (or none compared): by name
            left = [f for f in rest if keys[f] not in emb]
            picked += left[::max(1, len(left) // sample)][:sample - len(picked)]
        return [f for f in files if keys[f] in previous] + picked

    def gpu_needed(self, ctx) -> bool:
        """Would running this stage now (with its calibration sample) need the teacher's GPU? False when every
        answer it needs is already cached here: a calibration round then runs in seconds on a machine without a
        GPU, the threshold or the centres being all that the user's corrections change."""
        files, keys = self._files(ctx)
        if not files:
            return True
        sample = self._sample()
        if task_of(ctx.spec).points:
            if sample:
                if any(not self._embedding_file(ctx, f).exists() for f in files):
                    return True
                files = self._pick(ctx, files, keys, self._previous(ctx), sample, compute=False)
            shared = self.answers_dir(ctx)
            return any(not (shared / f"{content_key(f)}.json").exists() for f in files)
        prompts = self._prompts(ctx)
        model = teacher.SIGLIP_MODEL.split("/")[-1]
        text_id = cache_key(prompts)
        if not (ctx.cache.root / "labels" / "siglip_text" / model / f"{text_id}.json").exists():
            return True
        return any(not self._embedding_file(ctx, f).exists() for f in files)

    def prefetch(self, ctx, rounds: int = 3) -> int:
        """Count: SAM 3's answers for every photo of the job, without labeling it. The next calibration rounds and
        the first run then find them cached: the answers do not depend on the user's corrections (they keep every
        score down to FLOOR; corrections move only the threshold). The photos the next `rounds` calibration rounds
        will show go first, so a quick user rarely waits. Returns how many photos SAM 3 looked at."""
        files, keys = self._files(ctx)
        shared = self.answers_dir(ctx)
        shown = dict.fromkeys(self._previous(ctx))
        order, sample = [], self._sample() or 6
        self._embeddings(ctx, files, keys)
        for _ in range(rounds):
            new = [f for f in self._pick(ctx, files, keys, shown, sample, compute=False) if keys[f] not in shown]
            order += new
            shown.update(dict.fromkeys(keys[f] for f in new))
        order += [f for f in files if f not in order]
        todo = [f for f in order if not (shared / f"{content_key(f)}.json").exists()]
        ctx.log(f"prefetch: {len(todo)} of {len(files)} photos for {teacher.NAME}")
        if todo:
            teacher.detect([(f, shared / f"{content_key(f)}.json") for f in todo], self._phrases(ctx), log=ctx.log,
                           timeline=self._timeline(ctx))
        return len(todo)

    def _count(self, ctx, files, keys, previous, sample, corrections):
        spec = ctx.spec
        ids = [c.id for c in spec.classes]
        phrases = self._phrases(ctx)
        if sample:  # new photos, as different as possible
            files = self._pick(ctx, files, keys, previous, sample)

        shared = self.answers_dir(ctx)
        answer_file = {f: shared / f"{content_key(f)}.json" for f in files}
        todo = [f for f in files if not answer_file[f].exists()]
        ctx.log(f"labeling {len(todo)} photos with {teacher.NAME} ({len(files) - len(todo)} already labeled), "
                "looking for " + "; ".join(f"{k}: {', '.join(v)}" for k, v in phrases.items()))
        if todo:
            teacher.detect([(f, answer_file[f]) for f in todo], phrases, log=ctx.log, timeline=self._timeline(ctx))
        dets = {}
        for f in files:
            a = json.loads(answer_file[f].read_text()) if answer_file[f].exists() else {"error": "no answer"}
            if "error" in a:
                ctx.log(f"  {keys[f]}: {a['error']}")
                answer_file[f].unlink(missing_ok=True)  # asked again on the next run
            else:
                dets[keys[f]] = [d for d in a["detections"] if d["class"] in ids]
        if not dets:
            raise RuntimeError(f"the teacher failed on all {len(files)} photos, see the lines above")

        fix_points = corrections.get("points", {})
        truth = {k: [p for p in v if p[2] in ids] for k, v in fix_points.items()}
        chosen = teacher.choose(spec.errors or "balanced", dets, truth)
        th = chosen["threshold"]
        ctx.log(f"threshold {th} ({chosen['preference']}, from {chosen['photos']} corrected photos)")
        found = sum(len(teacher.objects(d, th)) for d in dets.values())
        marked = sum(len(v) for v in truth.values())
        if not found and marked:
            # Every photo would teach "nothing here" while the user's own marks say otherwise.
            raise RuntimeError(f"{teacher.NAME} found nothing in any of the {len(dets)} photos while your marked "
                               f"photos have {marked} objects: it looked for "
                               + "; ".join(f"{k}: {', '.join(v)}" for k, v in phrases.items())
                               + ". Describe the class with its common English name and run again")
        verdicts = {}
        for f in files:
            key = keys[f]
            v = {"objects": teacher.objects(dets[key], th) if key in dets else None,
                 "reason": f"{teacher.NAME}, threshold {th}" if key in dets else "no valid answer",
                 "model": teacher.MODEL, "threshold": th}
            if key in fix_points:  # the user's marks win over the teacher's
                v = {**v, "teacher_objects": v["objects"], "judge": "user",
                     "objects": [{"x": x, "y": y, "class": c} for x, y, c in fix_points[key] if c in ids]}
            verdicts[key] = {**v, "assigned": "unsure" if v["objects"] is None else "images"}
        learned = {**chosen, "model": teacher.MODEL, "phrases": phrases, "note": teacher.explain(chosen)}
        boxes = [[o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]] for v in verdicts.values()
                 for o in v["objects"] or [] if o.get("box")]
        small = counting.small_objects(boxes, ctx.hardware.count_input())
        if small:
            ctx.log("WARNING: " + small["note"])
            learned["small_objects"] = small
            learned["note"] += " " + small["note"]
        dense = counting.crowded([len(v["objects"]) for v in verdicts.values() if v["objects"] is not None])
        if dense:
            ctx.log("WARNING: " + dense["note"])
            learned["crowded"] = dense
            learned["note"] += " " + dense["note"]
        return verdicts, learned, {"threshold": th}
