"""The teacher (label stage) with a fake SAM 3: no GPU, no weights."""
import hashlib
import json

from PIL import Image

from sim2chip import review, teacher
from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.stages.ingest import Ingest
from sim2chip.stages.label import Label
from sim2chip.workspace import Cache, link_input, open_job

SPEC = {"name": "t", "classes": [{"id": "cow", "description": "cows", "size_m": {"measure": "length", "range": [2, 2.5]},
                                  "assets": {"categories": ["cow"]}},
                                 {"id": "dog", "size_m": {"measure": "length", "range": [0.5, 1]},
                                  "assets": {"source": "search", "queries": ["dog", "puppy"]}}]}


def img(path, color=(120, 120, 120)):
    """A plain photo with its bottom-right corner set from its path: exact copies count as one photo in ingest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (64, 64), color)
    im.paste(tuple(hashlib.sha256(str(path).encode()).digest()[:3]), (48, 48, 64, 64))  # survives JPEG
    im.save(path)


def fake_sam3(monkeypatch, answer):
    asked = []

    # Each class's objects come from its first phrase (SAM 3 answers each phrase on its own).
    first = {"cow": "cow", "dog": "dog"}

    def detect(items, log=print, timeline=None):
        for path, outs in items:
            assert set(outs) <= {"cow", "dog", "puppy"}
            asked.append(path.name)
            for phrase, out in outs.items():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"detections": [{"box": d["box"], "score": d["score"]}
                                                          for d in answer(path.name) if first[d["class"]] == phrase]}))
    monkeypatch.setattr(teacher, "detect", detect)
    return asked


TEXT = {"empty": [1, 0, 0], "cow": [0, 1, 0], "dog": [0, 0, 1]}


def fake_siglip(monkeypatch, vector):
    """SigLIP 2 without a GPU: each photo's embedding from its name, each label's description along one axis."""
    asked = []

    def embed(items, texts, text_out, log=print, timeline=None):
        for path, out in items:
            asked.append(path.name)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"embedding": teacher._unit(vector(path.name))}))
        text_out.parent.mkdir(parents=True, exist_ok=True)
        text_out.write_text(json.dumps({"embeddings": [TEXT[t.removeprefix("a photo of ")] for t in texts]}))
    monkeypatch.setattr(teacher, "embed", embed)
    monkeypatch.setattr(teacher, "prompts", lambda spec, client=None: {k: f"a photo of {k}" for k in spec.labels})
    return asked


def test_label_then_ingest(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for i in range(6):
        for kind in ("cow", "dog", "blur", "none"):
            img(raw / f"{kind}_{i}.jpg")
    # A blurry photo still gets a label (closest: cow), never "unsure".
    near = {"cow": [0, 1, 0.1], "dog": [0.1, 0, 1], "blur": [0.5, 0.6, 0.1], "none": [1, 0.1, 0]}
    fake_siglip(monkeypatch, lambda name: near[name.split("_")[0]])
    ctx = Context(from_dict({**SPEC, "user_data": {"raw": "raw", "test_per_class": 2, "train_per_class": 2}}),
                  open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    link_input(ctx.job, "raw", raw)
    info = Label().run(ctx)
    assert info["labels"] == {"cow": 12, "dog": 6, "empty": 6}
    counts = Ingest().run(ctx)["counts"]
    assert counts["cow"] == {"test": 2, "train": 2} and counts["dog"]["train"] == 2


def test_label_reuses_answers_and_learns_from_corrections(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for name in ("a.jpg", "sub/b.jpg", "c.jpg", "d.jpg"):
        img(raw / name)
    near = {"a.jpg": [0, 1, 0.1], "b.jpg": [0, 0.2, 1], "c.jpg": [0.6, 0.8, 0], "d.jpg": [0.62, 0.78, 0]}
    asked = fake_siglip(monkeypatch, lambda name: near[name])
    ctx = Context(from_dict({**SPEC, "user_data": {"raw": "raw"}}), open_job(tmp_path, "job"), Cache(tmp_path / "c"))
    link_input(ctx.job, "raw", raw)
    Label().run(ctx)
    v = json.loads((ctx.job.stage_dir("label") / "verdicts.json").read_text())
    # From the descriptions alone, c and d look more like a cow than like nothing.
    assert len(asked) == 4 and v["c.jpg"]["class"] == v["d.jpg"]["class"] == "cow" and v["sub/b.jpg"]["class"] == "dog"
    # The user says c is empty: with an example of every label, d (like c) follows, and the user's label wins.
    assert review.apply(ctx.job, ctx.cache, {"a.jpg": "cow", "sub/b.jpg": "dog", "c.jpg": "empty"}, {}) == "label"
    assert Label().run(ctx)["examples"] == 3 and len(asked) == 4  # no photo embedded twice
    out = ctx.job.stage_dir("label")
    v = json.loads((out / "verdicts.json").read_text())
    assert v["c.jpg"]["judge"] == "user" and v["d.jpg"]["class"] == "empty"
    assert (out / "dog" / "sub_b.jpg").exists() and (out / "empty" / "d.jpg").exists()
    got = review.gather(ctx.job, ctx.cache, ctx.spec.labels)
    assert "3 photos you corrected" in got["teacher_rules"][0]
    assert {p["key"]: p["label"] for p in got["photos"]} == {
        "a.jpg": "cow", "c.jpg": "empty", "d.jpg": "empty", "sub/b.jpg": "dog"}


def test_without_background_class_the_teacher_always_picks_one(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for name in ("cow.jpg", "nothing.jpg"):
        img(raw / name)
    fake_siglip(monkeypatch, lambda name: {"cow.jpg": [0, 1, 0], "nothing.jpg": [1, 0, 0.01]}[name])
    ctx = Context(from_dict({**SPEC, "background_class": None, "user_data": {"raw": "raw"}}),
                  open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    link_input(ctx.job, "raw", raw)
    assert ctx.spec.labels == ["cow", "dog"]
    assert Label().run(ctx)["labels"] == {"cow": 1, "dog": 1}  # nothing like any class still gets the closest


def test_calibration_asks_about_different_photos():
    same, other = [1, 0], [0, 1]
    emb = {"a": same, "a2": teacher._unit([0.99, 0.14]), "a3": teacher._unit([0.98, 0.2]), "b": other}
    assert teacher.spread(emb, [], 2) == ["a3", "b"]  # the most typical photo, then the most different one
    assert teacher.spread(emb, ["b"], 1) == ["a"]


def test_count_calibration_prefetch_and_rounds_without_a_gpu(tmp_path, monkeypatch):
    """Count calibration shows varied photos (SigLIP 2), the prefetch answers the rest with the next rounds' photos
    first, and a round after it needs no GPU: the corrections only move the threshold."""
    raw = tmp_path / "raw"
    for i in range(12):
        img(raw / f"p{i:02d}.jpg")
    # Photos 0-5 are near copies of one view, 6-11 each a view of their own.
    vec = lambda name: [1.0, 0.01 * int(name[1:3])] + [0.0] * 12 if int(name[1:3]) < 6 else \
        [0.0] * 2 + [1.0 if j == int(name[1:3]) else 0.0 for j in range(12)]  # noqa: E731
    embedded = fake_siglip(monkeypatch, vec)
    asked = fake_sam3(monkeypatch, lambda name: [{"class": "cow", "box": [0.1, 0.1, 0.3, 0.3], "score": 0.9}])
    spec = from_dict({**SPEC, "task": "count", "user_data": {"raw": "raw"}})
    ctx = Context(spec, open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    link_input(ctx.job, "raw", raw)
    stage = Label()
    stage.sample = 3
    assert stage.gpu_needed(ctx)
    stage.run(ctx)
    first = set(json.loads((ctx.job.stage_dir("label") / "verdicts.json").read_text()))
    assert len(embedded) == 12 and len(asked) == 3 and len(first) == 3
    assert sum(int(k[1:3]) < 6 for k in first) <= 1  # not three of the near copies
    asked.clear()
    assert stage.prefetch(ctx) == 9 and len(asked) == 9
    # The prefetch went through the next rounds' photos first: they are the ones the next round shows.
    stage.run(ctx)
    second = set(json.loads((ctx.job.stage_dir("label") / "verdicts.json").read_text())) - first
    assert set(asked[:3]) == second
    asked.clear()
    assert not stage.gpu_needed(ctx) and not Label().gpu_needed(ctx)
    Label().run(ctx)  # the whole folder, every answer cached
    assert not asked


def test_one_object_found_under_two_classes_gets_one(tmp_path, monkeypatch):
    """SAM 3 finds the same object as a cow and as a dog (one box, two phrases): SigLIP 2 looks at the crop and
    keeps one, even against SAM 3's scores; a separate object is left as it is."""
    raw = tmp_path / "raw"
    raw.mkdir()
    im = Image.new("RGB", (100, 100), (90, 90, 90))
    im.paste((230, 20, 20), (10, 10, 50, 50))  # the shared object: red, like the dog's description
    im.save(raw / "a.jpg")
    shared = [0.1, 0.1, 0.5, 0.5]
    fake_sam3(monkeypatch, lambda name: [{"class": "cow", "box": shared, "score": 0.9},
                                         {"class": "dog", "box": shared, "score": 0.8},
                                         {"class": "dog", "box": [0.6, 0.6, 0.9, 0.9], "score": 0.7}])
    crops = []

    def embed(items, texts, text_out, log=print, timeline=None):
        for path, out in items:
            r, g, b = Image.open(path).convert("RGB").resize((1, 1)).getpixel((0, 0))
            crops.append(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"embedding": teacher._unit([0.01, g / 255, r / 255])}))
        text_out.parent.mkdir(parents=True, exist_ok=True)
        text_out.write_text(json.dumps({"embeddings": [TEXT[t.removeprefix("a photo of ")] for t in texts]}))
    monkeypatch.setattr(teacher, "embed", embed)
    monkeypatch.setattr(teacher, "prompts", lambda spec, client=None: {k: f"a photo of {k}" for k in spec.labels})
    spec = from_dict({**SPEC, "task": "count", "user_data": {"raw": "raw"}})
    ctx = Context(spec, open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    link_input(ctx.job, "raw", raw)
    Label().run(ctx)
    objs = json.loads((ctx.job.stage_dir("label") / "verdicts.json").read_text())["a.jpg"]["objects"]
    assert sorted(o["class"] for o in objs) == ["dog", "dog"] and len(crops) == 1
    Label().run(ctx)
    assert len(crops) == 1  # the crop's answer is cached

    def broken(*a, **k):
        raise teacher.TeacherError("no GPU")
    monkeypatch.setattr(teacher, "embed", broken)
    ctx2 = Context(spec, open_job(tmp_path, "job2"), Cache(tmp_path / "cache2"))
    link_input(ctx2.job, "raw", raw)
    Label().run(ctx2)  # without SigLIP 2: the class SAM 3 was surest of, still one per object
    objs = json.loads((ctx2.job.stage_dir("label") / "verdicts.json").read_text())["a.jpg"]["objects"]
    assert sorted(o["class"] for o in objs) == ["cow", "dog"]


def test_same_objects_groups_only_different_classes():
    d = [{"class": "a", "box": [0, 0, 1, 1]}, {"class": "b", "box": [0, 0, 1, 0.95]},
         {"class": "a", "box": [0, 0, 1, 0.9]}, {"class": "c", "box": [0, 0, 0.2, 0.2]}]
    assert [sorted(g) for g in teacher.same_objects(d)] == [[0, 1, 2]]


def test_a_new_class_asks_sam3_only_for_it_and_marks_stand_for_their_classes(tmp_path, monkeypatch):
    """SAM 3's answers are kept per phrase: a later job that adds a class asks only for that class's phrases. The
    user's marks checked for one class only (the check asked about the new class) leave the teacher's marks of
    the others in place."""
    raw = tmp_path / "raw"
    img(raw / "a.jpg")
    img(raw / "b.jpg")
    asked = []
    found = {"cow": [0.1, 0.1, 0.3, 0.3], "dog": [0.6, 0.6, 0.8, 0.8]}

    def detect(items, log=print, timeline=None):
        for path, outs in items:
            asked.append(sorted(outs))
            for phrase, out in outs.items():
                out.parent.mkdir(parents=True, exist_ok=True)
                box = found.get(phrase)
                out.write_text(json.dumps({"detections": [{"box": box, "score": 0.9}] if box else []}))
    monkeypatch.setattr(teacher, "detect", detect)
    cache = Cache(tmp_path / "cache")
    cows = Context(from_dict({**SPEC, "task": "count", "classes": SPEC["classes"][:1], "user_data": {"raw": "raw"}}),
                   open_job(tmp_path, "cows"), cache)
    link_input(cows.job, "raw", raw)
    Label().run(cows)
    assert asked == [["cow"], ["cow"]]
    asked.clear()
    both = Context(from_dict({**SPEC, "task": "count", "user_data": {"raw": "raw"}}), open_job(tmp_path, "both"), cache)
    link_input(both.job, "raw", raw)
    Label().run(both)
    assert asked == [["dog", "puppy"], ["dog", "puppy"]]
    # The check asked about dogs only: the user's photo has no dog where the teacher saw one, and a cow mark the
    # page sent along stays the teacher's.
    assert review.apply(both.job, cache, {}, {}, {"a.jpg": [[0.2, 0.2, "cow"]]}, classes=["dog"]) == "label"
    assert review.load_corrections(both.job)["checked"] == {"a.jpg": ["dog"]}
    asked.clear()
    Label().run(both)
    v = json.loads((both.job.stage_dir("label") / "verdicts.json").read_text())
    assert not asked and v["a.jpg"]["judge"] == "user"
    assert [o["class"] for o in v["a.jpg"]["objects"]] == ["cow"] and "box" in v["a.jpg"]["objects"][0]
    assert sorted(o["class"] for o in v["b.jpg"]["objects"]) == ["cow", "dog"]
    # Fixed again in the review after training, with every class in view: the marks stand for all of them.
    review.apply(both.job, cache, {}, {}, {"a.jpg": [[0.3, 0.3, "cow"]]})
    assert review.load_corrections(both.job)["checked"] == {}
