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

    def detect(items, phrases, log=print):
        assert phrases == {"cow": ["cow"], "dog": ["dog", "puppy"]}
        for path, out in items:
            asked.append(path.name)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"detections": answer(path.name)}))
    monkeypatch.setattr(teacher, "detect", detect)
    return asked


TEXT = {"empty": [1, 0, 0], "cow": [0, 1, 0], "dog": [0, 0, 1]}


def fake_siglip(monkeypatch, vector):
    """SigLIP 2 without a GPU: each photo's embedding from its name, each label's description along one axis."""
    asked = []

    def embed(items, texts, text_out, log=print):
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
