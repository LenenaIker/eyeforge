"""Stages that call Token Factory, with a fake client (no network, no key)."""
import hashlib
import json
import threading

import pytest
from PIL import Image

from sim2chip import llm
from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.stages.ingest import Ingest
from sim2chip.stages.label import Label
from sim2chip.stages.qa_assets import QAAssets
from sim2chip.workspace import Cache, link_input, open_job

SPEC = {"name": "t", "classes": [{"id": "cow", "description": "cows", "size_m": {"measure": "length", "range": [2, 2.5]},
                                  "assets": {"categories": ["cow"]}},
                                 {"id": "dog", "size_m": {"measure": "length", "range": [0.5, 1]},
                                  "assets": {"categories": ["dog"]}}]}


class FakeClient:
    """Answers from a function of the prompt text; records calls."""
    answer = staticmethod(lambda text: "{}")
    calls = []

    def __init__(self, usage_log=None, **kw):
        pass

    def model_for(self, role):
        return f"fake-{role}"

    def chat(self, messages, model, purpose, **kw):
        content = messages[0]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        FakeClient.calls.append(purpose)
        return FakeClient.answer(text)


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    FakeClient.calls = []
    monkeypatch.setattr(llm, "Client", FakeClient)


def img(path, color=(120, 120, 120)):
    """A plain photo with its bottom-right corner set from its path: exact copies count as one photo in ingest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (64, 64), color)
    im.paste(tuple(hashlib.sha256(str(path).encode()).digest()[:3]), (48, 48, 64, 64))  # survives JPEG
    im.save(path)


def test_label_then_ingest(tmp_path):
    raw = tmp_path / "raw"
    for i in range(6):
        img(raw / f"cow_{i}.jpg")
        img(raw / f"dog_{i}.jpg")
        img(raw / f"blur_{i}.jpg")
    # the fake teacher cannot see file names, so it answers in turn: cow, dog, unsure-by-confidence
    seq = iter([{"class": "cow", "confidence": 0.9}, {"class": "dog", "confidence": 0.9},
                {"class": "cow", "confidence": 0.2}] * 6)
    lock = threading.Lock()

    def answer(text):
        with lock:  # the stage calls the teacher from several threads
            return json.dumps(next(seq))
    FakeClient.answer = staticmethod(answer)
    ctx = Context(from_dict({**SPEC, "user_data": {"raw": "raw", "test_per_class": 2, "train_per_class": 2}}),
                  open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    link_input(ctx.job, "raw", raw)
    info = Label().run(ctx)
    assert sum(info["labels"].values()) == 18 and info["labels"]["unsure"] == 6
    counts = Ingest().run(ctx)["counts"]
    assert counts["cow"] == {"test": 2, "train": 2} and counts["dog"]["train"] == 2
    assert FakeClient.calls.count("label") == 18


def test_qa_auto_with_manual_on_top(tmp_path):
    ctx = Context(from_dict({**SPEC, "qa": {"mode": "auto", "exclude": "r/ex.txt"}}), open_job(tmp_path, "job"),
                  Cache(tmp_path / "cache"))
    assets = [{"uid": u, "class": c, "name": u, "category": c, "usd": "x.usd", "dims": [2, 1, 1]}
              for u, c in [("a", "cow"), ("b", "cow"), ("c", "cow"), ("d", "dog")]]
    (ctx.out("convert_assets") / "assets.json").write_text(json.dumps(assets))
    prev = ctx.out("preview_assets")
    views = {}
    for a in assets:
        views[a["uid"]] = []
        for v in range(4):
            img(prev / f"{a['uid']}_{v}.png")
            views[a["uid"]].append(f"{a['uid']}_{v}.png")
    (prev / "views.json").write_text(json.dumps(views))
    (ctx.job.input / "r").mkdir()
    (ctx.job.input / "r" / "ex.txt").write_text("d\n")
    answers = {"a": {"decision": "keep", "class": "cow"},
               "b": {"decision": "discard", "class": "cow", "reason": "plinth"},
               "c": {"decision": "keep", "class": "dog", "no_color": True},
               "d": {"decision": "keep", "class": "dog"}}
    FakeClient.answer = staticmethod(lambda text: json.dumps(answers[text.split('name "')[1][0]]))
    info = QAAssets().run(ctx)
    kept = {a["uid"]: a for a in json.loads((ctx.job.stage_dir("qa_assets") / "assets.json").read_text())}
    assert set(kept) == {"a", "c"}            # b discarded by the model, d by the human list
    assert kept["c"]["class"] == "dog" and kept["c"]["no_color"]
    assert info["kept"] == {"cow": 1, "dog": 1}
    assert (ctx.job.stage_dir("qa_assets") / "sheets" / "a.jpg").exists()


def test_qa_describe_then_decide(tmp_path):
    ctx = Context(from_dict({**SPEC, "qa": {"mode": "auto", "strategy": "describe_then_decide"}}),
                  open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    assets = [{"uid": u, "class": c, "name": u, "category": c, "usd": "x.usd", "dims": [2, 1, 1]}
              for u, c in (("a", "cow"), ("b", "cow"), ("d", "dog"))]
    (ctx.out("convert_assets") / "assets.json").write_text(json.dumps(assets))
    prev = ctx.out("preview_assets")
    views = {a["uid"]: [] for a in assets}
    for u in views:
        for v in range(4):
            img(prev / f"{u}_{v}.png")
            views[u].append(f"{u}_{v}.png")
    (prev / "views.json").write_text(json.dumps(views))

    def answer(text):
        if text.startswith("The image shows"):  # the vision model only describes
            return json.dumps({"object": "cow", "on_plinth_or_base": False})
        name = text.split('name "')[1][0]      # the text model decides, with the description in the prompt
        assert '"object": "cow"' in text
        return json.dumps({"decision": "discard" if name == "b" else "keep"})
    FakeClient.answer = staticmethod(answer)
    info = QAAssets().run(ctx)
    assert info["kept"] == {"cow": 1, "dog": 1}
    assert FakeClient.calls.count("qa_describe") == 3 and FakeClient.calls.count("qa_decide") == 3


def test_label_reuses_verdicts_and_applies_corrections(tmp_path):
    from sim2chip import review
    raw = tmp_path / "raw"
    img(raw / "a.jpg")
    img(raw / "sub" / "b.jpg")
    prompts = []

    def answer(text):
        prompts.append(text)
        if "general rule" in text:
            assert 'The teacher said "cow"' in text and 'The user says "dog"' in text
            return json.dumps({"lesson": "Small animals are dogs."})
        return json.dumps({"class": "cow", "confidence": 0.9, "reason": "r"})
    FakeClient.answer = staticmethod(answer)
    ctx = Context(from_dict({**SPEC, "user_data": {"raw": "raw"}}), open_job(tmp_path, "job"), Cache(tmp_path / "c"))
    link_input(ctx.job, "raw", raw)
    Label().run(ctx)
    assert len(FakeClient.calls) == 2
    # The user fixes one photo in review: the teacher learns a rule from it and relabels with it, and the
    # user's label wins on that photo.
    assert review.apply(ctx.job, ctx.cache, {"sub/b.jpg": "dog"}, {}) == "label"
    Label().run(ctx)
    assert FakeClient.calls[2:].count("label_lesson") == 1 and FakeClient.calls[2:].count("label") == 2
    assert "- Small animals are dogs." in prompts[-1]
    out = ctx.job.stage_dir("label")
    assert (out / "dog" / "sub_b.jpg").exists() and (out / "cow" / "a.jpg").exists()
    v = json.loads((out / "verdicts.json").read_text())["sub/b.jpg"]
    assert v["judge"] == "user" and v["teacher"] == "cow"
    # A third run reuses the lesson and the teacher's verdicts.
    Label().run(ctx)
    assert len(FakeClient.calls) == 5
    photos = review.gather(ctx.job, ctx.cache, ctx.spec.labels)["photos"]
    assert {p["key"]: (p["label"], p["teacher"], p["image"] is not None) for p in photos} == {
        "a.jpg": ("cow", "cow", True), "sub/b.jpg": ("dog", "cow", True)}


def test_review_asset_decisions_go_to_the_knowledge_base(tmp_path):
    from sim2chip import review
    from sim2chip.asset_db import AssetDB
    ctx = Context(from_dict(SPEC), open_job(tmp_path, "job"), Cache(tmp_path / "c"))
    assets = [{"uid": "u1", "class": "cow", "category": "cow", "usd": "x.usd", "dims": [2, 1, 1]}]
    (ctx.out("convert_assets") / "assets.json").write_text(json.dumps(assets))
    assert review.apply(ctx.job, ctx.cache, {}, {"u1": "discard"}) == "qa_assets"
    r = AssetDB(ctx.cache.db).lookup("u1", "cow")
    assert r["decision"] == "discard" and r["judge"] == "user"
    assert review.apply(ctx.job, ctx.cache, {}, {}) is None
