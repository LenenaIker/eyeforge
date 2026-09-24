"""Stages that call Token Factory, with a fake client (no network, no key)."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


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
