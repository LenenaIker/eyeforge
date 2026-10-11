"""The scene photo is the cleanest of the typical ones, as the vision model rates them (roadmap B.3)."""
import json

from PIL import Image

from sim2chip import llm
from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.stages.scene_from_photo import PICK_AMONG, SceneFromPhoto
from sim2chip.workspace import Cache, open_job

SPEC = {"name": "b", "task": "count", "classes": [{"id": "bottle", "names": ["bottle"]}],
        "camera": {"pitch_deg": [60, 90]}, "user_data": {"raw": "photos"}}


class Rater:
    """A vision model whose score is written in the photo's red channel."""
    calls = 0

    def __init__(self, usage_log=None):
        pass

    def model_for(self, role):
        return "m"

    def chat(self, msg, model, purpose, **kw):
        import base64
        import io
        Rater.calls += 1
        url = msg[0]["content"][1]["image_url"]["url"]
        red = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).getpixel((0, 0))[0]
        return json.dumps({"score": red // 20, "why": f"red {red}"})


def test_the_cleanest_typical_photo_is_built_first(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "Client", Rater)
    job = open_job(tmp_path, "j")
    ctx = Context(from_dict(SPEC), job, Cache(tmp_path / "cache"))
    seen = []
    ctx.log = seen.append
    photos = []
    for i, red in enumerate([40, 200, 120, 0, 0, 0, 0, 220]):  # the last one is past PICK_AMONG
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 48), (red, 0, 0)).save(p, quality=100)
        photos.append((i, p))
    picked = SceneFromPhoto._pick(ctx, photos, ["bottle"])
    assert [pid for pid, _ in picked[:3]] == [1, 2, 0] and picked[PICK_AMONG:] == photos[PICK_AMONG:]
    assert any(line.startswith("picked photo 1 (1.jpg): 10/10, red") for line in seen)
    # Kept by photo and classes: asked again, the model is not called.
    before = Rater.calls
    assert SceneFromPhoto._pick(ctx, photos, ["bottle"]) == picked and Rater.calls == before


def test_no_answer_keeps_the_typical_order(tmp_path, monkeypatch):
    class Silent(Rater):
        def chat(self, *a, **kw):
            raise llm.LLMError("down")
    monkeypatch.setattr(llm, "Client", Silent)
    ctx = Context(from_dict(SPEC), open_job(tmp_path, "j"), Cache(tmp_path / "cache"))
    seen = []
    ctx.log = seen.append
    photos = []
    for i in range(3):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{i}.jpg")
        photos.append((i, tmp_path / f"{i}.jpg"))
    assert SceneFromPhoto._pick(ctx, photos, ["bottle"]) == photos and "did not answer" in seen[-1]
