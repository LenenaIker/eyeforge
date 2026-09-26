import json

import pytest

from sim2chip import interview


class Scripted:
    """Fake Nemotron that replays a list of JSON actions and records what it was told."""

    def __init__(self, actions):
        self.actions, self.seen = list(actions), []

    def chat(self, messages, model, purpose, **kw):
        self.seen.append(messages[-1]["content"])
        return json.dumps(self.actions.pop(0))


GOOD = {"name": "barn", "classes": [{"id": "cow", "description": "cows", "size_m": {"measure": "length",
        "range": [2, 2.5]}, "assets": {"categories": ["cow"], "count": 10}}], "qa": {"mode": "auto"}}


def test_interview_flow():
    bad = {**GOOD, "classes": [{**GOOD["classes"][0], "assets": {"categories": ["cattle_(beast)"]}}]}
    client = Scripted([{"say": "Where is the camera?"}, {"search": ["cows"]}, {"spec": bad}, {"spec": GOOD},
                       {"spec": GOOD}])
    answers = iter(["cows in my barn", "on a post, 3 m", "add night please", "yes"])
    told = []
    cats = interview.CategoryIndex(counts={"cow": 62, "calf": 11, "taxi": 3})
    spec = interview.run(client, "m", cats, ask=lambda q: next(answers), tell=told.append)
    assert spec.name == "barn"
    assert "cow" in client.seen[2] and "taxi" not in client.seen[2]    # search results went back
    assert "not found in Objaverse-LVIS" in client.seen[3]             # invalid category rejected
    assert "wants changes: add night please" in client.seen[4]         # user asked for a change
    assert "cow" in told[0]


def test_system_prompt_mentions_schema():
    p = interview.system_prompt()
    assert '"target_fill"' in p and "esp32cam" in p and "road" in p


def test_is_yes():
    for a in ["yes", "Sí, adelante.", "ok!", "vale", "de acuerdo", "Perfecto, gracias", "si"]:
        assert interview.is_yes(a), a
    for a in ["no", "sí, pero cambia la altura", "add night please", "", "the trucks should be 10 m long instead"]:
        assert not interview.is_yes(a), a


def test_scene_tags_are_checked():
    scene = {"template": "generic", "sky_tags": ["moon"], "ground_tags": ["grass"]}
    fixed = {"template": "generic", "sky_tags": ["forest"], "ground_tags": ["grass"]}
    client = Scripted([{"spec": {**GOOD, "scene": scene}}, {"spec": {**GOOD, "scene": fixed}}])
    check = lambda sc: "no Poly Haven sky matches" if "moon" in sc.sky_tags else None  # noqa: E731
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}), scene_check=check)
    t = s.reply("cows in a forest")
    assert t.kind == "confirm" and t.spec.scene.sky_tags == ["forest"]
    assert "no Poly Haven sky matches" in client.seen[1]


def test_photo_folders_are_checked(tmp_path):
    from PIL import Image

    from sim2chip.spec import SpecError, from_dict
    unsorted = tmp_path / "unsorted"
    unsorted.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8)).save(unsorted / f"{i}.jpg")
    cls = [{"id": "pot", "size_m": {"measure": "height", "range": [0.3, 1]},
            "assets": {"source": "search", "queries": ["flower pot"]}}]
    # Unsorted photos passed as labeled: every class would get the whole folder or nothing.
    labeled = from_dict({"name": "t", "classes": cls, "user_data": {"labeled": str(unsorted)}})
    assert "user_data.raw" in interview.check_photos(labeled)
    raw = from_dict({"name": "t", "classes": cls, "user_data": {"raw": str(unsorted)}})
    assert interview.check_photos(raw) is None
    missing = from_dict({"name": "t", "classes": cls, "user_data": {"raw": str(tmp_path / "nope")}})
    assert "does not exist" in interview.check_photos(missing)
    with pytest.raises(SpecError, match="subfolder names"):
        from_dict({"name": "t", "classes": cls, "user_data": {"labeled": "x", "class_map": {"pot": [""]}}})


def test_walls_are_not_ground():
    from sim2chip.spec import from_dict
    s = from_dict({"name": "t", "classes": [{"id": "pot"}], "scene": {"template": "generic", "ground_tags": ["wall"]}})
    assert "not a floor" in interview.check_scene(s.scene)
