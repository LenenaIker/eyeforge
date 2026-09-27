import json
from pathlib import Path

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


def test_interview_proposes_the_users_own_photos_for_3d(tmp_path):
    from PIL import Image
    photos = tmp_path / "phone"
    photos.mkdir()
    for n in ("close.jpg", "street.jpg", "blurry.jpg"):
        Image.new("RGB", (8, 8)).save(photos / n)
    scores = {"close.jpg": 0.9, "street.jpg": 0.2, "blurry.jpg": 0.7}

    def ranker(folder, query):
        return sorted(({"file": str(p), "score": scores[p.name], "reason": "r"} for p in Path(folder).iterdir()),
                      key=lambda c: -c["score"])
    pick = {"pick_photos": {"class": "pot", "query": "flower pot", "folder": str(photos), "say": "These?"}}
    client = Scripted([pick, {"say": "Thanks, anything else?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}), photo_ranker=ranker,
                          uploads=tmp_path / "up")
    t = s.reply("pots on my terrace, photos in " + str(photos))
    assert t.kind == "pick" and [Path(c["file"]).name for c in t.request["candidates"]] == ["close.jpg", "blurry.jpg"]
    # The user leaves one out; only proposed files are accepted.
    t = s.choose([str(photos / "close.jpg"), "/etc/passwd"], "pot")
    assert t.kind == "say" and [f.name for f in (tmp_path / "up" / "3d_pot").iterdir()] == ["close.jpg"]
    assert "1 photos to build 3D models of class pot" in client.seen[-1]


def test_interview_asks_for_photos_when_none_fit(tmp_path):
    (tmp_path / "phone").mkdir()
    pick = {"pick_photos": {"class": "pot", "query": "flower pot", "folder": str(tmp_path / "phone")}}
    ask = {"photos": {"for": "3d", "class": "pot", "say": "Do you have a clear photo of the pot?"}}
    client = Scripted([pick, ask, {"say": "ok"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}), photo_ranker=lambda f, q: [])
    t = s.reply("pots")
    assert t.kind == "photos" and t.request == {"for": "3d", "class": "pot"}
    assert "none of the 0 photos" in client.seen[1]
    t = s.add_photos(str(tmp_path / "phone"), 3, "3d", "pot")
    assert t.kind == "say" and "3 photos to build 3D models of class pot" in client.seen[-1]


def test_rejected_specs_stop_and_ask_the_user(tmp_path):
    bad = {**GOOD, "classes": [{**GOOD["classes"][0], "assets": {"categories": ["unicorn"]}}]}
    client = Scripted([{"spec": bad}] * interview.MAX_INVALID_SPECS + [{"say": "No encuentro unicornios, ¿vale un caballo?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}))
    s.transcript = tmp_path / "t.json"
    t = s.reply("unicorns")
    assert t.kind == "say" and "caballo" in t.text
    assert "Stop proposing it" in client.seen[-1]
    assert "not found in Objaverse-LVIS" in s.transcript.read_text()  # rejections are kept for later


def test_endless_rejections_end_the_interview():
    bad = {**GOOD, "classes": [{**GOOD["classes"][0], "assets": {"categories": ["unicorn"]}}]}
    client = Scripted([{"spec": bad}] * (interview.MAX_INVALID_SPECS * 2))
    t = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62})).reply("unicorns")
    assert t.kind == "failed" and "unicorn" in t.text


def test_no_photo_models_on_a_server_without_them():
    photos = {**GOOD, "classes": [{**GOOD["classes"][0], "assets": {"source": "photos", "photos": "user_data",
                                                                     "queries": ["cow"]}}], "user_data": {"raw": "x"}}
    client = Scripted([{"pick_photos": {"class": "cow", "query": "cow", "folder": "/tmp"}}, {"spec": photos},
                       {"cow": "cow"}, {"spec": GOOD}])  # the third answers the English query check
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}), photo_models=False)
    t = s.reply("cows, here are my photos")
    assert t.kind == "confirm" and t.spec.classes[0].assets.photos is None
    assert "cannot be made from the user's photos" in client.seen[1]  # pick_photos refused
    assert "cannot be made from the user's photos" in client.seen[3]  # photo-made spec refused
    assert "cannot be made from the user's photos" in s.messages[0]["content"]


def test_spec_sent_with_text_is_proposed():
    cats = interview.CategoryIndex(counts={"cow": 62})
    good = {"name": "barn", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                         "assets": {"categories": ["cow"], "count": 4}},
                                        {"id": "empty", "description": "no cow"}]}  # background listed too
    s = interview.Session(Scripted([{"say": "Here is the setup, check it.", "spec": good}]), "m", cats)
    turn = s.reply("go")
    assert turn.kind == "confirm" and turn.spec.name == "barn" and turn.spec.labels == ["empty", "cow"]
    assert turn.text.startswith("Here is the setup, check it.\n\nSpec 'barn'")


def test_revise_starts_from_the_trained_spec():
    from sim2chip import spec as spec_mod
    changed = {**GOOD, "hardware": "jetson_nano"}
    client = Scripted([{"say": "Sure, anything else?", "spec": changed}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}))
    first = s.revise(spec_mod.from_dict(GOOD))
    assert first.kind == "say" and "change" in first.text and client.seen == []  # no model call yet
    turn = s.reply("run it on a Jetson instead")
    assert turn.kind == "confirm" and turn.spec.hardware == "jetson_nano" and turn.spec.name == "barn"
    assert '"hardware"' in s.messages[-3]["content"] and "keep everything" in s.messages[-4]["content"]


def test_revise_does_not_reword_settled_queries():
    from sim2chip import spec as spec_mod
    base = {**GOOD, "classes": [{**GOOD["classes"][0], "assets": {"source": "search", "queries": ["cow", "vaca"]}}],
            "props": [{"id": "fence", "queries": ["railings"], "size_m": {"measure": "length", "range": [1, 2]}}]}
    base = json.loads(spec_mod.from_dict(base).to_json())
    client = Scripted([{"spec": {**base, "hardware": "jetson_nano"}}])  # no second answer: no translation call
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}))
    s.revise(spec_mod.from_dict(base))
    turn = s.reply("on a Jetson")
    assert turn.kind == "confirm" and len(client.seen) == 1
    assert turn.spec.props[0].queries == ["railings"] and turn.spec.classes[0].assets.queries == ["cow", "vaca"]
