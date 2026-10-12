import json
from pathlib import Path

import pytest

from sim2chip import interview


class Scripted:
    """Fake Nemotron that replays a list of JSON actions and records what it was told."""

    def __init__(self, actions):
        self.actions, self.seen = list(actions), []

    def chat(self, messages, model, purpose, **kw):
        if purpose == "typical_sizes" and not (self.actions and isinstance(self.actions[0], dict)
                                               and self.actions[0].get("_sizes")):
            return "{}"  # sizes unknown: nothing checked; a test that checks them scripts {"_sizes": {...}}
        if purpose == "typical_sizes":
            return json.dumps(self.actions.pop(0)["_sizes"])
        if purpose == "english_queries" and not (self.actions and isinstance(self.actions[0], dict)
                                                 and self.actions[0].get("_english")):
            # Words already in English stay as they are; a test that translates scripts {"_english": {...}}.
            return json.dumps({t: t for t in json.loads(messages[-1]["content"].split("Terms: ", 1)[1])})
        self.seen.append(messages[-1]["content"])
        action = self.actions.pop(0)
        return action if isinstance(action, str) else json.dumps(action.get("_english", action))


GOOD = {"name": "barn", "classes": [{"id": "cow", "description": "cows", "size_m": {"measure": "length",
        "range": [2, 2.5]}, "assets": {"categories": ["cow"], "count": 10}}], "qa": {"mode": "auto"},
        "user_data": {"raw": "photos"}}


def test_interview_flow():
    bad = {**GOOD, "classes": [{**GOOD["classes"][0], "size_m": {"measure": "length", "range": [3, 1]}}]}
    client = Scripted([{"say": "Where is the camera?"}, {"search": ["cows"]}, {"spec": bad}, {"spec": GOOD},
                       {"spec": GOOD}])
    answers = iter(["cows in my barn", "on a post, 3 m", "add night please", "yes"])
    told = []
    cats = interview.CategoryIndex(counts={"cow": 62, "calf": 11, "taxi": 3})
    spec = interview.run(client, "m", cats, ask=lambda q: next(answers), tell=told.append)
    assert spec.name == "barn"
    assert "cow" in client.seen[2] and "taxi" not in client.seen[2]    # search results went back
    assert "size_m.range" in client.seen[3]                            # invalid spec rejected
    assert "wants changes: add night please" in client.seen[4]         # user asked for a change
    assert "cow" in told[0]


def test_every_action_has_a_handler_and_unknown_ones_go_back_to_the_model():
    assert all(callable(getattr(interview.Session, m, None)) for _, m in interview.Session.ACTIONS)
    client = Scripted([{"dance": True}, {"photos": "not an object"}, {"say": "Where is the camera?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 1}))
    turn = s.reply("cows")
    assert turn.kind == "say" and turn.text == "Where is the camera?"
    assert client.seen[1].startswith("Unknown action") and client.seen[2].startswith("Unknown action")


def test_system_prompt_mentions_schema():
    p = interview.system_prompt()
    assert '"target_fill"' in p and "esp32cam" in p and "road" in p


def test_is_yes():
    for a in ["yes", "Sí, adelante.", "ok!", "vale", "de acuerdo", "Perfecto, gracias", "si"]:
        assert interview.is_yes(a), a
    for a in ["no", "sí, pero cambia la altura", "add night please", "", "the trucks should be 10 m long instead"]:
        assert not interview.is_yes(a), a


def test_scene_tags_are_checked(template_on):
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


def test_interview_proposes_the_users_own_photos_for_3d(tmp_path, template_on):
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


def test_turn_steps_show_what_the_interview_is_doing():
    client = Scripted(['{"say": ', {"search": ["cow"]}, {"spec": {"name": "x"}}, {"say": "Where is the camera?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}))
    s.begin("t1")
    assert s.reply("cows").text == "Where is the camera?"
    steps = [(a["text"], a["state"]) for a in s.activity]
    assert steps[0] == ("Thinking", "failed") and ("Searching the 3D model catalog: cow", "done") in steps
    assert ("Checking the proposed setup", "failed") in steps and steps[-1] == ("Thinking about what it found", "done")
    assert all(a["ended"] >= a["started"] for a in s.activity) and s.activity_id == "t1"
    s.begin("t2")
    assert s.activity == [] and s.activity_id == "t2"


def test_photo_steps_count_photos_when_the_helper_can():
    s = interview.Session(Scripted([]), "m", interview.CategoryIndex(counts={}))
    step = s._step("Looking at your photos")

    def helper(folder, progress=None):
        progress(3, 8)
        return folder
    assert s._counting(helper, step, "f") == "f" and (step["done"], step["total"]) == (3, 8)
    assert s._counting(lambda folder: folder, step, "g") == "g"  # helpers without progress still work


def test_interview_asks_for_photos_when_none_fit(tmp_path, template_on):
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



def test_photos_attached_to_a_message_make_one_turn_with_the_text(tmp_path):
    """The web stores attached photos without a turn, then sends the text: the model sees both in one turn,
    the photos' note before the user's words; photos already told about are not told again."""
    up = tmp_path / "uploads"
    (up / "training").mkdir(parents=True)
    (up / "library_x").mkdir()
    (up / "library_x" / "z.jpg").write_bytes(b"x")  # a library dataset opened by the chat: not attached
    for name in ("a.jpg", "b.jpg"):
        (up / "training" / name).write_bytes(b"x")
    client = Scripted([{"say": "Nice photos. Where is the camera?"}, {"say": "And at night?"},
                       {"say": "Got the third one."}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}), uploads=up)
    assert s.reply("count bottles on my desk").kind == "say"
    users = [m["content"] for m in s.messages if m["role"] == "user"]
    note = next(i for i, c in enumerate(users) if c.startswith("[The user provided 2 photos from their camera"))
    assert users.index("count bottles on my desk") > note and s.goal == "count bottles on my desk"
    s.reply("on a shelf")
    notes = lambda: [m["content"][:30] for m in s.messages  # noqa: E731
                     if m["role"] == "user" and m["content"].startswith("[The user provided")]
    assert len(notes()) == 1
    (up / "training" / "c.jpg").write_bytes(b"x")
    s.reply("")  # photos sent without text
    assert notes() == ["[The user provided 2 photos fr", "[The user provided 1 photos fr"]
    assert "" not in [m["content"] for m in s.messages]

def test_rejected_specs_stop_and_ask_the_user(tmp_path):
    bad = {**GOOD, "classes": [{**GOOD["classes"][0], "size_m": {"measure": "length", "range": [3, 1]}}]}
    client = Scripted([{"spec": bad}] * interview.MAX_INVALID_SPECS + [{"say": "No encuentro unicornios, ¿vale un caballo?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}))
    s.transcript = tmp_path / "t.json"
    t = s.reply("unicorns")
    assert t.kind == "say" and "caballo" in t.text
    assert "Stop proposing it" in client.seen[-1]
    assert "size_m.range" in s.transcript.read_text()  # rejections are kept for later


def test_bare_class_names_are_sent_back_not_a_crash():
    """Seen in the interview bench (2026-10-12): "classes": ["car", ...] broke the turn."""
    client = Scripted([{"spec": {**GOOD, "classes": ["cow"]}}, {"say": "ok"}])
    t = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62})).reply("cows")
    assert t.kind == "say" and "classes[0]" in client.seen[-1]


def test_endless_rejections_end_the_interview():
    bad = {**GOOD, "classes": [{**GOOD["classes"][0], "size_m": {"measure": "length", "range": [3, 1]}}]}
    client = Scripted([{"spec": bad}] * (interview.MAX_INVALID_SPECS * 2))
    t = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62})).reply("unicorns")
    assert t.kind == "failed" and "size_m.range" in t.text


def test_broken_answers_end_the_interview():
    # Seen 2026-10-03: the same malformed JSON 59 times in a row. The model is told why, then the interview stops.
    client = Scripted(['{"say": ', '{"spec": {', '{"say"', {"say": "never asked"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}))
    t = s.reply("cows")
    assert t.kind == "failed" and s.finished
    assert "not valid JSON" in client.seen[1]
    assert len(client.actions) == 1


def test_a_good_answer_resets_the_broken_count():
    client = Scripted(["oops", "oops", {"say": "What should it count?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}))
    assert s.reply("cows").kind == "say"


def test_no_photo_models_on_a_server_without_them():
    photos = {**GOOD, "classes": [{**GOOD["classes"][0], "assets": {"source": "photos", "photos": "user_data",
                                                                     "queries": ["cow"]}}], "user_data": {"raw": "x"}}
    client = Scripted([{"pick_photos": {"class": "cow", "query": "cow", "folder": "/tmp"}}, {"spec": photos},
                       {"spec": GOOD}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}), photo_models=False)
    t = s.reply("cows, here are my photos")
    assert t.kind == "confirm" and t.spec.classes[0].assets.photos is None
    assert "cannot be made from the user's photos" in client.seen[1]  # pick_photos refused
    assert "cannot be made from the user's photos" in client.seen[2]  # photo-made spec refused
    assert "cannot be made from the user's photos" in s.messages[0]["content"]


def test_spec_sent_with_text_is_proposed():
    cats = interview.CategoryIndex(counts={"cow": 62})
    good = {"name": "barn", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                         "assets": {"categories": ["cow"], "count": 4}},
                                        {"id": "empty", "description": "no cow"}],  # background listed too
            "user_data": {"raw": "photos"}}
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


def test_research_looks_facts_up_and_is_logged(tmp_path):
    """Nemotron asks the web instead of guessing a size; the search is logged for the usage panel and capped."""
    from sim2chip.research import ResearchError
    asked = []

    def researcher(q):
        asked.append(q)
        if "fail" in q:
            raise ResearchError("Tavily: HTTP 432")
        return {"answer": "A Holstein cow is about 2.4 m long.",
                "sources": [{"title": "Cattle", "url": "https://example.org/cow", "content": "Length 2.3-2.5 m"}]}
    client = Scripted([{"research": "Holstein cow body length"}, {"research": "fail please"},
                       {"say": "Where is the camera?"}] + [{"research": "more"}] * interview.MAX_RESEARCH
                      + [{"say": "ok"}])
    client.usage_log = tmp_path / "usage.jsonl"
    s = interview.Session(client, "m", interview.CategoryIndex(counts={"cow": 62}), researcher=researcher)
    assert '"research"' in s.messages[0]["content"]
    assert s.reply("cows").text == "Where is the camera?"
    assert "2.4 m long" in client.seen[1] and "https://example.org/cow" in client.seen[1]
    assert "research failed" in client.seen[2]
    assert s.reply("on a post").text == "ok"
    assert len(asked) == interview.MAX_RESEARCH and "No more research" in client.seen[-1]
    logged = [json.loads(line) for line in client.usage_log.read_text().splitlines()]
    assert len(logged) == interview.MAX_RESEARCH - 1 and logged[0]["model"] == "tavily/search"  # failures not billed
    # Without a key the action is not offered.
    assert '"research"' not in interview.Session(Scripted([]), "m", interview.CategoryIndex(counts={})).messages[0][
        "content"]


def test_folder_layout_is_read_not_asked(tmp_path):
    """A folder written in the message: its layout goes to the model before it answers, so it never asks."""
    raw, sorted_ = tmp_path / "raw", tmp_path / "sorted"
    for d, names in ((raw, ["a.jpg", "b.jpg"]), (sorted_ / "grass", ["c.jpg"]), (sorted_ / "edge", ["d.png", "e.png"])):
        d.mkdir(parents=True)
        for n in names:
            (d / n).write_bytes(b"x")
    assert "2 photos, no subfolders" in interview.folder_layout(raw)
    assert "edge (2), grass (1)" in interview.folder_layout(sorted_)
    (sorted_ / "loose.jpg").write_bytes(b"x")
    assert "mixed" in interview.folder_layout(sorted_)

    client = Scripted([{"say": "What device?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    s.reply(f"lawn, border or no grass; photos in {raw}")
    assert "[Folder layout" in client.seen[-1] and "no subfolders" in client.seen[-1]


def test_counting_asks_for_english_names_for_sam3():
    """SAM 3 marks the user's photos and reads only English: a count spec without class names goes back."""
    cls = {"id": "maceta", "description": "macetas", "size_m": {"measure": "length", "range": [0.1, 0.5]}}
    spec = {"name": "pots", "task": "count", "background_class": "empty", "classes": [cls],
            "user_data": {"raw": "x"}}
    named = {**spec, "classes": [{**cls, "names": ["flower pot"]}]}
    client = Scripted([{"spec": spec}, {"spec": named}, {"flower pot": "flower pot"}])  # the last: English check
    t = interview.Session(client, "m", interview.CategoryIndex(counts={})).reply("cuenta mis macetas")
    assert "understands only English" in client.seen[1]
    assert t.kind == "confirm" and t.spec.classes[0].names == ["flower pot"]


def test_a_failed_photo_check_lets_the_user_choose(tmp_path, template_on):
    """The vision model timed out on every photo (2026-09-30): that is not "no photo shows the pot"."""
    photos = [str(tmp_path / f"{i}.jpg") for i in range(3)]
    ranker = lambda f, q: [{"file": p, "score": 0.0, "reason": interview.NO_VERDICT[1]} for p in photos]
    pick = {"pick_photos": {"class": "pot", "query": "flower pot", "folder": str(tmp_path)}}
    s = interview.Session(Scripted([pick]), "m", interview.CategoryIndex(counts={}), photo_ranker=ranker)
    t = s.reply("pots")
    assert t.kind == "pick" and [c["file"] for c in t.request["candidates"]] == photos
    assert "could not check" in t.text


def test_counting_names_keep_only_english():
    """SAM 3 gets the names: the user's word is replaced, not kept after the English one."""
    cls = {"id": "pot", "size_m": {"measure": "length", "range": [0.1, 0.5]},
           "names": ["maceta", "flower pot"]}
    spec = {"name": "pots", "task": "count", "background_class": "empty", "classes": [cls], "user_data": {"raw": "x"}}
    client = Scripted([{"spec": spec}, {"_english": {"maceta": "flower pot", "flower pot": "flower pot"}}])
    t = interview.Session(client, "m", interview.CategoryIndex(counts={})).reply("cuenta mis macetas")
    assert t.kind == "confirm" and t.spec.classes[0].names == ["flower pot"]


def test_no_synthetic_images_means_no_3d_stages(template_on):
    """frames 0 with models built from photos (2026-09-30) ran the 3D stages for nothing."""
    from sim2chip.spec import from_dict
    from sim2chip.stages import STAGES
    cls = {"id": "pot", "size_m": {"measure": "length", "range": [0.1, 0.5]}, "names": ["flower pot"],
           "assets": {"source": "photos", "queries": ["flower pot"], "photos": "/nowhere"}}
    spec = {"name": "pots", "task": "count", "background_class": "empty", "classes": [cls], "user_data": {"raw": "x"}}
    run = lambda s: [st.name for st in STAGES if st.enabled(from_dict(s))]
    assert "render" in run(spec) and "generate_assets" in run(spec)
    only_photos = run({**spec, "generation": {"frames": 0}})
    assert not {"fetch_assets", "generate_assets", "render", "points"} & set(only_photos)
    assert {"label", "ingest", "train", "export"} <= set(only_photos)
    assert "synthetic images: none" in interview.summary(from_dict({**spec, "generation": {"frames": 0}}))


def test_a_dataset_id_is_never_a_3d_model():
    """The chat read "3D models kept" in a dataset's line and gave the dataset's id as the models."""
    from sim2chip.interview import _dataset_as_model
    from sim2chip import spec as spec_mod

    class S:
        dataset_ids = {"aaaaaaaaaaaaaaaa"}
    cls = {"id": "cow", "description": "cows", "size_m": {"measure": "length", "range": [1.0, 2.0]}}
    wrong = spec_mod.from_dict({"name": "x", "classes": [{**cls, "assets": {"source": "library",
                                                                             "models": ["library:aaaaaaaaaaaaaaaa"]}}]})
    right = spec_mod.from_dict({"name": "x", "classes": [{**cls, "assets": {"source": "library",
                                                                             "models": ["library:bbbbbbbbbbbbbbbb"]}}]})
    assert "search" in _dataset_as_model(S(), wrong) and _dataset_as_model(S(), right) is None


def test_class_ids_go_into_english_everywhere_the_spec_names_them():
    cls = {"id": "contenedor_azul", "description": "contenedores azules", "names": ["blue bin"]}
    spec = {"name": "bins", "classes": [cls, {**cls, "id": "nada", "names": ["nothing"]},
                                        {**cls, "id": "cubo", "names": ["bucket"]}],
            "user_data": {"labeled": "/x", "class_map": {"contenedor_azul": ["azul"], "nada": ["vacio"]}}}
    client = Scripted([{"_english": {"contenedor azul": "blue bin", "nada": "nothing", "cubo": "blue bin",
                                     "blue bin": "blue bin", "nothing": "nothing", "bucket": "bucket"}}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    s._english_queries(spec)
    assert [c["id"] for c in spec["classes"]] == ["blue_bin", "nothing", "cubo"]  # never two classes one id
    assert spec["user_data"]["class_map"] == {"blue_bin": ["azul"], "nothing": ["vacio"]}
    assert any("contenedor_azul -> blue_bin" in m["content"] for m in s.messages)


LID = "1111111111111111"


def card(names, lid=LID):
    """A library.dataset_card with one count labeling of these English names."""
    return {"id": "x", "name": "street", "photos": 38, "what_its_photos_show": None,
            "labelings": [{"id": lid, "name": "Bins", "task": "count",
                           "classes": [{"key": n.replace(" ", "_"), "names": [n], "description": ""} for n in names],
                           "photos_reviewed_by_user": 3, "photos_with_marks": 38, "models": []}]}


def test_a_dataset_labeled_for_other_classes_is_asked_about_once():
    """A spec that names no labeling of a dataset that has some, or changes one's classes, goes back to the model,
    which must ask the user; after the user answers, the same spec is proposed."""
    ds = "a7d9ef2d4b43c27b"
    blue = {"id": "blue_bin", "description": "blue bins", "names": ["blue bin"]}
    can = {"id": "can", "description": "drink can", "names": ["drink can"]}
    spec = {"name": "bins", "task": "count", "background_class": "empty", "classes": [blue, can],
            "user_data": {"raw": f"library:{ds}", "labeling": LID}}
    client = Scripted([{"spec": spec}, {"say": "It has blue bins: add cans to it?"}, {"spec": spec}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    s.use_library("- a dataset", {ds}, datasets={ds}, dataset_card=lambda i: card(["blue bin"]))
    t = s.reply("count bins and cans in my street photos")
    assert t.kind == "say" and "Labelings of dataset" in client.seen[1] and "adding can" in client.seen[1]
    assert f"(id {LID}): blue_bin" in client.seen[1]
    assert s.reply("add cans").kind == "confirm"
    # Its classes exactly: no question; no labeling named: a question.
    fresh = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    fresh.use_library("- a dataset", {ds}, datasets={ds}, dataset_card=lambda i: card(["blue bin"]))
    only = {**spec, "classes": [blue]}
    assert fresh.relabel_question(interview.spec_mod.from_dict(only)) is None
    none = {**only, "user_data": {"raw": f"library:{ds}"}}
    assert "names no labeling" in fresh.relabel_question(interview.spec_mod.from_dict(none))


def test_a_class_of_the_labeling_takes_its_english_names():
    """The chat named the blue bin class by its 3D search; the labeling calls it "blue bin": that name wins."""
    ds = "a7d9ef2d4b43c27b"
    s = interview.Session(Scripted([]), "m", interview.CategoryIndex(counts={}))
    s.use_library("- a dataset", {ds}, datasets={ds}, dataset_card=lambda i: card(["blue bin", "plastic bag"]))
    spec = {"task": "count", "user_data": {"raw": f"library:{ds}", "labeling": LID}, "classes": [
        {"id": "blue_bin", "names": ["blue recycling bin"], "assets": {"source": "search", "queries": ["blue recycling bin"]}},
        {"id": "papelera", "names": ["litter bin"]}]}
    s._reuse_label_names(spec)
    assert [c.get("names") for c in spec["classes"]] == [["blue bin"], ["litter bin"]]


def test_scene_tags_and_weather_the_code_can_settle(monkeypatch):
    from sim2chip import polyhaven
    monkeypatch.setattr(polyhaven, "catalog", lambda kind: kind)
    monkeypatch.setattr(polyhaven, "pick_textures", lambda cat, tags, n, seed: ["t"] if set(tags) <= {"asphalt"} else [])
    monkeypatch.setattr(polyhaven, "pick_skies", lambda cat, tags, weather, n, seed: ["s"])
    spec = {"scene": {"weather": [], "ground_tags": ["asphalt", "pavement"], "sky_tags": ["urban"]}}
    interview.Session(Scripted([]), "m", interview.CategoryIndex(counts={}))._fix_scene(spec)
    assert spec["scene"] == {"weather": ["clear"], "ground_tags": ["asphalt"], "sky_tags": ["urban"]}


def test_the_dataset_action_shows_the_dataset_as_it_is_now():
    ds = "a7d9ef2d4b43c27b"
    client = Scripted([{"dataset": f"library:{ds}"}, {"dataset": "0123456789abcdef"}, {"say": "Which bins?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    s.use_library("- a dataset", {ds}, datasets={ds}, dataset_card=lambda i: card(["blue bin"]))
    assert s.reply("count bins with my street dataset").text == "Which bins?"
    notes = [m["content"] for m in s.messages if m["role"] == "user"]
    assert any(n.startswith(f"[Dataset {ds}]") and '"blue bin"' in n and LID in n for n in notes)
    assert "not one of the listed datasets" in client.seen[2]


def test_plain_slips_are_fixed_without_another_turn():
    spec = {"scene": {"orientation": "any"}, "classes": [
        {"id": "item", "names": ["item"], "assets": {"source": "search"}},
        {"id": "big_cup", "assets": {"source": "search", "queries": []}}]}
    interview._fix_slips(spec)
    assert spec["training"] == {"orientation": "any"} and "orientation" not in spec["scene"]
    assert [c["assets"]["queries"] for c in spec["classes"]] == [["item"], ["big cup"]]


def test_more_plain_slips_are_fixed():
    spec = {"scene": {"distractors": ["tree", "fence"]}, "camera": {"pitch_deg": 30}, "classes": [
        {"id": "car", "assets": {"source": "objaverse_lvis", "categories": ["car"], "count": 0}},
        {"id": "cow", "assets": {"categories": ["cow"]}},
        {"id": "edge", "names": ["lawn edge"], "assets": {"source": "objaverse_lvis"}}]}
    interview._fix_slips(spec, {"cow", "car_(automobile)"})
    assert spec["scene"]["distractors"] == ["tree"] and spec["camera"]["pitch_deg"] == [20, 40]
    assert spec["classes"][0]["assets"] == {"source": "search", "queries": ["car"]}
    assert spec["classes"][1]["assets"] == {"categories": ["cow"]}
    assert spec["classes"][2]["assets"] == {"source": "search", "queries": ["lawn edge"]}


def test_an_english_id_is_not_renamed_by_a_longer_wording():
    spec = {"classes": [{"id": "cat"}], "background_class": "empty"}
    client = Scripted([{"_english": {"cat": "domestic cat", "empty": "empty space"}}])
    interview.Session(client, "m", interview.CategoryIndex(counts={}))._english_queries(spec)
    assert spec["classes"][0]["id"] == "cat" and spec["background_class"] == "empty"


def test_a_classification_on_the_users_labels_has_no_background_class():
    labels = {"raw": "library:a7d9ef2d4b43c27b", "labeling": "654f69c500ee4a24"}
    surfaces = {"task": "roi_classification", "background_class": "empty", "user_data": labels,
                "classes": [{"id": "grass"}, {"id": "border"}, {"id": "no_grass"}]}
    interview._drop_background_class(surfaces)
    assert surfaces["background_class"] is None and len(surfaces["classes"]) == 3
    counting = {"task": "count", "background_class": "empty", "user_data": labels, "classes": [{"id": "pot"}]}
    interview._drop_background_class(counting)
    assert counting["background_class"] == "empty"
    unlabeled = {"background_class": "empty", "user_data": {"raw": "photos"}, "classes": [{"id": "cat"}]}
    interview._drop_background_class(unlabeled)
    assert unlabeled["background_class"] == "empty"


def test_a_dataset_the_user_names_is_looked_at_first_and_never_searched_on_the_web():
    ds, other = "a7d9ef2d4b43c27b", "8c1b568d88d029d2"
    cards = {ds: {**card(["blue bin"]), "name": "Chat photos 09 Oct 2026 15:16"},
             other: {**card(["item"]), "name": "contador mesa"}}
    client = Scripted([{"research": f"dataset {other}"}, {"say": "Which bins?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    s.use_library("- two datasets", {ds, other}, datasets={ds, other}, dataset_card=cards.get)
    assert s._named_datasets("usa el dataset 'Chat photos 09 Oct' por favor") == [ds]
    assert s._named_datasets("con mi dataset contador mesa") == [other]
    s.reply("cuenta contenedores con 'Chat photos 09 Oct'")
    notes = [m["content"] for m in s.messages if m["role"] == "user"]
    assert notes[0].startswith(f"[Dataset {ds}]") and any(n.startswith(f"[Dataset {other}]") for n in notes)
    assert notes.index("cuenta contenedores con 'Chat photos 09 Oct'") > 0  # the user's words after the card
    assert not any(n.startswith("[Web research") for n in notes)


def test_plain_words_without_json_are_a_question():
    client = Scripted(["¿En qué dispositivo lo usarás?"])
    t = interview.Session(client, "m", interview.CategoryIndex(counts={})).reply("cuenta vacas")
    assert t.kind == "say" and t.text == "¿En qué dispositivo lo usarás?"


def test_the_proposal_says_which_labeling_and_which_new_classes():
    ds = "a7d9ef2d4b43c27b"
    blue = {"id": "blue_bin", "description": "blue bins", "names": ["blue bin"]}
    can = {"id": "trash_can", "description": "litter bins", "names": ["trash can"]}
    spec = {"name": "bins", "task": "count", "background_class": "empty", "classes": [blue, can],
            "user_data": {"raw": f"library:{ds}", "labeling": LID}}
    client = Scripted([{"spec": spec}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    s.use_library("- a dataset", {ds}, datasets={ds}, dataset_card=lambda i: card(["blue bin"]))
    s.relabel_asked[ds] = -1  # asked already
    t = s.reply("bins and litter bins")
    assert t.kind == "confirm"
    assert 'Labeling "Bins" (3 photos reviewed by you)' in t.intro and "new classes" in t.intro and "trash_can" in t.intro


def test_3d_models_with_zero_frames_get_the_default_frames():
    spec = {"generation": {"frames": 0}, "classes": [{"id": "cup", "assets": {"source": "search", "queries": ["cup"]}}]}
    interview._fix_slips(spec)
    assert "frames" not in spec["generation"]
    only = {"generation": {"frames": 0}, "classes": [{"id": "cup"}]}
    interview._fix_slips(only)
    assert only["generation"]["frames"] == 0  # photos only, as asked


def test_catch_all_classes_are_sent_back():
    from sim2chip import spec as spec_mod
    from sim2chip.interview import catch_all_problem
    base = {"name": "t", "task": "count", "user_data": {"raw": "photos"}}
    bad = spec_mod.from_dict({**base, "classes": [{"id": "litter", "names": ["litter", "trash"]},
                                                  {"id": "blue_bin", "names": ["blue bin"]}]})
    assert "['litter']" in catch_all_problem(bad)
    ok = spec_mod.from_dict({**base, "classes": [{"id": "trash_bag", "names": ["trash bag"]},
                                                 {"id": "waste_bin", "names": ["waste bin"]}]})
    assert catch_all_problem(ok) is None


def test_sizes_that_do_not_fit_the_thing_go_back():
    """A utility pole of 0.1-0.2 m by its length (trash_counter, 2026-10-11) is a tall thing measured wrong."""
    from sim2chip import spec as spec_mod
    pole = {"id": "utility_pole", "queries": ["utility pole"], "size_m": {"measure": "length", "range": [0.1, 0.2]}}
    hydrant = {"id": "fire_hydrant", "queries": ["fire hydrant"], "size_m": {"measure": "height", "range": [0.1, 0.2]}}
    bin_ = {"id": "bin", "description": "bins", "names": ["wheelie bin"], "size_m": {"measure": "height", "range": [0.9, 1.2]},
            "assets": {"source": "search", "queries": ["wheelie bin"]}}
    s = spec_mod.from_dict({"name": "t", "classes": [bin_], "props": [pole, hydrant]})
    client = Scripted([{"_sizes": {"utility pole": {"length": [0.2, 0.4], "height": [6, 12]},
                                   "fire hydrant": {"length": [0.3, 0.5], "height": [0.6, 1.0]},
                                   "wheelie bin": {"length": [0.5, 0.8], "height": [0.9, 1.1]}}}])
    session = interview.Session(client, "m", interview.CategoryIndex(counts={}))
    problem = session.sizes_problem(s)
    assert "utility_pole (utility pole) is tall" in problem and "fire_hydrant (fire hydrant): height 0.1-0.2 m" in problem
    assert "class bin" not in problem
    assert session.sizes_problem(spec_mod.from_dict({"name": "t", "classes": [bin_]})) is None  # asked once, kept


def test_the_photo_analysis_goes_into_the_spec_in_code():
    """Top-down photos become their surface seen from above; walls right behind objects on open ground stay."""
    from sim2chip import scene_profile
    table = [{"context": "indoor", "camera_pitch_deg": 75, "distance_m": 0.5, "supports": ["furniture"],
              "floor": "wood", "light": "indoor", "photo": f"{i}.jpg"} for i in range(4)]
    prof = scene_profile.aggregate(table)
    assert prof["contexts"] == [{"kind": "ground", "share": 1.0, "supports": ["ground"], "distance_m": [0.45, 0.55]}]
    assert prof["camera"]["pitch_deg"] == [70, 80] and prof["camera"]["height_m"][0] < 0.5
    street = [{"context": "ground", "camera_pitch_deg": 10, "distance_m": 4, "wall_behind": i < 3, "photo": f"{i}.jpg"}
              for i in range(4)]
    prof = scene_profile.aggregate(street)
    assert "camera" not in prof and prof["contexts"][0]["backdrop"] == 0.75
    s = interview.Session(Scripted([]), "m", interview.CategoryIndex(counts={}))
    s.measured = scene_profile.aggregate(table)
    spec = {"scene": {"contexts": [{"kind": "indoor", "share": 1}]}, "camera": {}}
    s._apply_measured(spec)
    assert spec["scene"]["contexts"][0]["kind"] == "ground" and spec["camera"]["pitch_deg"] == [70, 80]
    assert spec["scene"]["ground_tags"] == ["wood"]
    s.measured = {k: v for k, v in scene_profile.aggregate(street).items() if k != "suggest"}
    spec = {"scene": {"contexts": [{"kind": "ground", "share": 0.8}, {"kind": "facade", "share": 0.2}]}}
    s._apply_measured(spec)
    assert spec["scene"]["contexts"][0]["backdrop"] == 0.75 and "backdrop" not in spec["scene"]["contexts"][1]
