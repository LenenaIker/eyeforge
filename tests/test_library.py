import json

from PIL import Image

from sim2chip.library import Library
from sim2chip.workspace import Job


def photos(folder, colors):
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i, c in enumerate(colors):
        out.append(folder / f"{i}.jpg")
        Image.new("RGB", (32, 24), c).save(out[-1])
    return out


def job_with(home, name, colors):
    job = Job(home / "jobs" / name)
    photos(job.input / "photos", colors)
    job.root.mkdir(parents=True, exist_ok=True)
    (job.root / "spec.json").write_text(json.dumps({"classes": [{"id": "a"}], "user_data": {"raw": "photos"}}))
    return job


def test_jobs_with_the_same_photos_share_one_dataset(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    a, b = job_with(tmp_path, "a", ["red", "blue"]), job_with(tmp_path, "b", ["red", "blue"])
    lib.sync("u", [a, b])
    (ds,) = lib.datasets("u")
    assert ds["photos"] == 2 and ds["jobs"] == ["a", "b"]
    assert lib.datasets("other") == []


def test_chat_photos_make_a_dataset_the_job_then_uses(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    upload = photos(tmp_path / "uploads", ["red", "blue", "green"])
    chat = lib.add_chat_photos("u", "s1", upload, "train", None)
    job = job_with(tmp_path, "j", ["red", "blue"])  # two of the chat's three photos
    lib.sync("u", [job])
    (ds,) = lib.datasets("u")
    assert ds["id"] == chat and ds["photos"] == 3 and ds["jobs"] == ["j"]


def test_deleted_items_leave_kits_and_do_not_come_back(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = job_with(tmp_path, "j", ["red"])
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]["id"]
    kit = lib.save_kit("u", None, "K", {"datasets": [ds, "not-mine"]})
    assert lib.kits("u")[0]["datasets"] == [ds]
    assert lib.delete("u", "dataset", ds) == 1
    assert lib.kits("u")[0]["datasets"] == [] and lib.kits("u")[0]["id"] == kit
    assert not list((tmp_path / "library" / "files" / "photos").rglob("*.jpg"))
    (job.root / "spec.json").touch()  # the job changes: it is read again, the dataset stays deleted
    lib.db.execute("DELETE FROM synced")
    lib.sync("u", [job])
    assert lib.datasets("u") == []


def test_a_dataset_goes_into_a_job_by_label(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = Job(tmp_path / "jobs" / "j")
    photos(job.input / "lab" / "cat", ["red"])
    photos(job.input / "lab" / "dog", ["blue"])
    (job.root / "spec.json").write_text(json.dumps({"classes": [], "user_data": {"labeled": "lab"}}))
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]
    assert ds["labels"] == ["cat", "dog"]
    out = tmp_path / "out"
    assert lib.materialize_dataset("u", ds["id"], out, by_label=True) == 2
    assert sorted(p.parent.name for p in out.rglob("*.jpg")) == ["cat", "dog"]
    assert lib.materialize_dataset("intruder", ds["id"], tmp_path / "x", by_label=False) == 0


def test_the_chat_may_only_use_listed_library_ids():
    from types import SimpleNamespace

    from sim2chip import spec as spec_mod
    from sim2chip.interview import _library_refs
    data = {"name": "t", "classes": [{"id": "pot", "size_m": {"measure": "length", "range": [0.1, 0.3]},
                                      "assets": {"source": "library", "models": ["aaaaaaaaaaaaaaaa"]}}],
            "user_data": {"raw": "library:bbbbbbbbbbbbbbbb"}}
    s = spec_mod.from_dict(data)
    assert _library_refs(SimpleNamespace(library_ids={"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"}), s) is None
    assert "bbbbbbbbbbbbbbbb" in _library_refs(SimpleNamespace(library_ids={"aaaaaaaaaaaaaaaa"}), s)


def test_a_job_using_library_objects_does_not_copy_them(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    glb = tmp_path / "m.glb"
    glb.write_bytes(b"glTF fake")
    oid = lib.new_object("u", "Pot", "pot", glb, None)
    job = job_with(tmp_path, "j", ["red"])
    (job.stage_dir("convert_assets")).mkdir(parents=True)
    (job.stage_dir("convert_assets") / "assets.json").write_text(json.dumps(
        [{"uid": oid, "class": "pot", "glb": f"library/{oid}.glb"}]))
    lib.sync("u", [job])
    (obj,) = lib.objects("u")
    assert obj["id"] == oid and obj["jobs"] == ["j"]


def test_a_dataset_profile_is_made_once_and_read_by_the_chat(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    ds = lib.new_dataset("u", "Table", photos(tmp_path / "up", ["red", "blue", "green"]))
    asked = []

    def run(files):
        asked.extend(files)
        return [{"photo": f.name, "context": "indoor", "distance_m": 0.8, "light": "indoor", "floor": "wood",
                 "objects": [{"name": "Bottle", "count": 4}, {"name": "shoe", "count": 1}],
                 "background": "dark wooden table"} for f in files]
    assert lib.profile_stale("u", ds)
    lib.profile_dataset("u", ds, run)
    assert len(asked) == 3 and not lib.profile_stale("u", ds)
    prof = lib.datasets("u")[0]["profile"]
    assert prof["objects"][0] == {"name": "bottle", "share": 1.0, "count": 4}
    assert "dark wooden table" in prof["summary"] and "dark wooden table" in lib.for_chat("u")[0]
    lib.profile_dataset("u", ds, run)  # every photo already described
    assert len(asked) == 3
    assert lib.notes("other", [p for p in lib.dataset_notes("u", ds)], "scene", "x") == {}
    lib.delete("u", "dataset", ds)
    assert lib.db.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0


def test_corrections_of_one_job_come_back_in_the_next(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = job_with(tmp_path, "a", ["red", "blue"])
    (job.root / "spec.json").write_text(json.dumps({"task": "count", "classes": [{"id": "botella"}],
                                                    "user_data": {"raw": "photos"}}))
    label = job.stage_dir("label")
    label.mkdir(parents=True)
    (label / "teacher.json").write_text(json.dumps({"phrases": {"botella": ["bottle"]}}))
    (label / "verdicts.json").write_text(json.dumps({
        "0.jpg": {"objects": [{"x": 0.5, "y": 0.5, "class": "botella", "box": [0.4, 0.4, 0.6, 0.6], "score": 0.9}],
                  "model": "facebook/sam3", "threshold": 0.5, "assigned": "images"},
        "1.jpg": {"objects": [], "model": "facebook/sam3", "threshold": 0.5, "assigned": "images"}}))
    (job.input / "corrections.json").write_text(json.dumps({"points": {"1.jpg": [[0.2, 0.3, "botella"]]}}))
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]["id"]
    kinds = sorted((n["kind"], n["source"]) for v in lib.dataset_notes("u", ds).values() for n in v)
    assert kinds == [("boxes", "model"), ("boxes", "model"), ("points", "user")]
    # A new job with the same photos in another folder, its class named differently but the same in English.
    other = photos(tmp_path / "b" / "fotos", ["blue", "red"])
    got = lib.corrections_for("u", other[0].parent, ["bottle_cls"], {"bottle_cls": "bottle"}, [])
    assert got == {"points": {"0.jpg": [[0.2, 0.3, "bottle_cls"]]}}
    # A second class the user never marked: the photo stays the teacher's.
    assert lib.corrections_for("u", other[0].parent, ["bottle_cls", "cup"], {"bottle_cls": "bottle", "cup": "cup"}, []) == {}


def test_several_raw_sources_become_one_folder(tmp_path):
    from sim2chip import spec as spec_mod
    from sim2chip.server import merge_sources
    job = Job(tmp_path / "j")
    photos(job.input / "dataset_x", ["red", "blue"])
    photos(job.input / "photos", ["blue", "green"])
    assert merge_sources(job, ["dataset_x", "photos"]) == "raw"
    assert len(list((job.input / "raw").rglob("*.jpg"))) == 3
    s = spec_mod.from_dict({"name": "t", "classes": [{"id": "a"}], "user_data": {"raw": ["library:aaaaaaaaaaaaaaaa"]}})
    assert s.user_data.raw == "library:aaaaaaaaaaaaaaaa"


def test_photos_added_to_a_job_join_its_dataset(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    upload = photos(tmp_path / "uploads", ["red", "blue"])
    chat = lib.add_chat_photos("u", "s1", upload, "train", None)
    job = job_with(tmp_path, "j", ["red", "blue"])
    lib.sync("u", [job])
    photos(job.input / "photos" / "added", ["green"])  # taken in the live test
    (job.root / "spec.json").touch()
    lib.db.execute("DELETE FROM synced")
    lib.sync("u", [job])
    (ds,) = lib.datasets("u")
    assert ds["id"] == chat and ds["photos"] == 3


def test_forgetting_an_owner_leaves_others_alone(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    mine, theirs = job_with(tmp_path, "m", ["red"]), job_with(tmp_path, "t", ["blue"])
    lib.sync("u", [mine])
    lib.sync("v", [theirs])
    lib.save_kit("u", None, "K", {})
    lib.forget("u", ["m"])
    assert lib.datasets("u") == [] and lib.kits("u") == [] and len(lib.datasets("v")) == 1
