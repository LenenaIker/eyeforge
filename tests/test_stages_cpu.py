"""CPU stages on tiny fabricated data: ingest, qa_assets, crops and the C export."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.stages.crops import Crops
from sim2chip.stages.export import c_array, model_blob
from sim2chip.stages.ingest import Ingest
from sim2chip.stages.qa_assets import QAAssets
from sim2chip.workspace import Cache, link_input, open_job

SPEC = {"name": "t", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                  "assets": {"categories": ["cow"]}},
                                 {"id": "dog", "size_m": {"measure": "length", "range": [0.5, 1]},
                                  "assets": {"categories": ["dog"]}}]}


def ctx_for(tmp_path, **over):
    return Context(from_dict({**SPEC, **over}), open_job(tmp_path, "job"), Cache(tmp_path / "cache"))


def img(path, size=(64, 48), color=(200, 10, 10)):
    """A plain photo with its bottom-right corner set from its path: exact copies count as one photo in ingest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    im.paste(tuple(hashlib.sha256(str(path).encode()).digest()[:3]), (size[0] - 16, size[1] - 16, *size))  # survives JPEG
    im.save(path)


def test_ingest_maps_and_splits(tmp_path):
    photos = tmp_path / "photos"
    for i in range(10):
        img(photos / "vaca" / f"{i}.jpg")
        img(photos / "perro" / f"{i}.png")
        img(photos / "nada" / f"{i}.jpg")
    img(photos / "vaca" / "tiny.jpg", size=(10, 10))
    ctx = ctx_for(tmp_path, user_data={"labeled": "photos", "test_per_class": 4, "train_per_class": 3,
                                       "class_map": {"cow": ["vaca"], "dog": ["perro"], "empty": ["nada"]}})
    link_input(ctx.job, "photos", photos)
    info = Ingest().run(ctx)
    assert info["counts"]["cow"] == {"test": 4, "train": 3}
    out = ctx.job.stage_dir("ingest")
    files = list((out / "train" / "cow").glob("*.png"))
    assert len(files) == 3 and Image.open(files[0]).size == (96, 96)
    assert "tiny.png" not in {f.name for f in (out / "test" / "cow").glob("*")} | {f.name for f in files}


def test_ingest_separate_test_folder(tmp_path):
    photos, test = tmp_path / "photos", tmp_path / "test"
    for i in range(6):
        img(photos / "cow" / f"p{i}.jpg")
        img(test / "cow" / f"t{i}.jpg")
    # A phone photo stored landscape with an EXIF "rotate 90" tag must come out portrait before resizing.
    phone = Image.new("RGB", (80, 40), (255, 0, 0))
    phone.paste((0, 0, 255), (40, 0, 80, 40))  # left half red, right half blue
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    phone.save(photos / "cow" / "phone.jpg", exif=exif)
    ctx = ctx_for(tmp_path, hardware="generic",
                  user_data={"labeled": "photos", "test": "test", "test_per_class": 4, "train_per_class": 50})
    link_input(ctx.job, "photos", photos)
    link_input(ctx.job, "test", test)
    info = Ingest().run(ctx)
    assert info["counts"]["cow"] == {"test": 4, "train": 7}
    out = ctx.job.stage_dir("ingest")
    assert all(f.name.startswith("t") for f in (out / "test" / "cow").glob("*"))
    assert Image.open(out / "train" / "cow" / "p0.png").size == (224, 224)
    fixed = Image.open(out / "train" / "cow" / "phone.png")
    top, bottom = fixed.getpixel((112, 20)), fixed.getpixel((112, 200))
    assert top[0] > 200 and bottom[2] > 200  # upright: red on top, blue at the bottom


def test_ingest_takes_the_whole_test_folder(tmp_path):
    """A test folder smaller than test_per_class: every photo is a test photo (half of them went unused)."""
    photos, test = tmp_path / "photos", tmp_path / "test"
    for i in range(6):
        img(photos / "cow" / f"p{i}.jpg")
        img(test / "cow" / f"t{i}.jpg")
    ctx = ctx_for(tmp_path, user_data={"labeled": "photos", "test": "test", "test_per_class": 400,
                                       "train_per_class": 50})
    link_input(ctx.job, "photos", photos)
    link_input(ctx.job, "test", test)
    assert Ingest().run(ctx)["counts"]["cow"] == {"test": 6, "train": 6}


def test_ingest_keeps_photos_of_the_same_name(tmp_path):
    """a.jpg and a.png are two photos: both reach the model input, neither overwrites the other."""
    photos = tmp_path / "photos"
    img(photos / "cow" / "a.jpg")
    img(photos / "cow" / "a.png", color=(10, 200, 10))
    for i in range(4):
        img(photos / "cow" / f"p{i}.jpg")
    ctx = ctx_for(tmp_path, user_data={"labeled": "photos", "test_per_class": 0, "train_per_class": 50})
    link_input(ctx.job, "photos", photos)
    assert Ingest().run(ctx)["counts"]["cow"] == {"test": 0, "train": 6}
    assert len(list((ctx.job.stage_dir("ingest") / "train" / "cow").glob("*.png"))) == 6


def test_generate_assets_from_user_photos(tmp_path, monkeypatch):
    """TRELLIS is replaced by a fake: this checks photo selection, caching and the manifest."""
    import sys
    import types

    from sim2chip.stages.generate_assets import GenerateAssets

    calls = []

    class FakeModel:
        def detect(self, image, query):
            from sim2chip.img2mesh import Detection
            blue = image.getpixel((0, 0))[2] > 200  # blue photos "show" the object, red ones do not
            green = image.getpixel((0, 0))[1] > 200  # green: confident but tiny box, a false positive
            if green:
                return Detection(0.9, [0, 0, 2, 2])
            return Detection(0.8 if blue else 0.01, [0, 0, 10, 10] if blue else None)

        def generate(self, image, det, seed, glb_path):
            calls.append(seed)
            open(glb_path, "wb").write(b"glb")
            return Image.new("RGBA", (8, 8))

    fake = types.ModuleType("sim2chip.img2mesh")
    fake.MIN_SCORE, fake.ImageTo3D = 0.1, FakeModel
    from sim2chip import img2mesh
    fake.Detection = img2mesh.Detection
    monkeypatch.setitem(sys.modules, "sim2chip.img2mesh", fake)

    photos = tmp_path / "photos"
    img(photos / "vaca" / "a.jpg", color=(0, 0, 255))
    img(photos / "vaca" / "b.jpg", color=(200, 10, 10))
    img(photos / "vaca" / "c.jpg", color=(10, 250, 10))
    classes = [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                "assets": {"source": "photos", "photos": "user_data", "generate": 3, "queries": ["cow"]}}]
    ctx = ctx_for(tmp_path, classes=classes, user_data={"labeled": "photos", "class_map": {"cow": ["vaca"]}})
    link_input(ctx.job, "photos", photos)
    info = GenerateAssets().run(ctx)
    assert info["models"] == {"cow": 3} and calls == [1, 2, 3]  # one good photo, three seeds
    manifest = json.loads((ctx.job.stage_dir("generate_assets") / "manifest.json").read_text())
    assert {m["photo"] for m in manifest} == {"a.jpg"} and all(m["generated"] for m in manifest)
    assert all((ctx.cache.root / m["glb"]).exists() for m in manifest)
    GenerateAssets().run(ctx)
    assert calls == [1, 2, 3]  # second run: everything from the cache


def test_qa_manual(tmp_path):
    ctx = ctx_for(tmp_path, qa={"mode": "manual", "exclude": "r/ex.txt", "class_override": "r/ov.json",
                                "no_color": "r/nc.json"})
    conv = ctx.out("convert_assets")
    assets = [{"uid": u, "class": c, "usd": f"usd/{u}.usd", "dims": [2, 1, 1]}
              for u, c in [("a", "cow"), ("b", "cow"), ("c", "dog"), ("d", "cow")]]
    (conv / "assets.json").write_text(json.dumps(assets))
    r = ctx.job.input / "r"
    r.mkdir()
    (r / "ex.txt").write_text("a\n")
    (r / "ov.json").write_text(json.dumps({"b": {"class": "dog"}}))
    (r / "nc.json").write_text(json.dumps(["c"]))
    info = QAAssets().run(ctx)
    assert info["kept"] == {"dog": 2, "cow": 1}
    kept = json.loads((ctx.job.stage_dir("qa_assets") / "assets.json").read_text())
    assert [a["no_color"] for a in kept if a["uid"] == "c"] == [True]
    # a class left without models learns from the user's photos; with no models at all there is nothing to render
    ctx2 = ctx_for(tmp_path / "2")
    (ctx2.out("convert_assets") / "assets.json").write_text(json.dumps(assets[:1]))
    assert QAAssets().run(ctx2)["kept"] == {"cow": 1}
    ctx3 = ctx_for(tmp_path / "3")
    (ctx3.out("convert_assets") / "assets.json").write_text("[]")
    with pytest.raises(ValueError, match="dog"):
        QAAssets().run(ctx3)


def test_crops_from_fake_render(tmp_path):
    ctx = ctx_for(tmp_path)
    rd = ctx.out("render")
    labels = []
    for f, cls in enumerate(["cow", "empty", "dog"]):
        img(rd / f"rgb_{f:04d}.png", size=(320, 240))
        labels.append({"frame": f, "class": cls, "boxes": [] if cls == "empty" else [[cls, 100, 80, 180, 160]]})
    (rd / "labels.json").write_text(json.dumps(labels))
    info = Crops().run(ctx)
    assert info["crops"] == {"cow": 1, "empty": 1, "dog": 1}  # whole frames, one per frame


def test_c_array():
    h = c_array(bytes(range(20)), ["empty", "cow"], 96)
    assert "sim2chip_model[20]" in h and '"cow"' in h and "0x13" in h


def test_model_blob_header_matches_the_firmware():
    import struct
    import zlib
    model = bytes(range(256)) * 3
    b = model_blob(model, ["empty", "cow"], 96, False)
    assert b[:4] == b"EFM1"
    header, size, crc, side, classes, task = struct.unpack_from("<IIIHHB", b, 4)
    assert header % 16 == 0 and b[header:] == model and size == len(model) and crc == zlib.crc32(model)
    assert (side, classes, task) == (96, 2, 0) and b[24:header].rstrip(b"\0").split(b"\0") == [b"empty", b"cow"]


def test_export_report_is_portable(tmp_path):
    from sim2chip.stages.export import Export
    ctx = ctx_for(tmp_path)
    seed = ctx.job.stage_dir("train") / "seed_0"
    seed.mkdir(parents=True)
    (seed / "model_int8.tflite").write_bytes(b"\x00" * 2048)
    (seed / "model_float_io.tflite").write_bytes(b"\x01" * 1024)
    (ctx.job.stage_dir("train") / "summary.json").write_text(json.dumps({"0": {"val_acc": 0.9}}))
    Export().run(ctx)
    out = ctx.job.stage_dir("export")
    report = json.loads((out / "report.json").read_text())
    assert report["job"] == "t" and report["model_kb"] == 2.0
    assert (out / report["model_path"]).read_bytes() == b"\x00" * 2048
    assert (out / report["portable_model_path"]).read_bytes() == b"\x01" * 1024
    assert (out / "labels.txt").read_text().split() == ctx.spec.labels
    assert (out / "model_data.h").exists() == (ctx.hardware.runtime == "tflite-micro")
    assert (out / "model.bin").exists() == (ctx.hardware.runtime == "tflite-micro")


def test_export_ships_the_best_count_detector_as_its_recipe_says(tmp_path):
    from sim2chip.stages.export import Export
    ctx = ctx_for(tmp_path, task="count", hardware="generic", user_data={"raw": "photos"})
    for seed in (0, 1):
        d = ctx.job.stage_dir("train") / f"seed_{seed}"
        d.mkdir(parents=True)
        (d / "model.tflite").write_bytes(bytes([seed]) * 1024)
    seeds = {str(s): {"recipe": "picodet_s", "counter": "picodet_s", "input": 320, "val_f1": f1, "threshold": 0.4}
             for s, f1 in ((0, 0.6), (1, 0.8))}
    (ctx.job.stage_dir("train") / "summary.json").write_text(json.dumps(seeds))
    Export().run(ctx)
    report = json.loads((ctx.job.stage_dir("export") / "report.json").read_text())
    assert report["seed"] == 1 and report["metric"] == "f1" and report["val_score"] == 0.8
    assert report["model_path"] == "model.tflite" and report["quantization"] == "float16"
    assert report["input"] == [320, 320, 3] and report["output"]["kind"] == "boxes"
    assert report["output"]["model"] == "picodet_s"
    assert (ctx.job.stage_dir("export") / "model.tflite").read_bytes() == b"\x01" * 1024
    assert not (ctx.job.stage_dir("export") / "model_int8.tflite").exists()


def test_qa_caps_oversampled_search_classes(tmp_path):
    classes = [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                "assets": {"source": "search", "queries": ["cow"], "count": 2}},
               {"id": "dog", "size_m": {"measure": "length", "range": [0.5, 1]}, "assets": {"categories": ["dog"]}}]
    ctx = ctx_for(tmp_path, classes=classes)
    assets = [{"uid": u, "class": c, "usd": f"usd/{u}.usd", "dims": [2, 1, 1]}
              for u, c in [("a", "cow"), ("b", "cow"), ("c", "cow"), ("d", "cow"), ("e", "dog")]]
    (ctx.out("convert_assets") / "assets.json").write_text(json.dumps(assets))
    assert QAAssets().run(ctx)["kept"] == {"cow": 2, "dog": 1}
    kept = json.loads((ctx.job.stage_dir("qa_assets") / "assets.json").read_text())
    assert [a["uid"] for a in kept] == ["a", "b", "e"]  # search relevance order


def test_qa_props_drop_lookalikes(tmp_path, monkeypatch):
    classes = [{"id": "excavator", "size_m": {"measure": "length", "range": [6, 9]},
                "assets": {"source": "search", "queries": ["excavator"], "count": 5}}]
    props = [{"id": "cone", "queries": ["traffic cone"], "size_m": {"measure": "height", "range": [0.5, 0.9]},
              "count": 1}]
    ctx = ctx_for(tmp_path, classes=classes, props=props, qa={"mode": "auto"})
    assets = [{"uid": "t1", "class": "excavator", "category": "excavator", "usd": "u", "dims": [7, 3, 3]},
              {"uid": "t2", "class": "excavator", "category": "excavator", "usd": "u", "dims": [7, 3, 3]},
              {"uid": "p1", "class": "cone", "category": "traffic cone", "role": "prop", "usd": "u", "dims": [1, 1, 1]},
              {"uid": "p2", "class": "cone", "category": "traffic cone", "role": "prop", "usd": "u", "dims": [1, 1, 1]},
              {"uid": "p3", "class": "cone", "category": "traffic cone", "role": "prop", "usd": "u", "dims": [1, 1, 1]}]
    (ctx.out("convert_assets") / "assets.json").write_text(json.dumps(assets))
    verdicts = {"t1": {"decision": "keep", "class": "excavator"},
                "t2": {"decision": "keep", "class": "cone"},       # a target that looks like clutter
                "p1": {"decision": "keep", "class": "excavator"},  # a prop that looks like a target
                "p2": {"decision": "keep", "class": "cone"}, "p3": {"decision": "keep", "class": "cone"}}
    monkeypatch.setattr(QAAssets, "review", lambda self, ctx, assets: verdicts)
    assert QAAssets().run(ctx)["kept"] == {"excavator": 1, "cone": 1}  # p3 capped by count
    kept = json.loads((ctx.job.stage_dir("qa_assets") / "assets.json").read_text())
    assert [(a["uid"], a.get("role")) for a in kept] == [("t1", None), ("p2", "prop")]


def test_split_keeps_old_photos_and_halves_new_ones():
    from sim2chip.stages.ingest import split
    first = split([f"k{i}" for i in range(10)], set(), {}, test_cap=400, train_cap=3)
    assert list(first.values()).count("test") == 5 and list(first.values()).count("train") == 3  # 2 unused
    added = [f"n{i}" for i in range(6)]
    second = split([f"k{i}" for i in range(10)] + added, {"n0"}, first, test_cap=400, train_cap=3)
    assert all(second[k] == v for k, v in first.items())  # nothing old moves
    new = [second[k] for k in added]
    assert new.count("test") == 2 and new.count("train") == 4 and second["n0"] == "train"  # 5 usable // 2
    assert split(list(second), set(), second, test_cap=5, train_cap=3) == second  # stable when nothing changes
    capped = split([f"k{i}" for i in range(10)] + added, set(), first, test_cap=6, train_cap=3)
    assert [capped[k] for k in added].count("test") == 1  # room left under the test cap


def test_ingest_keeps_split_when_photos_are_added(tmp_path):
    photos = tmp_path / "photos"
    for i in range(8):
        img(photos / "cow" / f"{i}.jpg")
    ctx = ctx_for(tmp_path, user_data={"labeled": "photos", "test_per_class": 400, "train_per_class": 2})
    link_input(ctx.job, "photos", photos)
    Ingest().run(ctx)
    out = ctx.job.stage_dir("ingest")
    before = {s: sorted(f.name for f in (out / s / "cow").glob("*")) for s in ("test", "train")}
    for i in range(4):
        img(photos / "cow" / f"dry_{i}.jpg")
    info = Ingest().run(ctx)
    after = {s: sorted(f.name for f in (out / s / "cow").glob("*")) for s in ("test", "train")}
    assert set(before["test"]) <= set(after["test"]) and set(before["train"]) <= set(after["train"])
    assert info["counts"]["cow"] == {"test": 4 + 2, "train": 2 + 2}  # past the train cap of 2


def test_ingest_recovers_the_split_of_a_job_made_before_split_json(tmp_path):
    import os
    import time
    photos = tmp_path / "photos"
    for i in range(8):
        img(photos / "cow" / f"{i}.jpg")
    ctx = ctx_for(tmp_path, user_data={"labeled": "photos", "test_per_class": 400, "train_per_class": 2})
    link_input(ctx.job, "photos", photos)
    Ingest().run(ctx)
    out = ctx.job.stage_dir("ingest")
    (out / "split.json").unlink()  # as an older version left it
    ctx.job.mark_done("ingest", "old")
    ctx.job.clear_done("ingest")  # the pipeline does this before running the stage
    before = {s: sorted(f.name for f in (out / s / "cow").glob("*")) for s in ("test", "train")}
    later = time.time() + 60
    for i in range(4):
        img(photos / "cow" / f"dry_{i}.jpg")
        os.utime(photos / "cow" / f"dry_{i}.jpg", (later, later))
    info = Ingest().run(ctx)
    after = {s: sorted(f.name for f in (out / s / "cow").glob("*")) for s in ("test", "train")}
    assert set(before["test"]) <= set(after["test"]) and set(before["train"]) <= set(after["train"])
    assert info["counts"]["cow"] == {"test": 6, "train": 4}  # the 2 unused old photos stay out


def test_classes_without_synthetic_images_shift_the_weight_to_real_photos(tmp_path):
    from sim2chip.stages.train import real_repeats
    ctx = ctx_for(tmp_path)  # cow and dog
    assert real_repeats(ctx, 10, 2000, {"cow", "dog"}) == 50  # 1/4 of the synthetic set
    assert real_repeats(ctx, 10, 2000, {"cow"}) == 100  # only cows rendered: the real photos weigh 1/2
    assert real_repeats(ctx, 10, 0, set()) == 1
