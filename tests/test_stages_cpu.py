"""CPU stages on tiny fabricated data: ingest, qa_assets, crops and the C export."""
import json

import numpy as np
import pytest
from PIL import Image

from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.stages.crops import Crops
from sim2chip.stages.export import c_array
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
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


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
    # every class must keep at least one model
    ctx2 = ctx_for(tmp_path / "2")
    (ctx2.out("convert_assets") / "assets.json").write_text(json.dumps(assets[:1]))
    with pytest.raises(ValueError, match="dog"):
        QAAssets().run(ctx2)


def test_crops_from_fake_render(tmp_path):
    ctx = ctx_for(tmp_path)
    rd = ctx.out("render")
    dt = np.dtype([("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4"),
                   ("occlusionRatio", "<f4")])
    labels = []
    for f, cls in enumerate(["cow", "empty", "dog"]):
        img(rd / f"rgb_{f:04d}.png", size=(320, 240))
        np.save(rd / f"bounding_box_2d_tight_{f:04d}.npy", np.array([(0, 100, 80, 180, 160, 0.0)], dtype=dt))
        (rd / f"bounding_box_2d_tight_labels_{f:04d}.json").write_text(json.dumps({"0": {"class": cls}}))
        labels.append({"frame": f, "class": cls})
    (rd / "labels.json").write_text(json.dumps(labels))
    info = Crops().run(ctx)
    assert info["crops"] == {"cow": 1, "empty": 2, "dog": 1}


def test_c_array():
    h = c_array(bytes(range(20)), ["empty", "cow"], 96)
    assert "sim2chip_model[20]" in h and '"cow"' in h and "0x13" in h


def test_export_report_is_portable(tmp_path):
    from sim2chip.stages.export import Export
    ctx = ctx_for(tmp_path)
    seed = ctx.job.stage_dir("train") / "seed_0"
    seed.mkdir(parents=True)
    (seed / "model_int8.tflite").write_bytes(b"\x00" * 2048)
    (ctx.job.stage_dir("train") / "summary.json").write_text(json.dumps({"0": {"val_acc": 0.9}}))
    Export().run(ctx)
    report = json.loads((ctx.job.stage_dir("export") / "report.json").read_text())
    assert report["job"] == "t" and report["model_kb"] == 2.0
    assert (ctx.job.stage_dir("export") / report["model_path"]).read_bytes() == b"\x00" * 2048


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
