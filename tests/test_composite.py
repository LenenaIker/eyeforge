"""The arithmetic of the walk scene render (sim2chip/walkscene/composite.py), without Isaac: one prim per object
of a model, an object behind the splat hides, the shadow darkens only the ground, small objects lose their box."""
import numpy as np

from sim2chip.walkscene.composite import boxes, composite, copies, object_index, slots


def test_two_objects_of_one_model_take_two_prims():
    views = [{"objects": [{"key": "bin"}, {"key": "bin"}, {"key": "bench"}]},
             {"objects": [{"key": "bin"}, {"key": "bench"}, {"key": "bench"}, {"key": "bench"}]}]
    assert slots(views[0]) == [("bin", 0), ("bin", 1), ("bench", 0)]
    assert dict(copies(views)) == {"bin": 2, "bench": 3}


def test_the_index_follows_each_prim_and_its_children():
    ids = np.array([[0, 1], [2, 3]])
    labels = {"1": "/World/Objects/a0/Ref/mesh", "2": "/World/Objects/a1", "3": "/World/Objects/a10"}
    index = object_index(ids, labels, {"/World/Objects/a0": 1, "/World/Objects/a1": 2})
    assert index.tolist() == [[0, 1], [2, 0]]  # a10 is not a child of a1


def scene(h=20, w=20):
    """A flat ground 5 m away in both passes and in the splat, lit grey; pass A has an object on the left half."""
    b = np.full((h, w, 3), 100, np.float32)
    db = np.full((h, w), 5.0, np.float32)
    bg, sd = b * 1.5, db.copy()
    a, da = b.copy(), db.copy()
    a[:, :10], da[:, :10] = 200, 4.0
    index = np.zeros((h, w), np.uint8)
    index[:, :10] = 1
    return a, da, b, db, bg, sd, index


def test_the_object_shows_in_front_and_takes_the_background_gain():
    a, da, b, db, bg, sd, index = scene()
    out, obj, index, lit, gain = composite(a, da, b, db, bg, sd, index, np.ones(3))
    assert obj[:, :10].all() and not obj[:, 10:].any()
    assert np.allclose(gain, 1.5) and np.allclose(out[:, :10], 300)  # A at the gain that maps B to the splat
    assert np.allclose(out[:, 16:], bg[:, 16:])  # ground past the blur's reach (6 px) from the object: no shadow


def test_an_object_behind_the_splat_hides_and_loses_its_box():
    a, da, b, db, bg, sd, index = scene()
    sd[:, :10] = 2.0  # a hedge of the scene, nearer than the object
    out, obj, index, lit, gain = composite(a, da, b, db, bg, sd, index, np.full(3, 0.7))
    assert not obj.any() and boxes(index, 1, 5) == [None]
    assert np.allclose(gain, 1.5)  # the ground right of the hedge is still lit


def test_too_little_lit_ground_keeps_the_last_gain():
    a, da, b, db, bg, sd, index = scene(8, 8)  # 64 pixels, under the lit minimum
    *_, gain = composite(a, da, b, db, bg, sd, index, np.full(3, 0.7))
    assert np.allclose(gain, 0.7)


def test_boxes_are_tight_and_skip_small_objects():
    index = np.zeros((10, 10), np.uint8)
    index[2:5, 3:7] = 1
    index[8, 8] = 2
    assert boxes(index, 2, 4) == [{"pixels": 12, "box": [3, 2, 7, 5]}, None]
