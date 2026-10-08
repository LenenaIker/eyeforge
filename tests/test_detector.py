"""Count detector helpers without PaddlePaddle, TensorFlow or a GPU: boxes from points, decoding, threshold."""
import numpy as np
from pytest import approx

from sim2chip import detector, hardware

LABELS = ["empty", "pot", "cat"]


def test_boxes_from_points_take_the_photo_size_then_the_job_size():
    pts = [[0.5, 0.5, "pot", 0.2, 0.1], [0.1, 0.9, "cat"]]  # the user's point has no size: the photo's median
    assert detector.boxes_of(pts, LABELS, (0.05, 0.05)) == [
        approx([0.4, 0.45, 0.6, 0.55, 0]), approx([0.0, 0.85, 0.2, 0.95, 1])]
    # A photo with no sized point: the job's median, clipped to the image.
    assert detector.boxes_of([[0.99, 0.5, "pot"]], LABELS, (0.1, 0.2)) == [approx([0.94, 0.4, 1.0, 0.6, 0])]


def test_box_side_is_the_median_of_sized_points():
    sides = [[[0, 0, "pot", 0.1, 0.2]], [[0, 0, "pot", 0.3, 0.4], [0, 0, "cat"]]]
    assert detector.box_side(sides) == approx((0.2, 0.3))
    assert detector.box_side([[[0, 0, "pot"]]]) == (detector.DEFAULT_BOX, detector.DEFAULT_BOX)


def test_decode_keeps_the_best_of_overlapping_boxes_per_class():
    boxes = np.array([[0.1, 0.1, 0.3, 0.3], [0.11, 0.1, 0.31, 0.3], [0.6, 0.6, 0.8, 0.8], [0.1, 0.1, 0.3, 0.3]],
                     np.float32)
    scores = np.array([[0.9, 0.8, 0.3, 0.0],    # pot: two boxes of one pot, one under the threshold
                       [0.0, 0.0, 0.0, 0.7]],   # cat: same place as the pot, another class, kept
                      np.float32)
    assert detector.decode((boxes, scores), LABELS, 0.5) == [(0.2, 0.2, "pot"), (0.2, 0.2, "cat")]
    assert len(detector.decode((boxes, scores), LABELS, 0.2)) == 3


def test_pick_threshold_prefers_the_usual_one_among_equals():
    assert detector.pick_threshold({0.3: 0.80, 0.4: 0.795, 0.5: 0.70}) == 0.4
    assert detector.pick_threshold({0.3: 0.80, 0.4: 0.70}) == 0.3


def test_profiles_count_with_a_detector_only_where_it_fits():
    assert hardware.get("generic").counter == "picodet_s" and hardware.get("generic").count_input() == 320
    esp = hardware.get("esp32cam")
    assert esp.counter == "grid" and esp.count_input() == esp.input_size


def test_export_and_evaluate_read_the_recipe_each_seed_recorded():
    from sim2chip.stages.train import is_detector
    assert is_detector({"recipe": "picodet_s", "counter": "picodet_s", "val_f1": 0.8})
    for grid in ({"recipe": "fine_tune", "val_f1": 0.6}, {"recipe": "head", "val_acc": 0.9}, {}):
        assert not is_detector(grid)
