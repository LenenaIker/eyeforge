"""Training helpers (student): augmentation by spec.training.orientation, last-layer-only head."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize("orientation", [None, "any", "fixed"])
def test_count_augment_keeps_every_point(orientation):
    """Turns and tilts move the points with the pixels; a one-pixel point near the centre is never lost."""
    from sim2chip.hardware import get
    from sim2chip import student as tu

    hw, rng = get("esp32cam"), np.random.default_rng(0)
    size = hw.input_size
    augment = tf.function(tu.make_count_augment(hw, 3, orientation))
    x = rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
    for _ in range(20):
        m = np.zeros((size, size), np.uint8)
        m[rng.integers(size // 3, 2 * size // 3), rng.integers(size // 3, 2 * size // 3)] = 2
        img, (y, d) = augment(x, m)
        assert img.shape == (size, size, 3)
        assert y.numpy()[..., 2].max() == 1 and y.numpy()[..., 1].max() == 0
        assert d.shape == (*y.shape[:2], 2) and d.numpy()[..., 0].sum() == 0
    for _ in range(20):
        # The density moves with the point and still holds one object (a tilt stretches it, the area scale undoes
        # that). Near the centre: a crop can push a point near the border, where part of it rightly falls out.
        m = np.zeros((size, size), np.uint8)
        m[size // 2 + rng.integers(-4, 5), size // 2 + rng.integers(-4, 5)] = 2
        _, (_, d) = augment(x, m)
        assert abs(d.numpy()[..., 1].sum() - 1) < 0.05


def test_augment_runs_in_a_dataset():
    from sim2chip.hardware import get
    from sim2chip import student as tu

    hw = get("esp32cam")
    x = np.zeros((4, hw.input_size, hw.input_size, 3), np.uint8)
    ds = tf.data.Dataset.from_tensor_slices((x, np.arange(4))).map(tu.make_augment(hw, "any")).batch(4)
    assert next(iter(ds))[0].shape == (4, hw.input_size, hw.input_size, 3)


def test_head_classifies_raw_features():
    """The head classifies raw features through its own standardization."""
    from sim2chip import student as tu

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    f = rng.normal(0, 1, (60, 16)) * 5 + 100
    f[np.arange(60), y] += 20  # one feature per class gives it away, on a large offset
    head = tu._fit_head(f.astype(np.float32), y, 3, seed=0)
    assert np.mean(tu._head_predict(f, head) == y) == 1.0
    folds = tu._folds(y, 5, 0)
    assert all(np.bincount(y[folds == i]).tolist() == [4, 4, 4] for i in range(5))


def test_count_targets_hold_one_object_per_point():
    """The density grid of a photo sums to its number of points, by class."""
    from sim2chip import student as tu

    m = np.zeros((96, 96, 1), np.float32)
    m[40, 30], m[50, 60], m[20, 70] = 1, 1, 2
    y, d = tu.count_target(tf.constant(m), 3)
    assert y.shape == (12, 12, 3) and d.shape == (12, 12, 2)
    assert np.allclose(d.numpy().sum((0, 1)), [2, 1], atol=0.01)


def test_count_model_has_points_and_density_outputs():
    """Two outputs in the int8 model, found by their channels whatever order the converter gives them."""
    from sim2chip.hardware import get
    from sim2chip import student as tu

    hw = get("esp32cam")
    model, _ = tu.build_count_model(3, hw)
    x = np.random.default_rng(0).integers(0, 255, (4, hw.input_size, hw.input_size, 3)).astype(np.uint8)
    grids, dens = tu.predict_int8_count(tu.to_int8(model, x), x[:2])
    g = hw.input_size // 8
    assert grids[0].shape == (g, g, 3) and dens[0].shape == (g, g, 2)
