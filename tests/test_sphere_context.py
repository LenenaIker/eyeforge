import numpy as np

from sim2chip.photo2scene.sphere import context


def test_the_ground_plane_puts_the_camera_above_it():
    rng = np.random.default_rng(0)
    xz = rng.uniform(-5, 5, (400, 2))
    pts = np.stack([xz[:, 0], np.full(400, 1.5), xz[:, 1] + 8], 1)  # OpenCV: +y down, ground 1.5 m under
    n, d = context.fit_ground(pts, rng)
    assert abs(d - 1.5) < 1e-6 and abs(n @ np.array([0, 1.5, 8.0]) + d) < 1e-6


def test_a_target_hanging_off_the_ground_counts_as_mounted():
    H, W = 60, 80
    v, u = np.mgrid[0:H, 0:W]
    # A camera 1.5 m up looking along +z at a flat ground, a box (rows 20-35) standing 0.6 m above it.
    z = np.where(v > 30, 1.5 * 40 / np.maximum(v - 30, 1), 30.0)
    pts = np.stack([(u - 40) / 40 * z, np.where(v > 30, 1.5, (v - 30) / 40 * z), z], -1).astype(np.float32)
    ground = v > 30
    bucket = (v >= 20) & (v <= 26) & (u >= 35) & (u <= 45)
    pts[bucket] = np.stack([(u[bucket] - 40) / 40 * 4, 1.5 - 0.6 - (26 - v[bucket]) * 0.05, np.full(bucket.sum(), 4.0)], -1)
    ground &= ~bucket
    stuff = {"sidewalk": ground, "structure": np.zeros((H, W), bool)}
    from PIL import Image
    out = context.measure_photo(Image.new("RGB", (W, H)), {"points": pts, "mask": np.ones((H, W), bool)}, stuff,
                                [{"class": "brown_bin", "box": [35 / W, 20 / H, 46 / W, 27 / H]}],
                                np.random.default_rng(0))
    assert out[0]["mounted"] and abs(out[0]["base_m"] - 0.6) < 0.05


def test_the_prior_counts_standing_targets_only_for_where_they_stand():
    m = {"a.jpg": [{"class": "x", "colour": [0.1, 0.2, 0.3], "mounted": False, "ground": "sidewalk", "height_m": 1.0,
                    "distance_m": 4.0, "wall_gap_m": 0.2},
                   {"class": "x", "colour": [0.1, 0.2, 0.3], "mounted": True, "ground": "road", "height_m": 0.4,
                    "distance_m": 3.0}]}
    p = context.prior(m)["x"]
    assert p["mounted"] == 0.5 and p["ground"] == {"sidewalk": 1.0} and p["heights"] == [1.0] and p["wall_share"] == 1.0
