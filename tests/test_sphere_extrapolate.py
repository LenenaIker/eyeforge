import numpy as np

from sim2chip.photo2scene.sphere import extrapolate


def test_footprints_overlap_only_when_they_come_closer_than_the_gap():
    car = (4.0, 1.8)
    assert extrapolate.overlaps((0, 0), 0, car, (4.2, 0), 0, car)            # end to end, 0.2 m apart
    assert not extrapolate.overlaps((0, 0), 0, car, (5.0, 0), 0, car)        # 1 m apart
    assert not extrapolate.overlaps((0, 0), 90, car, (0, 3.5), 0, car)      # a T, 0.6 m apart
    assert extrapolate.overlaps((0, 0), 90, car, (0, 2.1), 90, car)         # side by side across x


def test_the_street_runs_along_the_nearest_wall():
    walls = [{"ends_xz": [[5.0, -10.0], [5.0, 10.0]]}, {"ends_xz": [[-30.0, 0.0], [-30.0, 5.0]]}]
    yaw = extrapolate.wall_axis(walls, np.array([2.0, 0.0]))
    assert abs(abs(yaw) - 90) < 1e-6
    assert extrapolate.wall_axis(walls, np.array([-12.0, 30.0]), max_m=5) is None


def test_an_asset_takes_a_width_from_its_class_range():
    rng = np.random.default_rng(0)
    L, W, H = extrapolate.asset_size({"dims": [400.0, 180.0, 150.0], "width_range_m": (1.7, 1.9)}, rng)
    assert 1.7 <= W <= 1.9 and abs(L / W - 400 / 180) < 1e-9 and abs(H / W - 150 / 180) < 1e-9


def test_labels_keep_the_visible_targets_only():
    plan = {"views": [{"name": "x000", "objects": [{"label": "car", "uid": "a"}, {"label": "car", "uid": "b"},
                                                   {"label": "car", "uid": "c"}]}]}
    report = {"views": [{"name": "x000", "objects": [{"pixels": 900, "box": [1, 2, 30, 40]},
                                                     {"pixels": 20, "box": [5, 5, 8, 8]}, None]}]}
    out = extrapolate.labels(report, plan)
    assert [b["uid"] for b in out["x000"]] == ["a"]


def test_views_stay_inside_the_captured_band():
    lo, hi = extrapolate.pitch_range({"up_deg": 30.0, "down_deg": 30.0}, 68.0, (1024, 768))
    vfov = 2 * np.degrees(np.arctan(np.tan(np.radians(34)) * 0.75))
    assert abs(hi - (30 - vfov / 2 - 1)) < 1e-6 and abs(lo + hi) < 1e-6


class FlatGround:
    """A flat ground 1.6 m under the camera, seen everywhere (render_plan.Ground's interface)."""
    h = 1.6

    def look(self, p):
        p = np.asarray(p, float)
        return np.ones(len(p), bool), np.linalg.norm(p, axis=1)


def test_a_target_against_a_wall_stands_in_front_of_it_facing_the_camera_side():
    walls = [{"ends_xz": [[-10.0, 6.0], [10.0, 6.0]], "normal": [0.0, 0.0, 1.0]}]  # a wall 6 m ahead
    rng = np.random.default_rng(0)
    spots = [extrapolate.wall_spot(FlatGround(), walls, 0.6, 0.7, 0.15, rng, 0.0, 70.0, (2.0, 12.0))
             for _ in range(50)]
    spots = [s for s in spots if s is not None]
    assert spots
    for p, front in spots:
        assert abs(p[2] - (6.0 - 0.38)) < 1e-6 and abs(p[1] - 1.6) < 1e-6  # half its depth plus 3 cm off the wall
        assert abs((front + 90) % 360) < 1e-6 or abs(front - 270) < 1e-6  # facing -z, away from the wall


def test_a_small_bin_takes_a_small_bins_shape():
    pool = [{"uid": "dumpster", "height_m": 1.4}, {"uid": "bucket", "height_m": 0.45}]
    rng = np.random.default_rng(0)
    assert extrapolate.pick_sized(pool, 0.4, rng, among=2)["uid"] == "bucket"
    L, W, H = extrapolate.sized({"dims": [60.0, 50.0, 100.0]}, 1.0)
    assert (L, W, H) == (0.6, 0.5, 1.0)


def test_a_hung_rig_goes_on_a_wall_with_its_bins_at_their_heights_in_front_of_the_holder():
    walls = [{"ends_xz": [[-10.0, 6.0], [10.0, 6.0]], "normal": [0.0, 0.0, 1.0]}]
    rig = {"standing": False,
           "holders": [{"x": 0.0, "y": 0.0, "w": 0.1, "d": 0.05, "bottom": 0.9, "top": 2.0, "colour": [0.6, 0.6, 0.6]}],
           "targets": [{"class": "brown_bin", "x": 0.0, "y": 0.1, "base_m": 1.2, "height_m": 0.4}]}
    pool = [{"uid": "bucket", "usd": "b.usd", "dims": [0.4, 0.35, 0.4], "height_m": 0.4, "front_deg": 0.0}]
    prior = {"brown_bin": {"distance_m": [2.0, 10.0], "colour_srgb": [0.3, 0.25, 0.2]}}
    rng = np.random.default_rng(0)
    out = extrapolate.place_rig(FlatGround(), None, walls, pool, prior, [rig], [], rng, 0.0, 70.0, 1.6)
    assert out is not None
    holder, bin_ = out
    assert holder["label"] is None and holder["box_m"] == [0.1, 0.05, 1.1] and abs(holder["pos"][2] - 0.9) < 1e-6
    assert bin_["label"] == "brown_bin" and abs(bin_["pos"][2] - 1.2) < 1e-6
    assert 6.0 - 0.06 < holder["pos"][1] < 6.0 - 0.025  # its back on the wall (wall_spot keeps 3 cm)
    assert bin_["pos"][1] < holder["pos"][1]  # hung on the holder's front, on the camera's side
    assert abs((bin_["render_yaw"] - 270) % 360) < 1e-6  # facing away from the wall
