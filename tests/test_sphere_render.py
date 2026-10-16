import math
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from sim2chip.photo2scene.sphere import camera_fx, capture, render_plan, sun


def test_world_frame_puts_the_ground_at_zero_and_forward_on_y():
    p = render_plan.to_world(np.array([1.0, 1.6, 5.0]), 1.6)  # 1 m right, on the ground, 5 m ahead
    assert np.allclose(p, [1.0, 5.0, 0.0])


def test_camera_pose_looks_along_its_heading_and_down_its_pitch():
    pose = np.array(render_plan.camera_pose({"yaw_deg": 90.0, "pitch_deg": 20.0}, 1.5))
    forward = pose[:3, 2]  # OpenCV camera +z in the world
    assert np.allclose(pose[:3, 3], [0, 0, 1.5])
    assert forward[0] > 0.9 and abs(forward[1]) < 1e-9 and math.isclose(forward[2], -math.sin(math.radians(20)))


def test_a_level_view_on_a_tilted_sphere_samples_the_level_horizon():
    """With a levelling R, the background cut must follow the levelled rays taken back to the panorama."""
    R = np.array([[1, 0, 0], [0, math.cos(0.1), -math.sin(0.1)], [0, math.sin(0.1), math.cos(0.1)]])
    d_level = np.array([[0.0, 0.0, 1.0]])
    d_pano = render_plan.pano_rays(d_level, R)
    assert np.allclose(d_pano @ R.T, d_level)  # back through R lands on the levelled ray


def test_proxy_albedo_divides_out_the_light_and_fills_the_unseen_ground():
    W, H = 64, 32
    pano = np.full((H, W, 3), [200, 160, 120], np.uint8)
    seen = np.zeros((H, W), bool)
    seen[: H * 3 // 4] = True
    v = np.array([[0.0, 1.0, 3.0], [0.0, 3.0, 0.2]])  # one seen ground point ahead, one under the camera
    col = render_plan.proxy_colors(v, pano, seen, np.eye(3), np.array([True, True]), [1.0, 0.8, 0.6])
    assert np.allclose(col[0], col[1], atol=1e-3)  # the unseen one takes the seen ground's colour
    assert col.max() <= render_plan.ALBEDO_MAX + 1e-6
    ratio = render_plan.srgb_to_linear(np.array([200, 160, 120])) / np.array([1.0, 0.8, 0.6])
    assert np.allclose(col[0] / col[0].max(), ratio / ratio.max(), atol=1e-3)


def test_the_sun_is_high_at_noon_in_summer_and_down_at_night():
    madrid = (40.42, -3.70)
    elev, az = sun.solar_position(datetime(2024, 6, 21, 14, 15, tzinfo=timezone(timedelta(hours=2))), *madrid)
    assert 72 < elev < 75 and 170 < az < 190
    night, _ = sun.solar_position(datetime(2024, 6, 21, 1, 0, tzinfo=timezone(timedelta(hours=2))), *madrid)
    assert night < 0


def test_the_sun_lands_in_the_panorama_by_the_compass_heading():
    info = {"taken_at": "2024-06-21T14:15:00+02:00", "lat": 40.42, "lon": -3.70, "heading_deg": 150.0}
    s = sun.sun_for(info)
    assert s["up"] and abs(s["yaw_deg"] - (s["azimuth_deg"] - 150.0)) < 1e-6
    travel = np.array(s["travel_level"])
    assert travel[1] > 0.9  # high sun: its light travels down (+y)
    assert sun.sun_for({"lat": 1.0}) is None


def test_the_esp32_camera_gives_a_qvga_rgb565_frame():
    img = Image.fromarray(np.full((480, 640, 3), 128, np.uint8))
    out = np.asarray(camera_fx.apply(img, "esp32cam", np.random.default_rng(0)))
    assert out.shape == (240, 320, 3)
    red = out[..., 0][out[..., 0] < 255]
    assert (red % 8 == 4).all()  # 5-bit red: the centres of its 32 levels
    assert camera_fx.sensor_for("unknown").jpeg is not None  # falls back to the generic camera


def test_resolve_asset_leaves_a_missing_file_alone(tmp_path):
    assert render_plan.resolve_asset(str(tmp_path / "none.usd"), tmp_path / "out") == str(tmp_path / "none.usd")


def test_view_rays_and_level_rays_agree():
    view = {"yaw_deg": 30.0, "pitch_deg": 5.0, "fov_deg": 66.0, "size": [8, 6]}
    assert np.allclose(render_plan.level_rays(view), capture.view_rays(view))
