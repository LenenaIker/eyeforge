import json
import math

import numpy as np

from sim2chip.photo2scene.sphere import capture, render_plan, replica

H_CAM = 1.6


def box_mask(W, H, centre_xz, yaw_deg, size):
    """Panorama pixels whose rays hit a box standing on the ground (slab test, identity levelling)."""
    rays = capture.pixel_rays(W, H)
    a = math.radians(yaw_deg)
    u, v = np.array([math.cos(a), 0, math.sin(a)]), np.array([-math.sin(a), 0, math.cos(a)])
    c = np.array([centre_xz[0], H_CAM - size[2] / 2, centre_xz[1]])
    t0, t1 = np.full(rays.shape[:2], -np.inf), np.full(rays.shape[:2], np.inf)
    for axis, half in ((u, size[0] / 2), (v, size[1] / 2), (np.array([0, 1.0, 0]), size[2] / 2)):
        d, o = rays @ axis, -(c @ axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (-half - o) / d, (half - o) / d
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        inside = np.abs(o) <= half
        lo = np.where(d == 0, np.where(inside, -np.inf, np.inf), lo)
        hi = np.where(d == 0, np.where(inside, np.inf, -np.inf), hi)
        t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
    return (t1 >= t0) & (t1 > 0)


def flat_proxy(W, H):
    rays = capture.pixel_rays(W, H)
    down = rays[..., 1] > 0.01
    dist = np.where(down, H_CAM / np.maximum(rays[..., 1], 1e-6), 50.0).astype(np.float32)
    return {"distance": dist, "valid": np.ones((H, W), bool), "ground": down}


def test_a_box_on_flat_ground_is_found_where_it_stands():
    W, H = 2048, 1024  # the masks' size; at half of it a shorter, turned box fits nearly as well
    mask = box_mask(W, H, (1.0, 8.0), 30.0, (4.2, 1.8, 1.5))
    f = replica.fit_box(mask, np.zeros_like(mask), np.eye(3), flat_proxy(W, H), (4.2, 1.8, 1.5))
    assert f["iou"] > 0.8
    assert np.hypot(f["centre_xz"][0] - 1.0, f["centre_xz"][1] - 8.0) < 0.4
    assert abs((f["yaw_deg"] - 30.0 + 90) % 180 - 90) < 8
    assert abs(f["ground_y"] - H_CAM) < 0.05


def test_the_models_heading_picks_the_front_end_of_the_box():
    W, H = 1024, 512
    mask = box_mask(W, H, (0.0, 9.0), 10.0, (4.2, 1.8, 1.5))
    f = replica.fit_box(mask, np.zeros_like(mask), np.eye(3), flat_proxy(W, H), (4.2, 1.8, 1.5), yaw_prior=185.0)
    assert abs((f["yaw_deg"] - 190.0 + 180) % 360 - 180) < 10


def test_front_yaw_faces_the_camera_at_azimuth_zero():
    # Looking straight ahead (heading 0, +z): a front facing the camera points along -z, world yaw 270.
    assert math.isclose(replica.front_yaw(0, 0), 270)
    # Seen from the side, front to the right (+x): world yaw 0.
    assert math.isclose(replica.front_yaw(270, 0) % 360, 0)


def test_an_asset_drawn_backwards_is_turned_round(tmp_path):
    d = tmp_path
    (d / "replica").mkdir()
    fits = [{"id": 1, "kept": True, "front": True, "uid": "a", "yaw_deg": 0.0},
            {"id": 2, "kept": True, "front": True, "uid": "a", "yaw_deg": 90.0},
            {"id": 3, "kept": True, "front": True, "uid": "b", "yaw_deg": 0.0}]
    (d / "replica" / "fits.json").write_text(json.dumps(fits))
    seen = {"o01": {"yaw_deg": 178.0, "front": True}, "o02": {"yaw_deg": 265.0, "front": True},
            "o03": {"yaw_deg": 5.0, "front": True}}
    assert replica.asset_flips(d, seen) == {"a": 180.0}


def test_match_objects_blurs_and_adds_noise_only_around_objects():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, 20:] = 200
    obj = np.zeros((40, 40), bool)
    obj[10:30, 22:30] = True
    out = replica.match_objects(img, obj, 1.0, 3.0, np.random.default_rng(0))
    assert (out[:, :5] == 0).all()  # far from the object nothing changes
    assert out[15, 21].mean() not in (0, 200)  # the edge next to it is softened


def test_overcast_sky_is_three_times_brighter_at_the_zenith(tmp_path):
    p = tmp_path / "sky.hdr"
    render_plan.overcast_sky(p, [1.0, 1.0, 1.0], size=(8, 90))
    raw = p.read_bytes()
    rgbe = np.frombuffer(raw[raw.index(b"+X 8\n") + 5:], np.uint8).reshape(90, 8, 4).astype(float)
    lum = rgbe[:, 0, 0] * 2.0 ** (rgbe[:, 0, 3] - 136)
    assert math.isclose(lum[0] / lum[44], 3.0, rel_tol=0.05)
    assert lum[-1] < lum[44] / 3
