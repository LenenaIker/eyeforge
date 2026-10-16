import json
import math

import numpy as np

from sim2chip.photo2scene.sphere import capture, geometry


def _tilted_ground(height=1.7, tilt_deg=2.0, n=5000, seed=0):
    """Points of a ground plane `height` under the origin, tilted about the x axis, with a little noise."""
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(-15, 15, n), rng.uniform(2, 20, n)
    t = math.radians(tilt_deg)
    y = height + z * math.tan(t)
    p = np.stack([x, y, z], 1)
    return p + rng.normal(0, 0.01, p.shape)


def test_the_ground_gives_height_and_tilt():
    p = _tilted_ground()
    clutter = np.random.default_rng(1).uniform([-10, -5, 2], [10, 1, 20], (1500, 3))  # walls, cars: above it
    g = geometry.fit_ground(np.concatenate([p, clutter]))
    assert abs(g["tilt_deg"] - 2.0) < 0.2
    assert abs(g["height_m"] - 1.7 * math.cos(math.radians(2))) < 0.03
    assert g["normal"][1] < 0 and g["inliers"] > 0.7


def test_levelling_turns_the_ground_flat():
    n = np.array([0.03, -0.99, 0.05])
    n /= np.linalg.norm(n)
    R = geometry.levelling(n)
    assert np.allclose(R @ n, geometry.UP, atol=1e-9) and np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_faces_skip_depth_jumps_and_wrap_the_seam():
    keep = np.ones((3, 4), bool)
    dist = np.full((3, 4), 5.0)
    dist[:, 2] = 9.0  # a jump between columns 1 and 2, and between 2 and 3
    faces = geometry.grid_faces(keep, dist)
    quads = {tuple(sorted(f)) for f in faces}
    cols = {frozenset(v % 4 for v in f) for f in quads}
    assert cols == {frozenset({0, 1}), frozenset({3, 0})}  # only the quads without the jump, one across the seam
    assert len(faces) == 2 * 2 * 2


def test_solve_scales_and_fills_under_the_capture(tmp_path):
    """Ground 2 m under the camera in MoGe's scale, a shell 8 m away all around, taken at 1.5 m; nothing captured
    more than 40 degrees down, which the plane fills."""
    from PIL import Image
    W, H = 256, 128
    dirs = capture.pixel_rays(W, H)
    down = dirs[..., 1] > 0.05
    dist = np.minimum(np.where(down, 2.0 / np.maximum(dirs[..., 1], 1e-6), 8.0), 8.0).astype(np.float32)
    ground = down & (dist < 8.0)
    seen = dirs[..., 1] < np.sin(np.radians(40))
    Image.fromarray(seen.astype(np.uint8) * 255).save(tmp_path / "seen.png")
    np.savez(tmp_path / "depth.npz", distance=dist, valid=seen)
    np.savez(tmp_path / "stuff.npz", ground=ground.astype(np.float16), sky=np.zeros((H, W), np.float16))
    out = geometry.solve(tmp_path, 1.5)
    assert abs(out["scale"] - 0.75) < 0.01 and out["ground"]["tilt_deg"] < 0.5
    proxy = np.load(tmp_path / "proxy.npz")
    v, g = proxy["vertices"], proxy["ground"]
    assert np.allclose(v[g, 1], 1.5, atol=1e-3)  # the ground on the plane, at the capture height
    under = v[g][np.hypot(v[g, 0], v[g, 2]) < 1.5 / np.tan(np.radians(41))]
    assert len(under) > 50  # filled under the capture's lower edge, down to the camera's feet
    assert np.allclose(np.linalg.norm(v[~g], axis=1), 6.0, atol=0.05)  # the far shell, 8 m scaled
    assert json.loads((tmp_path / "geometry.json").read_text())["proxy"]["faces"] == len(proxy["faces"])
    own = geometry.solve(tmp_path)  # no height given: MoGe's measure stands
    assert own["scale"] == 1 and abs(own["camera_height_m"] - 2.0) < 0.02 and own["height_from"] == "moge"
    close = geometry.solve(tmp_path, 2.1)  # within 10 % of MoGe's 2 m: MoGe's stands
    assert close["scale"] == 1 and close["height_from"] == "moge" and close["height_given_m"] == 2.1


def test_the_proxy_can_come_from_the_erased_depth_at_the_first_scale(tmp_path):
    """After S.3 the proxy is rebuilt from the erased panorama's depth, brought to the first depth's scale."""
    from PIL import Image
    W, H = 128, 64
    dirs = capture.pixel_rays(W, H)
    down = dirs[..., 1] > 0.05
    dist = np.minimum(np.where(down, 2.0 / np.maximum(dirs[..., 1], 1e-6), 8.0), 8.0).astype(np.float32)
    Image.fromarray(np.full((H, W), 255, np.uint8)).save(tmp_path / "seen.png")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(tmp_path / "erased.png")
    np.savez(tmp_path / "depth.npz", distance=dist, valid=np.ones((H, W), bool))
    np.savez(tmp_path / "depth_bg.npz", distance=dist / 3, valid=np.ones((H, W), bool))  # another scale
    np.savez(tmp_path / "stuff.npz", ground=(down & (dist < 8)).astype(np.float16))
    first = geometry.solve(tmp_path)
    v1 = np.load(tmp_path / "proxy.npz")["vertices"]
    again = geometry.solve(tmp_path, proxy_depth="depth_bg.npz")
    assert again["proxy_from"] == "depth_bg.npz" and again["scale"] == first["scale"]
    assert np.allclose(np.load(tmp_path / "proxy.npz")["vertices"], v1, atol=1e-4)


def test_a_bowed_wall_comes_out_straight_and_on_its_rays():
    """A wall 6 m ahead, bowed 15 cm in the middle as MoGe's merge leaves it: found and moved onto a plane, each
    point along its own ray."""
    W, H = 512, 256
    rays = capture.pixel_rays(W, H)
    ahead = (rays[..., 2] > 0.6) & (np.abs(rays[..., 1]) < 0.4)
    x = rays[..., 0] / np.maximum(rays[..., 2], 1e-6) * 6.0  # where each ray meets the plane z = 6
    bow = 0.15 * np.clip(1 - (x / 4) ** 2, 0, 1)  # the middle 15 cm farther, the ends on the plane
    dist = np.where(ahead, (6.0 + bow) / np.maximum(rays[..., 2], 1e-6), 0.0)
    P = rays * dist[..., None]
    normals = geometry.grid_normals(P)
    walls = geometry.fit_walls(P, normals, ahead & (np.abs(normals[..., 1]) < geometry.WALL_MAX_UP))
    assert len(walls) == 1 and abs(abs(walls[0]["normal"][2]) - 1) < 0.01
    out, moved = geometry.straighten(dist, rays, walls)
    z = (rays * out[..., None])[..., 2][moved]
    assert moved.sum() > 0.9 * ahead.sum() and np.ptp(z) < 0.02  # straight, within 2 cm
    assert abs(np.median(z) - 6.07) < 0.06
