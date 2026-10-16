import json

import numpy as np
from PIL import Image

from sim2chip.photo2scene.sphere import capture


def _pano(w=720):
    """An equirectangular test card: red channel = longitude, green = latitude (both 0-255)."""
    h = w // 2
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([x * 255 // (w - 1), y * 255 // (h - 1), np.full_like(x, 128)], -1).astype(np.uint8)


def _with_gpano(path, top, full=(720, 360)):
    """Write a GPano XMP block into a JPEG: a band of the full panorama starting `top` rows down."""
    w, h = Image.open(path).size
    xmp = (f'<x:xmpmeta><rdf:Description GPano:FullPanoWidthPixels="{full[0]}" '
           f'GPano:FullPanoHeightPixels="{full[1]}" GPano:CroppedAreaLeftPixels="0" '
           f'GPano:CroppedAreaTopPixels="{top}" GPano:CroppedAreaImageWidthPixels="{w}" '
           f'GPano:CroppedAreaImageHeightPixels="{h}" GPano:PoseHeadingDegrees="154.0"/></x:xmpmeta>')
    raw = path.read_bytes()
    path.write_bytes(raw[:2] + b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp.encode() + raw[2:])


def test_a_cut_looks_where_it_is_turned():
    pano = _pano()
    centre = lambda yaw, pitch: capture.sample(pano, capture.rays(9, 9, 60, yaw, pitch))[4, 4]  # noqa: E731
    assert abs(centre(0, 0)[0] - 127.5) < 2 and abs(centre(0, 0)[1] - 127.5) < 2  # straight ahead: the centre
    assert abs(centre(90, 0)[0] - 191) < 2  # turned right a quarter: three quarters across
    assert abs(centre(-90, 0)[0] - 64) < 2
    assert centre(0, 30)[1] > centre(0, 0)[1] + 35  # pitched down 30 degrees: a sixth of the height lower


def test_rays_land_on_their_own_pixels():
    """Pixel centres sent to rays and back land where they started, across the seam too."""
    W, H = 64, 32
    x, y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    px, py = capture.to_pixel(capture.pixel_rays(W, H), W, H)
    assert np.allclose(np.mod(px - x + W / 2, W) - W / 2, 0, atol=1e-6) and np.allclose(py, y, atol=1e-6)


def test_a_view_projects_its_own_rays_back():
    view = {"size": [40, 30], "fov_deg": 70.0, "yaw_deg": 250.0, "pitch_deg": 35.0}
    x, y, inside = capture.project(view, capture.view_rays(view))
    gx, gy = np.meshgrid(np.arange(40), np.arange(30))
    assert inside.all() and np.allclose(x, gx, atol=1e-6) and np.allclose(y, gy, atol=1e-6)
    _, _, behind = capture.project(view, -capture.view_rays(view))
    assert not behind.any()


def test_a_band_is_placed_in_its_canvas(tmp_path):
    p = tmp_path / "sphere.jpg"
    Image.fromarray(_pano()[60:300]).save(p, quality=95)
    _with_gpano(p, top=60)
    pano, seen = capture.read(p, width=360)  # half the size the metadata says
    assert pano.shape == (180, 360, 3) and seen[30:150].all() and not seen[:30].any() and not seen[150:].any()
    assert abs(int(pano[90, 180, 1]) - 127) < 4  # the band's middle row is the sphere's horizon
    ext = capture.captured_extent(seen)
    assert abs(ext["up_deg"] - 60) < 1 and abs(ext["down_deg"] - 60) < 1 and ext["around_share"] == 1


def test_a_full_turn_wraps_when_resized(tmp_path):
    """A bright line just right of the seam bleeds into the last column too, as it would anywhere else."""
    p = tmp_path / "sphere.png"
    pano = np.zeros((200, 400, 3), np.uint8)
    pano[:, :4] = 255
    Image.fromarray(pano).save(p)
    small, seen = capture.read(p, width=100)
    assert seen.all() and small[25, 0, 0] > 100 and small[25, -1, 0] > 0 and small[25, 50, 0] == 0


def test_not_a_photosphere_is_refused(tmp_path):
    p = tmp_path / "photo.jpg"
    Image.fromarray(np.zeros((300, 400, 3), np.uint8)).save(p)
    try:
        capture.read(p)
    except ValueError as e:
        assert "not a photosphere" in str(e)
    else:
        raise AssertionError("a 4:3 photo was taken as a sphere")


def test_prepare_cuts_only_what_was_captured(tmp_path, monkeypatch):
    """A band around the horizon: the level views are cut, the ones looking down are not."""
    p = tmp_path / "sphere.jpg"
    Image.fromarray(_pano()[120:240]).save(p, quality=95)
    _with_gpano(p, top=120)
    monkeypatch.setattr(capture, "WORK_WIDTH", 720)
    monkeypatch.setattr(capture, "CUT_SIZE", (64, 48))
    monkeypatch.setattr(capture, "CUT_PITCHES", (0.0, 75.0))
    info = capture.prepare(p, tmp_path / "place")
    assert [v["pitch_deg"] for v in info["cuts"]] == [0.0] * capture.CUT_HEADINGS
    assert info["heading_deg"] == 154.0
    assert all((tmp_path / "place" / "cuts" / v["file"]).exists() for v in info["cuts"])
    assert json.loads((tmp_path / "place" / "sphere.json").read_text()) == info
    assert capture.prepare(p, tmp_path / "place") == info  # done once
