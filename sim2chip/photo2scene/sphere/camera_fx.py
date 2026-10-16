"""The user's camera on a rendered frame: the composite (a phone's clean, stitched panorama with objects on it) made
to look like what the target hardware's sensor gives. Low-level camera realism is what moved Alhaija et al. (2018)
most, 43.8 to 49.7 AP50; Carlson et al. (2018) vary chromatic aberration, blur, exposure, noise and colour cast
(docs/conocimiento_photosphere.md, camera retouching). Applied to the whole image, background and objects alike, so
they share one camera. Each effect is drawn from its profile's range per frame. Numpy and PIL.

Profiles by the spec's hardware (hardware.py names):
    esp32cam, esp32s3  OV2640 (66 degree lens) read as RGB565 at QVGA (firmware/esp32cam/src/main.cpp
                       initCamera; lora_alarm reads QQVGA): soft optics, visible noise, colour cast, 5-6-5 bit
                       colour, no JPEG
    generic            a phone or a PC camera (a phone's main camera, about 68 degrees): light noise and
                       blur, JPEG

The noise, cast and blur ranges are a first guess from the sensors' reputation, to be measured on the user's own
photos (S.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Sensor:
    fov_deg: float                         # horizontal field of view of its lens
    size: tuple[int, int] | None           # output (width, height); None keeps the frame's
    blur_px: tuple[float, float]           # Gaussian sigma at the output size
    aberration_px: tuple[float, float]     # red and blue shifted apart, at the output size (lateral, radial)
    noise: tuple[float, float]             # read noise sigma, 0-255; shot noise scales with the square root
    cast: float                            # per-channel gain drawn in 1 +- cast
    exposure: tuple[float, float]          # overall gain
    rgb565: bool                           # colour quantized to 5-6-5 bits
    jpeg: tuple[int, int] | None           # quality range, None for raw frames


SENSORS = {
    "esp32cam": Sensor(66.0, (320, 240), (0.5, 1.3), (0.3, 1.0), (2.0, 7.0), 0.08, (0.8, 1.15), True, None),
    "generic": Sensor(68.0, None, (0.2, 0.7), (0.0, 0.4), (0.5, 2.5), 0.04, (0.9, 1.1), False, (80, 95)),
}
SENSORS["esp32s3"] = SENSORS["esp32cam"]


def sensor_for(hardware: str) -> Sensor:
    return SENSORS.get(hardware, SENSORS["generic"])


def _radial_shift(channel: np.ndarray, px: float) -> np.ndarray:
    """A channel scaled about the centre so its edges move by px (lateral chromatic aberration)."""
    h, w = channel.shape
    s = 1 + 2 * px / max(w, h)
    big = Image.fromarray(channel).resize((round(w * s), round(h * s)), Image.Resampling.BILINEAR)
    x0, y0 = (big.width - w) // 2, (big.height - h) // 2
    return np.asarray(big.crop((x0, y0, x0 + w, y0 + h)))


def apply(image: Image.Image, hardware: str, rng: np.random.Generator) -> Image.Image:
    sensor = sensor_for(hardware)
    img = image.convert("RGB")
    if sensor.size:
        img = img.resize(sensor.size, Image.Resampling.LANCZOS)
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    a = np.asarray(img, np.float32)
    ab = u(sensor.aberration_px)
    if ab > 0.05:
        a = np.stack([_radial_shift(a[..., 0], ab), a[..., 1], _radial_shift(a[..., 2], -ab)], -1)
    a = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(u(sensor.blur_px))), np.float32)
    a *= u(sensor.exposure) * (1 + rng.uniform(-sensor.cast, sensor.cast, 3))
    read = u(sensor.noise)
    a += rng.normal(0, 1, a.shape) * np.sqrt(read ** 2 + read * np.clip(a, 0, 255) / 16)
    a = np.clip(a, 0, 255)
    if sensor.rgb565:
        a = np.floor(a / [8, 4, 8]) * [8, 4, 8] + [4, 2, 4]
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if sensor.jpeg:
        import io
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=int(rng.integers(sensor.jpeg[0], sensor.jpeg[1] + 1)))
        out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return out
