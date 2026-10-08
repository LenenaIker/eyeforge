"""Target hardware profiles. The spec only names one; the profile fixes input size, model and limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Hardware:
    """A device profile. Flyweight: immutable and shared by every job (PROFILES) (patterns: docs/architecture.md)."""
    name: str
    input_size: int         # square RGB input, pixels
    backbone: str           # classification model family understood by the train stage (counting: see counter)
    alpha: float            # width multiplier of MobileNetV2
    quantization: str       # "int8" for microcontrollers
    max_model_kb: int       # flash budget for the model blob
    runtime: str            # what runs the model on the device
    description: str = ""   # one line for the interview
    # How sharp the camera's image is, as a fraction of the input size: training blurs every image to a
    # random resolution in this range (a cheap OV2640 at 96 px resolves about 20 to 64 px of detail).
    sharpness: tuple[float, float] = (0.45, 1.0)
    # What counts: "grid", our point and density head on MobileNetV2 (the only one that fits a microcontroller),
    # or "picodet_s", a pretrained detector (detector.py) at counter_size px, for devices with room for it.
    counter: Literal["grid", "picodet_s"] = "grid"
    counter_size: int = 0

    def count_input(self) -> int:
        """Input side of the count model: the detector's own, else the profile's."""
        return self.counter_size if self.counter != "grid" else self.input_size


PROFILES = {
    # AI-Thinker style ESP32-CAM: 4 MB flash, 4 MB PSRAM for the tensor arena, OV2640 sensor.
    "esp32cam": Hardware("esp32cam", 96, "mobilenetv2", 0.35, "int8", 1024, "tflite-micro",
                         "ESP32-CAM microcontroller: 96 px, 2 to 5 coarse classes", (0.2, 0.67)),
    "esp32s3": Hardware("esp32s3", 96, "mobilenetv2", 0.35, "int8", 2048, "tflite-micro",
                        "ESP32-S3 microcontroller: 96 px, 2 to 5 coarse classes", (0.2, 0.67)),
    "jetson_nano": Hardware("jetson_nano", 224, "mobilenetv2", 1.0, "int8", 16384, "tflite",
                            "NVIDIA Jetson Nano: 224 px, finer distinctions"),
    # No device, or not decided yet: a model to download and run on a phone or a PC. EfficientNetV2-B0 classifies
    # better than MobileNetV2 1.0 with the few-photos recipe (lawn 84.6 vs 75.4%, MIO-TCD 73.0 vs 69.7%, int8) and
    # takes 64 ms a photo in a Pixel 7a's browser (docs/hardware_makers.md, 2026-10-02). Counting uses PicoDet-S
    # at 320 px: on FSC-147 points F1 0.83-0.91 against 0.41-0.80 for our grid head (same doc, 2026-10-02).
    "generic": Hardware("generic", 224, "efficientnetv2b0", 1.0, "int8", 16384, "tflite",
                        "no specific device (phone, PC, testing): 224 px model to download",
                        counter="picodet_s", counter_size=320),
}


def get(name: str) -> Hardware:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown hardware '{name}', known: {sorted(PROFILES)}") from None
