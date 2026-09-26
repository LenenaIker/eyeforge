"""Target hardware profiles. The spec only names one; the profile fixes input size, model and limits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    input_size: int         # square RGB input, pixels
    backbone: str           # model family understood by the train stage
    alpha: float            # width multiplier of the backbone
    quantization: str       # "int8" for microcontrollers
    max_model_kb: int       # flash budget for the model blob
    runtime: str            # what runs the model on the device
    description: str = ""   # one line for the interview


PROFILES = {
    # AI-Thinker style ESP32-CAM: 4 MB flash, 4 MB PSRAM for the tensor arena, OV2640 sensor.
    "esp32cam": Hardware("esp32cam", 96, "mobilenetv2", 0.35, "int8", 1024, "tflite-micro",
                         "ESP32-CAM microcontroller: 96 px, 2 to 5 coarse classes"),
    "esp32s3": Hardware("esp32s3", 96, "mobilenetv2", 0.35, "int8", 2048, "tflite-micro",
                        "ESP32-S3 microcontroller: 96 px, 2 to 5 coarse classes"),
    "jetson_nano": Hardware("jetson_nano", 224, "mobilenetv2", 1.0, "int8", 16384, "tflite",
                            "NVIDIA Jetson Nano: 224 px, finer distinctions"),
    # No device, or not decided yet: a model to download and run on a phone or a PC.
    "generic": Hardware("generic", 224, "mobilenetv2", 1.0, "int8", 16384, "tflite",
                        "no specific device (phone, PC, testing): 224 px model to download"),
}


def get(name: str) -> Hardware:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown hardware '{name}', known: {sorted(PROFILES)}") from None
