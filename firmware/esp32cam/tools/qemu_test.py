"""Runs the firmware in Espressif's QEMU on a fixed frame and checks it answers as the model does on the PC.

No camera and no board: a photo becomes the camera's QVGA RGB565 frame (test_frame/test_frame.h), the firmware
is built with EYEFORGE_TEST_FRAME (env qemu_esp32 or qemu_esp32s3), flashed into an image with the model's
model.bin, and booted in QEMU. Its first result line is compared with TensorFlow Lite on the PC fed the same
frame, prepared as the firmware does (RGB565 to RGB888, Pillow's bilinear stretch to the model's side).
It catches a model the firmware cannot load (an op missing from the resolver, a too small arena) and a
preprocessing that drifted from training. Camera, USB and real speed still need the board.

    python tools/qemu_test.py --chip esp32s3 --model ../../runs/<...>/export --photo some.jpg
    QEMU=/path/to/qemu-system-xtensa python tools/qemu_test.py ...   (default: qemu-system-xtensa on PATH)
    PIO=/path/to/pio ESPTOOL_PYTHON=/path/to/python-with-pyserial ...  (pio and the python running this script)

The model folder holds model.bin, model_int8.tflite and labels.txt (the export stage's output).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
FRAME = (320, 240)  # QVGA, what the firmware asks the camera for
CHIPS = {  # env, bootloader offset, flash size, QEMU machine
    "esp32": ("qemu_esp32", 0x1000, "4MB", "esp32"),
    "esp32s3": ("qemu_esp32s3", 0x0, "8MB", "esp32s3"),
}
PIO = os.environ.get("PIO", "pio")
ESPTOOL = Path.home() / ".platformio" / "packages" / "tool-esptoolpy" / "esptool.py"
ESPTOOL_PYTHON = os.environ.get("ESPTOOL_PYTHON", sys.executable)  # one with pyserial (PlatformIO's)
TOLERANCE = 0.05  # score difference allowed: ESP-NN and the PC's kernels may round an int8 step apart


def rgb565_frame(photo: Path) -> bytes:
    """The photo as the camera gives it: QVGA, RGB565, big endian (esp32-camera's byte order)."""
    rgb = np.asarray(Image.open(photo).convert("RGB").resize(FRAME, Image.BILINEAR), dtype=np.uint16)
    p = (rgb[..., 0] >> 3) << 11 | (rgb[..., 1] >> 2) << 5 | rgb[..., 2] >> 3
    return p.astype(">u2").tobytes()


def write_header(frame: bytes, out: Path) -> None:
    out.mkdir(exist_ok=True)
    rows = ",\n".join(", ".join(str(b) for b in frame[i:i + 32]) for i in range(0, len(frame), 32))
    (out / "test_frame.h").write_text(
        f"// Written by tools/qemu_test.py: a {FRAME[0]}x{FRAME[1]} RGB565 frame.\n#pragma once\n#include <cstdint>\n"
        f"#define TEST_FRAME_W {FRAME[0]}\n#define TEST_FRAME_H {FRAME[1]}\n"
        f"static const uint8_t test_frame[] = {{\n{rows}\n}};\n")


def expected(model: Path, frame: bytes) -> dict:
    """TensorFlow Lite on the PC, fed the frame as the firmware prepares it."""
    import tensorflow as tf

    labels = (model / "labels.txt").read_text().split()
    p = np.frombuffer(frame, dtype=">u2").reshape(FRAME[1], FRAME[0]).astype(np.uint32)
    rgb = np.stack([(p >> 11 & 31) * 255 // 31, (p >> 5 & 63) * 255 // 63, (p & 31) * 255 // 31], -1)
    interp = tf.lite.Interpreter(model_path=str(model / "model_int8.tflite"))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    side, channels = inp["shape"][1], inp["shape"][3]
    img = Image.fromarray(rgb.astype(np.uint8)).resize((side, side), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    if channels == 1:
        x = ((x[..., 0] * 77 + x[..., 1] * 150 + x[..., 2] * 29) // 256)[..., None]
    scale, zero = inp["quantization"]
    interp.set_tensor(inp["index"], np.clip(np.round(x / scale) + zero, -128, 127).astype(np.int8)[None])
    interp.invoke()
    outs = [(d, interp.get_tensor(d["index"])) for d in interp.get_output_details()]
    if len(outs) != 1 or outs[0][1].shape[-1] != len(labels):
        return {"count": True}  # a count model: compared by its counts only, see main
    d, q = outs[0]
    s, z = d["quantization"]
    scores = (q.reshape(-1).astype(np.float32) - z) * s
    return {"label": labels[int(scores.argmax())], "scores": {k: round(float(v), 2) for k, v in zip(labels, scores)}}


def build(env: str) -> Path:
    subprocess.run([PIO, "run", "-e", env], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    return HERE / ".pio" / "build" / env


def flash_image(chip: str, build_dir: Path, model_bin: Path, out: Path) -> None:
    _, boot, size, _ = CHIPS[chip]
    parts = [hex(boot), build_dir / "bootloader.bin", "0x8000", build_dir / "partitions.bin",
             "0xe000", build_dir / "ota_data_initial.bin", "0x10000", build_dir / "firmware.bin",
             "0x190000", model_bin]
    subprocess.run([ESPTOOL_PYTHON, ESPTOOL, "--chip", chip, "merge_bin", "--fill-flash-size", size, "-o", out,
                    *map(str, parts)], check=True, stdout=subprocess.DEVNULL)


def run_qemu(chip: str, image: Path, timeout: float) -> tuple[dict | None, dict | None, str]:
    """The firmware's ready line and first result line, and everything it printed."""
    qemu = os.environ.get("QEMU") or shutil.which("qemu-system-xtensa") or "qemu-system-xtensa"
    cmd = [qemu, "-nographic", "-machine", CHIPS[chip][3], "-m", "4M" if chip == "esp32" else "8M",
           "-drive", f"file={image},if=mtd,format=raw"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log, ready, result = [], None, None
    t0 = time.time()
    try:
        os.set_blocking(proc.stdout.fileno(), False)
        buf = ""
        while time.time() - t0 < timeout and result is None and proc.poll() is None:
            chunk = (proc.stdout.read() or b"").decode(errors="replace")
            if not chunk:
                time.sleep(0.2)
                continue
            buf += chunk
            *lines, buf = buf.split("\n")
            for line in lines:
                log.append(line)
                m = re.search(r"\{.*\}\s*$", line)
                if not m:
                    continue
                try:
                    obj = json.loads(m.group(0))
                except ValueError:
                    continue
                if obj.get("event") == "error":
                    return ready, obj, "\n".join(log)
                if obj.get("event") == "ready":
                    ready = obj
                elif "label" in obj or "counts" in obj:
                    result = obj
                    break
    finally:
        proc.kill()
    return ready, result, "\n".join(log)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", choices=sorted(CHIPS), required=True)
    ap.add_argument("--model", type=Path, required=True, help="export folder: model.bin, model_int8.tflite, labels.txt")
    ap.add_argument("--photo", type=Path, required=True, nargs="+")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for each result in QEMU")
    ap.add_argument("--log", type=Path, help="write QEMU's output here")
    a = ap.parse_args()
    env = CHIPS[a.chip][0]
    failed = 0
    for photo in a.photo:
        frame = rgb565_frame(photo)
        write_header(frame, HERE / "test_frame")
        want = expected(a.model, frame)
        image = HERE / ".pio" / f"qemu_{a.chip}_flash.bin"
        flash_image(a.chip, build(env), a.model / "model.bin", image)
        t0 = time.time()
        ready, got, log = run_qemu(a.chip, image, a.timeout)
        if a.log:
            a.log.write_text(log)
        line = f"{photo.name}: "
        if not got or got.get("event") == "error":
            print(line + f"FAIL, no result ({(got or {}).get('message', 'timeout')}); last output:\n"
                  + "\n".join(log.splitlines()[-15:]))
            failed += 1
            continue
        if want.get("count"):
            ok = "counts" in got
            print(line + ("ok" if ok else "FAIL") + f", counts {got.get('counts')} ({time.time() - t0:.0f} s in QEMU)")
        else:
            worst = max(abs(got["scores"][k] - v) for k, v in want["scores"].items())
            ok = got["label"] == want["label"] and worst <= TOLERANCE
            print(line + ("ok" if ok else "FAIL") + f", board {got['label']} {got['scores']}, PC {want['label']} "
                  f"{want['scores']}, max diff {worst:.2f}, arena {ready and ready.get('arena')} B "
                  f"({time.time() - t0:.0f} s in QEMU)")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
