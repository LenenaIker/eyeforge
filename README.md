# Sim2Chip

**Describe what your camera should see. Get a tiny model ready to flash on a $10 chip.**

Sim2Chip trains vision models for low-power hardware such as the ESP32-CAM from a short spec, a handful
of real photos and synthetic images rendered with NVIDIA Isaac Sim. The output is an int8 TFLite model
(and a C array for TFLite Micro) that fits in a microcontroller.

Status: work in progress for the Nebius x NVIDIA Global AI Hackathon (Physical AI track).

## How it works

```
spec.json ──► ingest ──► fetch_assets ──► convert_assets ──► qa_assets ──► render ──► crops ──► train ──► evaluate ──► export
             (user      (3D models,      (USD, upright,     (keep/drop,    (Isaac Sim   (model    (student   (real test   (.tflite,
              photos)    Objaverse)       measured)          reclass)       Replicator)  input)    model)     photos)      C header)
```

- **One spec drives everything**: classes, real-world sizes, scene, camera, target hardware and the
  user's photos. No stage hardcodes a class or an environment. See `specs/lora_alarm_traffic.json`.
- **Stages** read and write only inside the job folder, so each one can run on a different machine.
  A stage reruns only when the spec sections it uses or its upstream stages change.
- **Backends** decide where stages run. `local` runs each stage as a subprocess with the right
  interpreter (Isaac Sim's Python for rendering, a TensorFlow environment for training). A Nebius
  backend (Serverless AI jobs on L40S GPUs) is planned.

Design notes: `docs/architecture.md`.

## First results

Vehicle classifier (empty, small vehicle, large vehicle, motorcycle, bicycle), MobileNetV2 0.35 at
96x96, int8, tested on real traffic-camera crops (MIO-TCD, 400 per class):

| Training data | Real accuracy (int8) |
|---|---|
| 50 real photos per class | 0.58 |
| Synthetic only | 0.55 |
| Synthetic + the same 50 real photos per class | **0.77** |

A few real photos plus synthetic data beat either one alone by about 20 points.

## Setup

Requirements: Linux, an NVIDIA RTX GPU (rendering needs RT cores), Isaac Sim 5.1, and a Python 3.11
environment with TensorFlow 2.18.

```bash
conda create -n sim2chip python=3.11
conda activate sim2chip
pip install -e ".[train,assets,dev]"
export SIM2CHIP_ISAAC_PYTHON=/path/to/isaac-sim/python.sh   # used for convert_assets and render
export SIM2CHIP_HOME=/path/with/space                        # jobs/ and cache/ (default: ./data)
pytest
```

## Usage

```bash
# Full pipeline; --input exposes a folder of the user's photos inside the job
sim2chip run specs/lora_alarm_traffic.json --input mio_tcd=/path/to/photos --input review=/path/to/review

sim2chip stages specs/lora_alarm_traffic.json        # which stages are done or pending
sim2chip run specs/lora_alarm_traffic.json --from train   # retrain without rendering again
sim2chip run specs/lora_alarm_traffic.json --only render  # one stage (upstream must be done)
```

The result lands in `<SIM2CHIP_HOME>/jobs/<name>/export/`: `model_int8.tflite`, `model_data.h` and
`report.json` (accuracy on the real test photos and credits for the 3D models used).

## Data and licenses

- 3D models come from Objaverse and only CC-BY, CC-BY-SA and CC0 models are used. Their authors are
  listed in each job's `export/report.json`.
- Sky maps come from NVIDIA's public Omniverse asset bucket.
- MIO-TCD is used only for evaluation and is not redistributed (non-commercial research license).
