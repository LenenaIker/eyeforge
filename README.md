# EyeForge

**Describe what your camera should see. Get a tiny model ready to flash on a $10 chip.**

EyeForge trains small vision models that run on the device itself, from a short conversation, a handful
of real photos and synthetic images rendered with NVIDIA Isaac Sim. The target hardware is part of the
spec: an ESP32-CAM gets an int8 TFLite Micro model (and a C array) that fits in a microcontroller, while
a Jetson or a PC gets a larger model for finer distinctions. The model can classify a region of interest
or count objects.

The engine and CLI are still called `sim2chip`.

Status: work in progress for the Nebius x NVIDIA Global AI Hackathon (Physical AI track).

## How it works

```
interview ──► spec.json ──► label ──► ingest ──► fetch_assets ──► convert_assets ──► preview/qa_assets ──► render ──► crops/points ──► train ──► evaluate ──► export
(Nemotron)                  (teacher  (user      (3D models:     (GLB to USD,       (vision model       (Isaac Sim  (model input)  (student   (real test   (.tflite,
                             labels    photos)    web search,     upright,           keeps or drops      Replicator)                model)     photos)      C header)
                             raw                  optional 3D     measured)          each model)
                             photos)              from photos)
```

- **An interview writes the spec**: Nemotron on Nebius Token Factory asks about the objects, the place,
  the camera and the hardware, and can look at the user's photos. `sim2chip serve` runs it in the browser.
- **One spec drives everything**: classes, real-world sizes, scene, camera, target hardware and the
  user's photos. No stage hardcodes a class or an environment. See `specs/`.
- **Stages** read and write only inside the job folder, so each one can run on a different machine.
  A stage reruns only when the spec sections it uses or its upstream stages change.
- **Backends** decide where stages run. `local` runs each stage as a subprocess with the right
  interpreter (Isaac Sim's Python for rendering, a TensorFlow environment for training). `nebius` runs
  the whole pipeline in one Serverless AI job on an L40S GPU in Nebius AI Cloud and brings the results
  back; the job stops when it ends.

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

Requirements: Linux and a Python 3.11 environment. Running the pipeline locally also needs an NVIDIA RTX
GPU (rendering needs RT cores) and Isaac Sim 5.1; with the Nebius backend the GPU work runs in the cloud.

```bash
conda create -n sim2chip python=3.11
conda activate sim2chip
pip install -e ".[train,assets,web,cloud,dev]"
cp .env.example .env    # then fill in the keys and paths (see the comments in it)
pytest
```

For the Nebius backend: install the `nebius` CLI and log in (a service account on a server), create an
object storage key and two MysteryBox secrets (the storage key for the jobs, `SIM2CHIP_S3_SECRET`, and the
Token Factory key, `SIM2CHIP_NEBIUS_SECRET`), and set `SIM2CHIP_NEBIUS_PROJECT`. Nothing else is installed
in advance: each job starts from NVIDIA's public Isaac Sim image.

## Usage

```bash
sim2chip serve --backend nebius        # web UI on http://127.0.0.1:8000: interview, photos, run, results
sim2chip interview                     # the same interview in the terminal, writes a spec

# Full pipeline; --input exposes a folder of the user's photos inside the job
sim2chip run specs/macetas_count.json --input maceta=/path/to/photos
sim2chip run specs/macetas_count.json --input maceta=/path/to/photos --backend nebius

sim2chip stages specs/lora_alarm_traffic.json             # which stages are done or pending
sim2chip run specs/lora_alarm_traffic.json --from train   # retrain without rendering again
sim2chip run specs/lora_alarm_traffic.json --only render  # one stage (upstream must be done)
```

The result lands in `<SIM2CHIP_HOME>/jobs/<name>/export/`: `model_int8.tflite`, `model_data.h` and
`report.json` (the score on the real test photos and credits for the 3D models used).

## Data and licenses

- 3D models come from Objaverse (Sketchfab uploads) and only CC-BY, CC-BY-SA and CC0 models are used.
  Their authors are listed in each job's `export/report.json`.
- Sky maps and textures come from Poly Haven (CC0) and NVIDIA's public Omniverse asset bucket.
- 3D from photos (optional, local only) uses TRELLIS (MIT); its GLB export depends on non-commercial code.
- The code is MIT licensed (`LICENSE`).
- MIO-TCD is used only for evaluation and is not redistributed (non-commercial research license).
