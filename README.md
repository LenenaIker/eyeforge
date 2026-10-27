# EyeForge

**Describe what your camera should see. Get a tiny model ready to flash on a $10 chip.**

EyeForge trains small vision models that run on the device itself, from a short conversation, a handful
of the user's own photos and synthetic images rendered with NVIDIA Isaac Sim. The target hardware is part
of the spec: an ESP32-CAM or a XIAO ESP32-S3 gets an int8 TFLite Micro model (and a C array) that fits in a
microcontroller, while a phone or a PC gets a larger model for finer distinctions. The model can classify a
region of interest or count objects.

The engine and CLI are still called `sim2chip`.

Status: built for the Nebius x NVIDIA Global AI Hackathon (Apps track).

## New: synthetic data rendered in the user's own place

Until now, synthetic training images showed the objects in generic 3D places. A small model learns the
background as much as the object, so images of somewhere else teach it the wrong things: the gap between
those renders and the real photos was the main limit of synthetic data for us.

EyeForge now turns **one of the user's photos into a walkable 3D scene of that same place** (a "walk
scene"), and the training renders the objects to learn inside it:

- **The place is the user's.** The street, the wall, the light and the camera height of the photos the model
  will see in use, not a stand-in.
- **One photo, many viewpoints.** The scene is a 3D Gaussian splat the camera can move through, so the
  training gets thousands of views of the place, not crops of a single picture.
- **Objects sit where they could really be.** Each scene keeps a map of its free ground; the class's 3D
  models are placed on it, rendered in Isaac Sim with their shadows and composited by depth over the
  splat, so things in front hide them. The boxes come from what is visible.
- **Any class, any scene.** A scene is only the place: the large objects in the photo are erased before it
  is built, and the objects to learn are added at training time. The same scene serves a bin counter today
  and a bicycle detector tomorrow, and public scenes (added by the admin) are shared by every user. This is
  what keeps the pipeline general: nothing in a scene is tied to a class.
- **Made once, apart from training.** A scene takes about 2 hours and about $1 of cloud GPU, and is
  started from the dataset page (EyeForge proposes six of the user's photos; *Extract* builds one) or
  from the CLI (`sim2chip scene make`). A training run then picks ready scenes and starts at once.

How a scene is made (recipe "F3c", `experiments/worldstereo/`):

```
one photo ──► erase ──► caption ──► 360° panorama ──► WorldStereo walks ──► depth ──► splat ──► repair ──► free ground ──► bundle ──► library
             (SAM 3,    (vision      (Qwen-Image-Edit   (generated video    (Map-     (gsplat)  (NVIDIA    (grid of     (splat, sky,
              Qwen       model)       + 360 LoRA)        along camera        Anything)           Fixer,     where an      map, card)
              eraser)                                    paths, Nebius VM)                       3 rounds)  object fits)
```

Status, honestly: the whole loop runs end to end (2026-10-27: a user's photo became a scene in 1 h 59 min
for about $0.78 of VM, and a bin counter trained in it in 23 minutes; report in `docs/e2e_2026-10-27.md`).
That first test had 20 test photos with unreviewed labels, too few to show whether the user's own scene
beats a public one; the measured gain of walk scenes is still to come.

## How it works

```
interview ──► spec.json ──► label ──► ingest ──► fetch_assets ──► convert_assets ──► preview/qa_assets ──► scene_render ──► render ──► crops/points ──► train ──► evaluate ──► export
(Nemotron                   (teacher  (user      (3D models:      (GLB to USD,       (vision model        (objects in the  (keeps     (model input)  (student   (real test   (.tflite,
 3.5 Super VL)              labels    photos:    web search,      upright,           keeps or drops       walk scenes:     frames     model)    photos)      C header)
                            raw       train,     optional 3D      measured)          each model)          gsplat + Isaac)  like the
                            photos)   val, test) from photos)                                                              user's)
```

- **An interview writes the spec**: Nemotron 3.5 Super VL on Nebius Token Factory talks with the user, looks
  at a sample of their photos and writes the spec (objects, place, camera, hardware). `sim2chip serve` runs
  it in the browser.
- **The interview looks facts up** (Tavily, optional with `TAVILY_API_KEY`): the real size of an object or a
  part and its English name, instead of guesses, so the check "can this camera see it from there" uses real numbers.
- **Each model shows what it took**: the Token Factory models it called (tokens and dollars, the interview
  included), web searches and GPU minutes on Nebius AI Cloud or the local GPU.
- **A teacher labels the user's photos**, chosen by task. To classify, SigLIP 2 puts each photo with the
  class it looks closest to: the class description plus the few photos the user corrects before the first
  run, so it learns their own criterion (on a lawn test: 48% right from the descriptions, 83% after 6
  corrections). It always commits to one of the user's classes. To count, SAM 3 finds every object of
  each class and scores it; how sure it must be follows what hurts the user more (missing an object or a
  false alarm) and is tuned to the corrected photos. To follow a line or a path, Molmo2 points at its far end.
- **Synthetic data completes the user's photos, it does not replace them.** A run without photos of the
  user does not train. Synthetic frames that do not look like the user's training photos are left out
  (SigLIP 2 likeness), and a vision model drops broken 3D models before rendering.
- **The student fits the hardware**: MobileNetV2 for microcontrollers, EfficientNetV2-B0 for phones,
  PicoDet-S to count; all quantized to int8.
- **One spec drives everything**: classes, real-world sizes, scene, camera, target hardware and the
  user's photos. No stage hardcodes a class or an environment. See `specs/`.
- **Stages** read and write only inside the job folder, so each one can run on a different machine.
  A stage reruns only when the spec sections it uses or its upstream stages change; `--from X` reruns X
  and what depends on it, nothing else. What is not essential and fails (the synthetic branch, the
  evaluation, one broken 3D model) is left out with a warning, and the report and the web say so.
- **A first model is rarely the last.** On the job page, *Make it better* takes photos of what the model gets
  wrong (picked, shot, or saved with the flag button in the live test). The teacher marks them, half join
  the test set and half the training set, the old photos keep their side, and only the real-photo steps run
  again: the synthetic data is reused. *Change the setup* reopens the interview from the job's spec and
  reruns what the change affects. Each earlier model stays as a version (`versions/v<n>/`), and the new
  report scores the previous model on the same test photos (`report.json`, `previous`).
- **Backends** decide where stages run. `local` runs each stage as a subprocess with the right
  interpreter (Isaac Sim's Python for rendering, a TensorFlow environment for training). `nebius` runs
  the work on an L40S GPU in Nebius AI Cloud and brings the results back. A shared warm GPU starts when
  the user uploads photos, takes the labeling, calibration and training of several users as tasks, and
  stops 20 s after its last one (it boots a light image in about 92 s; runs with synthetic data get their
  own GPU with Isaac Sim). Walk scenes run their WorldStereo steps on a stopped Nebius VM that starts only
  for them.

Design notes: `docs/architecture.md`.

## First results

Vehicle classifier (empty, small vehicle, large vehicle, motorcycle, bicycle), MobileNetV2 0.35 at
96x96, int8, tested on real traffic-camera crops (MIO-TCD, 400 per class). This test predates the walk
scenes: the synthetic images came from generic Isaac Sim scenes.

| Training data | Real accuracy (int8) |
|---|---|
| 50 real photos per class | 0.58 |
| Synthetic only (no longer offered) | 0.55 |
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
sim2chip config         # checks the settings: typos, numbers, interpreters and folders
pytest
ruff check sim2chip tests experiments/worldstereo/wsroute experiments/worldstereo/ws_route.py
(cd web && npm install && npm run build)   # the web app (Node 20+), served by `sim2chip serve`
```

The web app is React + Vite + Tailwind in `web/` and installs as an app on phones (PWA). To work on it,
run `sim2chip serve` and, in `web/`, `npm run dev`: it serves the app with live reload and forwards `/api`
to the server. Design notes in `web/DESIGN.md`.

On a phone or tablet, a finished model can be tried live with the camera ("Try it live"): it runs in the
browser with LiteRT.js, on the device, and keeps working offline once opened.

The teachers (SigLIP 2, SAM 3) run in their own Python with a GPU: `pip install torch transformers>=5 accelerate
pillow` in another environment and set `SIM2CHIP_SAM3_PYTHON`. Its weights are gated: accept the license on
https://huggingface.co/facebook/sam3 and log in (`hf auth login`) or set `HF_TOKEN`. With the Nebius backend
and no local GPU, the teacher runs in the cloud job.

For the Nebius backend: install the `nebius` CLI and log in (a service account on a server), create an
object storage key and two MysteryBox secrets (the storage key for the jobs, `SIM2CHIP_S3_SECRET`, and the
Token Factory key, `SIM2CHIP_NEBIUS_SECRET`), and set `SIM2CHIP_NEBIUS_PROJECT`. With nothing else, each job starts from
NVIDIA's public Isaac Sim image and installs what it needs. Faster: build the worker's two images and its
Python kits (`images/build.sh`: a light image and one with Isaac Sim, chosen per run, go to a Nebius registry;
the Python environments go to the cache bucket and each run streams in only the ones it uses), set
`SIM2CHIP_NEBIUS_REGISTRY` and `SIM2CHIP_IMAGE_TAG`, and copy the teachers' weights to the cache bucket
(`scripts/upload_weights.py`), so no job needs a Hugging Face token.

Making walk scenes needs a few more Python environments (Qwen-Image-Edit, MoGe-2, MapAnything, gsplat,
3DGRUT) and the upstream checkouts in `SIM2CHIP_TOOLS`; `experiments/worldstereo/wsroute/paths.py` lists the
variables. WorldStereo runs on a Nebius VM (`experiments/worldstereo/ws_vm.py`). Training in a ready scene
needs only gsplat's environment next to Isaac Sim.

## Usage

```bash
sim2chip serve --backend nebius        # web app on http://127.0.0.1:8000: interview, photos, scenes, run, review
sim2chip interview                     # the same interview in the terminal, writes a spec

# Full pipeline; --input exposes a folder of the user's photos inside the job
sim2chip run specs/examples/pots_count.json --input maceta=/path/to/photos
sim2chip run specs/examples/pots_count.json --input maceta=/path/to/photos --backend nebius

sim2chip stages specs/examples/lora_alarm_traffic.json             # which stages are done or pending
sim2chip run specs/examples/lora_alarm_traffic.json --from train   # retrain without rendering again
sim2chip run specs/examples/lora_alarm_traffic.json --only render  # one stage (upstream must be done)

# Walk scenes (a training run uses the ones in its spec's generation.scenes)
sim2chip scene list                    # the scenes in the library, with their state
sim2chip scene make <photo id>         # build one from a library photo (about 2 h and $1 of cloud GPU)
sim2chip scene add <bundle folder>     # add a finished scene; --public shares it with every user
```

To let others use it (judges, testers), serve it in public mode: every visitor needs the access link, specs
may only use photos uploaded through the web, runs are limited in number and by a 24 h cloud budget.

```bash
sim2chip serve --backend nebius --public --host 0.0.0.0     # needs SIM2CHIP_WEB_KEY and SIM2CHIP_HOME
docker build -t eyeforge . && docker run -p 8000:8000 --env-file .env -v eyeforge-data:/data \
    -v ~/.nebius:/root/.nebius eyeforge                       # the same in a container (no GPU needed)
sim2chip accounts                      # public accounts: list, mark a judge (free use), pause
sim2chip credits                       # add dollars to an account's balance
```

The result lands in `<SIM2CHIP_HOME>/jobs/<name>/export/`: `model_int8.tflite`, `model_data.h` and
`report.json` (the score on the real test photos and credits for the 3D models used).

## Data and licenses

- 3D models come from Objaverse (Sketchfab uploads) and only CC-BY, CC-BY-SA and CC0 models are used.
  Their authors are listed in each job's `export/report.json`.
- Sky maps and textures come from Poly Haven (CC0) and NVIDIA's public Omniverse asset bucket.
- 3D from photos (optional, local only) uses TRELLIS (MIT); its GLB export depends on non-commercial code.
- The teachers are SigLIP 2 (Google, Apache 2.0), SAM 3 (Meta, SAM License: commercial use allowed; the
  license travels with the weights) and Molmo2 (Ai2, Apache 2.0). They are downloaded from Hugging Face and
  not redistributed here.
- Walk scenes use WorldStereo (code Apache 2.0, weights MIT), Qwen-Image-Edit-2511 (Apache 2.0) with the
  QWEN-EDIT_360 LoRA (MIT) and the Lightning LoRA (Apache 2.0), MoGe-2 (MIT), MapAnything (Apache 2.0
  weights), gsplat (Apache 2.0), 3DGRUT (Apache 2.0) and NVIDIA Fixer (NVIDIA Open Model License). None of
  them is redistributed here.
- Copyright (c) 2026 Iker Lenena. The code is licensed under the Apache License 2.0 (`LICENSE`): anyone
  may use, change and share it, also commercially, keeping the license and copyright notices.
- MIO-TCD is used only for evaluation and is not redistributed (non-commercial research license).
