# Experiments

The product is the `sim2chip` package. This folder holds the benchmarks and tests that led to its design
decisions, kept so the numbers in the README and in the write-ups can be checked. Nothing here is imported by
the package. The lab notebook, with every finding and result by date, is `NOTES.md` (in Spanish).

## Benchmarks behind product decisions

| Folder or script | Question it answers |
|---|---|
| `interview_bench/` | Which Nemotron model, prompt and call settings run the interview best (simulated users, automatic judge) |
| `library_bench/` | Whether the chat reuses the datasets and labels a user already has |
| `teacher_bench/`, `fsc147_count.py` | Which teacher labels raw photos best, and how counting scores on FSC-147 |
| `line_follow/` | Which teacher points the way on path-following photos (Molmo2 won) |
| `picodet/`, `picodet_speed.py` | PicoDet-S configs per counting dataset, and where the count detector's training time goes |
| `phone_speed/` | How fast each student backbone runs in a phone browser |
| `road_grid/` | Two ways to train a 3x3 "walkable or not" grid |
| `data_warnings/` | Prototype of the warnings on a labeling's marks and on the student's input size |
| `vlm_bench/` | Whether a Token Factory VLM can check scene photos and renders |
| `upload_bench/` | How long a phone dataset takes to upload, and how much of it is the server |
| `nebius_parallel_probe.py` | How Nebius Serverless AI starts several Isaac Sim jobs at once |
| `e2e/` | A whole run through the web API, as a user, with a simulated user in the chat |

## Walk scenes (synthetic images in the user's own place)

`worldstereo/` is the scene route the product uses: `ws_route.py` drives it and `wsroute/` is its code
(`sim2chip/scene_job.py` runs it as a job). `worldstereo/closed/` is its frozen earlier version.

## Closed research

Routes tried before the walk scenes, kept for their results (see `NOTES.md` and `docs/`): `sphere/`,
`sphere_walk/` and `panogen/` (360 panoramas and splats), `scene3d/` (one photo to a 3D scene), `facade/`
(objects on a real building), `img2mesh/` (3D models from photos), and probes of other scene generators
(`lyra1/`, `flashworld/`, `lingbot/`). Some of them use weights with non-commercial licenses; none of those
reached the product.

`legacy/` is the first feasibility test (2026-09-23), before the package existed.
