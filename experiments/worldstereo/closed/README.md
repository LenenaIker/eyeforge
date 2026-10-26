# Closed WorldStereo experiments (frozen)

Snapshot of `experiments/worldstereo/` at commit 2317141 (2026-10-25), moved here when the route was refactored into
`../wsroute/` (driver: `../ws_route.py`). Nothing here is maintained: these scripts are kept so the closed tests in
`docs/pruebas_reconstruccion_worldstereo.md` and `docs/alternativas_agujeros_sin_datos_2026-10-25.md` can be read and
rerun as they were. Run them from the repo root, e.g. `<gsplat env>/bin/python experiments/worldstereo/closed/ws_splat.py ...`.

The live recipe (F3c) is `wsroute/`; it reproduces these scripts' outputs on photo21_np (checked step by step against
them on 2026-10-26). Closed, do not retry: meshes (ws_mesh, ws_colmap), unmasked diffusion repair (ws_genfusion,
Fixer F0/F1), far fill (ws_farfill), ground (ws_ground), the growing 3D cache (ws_cache, ws_cache_job, ws_knobs,
ws_upstream), 360 spins, consensus masks, pruning by Gaussian evidence, ws_objects, ws_diag, and the SAM 3 text-prompt
eraser (ws_nocar, replaced by wsroute/erase.py's VLM list).
