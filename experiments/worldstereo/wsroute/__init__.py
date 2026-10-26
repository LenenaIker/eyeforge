"""The WorldStereo scene route: one photo to a walkable NuRec splat (USDZ) for Isaac Sim, recipe F3c.

Each step runs in its own Python environment (MoGe, SAM 3, Qwen-Image-Edit, MapAnything, gsplat, 3DGRUT do not share one);
experiments/worldstereo/ws_route.py is the command line and the driver. Modules:

    paths    where data, tools and each step's interpreter live (all from the environment or .env)
    config   the per-scene settings (route.json in the case folder)
    camera   cameras, panorama crops and depth helpers (upstream WorldStereo's conventions)
    erase    the photo emptied of movable objects (VLM list, SAM 3 masks, MoGe-2 distances, Qwen eraser)
    pano     the erased photo outpainted to a 360 panorama (QWEN-EDIT_360 LoRA on Qwen-Image-Edit-2511)
    walks    the panorama's depth, the walks out of it and WorldStereo's scene folder
    cloud    the WorldStereo job script, the upload and the fetch
    vm       the stopped Nebius GPU VM that runs that job
    depth    MapAnything's depth per generated leg and the cross-leg consistency
    splat    the gsplat fit of the panorama cuts and generated frames
    student  where a synthetic camera may stand
    conflict what the generated frames say about each splat pixel, and its sheets
    nbv      the next legs, aimed at what no frame backs
    fixer    NVIDIA Fixer's pseudo views, three rounds (F3)
    steps    the recipe, step by step"""
