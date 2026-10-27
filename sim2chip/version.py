"""Which EyeForge made a model (user, 2026-10-09): the release, raised by hand at milestones, and the commit the
server runs, e.g. 0.9.0+1c20634 (".dirty" when it had changes not committed). Every run notes it in its job
(logs/runs.jsonl) and every kept model in its meta.json, so a result can be traced to the code that made it.
A server image gets the commit in build.json at build time; on a workstation it comes from git."""
from __future__ import annotations

import functools
import json
import subprocess
from pathlib import Path

RELEASE = "0.9.0"
BUILD = Path(__file__).with_name("build.json")  # {"commit", "dirty"}, written when building a server image (not in git)


@functools.lru_cache(maxsize=1)
def engine() -> str:
    try:
        b = json.loads(BUILD.read_text())
        commit, dirty = b["commit"], b.get("dirty", False)
    except (OSError, ValueError, KeyError):
        try:
            root = Path(__file__).parent
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                    text=True, check=True).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                        capture_output=True, text=True).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            return RELEASE
    return f"{RELEASE}+{commit}" + (".dirty" if dirty else "")
